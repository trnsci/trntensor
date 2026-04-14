#!/usr/bin/env bash
#
# Run the NKI simulator pytest suite on the trntensor CI instance.
#
# Unlike run_neuron_tests.sh, this routes kernel dispatch through
# `nki.simulate(kernel)(numpy_args)` on the instance's CPU — no NEFF
# compile, no XLA, no Tensor Engine time. Correctness gate only.
#
# The simulator suite also runs on ubuntu-latest in GH Actions
# (`nki-simulator` job in .github/workflows/ci.yml). This script
# exists for parity with the hardware runner and for debugging under
# the same DLAMI environment.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_simulator_tests.sh [instance_type]

set -euo pipefail

WARM=0
if [[ "${1:-}" == "--warm" ]]; then
  WARM=1
  shift
fi

INSTANCE_TYPE="${1:-trn1}"
TAG="trntensor-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-west-2}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_neuron_tests.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  echo "Provision with: cd infra/terraform && terraform apply" >&2
  exit 1
fi

echo "Instance: $INSTANCE_ID"

cleanup() {
  local exit_code=$?
  echo ""
  echo "Stopping $INSTANCE_ID..."
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  exit "$exit_code"
}
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)

if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi

echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
# `aws ssm wait instance-information` isn't available in all CLI versions —
# poll describe-instance-information instead.
for _ in $(seq 1 60); do
  PING=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$PING" == "Online" ]] && break
  sleep 5
done
if [[ "$PING" != "Online" ]]; then
  echo "ERROR: SSM agent not Online after 5 minutes (last PingStatus=$PING)" >&2
  exit 1
fi

# --warm: run the suite twice to expose the NEFF cache delta — the second
# pass gets warm /var/tmp/neuron-compile-cache/. -s surfaces the perf
# prints from TestPerformance.
PYTEST_INVOCATION="TRNTENSOR_USE_SIMULATOR=1 \$NEURON_VENV/bin/pytest /home/ubuntu/trntensor/tests/ -v -m nki_simulator --tb=short"

echo "Sending simulator test command (SHA=$SHA)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trntensor simulator tests @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trntensor && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && NEURON_VENV=\$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1) && sudo -u ubuntu \$NEURON_VENV/bin/pip install -e /home/ubuntu/trntensor[dev] --quiet && sudo -u ubuntu bash -c \\\"$PYTEST_INVOCATION\\\"'\"
  ]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (this may take several minutes)..."

# Poll for terminal status. `aws ssm wait command-executed` maxes at
# ~100 seconds (20 * 5s), which is too short — kernel compilation + test
# runs routinely take 5+ minutes.
STATUS="InProgress"
for _ in $(seq 1 80); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 15
done

echo ""
echo "=== STDOUT ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardOutputContent' --output text

echo ""
echo "=== STDERR ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardErrorContent' --output text

echo ""
echo "=== Status: $STATUS ==="

[[ "$STATUS" == "Success" ]]
