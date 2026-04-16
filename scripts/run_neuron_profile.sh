#!/usr/bin/env bash
#
# Capture a Neuron profiler trace of trntensor matmul/bmm kernels on the
# trntensor CI instance via SSM, using Neuron Profiler 2.0 (Neuron 2.29+).
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --probe            # Phase A: tool discovery
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh                    # full capture, matmul
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --kernel bmm       # batched matmul
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --shape large      # 2048×2048
#
# Context (#33):
#   Per-dispatch overhead in nki_matmul dominates at chemistry sizes (<1 GFLOPs):
#   each call pays ~1ms XLA/NEFF launch cost regardless of kernel work. The
#   profiler output tells us where that time actually goes:
#     - Is it `_to_xla` (host→device transfer)?
#     - Is it graph compile?
#     - Is it NEFF dispatch latency?
#   Those answers determine whether the fix is: (a) lower `_MIN_NKI_FLOPS`
#   threshold, (b) persistent XLA residency, or (c) torch_xla.compile fusion.
#
# See also: scripts/autotune_dispatch.py for threshold calibration.
#
# Adapted from trnblas/scripts/run_neuron_profile.sh (same SSM + double-base64
# pattern). Instance tag: trntensor-ci-trn1 (per trnsci naming convention).

set -euo pipefail

PROBE=false
KERNEL="matmul"
SHAPE="medium"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)
      PROBE=true
      shift
      ;;
    --kernel)
      KERNEL="$2"
      shift 2
      ;;
    --shape)
      SHAPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

case "$SHAPE" in
  small)  SHAPE_MKN="256 256 256" ;;
  medium) SHAPE_MKN="1024 1024 1024" ;;
  large)  SHAPE_MKN="2048 2048 2048" ;;
  *)
    echo "Unknown --shape '$SHAPE'; use small|medium|large" >&2
    exit 1
    ;;
esac

TAG="trntensor-ci-trn1"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_neuron_profile.sh}"

NP="/opt/aws/neuron/bin/neuron-profile"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
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

if [[ "$STATE" == "stopping" ]]; then
  echo "Instance is stopping — waiting for it to reach stopped..."
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
  STATE=stopped
fi

if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi

echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
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

# ---------------------------------------------------------------------------
# Phase A — probe: discover API surface, confirm profiler version
# ---------------------------------------------------------------------------
if [[ "$PROBE" == "true" ]]; then
  echo "Running Phase A probe (SHA=$SHA)..."
  PROBE_SCRIPT=$(cat <<'PROBE_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
printf '%s\n' ==NP_VERSION==
$NP --version 2>&1 || true
printf '%s\n' ==NP_CAPTURE_HELP==
$NP capture --help 2>&1 || true
printf '%s\n' ==NP_VIEW_FORMATS==
$NP view --help 2>&1 | grep -i "output.format" || echo "NOT FOUND"
printf '%s\n' ==NEFF_CACHE==
find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -10 || echo none
printf '%s\n' ==OLD_PROFILES==
find /home/ubuntu/profiles -type f 2>/dev/null | head -20 || echo none
PROBE_EOF
)
  B64=$(printf '%s' "$PROBE_SCRIPT" | base64 | tr -d '\n')

  CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "trntensor neuron-profile probe @ $SHA" \
    --parameters "commands=[\"printf '%s' $B64 | base64 -d | bash\"]" \
    --region "$REGION" \
    --output text --query 'Command.CommandId')

  echo "Command ID: $CMD_ID"
  STATUS=InProgress
  for _ in $(seq 1 30); do
    STATUS=$(aws ssm get-command-invocation \
      --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
      --region "$REGION" --query 'Status' --output text 2>/dev/null || echo "InProgress")
    [[ "$STATUS" != "InProgress" && "$STATUS" != "Pending" ]] && break
    sleep 10
  done

  echo ""
  echo "=== PROBE STDOUT ==="
  aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardOutputContent' --output text
  echo ""
  echo "=== PROBE STDERR ==="
  aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text
  echo ""
  echo "=== Status: $STATUS ==="
  [[ "$STATUS" == "Success" ]]
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase B/C — full capture using Neuron Profiler 2.0
#
# Double-base64 encoding (same technique as trnblas):
#   1. Python warmup script → PY_B64
#   2. Bash capture body with PY_B64 embedded → B64
#   3. SSM sends: printf '%s' B64 | base64 -d | bash
# ---------------------------------------------------------------------------
echo "Building capture command (SHA=$SHA, kernel=$KERNEL, shape=$SHAPE)..."

if [[ "$KERNEL" == "bmm" ]]; then
  read -r M K N <<< "$SHAPE_MKN"
  PY_WARMUP=$(cat <<PYEOF
import sys
sys.path.insert(0, '/home/ubuntu/trntensor')
import torch, os
os.environ['TRNTENSOR_REQUIRE_NKI'] = '1'
import trntensor
trntensor.set_backend('nki')
M, K, N = $M, $K, $N
BSZ = 4
A = torch.randn(BSZ, M, K)
B = torch.randn(BSZ, K, N)
print(f"Compiling batched_matmul_kernel: ({BSZ},{M},{K}) x ({BSZ},{K},{N})", flush=True)
result = trntensor.einsum("bij,bjk->bik", A, B)
print(f"Done. result.shape={tuple(result.shape)}", flush=True)
PYEOF
)
else
  read -r M K N <<< "$SHAPE_MKN"
  PY_WARMUP=$(cat <<PYEOF
import sys
sys.path.insert(0, '/home/ubuntu/trntensor')
import torch, os
os.environ['TRNTENSOR_REQUIRE_NKI'] = '1'
import trntensor
trntensor.set_backend('nki')
M, K, N = $M, $K, $N
A = torch.randn(M, K)
B = torch.randn(K, N)
print(f"Compiling matmul_kernel: ({M},{K}) x ({K},{N})", flush=True)
result = trntensor.einsum("ij,jk->ik", A, B)
print(f"Done. result.shape={tuple(result.shape)}", flush=True)
PYEOF
)
fi

PY_B64=$(printf '%s' "$PY_WARMUP" | base64 | tr -d '\n')

CAPTURE_BODY=$(cat <<'CAPTURE_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1)
test -n "$NEURON_VENV" || { echo "ERROR: no Neuron venv" >&2; exit 1; }
PYTHON="$NEURON_VENV/bin/python"

cd /home/ubuntu
sudo -u ubuntu git -C /home/ubuntu/trntensor fetch --all --quiet
sudo -u ubuntu git -C /home/ubuntu/trntensor checkout __SHA__
sudo -u ubuntu env PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  "$PYTHON" -m pip install -e /home/ubuntu/trntensor --quiet

PROFILE_DIR=/home/ubuntu/profiles/trntensor-run-$(date +%s)
sudo -u ubuntu mkdir -p "$PROFILE_DIR"
chown -R ubuntu:ubuntu /home/ubuntu/profiles

printf '%s\n' ==STEP1_WRITE_WARMUP==
printf '%s' __PY_B64__ | base64 -d > /tmp/trntensor_warmup.py
chown ubuntu:ubuntu /tmp/trntensor_warmup.py
echo "Warmup script written."

printf '%s\n' ==STEP2_CLEAR_CACHE_AND_COMPILE==
rm -rf /var/tmp/neuron-compile-cache/* 2>/dev/null || true
echo "Compile cache cleared."
sudo -u ubuntu env \
  PATH="$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin" \
  "$PYTHON" /tmp/trntensor_warmup.py 2>&1

printf '%s\n' ==STEP3_FIND_NEFF==
NEFF=$(find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -1)
test -n "$NEFF" || { echo "ERROR: no model.neff after warmup" >&2; exit 1; }
echo "NEFF: $NEFF"
ls -lah "$NEFF"

printf '%s\n' ==STEP4_CAPTURE==
sudo -u ubuntu HOME=/home/ubuntu "$NP" capture \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" 2>&1

printf '%s\n' ==STEP5_SUMMARY_TEXT==
sudo -u ubuntu HOME=/home/ubuntu "$NP" view \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" \
  --output-format summary-text 2>&1

printf '%s\n' ==STEP6_SUMMARY_JSON==
sudo -u ubuntu HOME=/home/ubuntu "$NP" view \
  -n "$NEFF" -s "$PROFILE_DIR/profile.ntff" \
  --output-format summary-json 2>&1 | head -300

printf '%s\n' ==ARTIFACTS==
ls -laR "$PROFILE_DIR" 2>&1 | head -40
CAPTURE_EOF
)

CAPTURE_BODY="${CAPTURE_BODY//__SHA__/$SHA}"
CAPTURE_BODY="${CAPTURE_BODY//__PY_B64__/$PY_B64}"

B64=$(printf '%s' "$CAPTURE_BODY" | base64 | tr -d '\n')

echo "Sending capture command (SHA=$SHA)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trntensor neuron-profile 2.0 @ $SHA ($KERNEL $SHAPE)" \
  --parameters "commands=[\"printf '%s' $B64 | base64 -d | bash\"]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command (poll 30s, up to 60min)..."

STATUS=InProgress
for _ in $(seq 1 120); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'Status' --output text 2>/dev/null || echo "InProgress")
  [[ "$STATUS" != "InProgress" && "$STATUS" != "Pending" ]] && break
  sleep 30
done

echo ""
echo "=== STDOUT ==="
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text

echo ""
echo "=== STDERR ==="
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardErrorContent' --output text

echo ""
echo "=== Status: $STATUS ==="
[[ "$STATUS" == "Success" ]]
