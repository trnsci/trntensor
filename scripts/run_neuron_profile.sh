#!/usr/bin/env bash
#
# Capture Neuron Profiler 2.0 traces of trntensor NKI kernels on the
# trntensor CI instance via SSM. Adapted from trnfft/scripts/run_neuron_profile.sh.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --probe
#     Phase A: confirm neuron-profile 2.0 API, list cached NEFFs
#
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --kernel matmul
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --kernel bmm
#     Phase B: engine-utilization + HBM bandwidth for one kernel
#
#   AWS_PROFILE=aws ./scripts/run_neuron_profile.sh --dispatch-timing
#     Phase C: wall-clock decomposition of nki_matmul overhead at
#     multiple (M, K, N) sizes.  Answers #33: where does the ~1 ms per-call
#     cost live — _to_xla, kernel dispatch, or .to(cpu)?
#
# Output:
#   summary-text: per-engine utilization % to stdout
#   summary-json: raw metrics to stdout (head 200 lines)
#   profiles saved on instance: /home/ubuntu/profiles/trntensor-<kernel>-<sha>/
#
# ## Neuron Profiler 2.0 API (Neuron SDK 2.29)
#   neuron-profile capture -n <model.neff> -s <trace.ntff>
#   neuron-profile view   -n <model.neff> -s <trace.ntff> --output-format summary-text
#
# Double-base64 encoding bypasses all shell-quoting/heredoc-in-pipe-to-bash
# issues when sending scripts via SSM (same strategy as trnblas / trnfft).

set -euo pipefail

KERNEL=""
PROBE=false
DISPATCH_TIMING=false
TAG_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)             PROBE=true;               shift ;;
    --kernel)            KERNEL="$2";              shift 2 ;;
    --dispatch-timing)   DISPATCH_TIMING=true;     shift ;;
    --instance-tag)      TAG_OVERRIDE="$2";        shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

TAG="${TAG_OVERRIDE:-trntensor-ci-trn1}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"
NP="/opt/aws/neuron/bin/neuron-profile"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_neuron_profile.sh}"

# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------
echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  echo "  trntensor-ci-trn1 can be provisioned via: cd infra/terraform && terraform apply" >&2
  echo "  To run on another instance: --instance-tag trnblas-ci-trn1" >&2
  exit 1
fi
echo "Instance: $INSTANCE_ID"

cleanup() {
  local ec=$?
  echo ""
  echo "Stopping $INSTANCE_ID..."
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  exit "$ec"
}
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
if [[ "$STATE" == "stopping" ]]; then
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"; STATE=stopped
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
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$PING" == "Online" ]] && break
  sleep 5
done
[[ "$PING" == "Online" ]] || { echo "ERROR: SSM agent not Online after 5 minutes" >&2; exit 1; }

# ---------------------------------------------------------------------------
# SSM helper: send encoded body, poll, print output
# ---------------------------------------------------------------------------
_run_ssm() {
  local comment="$1" body_b64="$2" wait_iters="${3:-60}" poll_interval="${4:-10}"
  local cmd_id status
  cmd_id=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "$comment" \
    --parameters "commands=[\"printf '%s' $body_b64 | base64 -d | bash\"]" \
    --region "$REGION" \
    --output text --query 'Command.CommandId')
  echo "Command ID: $cmd_id"
  status=InProgress
  for _ in $(seq 1 "$wait_iters"); do
    status=$(aws ssm get-command-invocation \
      --command-id "$cmd_id" --instance-id "$INSTANCE_ID" --region "$REGION" \
      --query 'Status' --output text 2>/dev/null || echo "InProgress")
    [[ "$status" != "InProgress" && "$status" != "Pending" ]] && break
    sleep "$poll_interval"
  done
  echo ""
  echo "=== STDOUT ==="
  aws ssm get-command-invocation --command-id "$cmd_id" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardOutputContent' --output text
  echo ""
  echo "=== STDERR ==="
  aws ssm get-command-invocation --command-id "$cmd_id" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text
  echo ""
  echo "=== Status: $status ==="
  [[ "$status" == "Success" ]]
}

# ---------------------------------------------------------------------------
# Build a parameterized SSM capture body (same pattern as trnfft)
# ---------------------------------------------------------------------------
_build_capture_body() {
  local kernel_name="$1" py_b64="$2"
  local body
  body=$(cat <<'BODY_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1)
test -n "$NEURON_VENV" || { echo "ERROR: no Neuron venv" >&2; exit 1; }
PYTHON="$NEURON_VENV/bin/python"

cd /home/ubuntu
if [ ! -d /home/ubuntu/trntensor/.git ]; then
  sudo -u ubuntu git clone https://github.com/trnsci/trntensor.git /home/ubuntu/trntensor
fi
sudo -u ubuntu git -C /home/ubuntu/trntensor fetch --all --quiet
sudo -u ubuntu git -C /home/ubuntu/trntensor checkout __SHA__
sudo -u ubuntu env PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  "$PYTHON" -m pip install -e /home/ubuntu/trntensor --quiet

KNAME=__KERNEL_NAME__
PROFILE_DIR=/home/ubuntu/profiles/trntensor-${KNAME}-$(date +%s)
sudo -u ubuntu mkdir -p "$PROFILE_DIR"
chown -R ubuntu:ubuntu /home/ubuntu/profiles

printf '%s\n' ==STEP1_WRITE_WARMUP==
printf '%s' __PY_B64__ | base64 -d > /tmp/trntensor_warmup_${KNAME}.py
chown ubuntu:ubuntu /tmp/trntensor_warmup_${KNAME}.py
echo "Warmup script written."

printf '%s\n' ==STEP2_CLEAR_CACHE_AND_COMPILE==
rm -rf /var/tmp/neuron-compile-cache/* 2>/dev/null || true
sudo -u ubuntu env \
  PATH="$NEURON_VENV/bin:/opt/aws/neuron/bin:/usr/bin:/bin" \
  TRNTENSOR_REQUIRE_NKI=1 \
  "$PYTHON" /tmp/trntensor_warmup_${KNAME}.py 2>&1

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
  --output-format summary-json 2>&1 | head -200

printf '%s\n' ==ARTIFACTS==
ls -laR "$PROFILE_DIR" 2>&1 | head -20
BODY_EOF
)
  body="${body//__SHA__/$SHA}"
  body="${body//__KERNEL_NAME__/$kernel_name}"
  body="${body//__PY_B64__/$py_b64}"
  printf '%s' "$body"
}

# ---------------------------------------------------------------------------
# Phase A — probe
# ---------------------------------------------------------------------------
if [[ "$PROBE" == "true" ]]; then
  echo "Running Phase A probe (SHA=$SHA)..."
  PROBE_BODY=$(cat <<'PROBE_EOF'
set -euo pipefail
NP=/opt/aws/neuron/bin/neuron-profile
printf '%s\n' ==NP_VERSION==
$NP --version 2>&1 || true
printf '%s\n' ==CAPTURE_HELP==
$NP capture --help 2>&1 | head -30 || true
printf '%s\n' ==VIEW_OUTPUT_FORMAT==
$NP view --help 2>&1 | grep -i "output.format" || echo "NOT FOUND"
printf '%s\n' ==NEFF_CACHE==
find /var/tmp/neuron-compile-cache -name model.neff 2>/dev/null | head -10 || echo none
printf '%s\n' ==OLD_PROFILES==
find /home/ubuntu/profiles -type f 2>/dev/null | head -20 || echo none
PROBE_EOF
)
  B64=$(printf '%s' "$PROBE_BODY" | base64 | tr -d '\n')
  _run_ssm "trntensor neuron-profile probe @ $SHA" "$B64" 30 5
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase B — profile a specific kernel
# ---------------------------------------------------------------------------
_profile_kernel() {
  local name="$1"
  echo ""
  echo "========================================"
  echo "Profiling kernel: $name (SHA=$SHA)"
  echo "========================================"

  case "$name" in
    matmul)
      PY_WARMUP=$(cat <<'PYEOF'
import sys, torch, os
sys.path.insert(0, '/home/ubuntu/trntensor')
os.environ['TRNTENSOR_REQUIRE_NKI'] = '1'
import trntensor
trntensor.set_backend('nki')
M, K, N = 1024, 1024, 1024
A = torch.randn(M, K)
B = torch.randn(K, N)
print(f"Compiling matmul_kernel: ({M},{K}) x ({K},{N})", flush=True)
result = trntensor.einsum("ij,jk->ik", A, B)
print(f"Done. shape={tuple(result.shape)}", flush=True)
PYEOF
)
      ;;
    bmm)
      PY_WARMUP=$(cat <<'PYEOF'
import sys, torch, os
sys.path.insert(0, '/home/ubuntu/trntensor')
os.environ['TRNTENSOR_REQUIRE_NKI'] = '1'
import trntensor
trntensor.set_backend('nki')
BSZ, M, K, N = 4, 512, 512, 512
A = torch.randn(BSZ, M, K)
B = torch.randn(BSZ, K, N)
print(f"Compiling batched_matmul_kernel: ({BSZ},{M},{K}) x ({BSZ},{K},{N})", flush=True)
result = trntensor.einsum("bij,bjk->bik", A, B)
print(f"Done. shape={tuple(result.shape)}", flush=True)
PYEOF
)
      ;;
    *)
      echo "ERROR: unknown kernel '$name' (choose: matmul, bmm)" >&2
      exit 1
      ;;
  esac

  PY_B64=$(printf '%s' "$PY_WARMUP" | base64 | tr -d '\n')
  CAPTURE_BODY=$(_build_capture_body "$name" "$PY_B64")
  B64=$(printf '%s' "$CAPTURE_BODY" | base64 | tr -d '\n')
  _run_ssm "trntensor neuron-profile $name @ $SHA" "$B64" 120 30
}

# ---------------------------------------------------------------------------
# Phase C — dispatch overhead timing (#33)
#
# Measures wall-clock time of each sub-step in nki_matmul:
#   (a) _to_xla  — host→device transfer
#   (b) kernel   — matmul_kernel dispatch
#   (c) to_cpu   — device→host transfer
#   (d) total    — sum of all three (matches nki_matmul's actual cost)
#
# Run at a range of (M,K,N) sizes to find:
#   1. Which step dominates at small sizes (likely transfer or launch overhead)
#   2. The crossover point where NKI total beats torch.matmul
#   3. Whether persistent XLA residency (to_xla / from_xla) removes the
#      dominant overhead — tested by a second sweep with pre-pinned tensors
# ---------------------------------------------------------------------------
if [[ "$DISPATCH_TIMING" == "true" ]]; then
  echo "Running Phase C — dispatch overhead timing (SHA=$SHA)..."

  PY_TIMING=$(cat <<'PYEOF'
"""
Per-step timing of nki_matmul dispatch overhead (#33).

Measures: _to_xla, kernel, .to(cpu), total nki_matmul vs torch.matmul
at square sizes from 128×128 to 2048×2048.

Second sweep: tensors pre-pinned via to_xla() — measures the
cost after host→device transfer has been paid once.
"""
import sys, os, time, torch
sys.path.insert(0, '/home/ubuntu/trntensor')
os.environ['TRNTENSOR_REQUIRE_NKI'] = '1'

import trntensor
from trntensor.nki.dispatch import _to_xla, to_xla, from_xla
from trntensor.nki._kernels import matmul_kernel, TILE_M, TILE_K, TILE_N

trntensor.set_backend('nki')

SIZES = [128, 256, 512, 768, 1024, 1536, 2048]
WARMS = 3
REPS  = 10

def sync_and_time(fn, reps):
    fn()  # warmup
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1e3  # ms

def roundup(n, m): return ((n + m - 1) // m) * m

print("=" * 95)
print(f"{'Size':>6}  {'FLOPs':>12}  {'PT ms':>8}  {'to_xla ms':>10}  {'kern ms':>8}  {'to_cpu ms':>10}  {'nki_tot ms':>11}  {'winner':>7}")
print("-" * 95)

results = []
for S in SIZES:
    M = K = N = S
    flops = 2 * M * K * N

    A = torch.randn(M, K)
    B = torch.randn(K, N)

    # Pad to tile boundaries
    Mp = roundup(M, TILE_M); Kp = roundup(K, TILE_K)
    Np = N if N <= TILE_N else roundup(N, TILE_N)
    A_p = torch.zeros(Mp, Kp); A_p[:M, :K] = A; A_f = A_p.contiguous()
    B_p = torch.zeros(Kp, Np); B_p[:K, :N] = B; B_f = B_p.contiguous()

    # (0) PyTorch baseline
    pt_ms = sync_and_time(lambda: torch.matmul(A, B), REPS)

    # Compile kernel once (cold)
    (a_x, b_x), orig = _to_xla(A_f, B_f)
    _ = matmul_kernel(a_x, b_x).to(orig)

    # (a) _to_xla step alone
    def step_to_xla():
        (a_x2, b_x2), _ = _to_xla(A_f.clone(), B_f.clone())
        return a_x2, b_x2

    (a_x, b_x), orig = _to_xla(A_f, B_f)

    t_toxla = sync_and_time(step_to_xla, REPS)

    # (b) kernel step alone (tensors already on XLA)
    t_kern = sync_and_time(lambda: matmul_kernel(a_x, b_x), REPS)

    # (c) to_cpu step alone
    c_x = matmul_kernel(a_x, b_x)
    t_tocpu = sync_and_time(lambda: c_x.to(orig), REPS)

    # (d) total nki_matmul (includes transfer + kernel + sync)
    t_nki = sync_and_time(lambda: trntensor.einsum("ij,jk->ik", A, B), REPS)

    winner = "NKI" if t_nki < pt_ms else "PyTorch"
    results.append((S, flops, pt_ms, t_toxla, t_kern, t_tocpu, t_nki, winner))

    print(f"{S:>6}  {flops:>12,}  {pt_ms:>8.3f}  {t_toxla:>10.3f}  {t_kern:>8.3f}  "
          f"{t_tocpu:>10.3f}  {t_nki:>11.3f}  {winner:>7}")

print()
print("=" * 95)
print("Pre-pinned sweep (to_xla applied once, kernel + to_cpu only):")
print("-" * 95)
print(f"{'Size':>6}  {'FLOPs':>12}  {'PT ms':>8}  {'kern ms':>8}  {'to_cpu ms':>10}  {'pinned_tot ms':>13}  {'winner':>7}")
print("-" * 95)

for S in SIZES:
    M = K = N = S
    flops = 2 * M * K * N
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    Mp = roundup(M, TILE_M); Kp = roundup(K, TILE_K)
    Np = N if N <= TILE_N else roundup(N, TILE_N)
    A_p = torch.zeros(Mp, Kp); A_p[:M, :K] = A; A_f = A_p.contiguous()
    B_p = torch.zeros(Kp, Np); B_p[:K, :N] = B; B_f = B_p.contiguous()

    pt_ms = sync_and_time(lambda: torch.matmul(A, B), REPS)

    # Pre-pin both operands to XLA
    a_x = to_xla(A_f)
    b_x = to_xla(B_f)

    # Kernel only
    t_kern = sync_and_time(lambda: matmul_kernel(a_x, b_x), REPS)

    # to_cpu only
    c_x = matmul_kernel(a_x, b_x)
    t_tocpu = sync_and_time(lambda: c_x.to('cpu'), REPS)

    t_pinned = t_kern + t_tocpu

    winner = "NKI" if t_pinned < pt_ms else "PyTorch"
    print(f"{S:>6}  {flops:>12,}  {pt_ms:>8.3f}  {t_kern:>8.3f}  "
          f"{t_tocpu:>10.3f}  {t_pinned:>13.3f}  {winner:>7}")

print()
print("Done.")
PYEOF
)

  PY_B64=$(printf '%s' "$PY_TIMING" | base64 | tr -d '\n')
  TIMING_BODY=$(cat <<'TEOF'
set -euo pipefail
NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1)
test -n "$NEURON_VENV" || { echo "ERROR: no Neuron venv" >&2; exit 1; }
PYTHON="$NEURON_VENV/bin/python"
cd /home/ubuntu
if [ ! -d /home/ubuntu/trntensor/.git ]; then
  sudo -u ubuntu git clone https://github.com/trnsci/trntensor.git /home/ubuntu/trntensor
fi
sudo -u ubuntu git -C /home/ubuntu/trntensor fetch --all --quiet
sudo -u ubuntu git -C /home/ubuntu/trntensor checkout __SHA__
sudo -u ubuntu env PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  "$PYTHON" -m pip install -e /home/ubuntu/trntensor --quiet
printf '%s' __PY_B64__ | base64 -d > /tmp/trntensor_dispatch_timing.py
chown ubuntu:ubuntu /tmp/trntensor_dispatch_timing.py
sudo -u ubuntu env \
  PATH="$NEURON_VENV/bin:/usr/bin:/bin" \
  TRNTENSOR_REQUIRE_NKI=1 \
  "$PYTHON" /tmp/trntensor_dispatch_timing.py 2>&1
TEOF
)
  TIMING_BODY="${TIMING_BODY//__SHA__/$SHA}"
  TIMING_BODY="${TIMING_BODY//__PY_B64__/$PY_B64}"
  B64=$(printf '%s' "$TIMING_BODY" | base64 | tr -d '\n')
  _run_ssm "trntensor dispatch-timing @ $SHA" "$B64" 60 15
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase B — profile specified kernel; default: both
# ---------------------------------------------------------------------------
if [[ -n "$KERNEL" ]]; then
  _profile_kernel "$KERNEL"
else
  echo "Profiling all kernels (matmul, bmm)..."
  _profile_kernel matmul
  _profile_kernel bmm
fi
