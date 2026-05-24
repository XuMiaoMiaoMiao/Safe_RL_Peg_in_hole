#!/bin/bash
# Continue the best Stage 1 LagSAC seed into Stage 2 and Stage 3.
#
# Selected Stage 1 seed:
#   seed=9 from final_bench_20260520
#   best_J=-51.672, best_geom=1.0, max_hold_mean=88.1, full_hold_epochs=48
#   checkpoint: results/checkpoints_lag/2026-05-20/06-53-51/best_hold.msh
#
# This script does not rerun Stage 1. It runs:
#   Stage 2 preaxis: warm-start actor from S1 best_hold, 120 ep
#   Stage 3 insert:  warm-start actor from S2 best_hold, 100 ep
#
# Safety policy:
#   Final Stage 1 used cost_limit_per_ep=10 at horizon=100, i.e. 0.1/step.
#   Keep that per-step tolerance across stages:
#     Stage 2 horizon=150 -> cost_limit=15
#     Stage 3 horizon=200 -> cost_limit=20
#
# Usage:
#   DRY_RUN=1 bash scripts/run_lagsac_s9_stage2_stage3_2026-05-20.sh
#   nohup bash scripts/run_lagsac_s9_stage2_stage3_2026-05-20.sh > /tmp/lagsac_s9_chain_main.log 2>&1 & disown
#   tail -F /tmp/lagsac_s9_chain_summary.txt

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-9}"
DATE_TAG="$(date +%Y%m%d)"
TAG="lagsac_s9_chain_${DATE_TAG}"

S1_CKPT="${S1_CKPT:-$PROJECT_ROOT/results/checkpoints_lag/2026-05-20/06-53-51/best_hold.msh}"
S2_BASE="$PROJECT_ROOT/conf/experiment/lag_stage2_preaxis_b_route.yaml"
S3_BASE="$PROJECT_ROOT/conf/experiment/lag_stage3_b_route_calib.yaml"

S2_EPOCHS="${S2_EPOCHS:-120}"
S3_EPOCHS="${S3_EPOCHS:-100}"
S2_COST_LIMIT="${S2_COST_LIMIT:-15.0}"
S3_COST_LIMIT="${S3_COST_LIMIT:-20.0}"
NOISE="${NOISE:-0.05}"

LOGDIR="/tmp/lagsac_s9_chain_logs"
SUMMARY="/tmp/lagsac_s9_chain_summary.txt"
TMPCFG="/tmp/lagsac_s9_chain_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

START_TIME="$(date +%s)"
START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"

cat > "$SUMMARY" <<EOF
=== LagSAC S9 Stage 2 -> Stage 3 chain started at $START_HUMAN ===
PROJECT_ROOT=$PROJECT_ROOT
TAG=$TAG
DRY_RUN=$DRY_RUN

Stage 1 source:
  seed=$SEED
  ckpt=$S1_CKPT

Stage 2:
  base=$S2_BASE
  n_epochs=$S2_EPOCHS
  cost_limit_per_ep=$S2_COST_LIMIT

Stage 3:
  base=$S3_BASE
  n_epochs=$S3_EPOCHS
  cost_limit_per_ep=$S3_COST_LIMIT

Shared:
  default_pose_variant=harder
  initial_joint_noise=$NOISE
  proxy_arm_radius=0.065
  proxy_ee_radius=0.04

EOF

set_yaml_scalar() {
    local file="$1" key="$2" value="$3" before_key="${4:-wandb_project}"
    if grep -q "^${key}:" "$file"; then
        sed -i -E "s|^${key}:.*$|${key}: ${value}|" "$file"
    else
        sed -i "/^${before_key}:/i ${key}: ${value}" "$file"
    fi
}

make_yaml() {
    local base="$1" out="$2" load_path="$3" stage="$4" n_epochs="$5" cost_limit="$6"
    sed -E "s|^load_agent:[[:space:]]+.*$|load_agent: $load_path|" "$base" > "$out"
    set_yaml_scalar "$out" "seed" "$SEED"
    set_yaml_scalar "$out" "n_epochs" "$n_epochs"
    set_yaml_scalar "$out" "cost_limit_per_ep" "$cost_limit"
    set_yaml_scalar "$out" "default_pose_variant" "harder"
    set_yaml_scalar "$out" "initial_joint_noise" "$NOISE"
    set_yaml_scalar "$out" "proxy_arm_radius" "0.065"
    set_yaml_scalar "$out" "proxy_ee_radius" "0.04"
    set_yaml_scalar "$out" "wandb_run_name" "lag_${stage}_from_s9_cost${cost_limit}_seed${SEED}"
    set_yaml_scalar "$out" "wandb_group" "$TAG"
}

find_newest_ckpt_after() {
    local since="$1"
    python3 - "$PROJECT_ROOT" "$since" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
since = float(sys.argv[2])
base = root / "results" / "checkpoints_lag"
candidates = []
for path in base.glob("*/*/best_hold.msh"):
    try:
        if path.stat().st_mtime >= since:
            candidates.append(path)
    except OSError:
        pass
if not candidates:
    sys.exit(1)
print(max(candidates, key=lambda p: p.stat().st_mtime))
PY
}

parse_completion() {
    local log="$1"
    grep -E "训练完成|best_geom_rate|best geom_hold_rate|final λ|checkpoint 写入" "$log" 2>/dev/null | tail -20
}

run_stage() {
    local name="$1" stage="$2" cfg="$3" timeout_min="$4"
    local log="$LOGDIR/${name}.log"
    local start_ts start_human rc end_ts end_human dur status
    start_ts="$(date +%s)"
    start_human="$(date '+%H:%M:%S')"

    local cmd=(
        python scripts/run_lagrangian_chain_local_from_yaml.py
        --start_stage "$stage" --stop_stage "$stage"
        "--stage${stage}_cfg" "$cfg"
        --tag "$TAG"
        --skip_snapshot
    )

    echo "[$start_human] START $name stage=$stage timeout=${timeout_min}m" | tee -a "$SUMMARY"
    echo "  cfg=$cfg" | tee -a "$SUMMARY"
    echo "  cmd=${cmd[*]}" >> "$SUMMARY"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[$start_human] DRY_RUN $name" | tee -a "$SUMMARY"
        echo "" >> "$SUMMARY"
        return 0
    fi

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s "${timeout_min}m" "${cmd[@]}" > "$log" 2>&1
    rc=$?

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    end_ts="$(date +%s)"
    end_human="$(date '+%H:%M:%S')"
    dur=$((end_ts - start_ts))
    status="OK"
    [ "$rc" -ne 0 ] && status="FAIL(rc=$rc)"

    echo "[$end_human] END   $name dur=${dur}s status=$status" | tee -a "$SUMMARY"
    parse_completion "$log" | sed 's/^/  /' | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
    return "$rc"
}

if [ "$DRY_RUN" != "1" ] && [ ! -f "$S1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint does not exist: $S1_CKPT" | tee -a "$SUMMARY"
    exit 1
fi

S2_CFG="$TMPCFG/lag_s2_from_s9.yaml"
S3_CFG="$TMPCFG/lag_s3_from_s9.yaml"
make_yaml "$S2_BASE" "$S2_CFG" "$S1_CKPT" "stage2" "$S2_EPOCHS" "$S2_COST_LIMIT"

echo "Generated Stage 2 cfg: $S2_CFG" | tee -a "$SUMMARY"
grep -E "^(seed|load_agent|n_epochs|cost_limit_per_ep|default_pose_variant|initial_joint_noise|proxy_arm_radius|wandb_run_name|wandb_group):" "$S2_CFG" | sed 's/^/  /' | tee -a "$SUMMARY"
echo "" >> "$SUMMARY"

if run_stage "lagsac_s9_stage2" 2 "$S2_CFG" 75; then
    if [ "$DRY_RUN" = "1" ]; then
        S2_CKPT="$PROJECT_ROOT/results/checkpoints_lag/<dry-run-stage2>/best_hold.msh"
    else
        S2_CKPT="$(find_newest_ckpt_after "$START_TIME")"
    fi
    echo "Stage 2 best_hold: $S2_CKPT" | tee -a "$SUMMARY"
else
    echo "ERROR: Stage 2 failed; skipping Stage 3." | tee -a "$SUMMARY"
    exit 1
fi

make_yaml "$S3_BASE" "$S3_CFG" "$S2_CKPT" "stage3" "$S3_EPOCHS" "$S3_COST_LIMIT"
echo "" | tee -a "$SUMMARY"
echo "Generated Stage 3 cfg: $S3_CFG" | tee -a "$SUMMARY"
grep -E "^(seed|load_agent|n_epochs|cost_limit_per_ep|default_pose_variant|initial_joint_noise|proxy_arm_radius|wandb_run_name|wandb_group):" "$S3_CFG" | sed 's/^/  /' | tee -a "$SUMMARY"
echo "" >> "$SUMMARY"

if run_stage "lagsac_s9_stage3" 3 "$S3_CFG" 110; then
    if [ "$DRY_RUN" = "1" ]; then
        S3_CKPT="$PROJECT_ROOT/results/checkpoints_lag/<dry-run-stage3>/best_hold.msh"
    else
        S3_CKPT="$(find_newest_ckpt_after "$START_TIME")"
    fi
    echo "Stage 3 best_hold: $S3_CKPT" | tee -a "$SUMMARY"
else
    echo "ERROR: Stage 3 failed." | tee -a "$SUMMARY"
    exit 1
fi

END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
END_TIME="$(date +%s)"
DUR=$((END_TIME - START_TIME))
{
    echo ""
    echo "============================================================"
    echo "=== LagSAC S9 chain DONE at $END_HUMAN ==="
    echo "Total: $((DUR / 60)) min"
    echo "Stage 1 ckpt: $S1_CKPT"
    echo "Stage 2 ckpt: $S2_CKPT"
    echo "Stage 3 ckpt: $S3_CKPT"
    echo "============================================================"
} | tee -a "$SUMMARY"
