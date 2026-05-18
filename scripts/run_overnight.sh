#!/bin/bash
# Overnight benchmark runner — SAC vs LagSAC, default + harder pose, multi-seed.
#
# Designed to run unattended for ~6-8 hours on single GPU. Logs each run to
# its own file; failures don't kill the whole script (set +e). After each
# run completes/fails, appends to summary file so progress is visible from
# any other terminal.
#
# Usage:
#   ./scripts/run_overnight.sh
# Dry-run generated schedule/commands without launching IsaacSim:
#   DRY_RUN=1 ./scripts/run_overnight.sh
# To run in background + survive terminal close:
#   nohup ./scripts/run_overnight.sh > /tmp/overnight_main.log 2>&1 &
#   disown
#
# Outputs:
#   /tmp/overnight_logs/<run_name>.log    — full stdout/stderr per run
#   /tmp/overnight_summary.txt             — append-only progress log
#   /tmp/overnight_status.json             — machine-readable status

set -uo pipefail
# Deliberately NOT set -e: we want to continue on individual run failures.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

# Paths
LOGDIR="/tmp/overnight_logs"
SUMMARY="/tmp/overnight_summary.txt"
STATUS="/tmp/overnight_status.json"
mkdir -p "$LOGDIR"

# Sandbox tmp dir for per-seed yaml copies
TMPCFG="/tmp/overnight_cfg"
mkdir -p "$TMPCFG"

START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
DRY_RUN="${DRY_RUN:-0}"

cat > "$SUMMARY" <<EOF
=== Overnight benchmark started at $START_HUMAN ===
PROJECT_ROOT=$PROJECT_ROOT
LOGDIR=$LOGDIR

Plan:
  P1-A: Stage 1 default 30ep × 4 new seeds × 2 algos (~65 min)
  P1-B: Stage 1 harder 60ep × 5 seeds × 2 algos (~150 min)
  P2-C: Stage 3 B route calibration 50ep × 3 configs (~80 min)
  Total expected: ~5 h. Budget: 8h.
  DRY_RUN=$DRY_RUN

EOF

echo '{"started_at":"'"$START_HUMAN"'","runs":[]}' > "$STATUS"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: make a per-seed yaml copy by sed-replacing seed and wandb_run_name
# Args: $1=base_yaml $2=seed $3=output_path
make_seed_yaml() {
    local base="$1" seed="$2" out="$3"
    # Replace seed field
    sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$base" > "$out"
    # Append seed suffix to wandb_run_name if not already there
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_s${seed}/" "$out"
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run a single experiment via the local runner, log it, update summary.
# Args: $1=run_name $2=yaml_path $3=tag $4=stage (1 or 3) $5=n_epochs_override (optional, "" to skip)
run_experiment() {
    local name="$1"
    local yaml="$2"
    local tag="$3"
    local stage="${4:-1}"
    local n_epochs_override="${5:-}"
    local log="$LOGDIR/${name}.log"
    local start_ts=$(date +%s)
    local start_human=$(date '+%H:%M:%S')

    # Override n_epochs in the yaml in-place if requested.
    if [ -n "$n_epochs_override" ]; then
        sed -i -E "s/^n_epochs: [0-9]+/n_epochs: $n_epochs_override/" "$yaml"
    fi

    # Per-run timeout: tightened (Stage 1 30ep ~8min, 60ep ~16min, Stage 3 50ep ~28min).
    # 给 ~1.5× 余量, 早 fail 早 unblock.
    local to="15m"
    [ "$stage" = "3" ] && to="40m"
    [ -n "$n_epochs_override" ] && [ "$n_epochs_override" -ge 60 ] && [ "$stage" = "1" ] && to="25m"

    local runner_cmd=(
        python scripts/run_lagrangian_chain_local_from_yaml.py
        --start_stage "$stage" --stop_stage "$stage"
        --stage${stage}_cfg "$yaml"
        --tag "$tag"
        --skip_snapshot
    )

    echo "[$start_human] START $name (yaml=$yaml tag=$tag stage=$stage to=$to)" | tee -a "$SUMMARY"
    echo "          CMD: ${runner_cmd[*]}" | tee -a "$SUMMARY"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[$start_human] DRY-RUN $name (not launched)" | tee -a "$SUMMARY"
        echo "" >> "$SUMMARY"
        return 0
    fi

    # 清掉上一个 run 可能遗留的 IsaacSim/python zombie 进程, 释放 GPU.
    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    # timeout -k: SIGTERM 后 60s 强制 SIGKILL (IsaacSim PhysX 卡死无视 SIGTERM 用)
    timeout -k 60s "$to" "${runner_cmd[@]}" > "$log" 2>&1
    local rc=$?

    # 二次 cleanup: 即使 timeout SIGKILL, child process 也可能 orphan, 强制清理
    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    local end_ts=$(date +%s)
    local dur=$((end_ts - start_ts))
    local end_human=$(date '+%H:%M:%S')

    local status="OK"
    [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    # Extract key metrics from log if completed
    local best_J=""
    local best_geom=""
    local final_lambda=""
    if [ -f "$log" ]; then
        best_J=$(grep -E "训练完成.*best J" "$log" | head -1 | grep -oE "best J = -?[0-9.]+" | awk '{print $4}')
        best_geom=$(grep -E "训练完成.*best[_ ]geom[_ ]?rate|训练完成.*best geom_hold_rate" "$log" | head -1 | grep -oE "best[_ ]geom[_ ]?rate[ =]+[0-9.]+" | grep -oE "[0-9.]+$" | head -1)
        final_lambda=$(grep -E "训练完成.*final λ" "$log" | head -1 | grep -oE "final λ = [0-9.]+" | awk '{print $4}')
    fi

    echo "[$end_human] END   $name dur=${dur}s status=$status bestJ=${best_J:-?} bestGeom=${best_geom:-?} finalLambda=${final_lambda:-?}" | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
}

# ──────────────────────────────────────────────────────────────────────────────
# Block P1-A: Stage 1 default pose, 4 new seeds × 2 algos
# Tag groups all into same wandb group for easy filter
TAG_DEFAULT="overnight_s1_default_$(date +%Y%m%d)"

LAG_DEFAULT_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"
SAC_DEFAULT_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Block P1-A: Stage 1 DEFAULT pose, seeds 1,2,3,42 × LagSAC + SAC" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

for seed in 1 2 3 42; do
    # LagSAC
    cfg="$TMPCFG/lag_default_s${seed}.yaml"
    make_seed_yaml "$LAG_DEFAULT_YAML" $seed "$cfg"
    run_experiment "lag_default_s${seed}" "$cfg" "$TAG_DEFAULT"

    # SAC
    cfg="$TMPCFG/sac_default_s${seed}.yaml"
    make_seed_yaml "$SAC_DEFAULT_YAML" $seed "$cfg"
    run_experiment "sac_default_s${seed}" "$cfg" "$TAG_DEFAULT"
done

# ──────────────────────────────────────────────────────────────────────────────
# Block P1-B: Stage 1 HARDER pose, 5 seeds × 2 algos
# Uses same base yamls but overrides via extra CLI arg: --default_pose_variant harder
TAG_HARDER="overnight_s1_harder_$(date +%Y%m%d)"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Block P1-B: Stage 1 HARDER pose, seeds 0,1,2,3,42 × LagSAC + SAC" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# Build per-seed yamls that also include default_pose_variant: harder
# Use sed to inject (or append) the field
make_harder_yaml() {
    local base="$1" seed="$2" out="$3"
    # Start from default yaml, set seed
    make_seed_yaml "$base" $seed "$out"
    # Inject default_pose_variant: harder (replace if exists, else append before wandb_project)
    if grep -q "^default_pose_variant:" "$out"; then
        sed -i -E "s/^default_pose_variant:.*$/default_pose_variant: harder/" "$out"
    else
        # Insert before wandb_project line
        sed -i "/^wandb_project:/i default_pose_variant: harder" "$out"
    fi
    # Append _hard to wandb_run_name
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_hard/" "$out"
}

for seed in 0 1 2 3 42; do
    # LagSAC — harder pose, bump to 60 ep (default 30 may converge too late)
    cfg="$TMPCFG/lag_harder_s${seed}.yaml"
    make_harder_yaml "$LAG_DEFAULT_YAML" $seed "$cfg"
    run_experiment "lag_harder_s${seed}" "$cfg" "$TAG_HARDER" 1 60

    # SAC — harder pose, 60 ep
    cfg="$TMPCFG/sac_harder_s${seed}.yaml"
    make_harder_yaml "$SAC_DEFAULT_YAML" $seed "$cfg"
    run_experiment "sac_harder_s${seed}" "$cfg" "$TAG_HARDER" 1 60
done

# ──────────────────────────────────────────────────────────────────────────────
# Block P2-C: Stage 3 B route calibration (default pose only — harder pose at
# Stage 3 is too risky for first calibration). 3 configs × 1 seed × 50 ep.
TAG_STAGE3="overnight_s3_calib_$(date +%Y%m%d)"

SAC_S3_YAML="$PROJECT_ROOT/conf/experiment/sac_stage3_b_route_calib.yaml"
LAG_S3_YAML="$PROJECT_ROOT/conf/experiment/lag_stage3_b_route_calib.yaml"
LAG_S3_STRICT_YAML="$PROJECT_ROOT/conf/experiment/lag_stage3_b_route_calib_strict.yaml"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Block P2-C: Stage 3 B route calibration (50ep, seed=1, default pose)" | tee -a "$SUMMARY"
echo "  3 configs: SAC monitor + LagSAC cost_lim=100 + LagSAC cost_lim=50" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# Each Stage 3 run uses its own yaml as-is (seed/run_name already baked in,
# warm-start ckpt path embedded). No per-seed copying needed.
for cfg_base in "$SAC_S3_YAML" "$LAG_S3_YAML" "$LAG_S3_STRICT_YAML"; do
    if [ ! -f "$cfg_base" ]; then
        echo "[SKIP] Stage 3 yaml missing: $cfg_base" | tee -a "$SUMMARY"
        continue
    fi
    base_name=$(basename "$cfg_base" .yaml)
    run_experiment "$base_name" "$cfg_base" "$TAG_STAGE3" 3
done

# ──────────────────────────────────────────────────────────────────────────────
# Final summary
END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

cat >> "$SUMMARY" <<EOF

============================================================
DONE at $END_HUMAN
Total duration: $((TOTAL_DUR / 60)) min ($((TOTAL_DUR / 3600))h $((TOTAL_DUR % 3600 / 60))m)
Logs in: $LOGDIR
============================================================
EOF

echo "Overnight benchmark complete."
