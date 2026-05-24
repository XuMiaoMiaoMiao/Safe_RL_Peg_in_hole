#!/bin/bash
# Stage 1 reward calibration — verify geom_d_sat=0.30 → 1.0 hypothesis.
#
# 假设 (from tonight Block 1 analysis):
#   harder pose initial d_err = 0.57-0.88m, 超过当前 geom_d_sat=0.30m.
#   r_geom_d 在 d_err > 0.30 时饱和 (= -8 × 0.30 = -2.4 常数), Q-critic 学
#   不到 "depth 越靠近 target 越好" 的 dense signal. 失败 seed (Mode 1, 4/6)
#   d_err 永远没掉到 sat 边界以下, 卡在 0.4-0.9m.
#
# Calibration 设计 — 只改一个变量, 干净 ablation:
#   geom_d_sat: 0.30 → 1.0    (aggressive variant, max penalty -8/step)
#   保持其他全部与昨晚 Block 1 一致 (harder pose, noise=0.05, 60ep, lag b_route).
#
# Round 1 (本脚本):
#   LagSAC × {0, 1, 2, 6, 7, 8, 9}     s0 sanity + 6 failed
#   SAC    × {0, 1, 2, 6, 7, 8, 9}     同 seed 对照
#   = 14 runs × ~19 min ≈ 4.4 h
#
# 多指标判断 (analyze_calibration.py 输出):
#   - d_err_min ever < 0.30 ?            (saturation barrier crossed)
#   - first_hold_epoch (geom_hold_rate > 0)
#   - best_geom_rate, best_J
#   - eval_ep_cost, cum_collision_sphere
#   - sanity s0 是否仍成功 (不 break previously-working)
#
# Round 2 (defer): 若 Mode 1 救回但 hold 不稳, 加 rew_geom_soft_success=2.0;
#                  若 d 项 dominate 让 radial 失控, 试 scale-preserving rew_geom_d=3.0.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/calib_logs"
SUMMARY="/tmp/calib_summary.txt"
TMPCFG="/tmp/calib_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

DRY_RUN="${DRY_RUN:-0}"
DATE_TAG=$(date +%Y%m%d)
DATESTR=$(date +%Y-%m-%d)
TONIGHT_NEXT=$(date -d "tomorrow" +%Y-%m-%d)

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"
SAC_S1_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"

# Sanity (s0 was ✓ in tonight's run) + 6 failed seeds (s1,s2,s6,s7,s8,s9)
SEEDS=(0 1 2 6 7 8 9)
TAG="calib_s1_dsat1p0_${DATE_TAG}"
HARDER_NOISE=0.05
HARDER_EPOCHS=60
NEW_D_SAT=1.0
RUN_TIMEOUT=30m

START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$SUMMARY" <<EOF
=== Stage 1 calibration started at $START_HUMAN ===
PROJECT_ROOT=$PROJECT_ROOT
LOGDIR=$LOGDIR

Hypothesis test: geom_d_sat: 0.30 → 1.0 (single-variable change)
  Other params identical to tonight Block 1 (harder pose, noise=0.05, 60ep)

Runs:
  SEEDS=${SEEDS[*]}
  HARDER_NOISE=$HARDER_NOISE
  HARDER_EPOCHS=$HARDER_EPOCHS
  NEW_D_SAT=$NEW_D_SAT
  TAG=$TAG
  14 total (7 seeds × 2 algos), ~4.4h expected; timeout/run=$RUN_TIMEOUT
  DRY_RUN=$DRY_RUN

EOF

# ──────────────────── Helpers ────────────────────

set_yaml_scalar() {
    local file="$1" key="$2" value="$3" before_key="${4:-wandb_project}"
    if grep -q "^${key}:" "$file"; then
        sed -i -E "s|^${key}:.*$|${key}: ${value}|" "$file"
    else
        sed -i "/^${before_key}:/i ${key}: ${value}" "$file"
    fi
}

make_calib_yaml() {
    local base="$1" seed="$2" out="$3"
    sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$base" > "$out"
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_s${seed}_dsat1p0/" "$out"
    set_yaml_scalar "$out" "n_epochs" "$HARDER_EPOCHS"
    set_yaml_scalar "$out" "default_pose_variant" "harder"
    set_yaml_scalar "$out" "initial_joint_noise" "$HARDER_NOISE"
    # ★ The one calibration variable ★
    set_yaml_scalar "$out" "geom_d_sat" "$NEW_D_SAT"
}

run_experiment() {
    local name="$1" yaml="$2" tag="$3" to="$4"
    local log="$LOGDIR/${name}.log"
    local start_ts=$(date +%s) start_human=$(date '+%H:%M:%S')

    local cmd=(
        python scripts/run_lagrangian_chain_local_from_yaml.py
        --start_stage 1 --stop_stage 1
        --stage1_cfg "$yaml"
        --tag "$tag"
        --skip_snapshot
    )

    echo "[$start_human] START $name (yaml=$yaml to=$to)" | tee -a "$SUMMARY"
    echo "          CMD: ${cmd[*]}" >> "$SUMMARY"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[$start_human] DRY-RUN $name (not launched)" | tee -a "$SUMMARY"
        echo "" >> "$SUMMARY"
        return 0
    fi

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s "$to" "${cmd[@]}" > "$log" 2>&1
    local rc=$?

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    local end_ts=$(date +%s) end_human=$(date '+%H:%M:%S')
    local dur=$((end_ts - start_ts))
    local status="OK"; [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    local best_J best_geom min_d_err min_d_err_mean final_lambda
    best_J=$(grep -E "训练完成.*best J" "$log" 2>/dev/null | head -1 | grep -oE "best J = -?[0-9.]+" | awk '{print $4}')
    best_geom=$(grep -E "训练完成.*best[ _]geom(_hold)?_rate" "$log" 2>/dev/null | head -1 | grep -oE "best[ _]geom(_hold)?_rate[ =]+[0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    final_lambda=$(grep -E "训练完成.*final λ" "$log" 2>/dev/null | head -1 | grep -oE "final λ = [0-9.]+" | awk '{print $4}')
    min_d_err=$(grep -oE "d_err_min=[0-9.]+" "$log" 2>/dev/null | grep -oE "[0-9.]+" | sort -n | head -1)
    min_d_err_mean=$(grep -oE "d_err_mean=[0-9.]+" "$log" 2>/dev/null | grep -oE "[0-9.]+" | sort -n | head -1)

    echo "[$end_human] END   $name dur=${dur}s status=$status bestJ=${best_J:-?} bestGeom=${best_geom:-?} minDErr=${min_d_err:-?} minMeanDErr=${min_d_err_mean:-?} λ=${final_lambda:-?}" | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
    return $rc
}

# ──────────────────── Run all ────────────────────
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Stage 1 calibration: geom_d_sat 0.30 → 1.0 (single-variable)" | tee -a "$SUMMARY"
echo "Seeds: ${SEEDS[*]} (s0 sanity + s1,s2,s6,s7,s8,s9 failed)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

for seed in "${SEEDS[@]}"; do
    cfg="$TMPCFG/lag_calib_s${seed}.yaml"
    make_calib_yaml "$LAG_S1_YAML" "$seed" "$cfg"
    run_experiment "lag_calib_s${seed}" "$cfg" "$TAG" "$RUN_TIMEOUT"

    cfg="$TMPCFG/sac_calib_s${seed}.yaml"
    make_calib_yaml "$SAC_S1_YAML" "$seed" "$cfg"
    run_experiment "sac_calib_s${seed}" "$cfg" "$TAG" "$RUN_TIMEOUT"
done

# ──────────────────── Done ────────────────────
END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

cat >> "$SUMMARY" <<EOF

============================================================
=== DONE at $END_HUMAN ===
Total duration: $((TOTAL_DUR / 60)) min ($((TOTAL_DUR / 3600))h $((TOTAL_DUR % 3600 / 60))m)

Next step: run scripts/analyze_calibration.py to compare against tonight Block 1 baseline.
  - Did failed seeds (1,2,6,7,8,9) cross d_err < 0.30?
  - first_hold_epoch ?
  - sanity seed 0 still succeeds?
  - LagSAC vs SAC violation gap preserved?
============================================================
EOF

echo "Calibration complete. Summary: $SUMMARY"
