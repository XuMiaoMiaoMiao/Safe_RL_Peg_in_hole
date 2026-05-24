#!/bin/bash
# Stage 1 sat=0.8 verification — 3 LagSAC runs to confirm hypothesis.
#
# Round 1 (lag_s7_sat0p8): saturation fix saved s7 (best_geom 0→1.0, cost 41→3.8).
# This script verifies seed-robustness + sanity:
#   - lag_s2 sat=0.8  (Mode 1 seed-robustness, was d_err stuck 0.50-0.91)
#   - lag_s8 sat=0.8  (Mode 1 seed-robustness, was d_err stuck 0.41-0.66)
#   - lag_s0 sat=0.8  (sanity: don't break previously-successful seed)
#
# Single variable change: geom_d_sat 0.30 → 0.8
# All other params identical to tonight Block 1 (rew_geom_d=8 unchanged).
#
# Success criteria (per codex + Claude):
#   - s2/s8: min_d_err_mean < 0.30 AND max_geom_hold_rate > 0.5 → seed-robust fix
#   - s0: best_geom > 0.90 → sanity preserved
#
# If all 3 pass → Stage 1 fix validated, move to Stage 2 smoke (30-50ep), NOT chain.
#
# 3 runs × ~11 min ≈ 35 min total.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/calib_logs"
SUMMARY="/tmp/calib_summary.txt"
TMPCFG="/tmp/calib_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"

SEEDS=(2 8 0)
N_EPOCHS=30
NEW_D_SAT=0.8
HARDER_NOISE=0.05
TAG="calib_verify_sat0p8_$(date +%Y%m%d)"
RUN_TO=15m

START_TIME=$(date +%s)

cat >> "$SUMMARY" <<EOF

============================================================
=== sat=0.8 verification (3 LagSAC runs) at $(date '+%Y-%m-%d %H:%M:%S') ===
Seeds: ${SEEDS[*]}  (s2/s8 = failed-seed rescue, s0 = sanity)
geom_d_sat: 0.30 → ${NEW_D_SAT}, all else identical to Block 1
n_epochs=${N_EPOCHS}, timeout=${RUN_TO}/run
============================================================
EOF

set_yaml_scalar() {
    local file="$1" key="$2" value="$3" before_key="${4:-wandb_project}"
    if grep -q "^${key}:" "$file"; then
        sed -i -E "s|^${key}:.*$|${key}: ${value}|" "$file"
    else
        sed -i "/^${before_key}:/i ${key}: ${value}" "$file"
    fi
}

run_one() {
    local seed="$1"
    local cfg="$TMPCFG/lag_s${seed}_sat0p8_verify.yaml"
    sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$LAG_S1_YAML" > "$cfg"
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_s${seed}_sat0p8_v/" "$cfg"
    set_yaml_scalar "$cfg" "n_epochs" "$N_EPOCHS"
    set_yaml_scalar "$cfg" "default_pose_variant" "harder"
    set_yaml_scalar "$cfg" "initial_joint_noise" "$HARDER_NOISE"
    set_yaml_scalar "$cfg" "geom_d_sat" "$NEW_D_SAT"

    local name="lag_s${seed}_sat0p8_v"
    local log="$LOGDIR/${name}.log"
    local start_ts=$(date +%s) start_human=$(date '+%H:%M:%S')

    local cmd=(
        python scripts/run_lagrangian_chain_local_from_yaml.py
        --start_stage 1 --stop_stage 1
        --stage1_cfg "$cfg"
        --tag "$TAG"
        --skip_snapshot
    )

    echo "[$start_human] START $name" | tee -a "$SUMMARY"
    echo "          CMD: ${cmd[*]}" >> "$SUMMARY"

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s "$RUN_TO" "${cmd[@]}" > "$log" 2>&1
    local rc=$?

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    local end_ts=$(date +%s) end_human=$(date '+%H:%M:%S')
    local dur=$((end_ts - start_ts))
    local status="OK"; [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    local best_J best_geom min_d_err_mean max_hold cum_sphere cum_table final_lambda
    best_J=$(grep -E "训练完成.*best J" "$log" | head -1 | grep -oE "best J = -?[0-9.]+" | awk '{print $4}')
    best_geom=$(grep -E "训练完成.*best[ _]geom(_hold)?_rate" "$log" | head -1 | grep -oE "best[ _]geom(_hold)?_rate[ =]+[0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    final_lambda=$(grep -E "训练完成.*final λ" "$log" | head -1 | grep -oE "final λ = [0-9.]+" | awk '{print $4}')
    min_d_err_mean=$(grep -oE "d_err_mean=[0-9.]+" "$log" | grep -oE "[0-9.]+" | sort -n | head -1)
    max_hold=$(grep -oE "geom_hold_rate=[0-9.]+" "$log" | grep -oE "[0-9.]+" | sort -nr | head -1)
    cum_sphere=$(grep -oE "epoch_collision_sphere=[0-9]+" "$log" | grep -oE "[0-9]+" | paste -sd+ | bc 2>/dev/null || echo "?")
    cum_table=$(grep -oE "epoch_collision_table=[0-9]+" "$log" | grep -oE "[0-9]+" | paste -sd+ | bc 2>/dev/null || echo "?")

    cat >> "$SUMMARY" <<EOF
[$end_human] END $name dur=${dur}s status=$status
  best_J=${best_J:-?}  best_geom=${best_geom:-?}
  min_d_err_mean=${min_d_err_mean:-?}   max_hold=${max_hold:-?}
  cum_collision sphere=${cum_sphere:-?} table=${cum_table:-?}
  final_λ=${final_lambda:-?}
EOF

    # Inline pass/fail judgment
    local verdict="?"
    if [ -n "$min_d_err_mean" ] && [ -n "$max_hold" ]; then
        if [ "$seed" = "0" ]; then
            # sanity: best_geom >= 0.90
            if awk "BEGIN{exit !(${best_geom:-0} >= 0.9)}"; then verdict="✅ sanity preserved"; else verdict="⚠️ s0 best_geom regressed"; fi
        else
            # rescue: min_d_err < 0.30 AND max_hold > 0.5
            if awk "BEGIN{exit !(${min_d_err_mean:-99} < 0.3 && ${max_hold:-0} > 0.5)}"; then
                verdict="✅ Mode 1 seed rescued"
            elif awk "BEGIN{exit !(${min_d_err_mean:-99} < 0.3)}"; then
                verdict="⚠️ barrier crossed but hold weak"
            else
                verdict="❌ still stuck > 0.30"
            fi
        fi
    fi
    echo "  verdict: $verdict" | tee -a "$SUMMARY"
}

for seed in "${SEEDS[@]}"; do
    run_one "$seed"
done

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF

============================================================
=== sat=0.8 verification DONE at $(date '+%Y-%m-%d %H:%M:%S') ===
Total: $((TOTAL_DUR / 60)) min

Baseline (tonight Block 1, sat=0.30):
  lag_s2  bestJ=-174.7 bestGeom=0  min_d_err=0.498  cum_sphere=10867
  lag_s8  bestJ=-175.2 bestGeom=0  min_d_err=0.410  cum_sphere=15574
  lag_s0  bestJ= -56.2 bestGeom=1.000  min_d_err=0.057  cum_sphere= 4766

Compare verdict above to decide next step:
  3/3 pass → Stage 1 fix validated; propose Stage 2 smoke (30-50ep, NOT chain)
  2/3 pass (s2 or s8 still stuck) → Mode 1 not fully reward-bound; may need
       additional fix (e.g., looser radial gate); investigate failed seed log
  s0 regressed (best_geom < 0.90) → sat=0.8 partially breaks refinement;
       try sat=0.6 or revisit
============================================================
EOF

echo "Verification complete. Summary: $SUMMARY"
