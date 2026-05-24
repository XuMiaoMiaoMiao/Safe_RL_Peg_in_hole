#!/bin/bash
# Targeted single-seed hypothesis test — clean ablation of geom_d_sat alone.
#
# Hypothesis: harder pose 失败 seed (lag_s7 = 干净 Mode 1) 被 geom_d_sat=0.30
# 卡死, 因为初始 d_err = 0.80m >> sat boundary, reward 在 d_err > 0.30 区域
# 完全平坦, Q-critic 看不到 depth gradient.
#
# 严格单变量 ablation: 只改 sat 0.30 → 0.8, 其余完全同昨晚 Block 1.
#   - rew_geom_d=8.0 不动 → 保证 d_err < 0.30 近距 refinement landscape 不变
#   - 不动 cost_limit / lr / noise / pose
#
# sat 选 0.8 而非 1.0 是为了温和 (前面 s0 在 sat=1.0 显示快速 dive 副作用).
# sat=0.8 cover 几乎所有 harder reset (max d_err ≈ 0.88), 不至于像 sat=1.0
# 把整个 reward 空间斜度都加大.
#
# 判断指标 (后续看 log):
#   - min_d_err_mean < 0.30 ?            (跨过 saturation barrier)
#   - any geom_step_rate > 0 ?           (policy 真进入 success zone)
#   - any geom_hold_rate > 0 ?           (短暂 hold 出现)
#   - best_geom > 0.3 ?                  (任务方向对)
#   - cum_collision 不爆炸 ?              (没乱撞)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/calib_logs"
SUMMARY="/tmp/calib_summary.txt"  # append to existing
TMPCFG="/tmp/calib_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"

SEED=7
N_EPOCHS=30
NEW_D_SAT=0.8
HARDER_NOISE=0.05
TAG="calib_s7_sat0p8_$(date +%Y%m%d)"
RUN_TO=15m  # 30 ep × ~21s/ep ≈ 10.5 min, give 50% buffer

cat >> "$SUMMARY" <<EOF

============================================================
=== Targeted test started at $(date '+%Y-%m-%d %H:%M:%S') ===
Single ablation: lag_s${SEED}, geom_d_sat: 0.30 → ${NEW_D_SAT}
All other params identical to lag_harder_s${SEED} tonight Block 1.
n_epochs=${N_EPOCHS}, timeout=${RUN_TO}
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

cfg="$TMPCFG/lag_s${SEED}_sat0p8.yaml"
sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $SEED/" "$LAG_S1_YAML" > "$cfg"
sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_s${SEED}_sat0p8/" "$cfg"
set_yaml_scalar "$cfg" "n_epochs" "$N_EPOCHS"
set_yaml_scalar "$cfg" "default_pose_variant" "harder"
set_yaml_scalar "$cfg" "initial_joint_noise" "$HARDER_NOISE"
set_yaml_scalar "$cfg" "geom_d_sat" "$NEW_D_SAT"

# Verify yaml
echo "Generated yaml (key fields):" | tee -a "$SUMMARY"
grep -E "^seed:|^n_epochs:|^geom_d_sat:|^rew_geom_d:|^default_pose_variant:|^initial_joint_noise:" "$cfg" | sed 's/^/  /' | tee -a "$SUMMARY"

name="lag_s${SEED}_sat0p8"
log="$LOGDIR/${name}.log"
start_ts=$(date +%s)
start_human=$(date '+%H:%M:%S')

cmd=(
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
rc=$?

pkill -9 -f train_sac_lagrangian.py 2>/dev/null
pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

end_ts=$(date +%s)
end_human=$(date '+%H:%M:%S')
dur=$((end_ts - start_ts))
status="OK"; [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

# Multi-metric extraction
best_J=$(grep -E "训练完成.*best J" "$log" | head -1 | grep -oE "best J = -?[0-9.]+" | awk '{print $4}')
best_geom=$(grep -E "训练完成.*best[ _]geom(_hold)?_rate" "$log" | head -1 | grep -oE "best[ _]geom(_hold)?_rate[ =]+[0-9.]+" | grep -oE "[0-9.]+$" | head -1)
final_lambda=$(grep -E "训练完成.*final λ" "$log" | head -1 | grep -oE "final λ = [0-9.]+" | awk '{print $4}')
min_d_err_mean=$(grep -oE "d_err_mean=[0-9.]+" "$log" | grep -oE "[0-9.]+" | sort -n | head -1)
max_geom_step=$(grep -oE "geom_step_rate=[0-9.]+" "$log" | grep -oE "[0-9.]+" | sort -nr | head -1)
max_hold=$(grep -oE "geom_hold_rate=[0-9.]+" "$log" | grep -oE "[0-9.]+" | sort -nr | head -1)
cum_sphere=$(grep -oE "epoch_collision_sphere=[0-9]+" "$log" | grep -oE "[0-9]+" | paste -sd+ | bc 2>/dev/null || echo "?")

cat >> "$SUMMARY" <<EOF
[$end_human] END $name dur=${dur}s status=$status
  best_J = ${best_J:-?}
  best_geom = ${best_geom:-?}
  min_d_err_mean = ${min_d_err_mean:-?}   ← crossed 0.30 ? (key Mode 1 test)
  max_geom_step_rate = ${max_geom_step:-?}
  max_geom_hold_rate = ${max_hold:-?}
  cum_collision_sphere = ${cum_sphere:-?}
  final_lambda = ${final_lambda:-?}
EOF

echo "============================================================" | tee -a "$SUMMARY"
echo "  Compare against tonight lag_harder_s7:" | tee -a "$SUMMARY"
echo "    best_J=-179.4 best_geom=0 min_d_err_mean=0.506 max_hold=0 cum_sphere=16395" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Done. Read summary: tail /tmp/calib_summary.txt"
