#!/bin/bash
# Stage 3 100-epoch comparison: SAC vs LagSAC(cost=100) vs LagSAC(cost=50 strict).
#
# 历史: 5/12 SAC v8 在 ep 74 才 breakthrough insert (ep 1-73 best_geom=0); 50ep
# calibration 完全踩不到. 100ep 给 ~25 epoch buffer 保证不被 seed variance 误伤.
#
# Usage:
#   nohup bash scripts/run_stage3_100ep.sh > /tmp/stage3_100ep_main.log 2>&1 &
#   disown
#
# Outputs:
#   /tmp/overnight_logs/<name>.log
#   /tmp/stage3_100ep_summary.txt

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/overnight_logs"
SUMMARY="/tmp/stage3_100ep_summary.txt"
mkdir -p "$LOGDIR"

START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
cat > "$SUMMARY" <<EOF
=== Stage 3 100-epoch comparison started at $START_HUMAN ===
3 runs × ~55min ≈ 2.75 h total
EOF

TAG="s3_100ep_$(date +%Y%m%d_%H%M)"

run() {
    local name="$1" yaml="$2"
    local log="$LOGDIR/${name}.log"
    local start=$(date '+%H:%M:%S')
    echo "[$start] START $name" | tee -a "$SUMMARY"

    # GPU cleanup between runs (防 zombie 进程占 GPU)
    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local 2>/dev/null
    sleep 3

    # 75m timeout (100ep × ~33s/ep = 55min, 给 1.4× buffer); -k 60s SIGKILL 兜底
    timeout -k 60s 75m python scripts/run_lagrangian_chain_local_from_yaml.py \
        --start_stage 3 --stop_stage 3 \
        --stage3_cfg "$yaml" --tag "$TAG" --skip_snapshot \
        > "$log" 2>&1
    local rc=$?

    local end=$(date '+%H:%M:%S')
    local best_J=$(grep -oE "best J = -?[0-9.]+" "$log" | head -1 | awk '{print $4}')
    local best_geom=$(grep -E "训练完成" "$log" | head -1 | grep -oE "best[_ ]geom[_ ]?(hold[_ ])?rate[^0-9]+[0-9.]+" | grep -oE "[0-9.]+$")
    local final_lam=$(grep -oE "final λ = [0-9.]+" "$log" | head -1 | awk '{print $4}')
    echo "[$end] END   $name rc=$rc bestJ=${best_J:-?} bestGeom=${best_geom:-?} λ=${final_lam:-?}" | tee -a "$SUMMARY"

    # 二次 cleanup
    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    sleep 3
}

# SAC baseline 先 (跟 5/12 v8 期望 ep 74-79 breakthrough)
run "sac_stage3_100ep" "$PROJECT_ROOT/conf/experiment/sac_stage3_b_route_calib.yaml"
# LagSAC cost=100 (proportional)
run "lag_stage3_100ep_cost100" "$PROJECT_ROOT/conf/experiment/lag_stage3_b_route_calib.yaml"
# LagSAC cost=50 strict
run "lag_stage3_100ep_cost50_strict" "$PROJECT_ROOT/conf/experiment/lag_stage3_b_route_calib_strict.yaml"

echo "=== DONE at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$SUMMARY"
