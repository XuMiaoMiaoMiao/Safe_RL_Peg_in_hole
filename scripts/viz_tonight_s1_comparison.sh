#!/bin/bash
# Record 4 Stage 1 harder pose policy videos: {LagSAC, SAC} × {seed 0, seed 5}.
#
# Same-seed apples-to-apples for tonight's Block 1 successful policies.
# Deterministic eval: initial_joint_noise=0.0.
#
# Outputs:
#   results/videos/tonight_s1_compare_20260519/<algo>_s<seed>_prepos.mp4
#   /tmp/tonight_viz_logs/<algo>_s<seed>.log

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

VIDEO_OUT="$PROJECT_ROOT/results/videos/tonight_s1_compare_20260519_train_cond"
LOGDIR="/tmp/tonight_viz_logs"
mkdir -p "$VIDEO_OUT" "$LOGDIR"

# (ckpt_path, algo, seed)
declare -a JOBS=(
    "results/checkpoints_lag/2026-05-19/00-28-11/best_hold.msh|lag|0"
    "results/checkpoints/2026-05-19/00-50-12/best_hold.msh|sac|0"
    "results/checkpoints_lag/2026-05-19/03-41-23/best_hold.msh|lag|5"
    "results/checkpoints/2026-05-19/04-02-57/best_hold.msh|sac|5"
)

for job in "${JOBS[@]}"; do
    IFS='|' read -r ckpt algo seed <<< "$job"
    name="${algo}_s${seed}"
    log="$LOGDIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] RECORD $name ckpt=$ckpt"
    if [ ! -f "$ckpt" ]; then
        echo "  SKIP: ckpt missing"
        continue
    fi

    pkill -9 -f isaac-sim 2>/dev/null; sleep 2

    timeout -k 30s 10m python scripts/record_geom_video.py \
        --agent_path "$ckpt" \
        --geom_stage prepos \
        --default_pose_variant harder \
        --initial_joint_noise 0.05 \
        --stochastic \
        --horizon 100 \
        --n_episodes 4 \
        --output_dir "$VIDEO_OUT" \
        --tag "${name}_prepos" \
        > "$log" 2>&1
    rc=$?

    pkill -9 -f isaac-sim 2>/dev/null

    if [ $rc -eq 0 ]; then
        mp4=$(ls -t "$VIDEO_OUT"/*${name}*.mp4 2>/dev/null | head -1)
        echo "  [$(date '+%H:%M:%S')] OK → $mp4"
    else
        echo "  [$(date '+%H:%M:%S')] FAIL rc=$rc (see $log)"
    fi
done

echo ""
echo "=== DONE ==="
echo "Videos: $VIDEO_OUT"
ls -la "$VIDEO_OUT"/*.mp4 2>&1 | tail -10
