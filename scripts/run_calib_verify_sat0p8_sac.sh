#!/bin/bash
# SAC counterpart of run_calib_verify_sat0p8.sh — verify saturation fix is
# algorithm-agnostic (not LagSAC-specific) by testing same 3 seeds with SAC.
#
# Why this matters (per user + Claude):
#   - Paper goal is SAC vs LagSAC, not just "LagSAC can learn"
#   - If SAC sat=0.8 also rescues s2/s8 → fix is reward-design issue, not algo
#   - If only LagSAC rescues → real algorithmic advantage
#   - Either way the data informs paper narrative
#
# Single variable change: geom_d_sat 0.30 → 0.8, all else identical to tonight
# Block 1 SAC config (which IS verified to run — sac_harder_s* all completed).
#
# Multi-metric judgment per seed (per user's revised criteria):
#   - best_geom (peak reached)
#   - hold@ep30 (late stability)
#   - hold=1.0 epoch count (sustained convergence)
#   - first_hold epoch (sample efficiency)
#   - cum_collision_sphere/table (safety profile)
#   - eval_ep_cost (rollout cost)
#
# Don't flunk on s8-style "peak then drift" — could be 30ep too short, defer to
# 60ep final benchmark.
#
# 3 runs × ~9 min (SAC slightly faster than LagSAC at 30ep) ≈ 28 min total.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/calib_logs"
SUMMARY="/tmp/calib_summary.txt"
TMPCFG="/tmp/calib_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

SAC_S1_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"

SEEDS=(2 8 0)
N_EPOCHS=30
NEW_D_SAT=0.8
HARDER_NOISE=0.05
TAG="calib_verify_sat0p8_sac_$(date +%Y%m%d)"
RUN_TO=15m

START_TIME=$(date +%s)

cat >> "$SUMMARY" <<EOF

============================================================
=== SAC sat=0.8 verification (3 runs) at $(date '+%Y-%m-%d %H:%M:%S') ===
Seeds: ${SEEDS[*]}  (s2/s8 = paired with LagSAC rescue, s0 = paired sanity)
geom_d_sat: 0.30 → ${NEW_D_SAT}, all else identical to Block 1 SAC config
Goal: test algorithm-agnosticism of saturation hypothesis
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

# Multi-metric extractor using Python for trajectory analysis
extract_metrics() {
    local log="$1"
    python3 - "$log" <<'PY'
import re, sys
log = open(sys.argv[1], errors='ignore').read()
m_J = re.search(r'训练完成.*?best J = (-?\d+\.\d+)', log)
m_geom = re.search(r'训练完成.*?best[ _]geom(_hold)?_rate[ =]+(\d+\.\d+)', log)
# Extract per-epoch hold rates + collisions
holds = [float(x) for x in re.findall(r'geom_hold_rate=(\d+\.\d+)', log)]
d_errs = [float(x) for x in re.findall(r'd_err_mean=(\d+\.\d+)', log)]
costs = [float(x) for x in re.findall(r'epoch_collision_sphere=(\d+)', log)]
tables = [float(x) for x in re.findall(r'epoch_collision_table=(\d+)', log)]
eval_costs = [float(x) for x in re.findall(r'eval_step_cost[: =]+(\d+\.\d+)', log)]

first_hold = next((i+1 for i, h in enumerate(holds) if h > 0), None)
first_full = next((i+1 for i, h in enumerate(holds) if h >= 1.0), None)
full_epochs = sum(1 for h in holds if h >= 1.0)
hold_last = holds[-1] if holds else 0.0
min_d = min(d_errs) if d_errs else None
cum_s = sum(costs) if costs else 0
cum_t = sum(tables) if tables else 0
last_ep_step_cost = eval_costs[-1] if eval_costs else 0

print(f"best_J={m_J.group(1) if m_J else '?'}")
print(f"best_geom={m_geom.group(2) if m_geom else '?'}")
print(f"first_hold_ep={first_hold}")
print(f"first_full_ep={first_full}")
print(f"hold_at_last={hold_last:.3f}")
print(f"full_hold_epochs={full_epochs}")
print(f"min_d_err_mean={min_d:.3f}" if min_d else "min_d_err_mean=?")
print(f"cum_collision_sphere={cum_s}")
print(f"cum_collision_table={cum_t}")
print(f"final_eval_step_cost={last_ep_step_cost:.3f}")
PY
}

run_one() {
    local seed="$1"
    local cfg="$TMPCFG/sac_s${seed}_sat0p8_verify.yaml"
    sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$SAC_S1_YAML" > "$cfg"
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_s${seed}_sat0p8_v/" "$cfg"
    set_yaml_scalar "$cfg" "n_epochs" "$N_EPOCHS"
    set_yaml_scalar "$cfg" "default_pose_variant" "harder"
    set_yaml_scalar "$cfg" "initial_joint_noise" "$HARDER_NOISE"
    set_yaml_scalar "$cfg" "geom_d_sat" "$NEW_D_SAT"

    local name="sac_s${seed}_sat0p8_v"
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

    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s "$RUN_TO" "${cmd[@]}" > "$log" 2>&1
    local rc=$?

    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    local end_ts=$(date +%s) end_human=$(date '+%H:%M:%S')
    local dur=$((end_ts - start_ts))
    local status="OK"; [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    echo "[$end_human] END $name dur=${dur}s status=$status" | tee -a "$SUMMARY"
    extract_metrics "$log" | sed 's/^/  /' | tee -a "$SUMMARY"
}

for seed in "${SEEDS[@]}"; do
    run_one "$seed"
done

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF

============================================================
=== SAC sat=0.8 verification DONE at $(date '+%Y-%m-%d %H:%M:%S') ===
Total: $((TOTAL_DUR / 60)) min

Baseline reference (tonight Block 1, sat=0.30):
  SAC seed 0: bestJ=-91.3  bestGeom=1.000 hold@60=0.06 cum_sphere=10604 cum_table= 158
  SAC seed 2: bestJ=-178.1 bestGeom=0     hold@60=0     cum_sphere=19348 cum_table=1821
  SAC seed 8: bestJ=-85.0  bestGeom=1.000 (regressed)   cum_sphere= 8256 cum_table= 126

LagSAC sat=0.8 reference (just completed):
  s2: bestJ=-62.0  bestGeom=1.000 hold@30=1.000 full_epochs=17 cum_sphere= 601 cum_table= 3
  s8: bestJ=-76.4  bestGeom=1.000 hold@30=0.312 full_epochs= 2 cum_sphere= 848 cum_table=37
  s0: bestJ=-77.1  bestGeom=0.938 hold@30=0.938 full_epochs= 0 cum_sphere= 656 cum_table=37

Decision rubric:
  All 3 SAC reach best_geom > 0.5 → fix is algorithm-agnostic (reward design)
    + SAC collision > LagSAC → LagSAC safety advantage holds (paper-grade)
  Some SAC fail but LagSAC ✓ → genuine LagSAC advantage from Q_C critic
  None SAC pass → SAC needs additional tuning beyond sat=0.8
============================================================
EOF

echo "Done. Summary: $SUMMARY"
