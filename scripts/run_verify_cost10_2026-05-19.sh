#!/bin/bash
# Test cost_limit=10 hypothesis on LagSAC.
#
# Background:
#   sat=0.8 fixed barrier, but s8 60ep shows λ→0 → no constraint → drift
#   (ep13 first_full, ep30+ peak hold=1, ep46+ drift back to hold=0, λ=0)
#   Hypothesis: cost_limit=50 too loose, λ decays before constraint becomes
#   useful. Tightening to 10 keeps λ active.
#
# Step 1: LagSAC s8, cost_limit=10 60ep    (stress seed test)
# Step 2: LagSAC s2, cost_limit=10 60ep    (sanity — don't break working seed)
#
# If both pass → cost_limit=10 confirmed, run 10-seed final
# If s8 ok but s2 broken → cost_limit too aggressive, try 20
# If s8 still drifts → cost not the answer, accept entropy drift in paper
#
# 2 × ~21min = ~42min total.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/verify_cost10_logs"
SUMMARY="/tmp/verify_cost10_summary.txt"
TMPCFG="/tmp/verify_cost10_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"

TAG="verify_cost10_$(date +%Y%m%d)"
START_TIME=$(date +%s)

cat > "$SUMMARY" <<EOF
=== cost_limit=10 verification started $(date '+%Y-%m-%d %H:%M:%S') ===
Step 1: LagSAC s8 (stress seed — was peak-drift @ cost_limit=50)
Step 2: LagSAC s2 (sanity — was stable @ cost_limit=50)
Config: geom_d_sat=0.8, cost_limit_per_ep=10, 60ep, noise=0.05, harder pose
NO reward changes.

EOF

set_yaml_scalar() {
    local file="$1" key="$2" value="$3" before_key="${4:-wandb_project}"
    if grep -q "^${key}:" "$file"; then
        sed -i -E "s|^${key}:.*$|${key}: ${value}|" "$file"
    else
        sed -i "/^${before_key}:/i ${key}: ${value}" "$file"
    fi
}

build_yaml() {
    local seed=$1 out=$2 name=$3
    sed -E "s/^seed:.*/seed: $seed/" "$LAG_S1_YAML" > "$out"
    set_yaml_scalar "$out" "n_epochs" "60"
    set_yaml_scalar "$out" "geom_d_sat" "0.8"
    set_yaml_scalar "$out" "initial_joint_noise" "0.05"
    set_yaml_scalar "$out" "default_pose_variant" "harder"
    set_yaml_scalar "$out" "cost_limit_per_ep" "10.0"      # ← the new variable
    sed -i -E "s/^wandb_run_name:.*/wandb_run_name: $name/" "$out"
}

extract_metrics() {
    local log="$1"
    python3 - "$log" <<'PY'
import re, sys
log = open(sys.argv[1], errors='ignore').read()
m_J = re.search(r'训练完成.*?best J = (-?\d+\.\d+)', log)
m_geom = re.search(r'训练完成.*?best[ _]geom(_hold)?_rate[ =]+(\d+\.\d+)', log)
holds = [float(x) for x in re.findall(r'geom_hold_rate=(\d+\.\d+)', log)]
spheres = [int(x) for x in re.findall(r'epoch_collision_sphere=(\d+)', log)]
tables = [int(x) for x in re.findall(r'epoch_collision_table=(\d+)', log)]
rollout_costs = [float(x) for x in re.findall(r'rollout_ep_cost=(-?[\d.]+)', log)]
lambdas = [float(x) for x in re.findall(r'lam: ([\d.e+-]+)', log)]

first_hold = next((i+1 for i, h in enumerate(holds) if h > 0), None)
first_full = next((i+1 for i, h in enumerate(holds) if h >= 1.0), None)
full_epochs = sum(1 for h in holds if h >= 1.0)
hold_last = holds[-1] if holds else 0.0
hold_last10_mean = sum(holds[-10:])/min(10, len(holds)) if holds else 0
mean_lambda_late = sum(lambdas[-20:])/min(20, len(lambdas)) if lambdas else 0
mean_cost_late = sum(rollout_costs[-20:])/min(20, len(rollout_costs)) if rollout_costs else 0

print(f"  best_J={m_J.group(1) if m_J else '?'}")
print(f"  best_geom={m_geom.group(2) if m_geom else '?'}")
print(f"  first_hold_ep={first_hold}  first_full_ep={first_full}")
print(f"  full_hold_epochs={full_epochs}")
print(f"  hold_at_last={hold_last:.3f}  hold_last10_mean={hold_last10_mean:.3f}")
print(f"  cum_collision_sphere={sum(spheres)}  cum_table={sum(tables)}")
print(f"  mean_lambda_late20={mean_lambda_late:.4f}  (was ~0 with cost_lim=50)")
print(f"  mean_rollout_cost_late20={mean_cost_late:.2f}")
PY
}

run_one() {
    local name="$1"
    local cfg="$2"
    local log="$LOGDIR/${name}.log"
    local start_ts=$(date +%s) start_human=$(date '+%H:%M:%S')

    echo "[$start_human] START $name" | tee -a "$SUMMARY"

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s 35m python scripts/run_lagrangian_chain_local_from_yaml.py \
        --start_stage 1 --stop_stage 1 \
        --stage1_cfg "$cfg" \
        --tag "$TAG" \
        --skip_snapshot > "$log" 2>&1
    local rc=$?

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    local end_ts=$(date +%s) end_human=$(date '+%H:%M:%S')
    local dur=$((end_ts - start_ts))
    local status="OK"; [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    echo "[$end_human] END $name dur=${dur}s status=$status" | tee -a "$SUMMARY"
    extract_metrics "$log" | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"
}

build_yaml 8 "$TMPCFG/lag_s8_cost10_60ep.yaml" "lag_s8_cost10_60ep"
build_yaml 2 "$TMPCFG/lag_s2_cost10_60ep.yaml" "lag_s2_cost10_60ep"

run_one "lag_s8_cost10_60ep" "$TMPCFG/lag_s8_cost10_60ep.yaml"
run_one "lag_s2_cost10_60ep" "$TMPCFG/lag_s2_cost10_60ep.yaml"

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF
============================================================
=== DONE at $(date '+%Y-%m-%d %H:%M:%S') ===
Total: $((TOTAL_DUR / 60)) min

Reference baselines (cost_limit=50):
  s8 60ep: bestG=1.0 first_full=13 full=3/60  hold_last=0  cum_s=2033 cum_t=82  λ_late=~0
  s2 30ep: bestG=1.0 first_full= 9 full=17/30 hold_last=1  cum_s= 601 cum_t= 3  λ_late=~0

Decision rubric:
  s8 cost_limit=10:
    full_hold_epochs ≥ 10 AND hold_last10_mean > 0.5 → drift fixed ✓
    λ_late > 0.01 → constraint stayed active ✓
    Either fails → cost_limit not the answer
  s2 cost_limit=10:
    full_hold_epochs ≥ 5 AND cum_sphere < 1500 → not broken ✓
    Either fails → cost_limit=10 too aggressive (try cost=20)

Both pass → final benchmark with cost_limit=10
Either fails → accept entropy drift, use mean ± std paper format
============================================================
EOF

echo "Done. cat $SUMMARY"
