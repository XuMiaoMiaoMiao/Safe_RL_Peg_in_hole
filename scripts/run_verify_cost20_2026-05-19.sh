#!/bin/bash
# Test cost_limit=20 on LagSAC Stage 1 harder pose.
#
# Background:
#   cost_limit=50 was too loose: s8 reached success, lambda decayed to ~0,
#   then the policy drifted and collisions returned.
#   cost_limit=10 fixed s8 drift but broke s2 stability.
#
# This script tests the middle point:
#   Step 1: LagSAC s8, cost_limit=20 60ep    (stress seed)
#   Step 2: LagSAC s2, cost_limit=20 60ep    (sanity seed)
#
# Everything else stays unchanged:
#   geom_d_sat=0.8, harder pose, noise=0.05, no reward changes.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/verify_cost20_logs"
SUMMARY="/tmp/verify_cost20_summary.txt"
TMPCFG="/tmp/verify_cost20_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"

TAG="verify_cost20_$(date +%Y%m%d)"
START_TIME=$(date +%s)

cat > "$SUMMARY" <<EOF
=== cost_limit=20 verification started $(date '+%Y-%m-%d %H:%M:%S') ===
Step 1: LagSAC s8 (stress seed; cost50 drifted, cost10 improved)
Step 2: LagSAC s2 (sanity seed; cost10 broke late hold)
Config: geom_d_sat=0.8, cost_limit_per_ep=20, 60ep, noise=0.05, harder pose
NO reward changes.

EOF

set_yaml_scalar() {
    local file="$1"
    local key="$2"
    local value="$3"
    local before_key="${4:-wandb_project}"
    if grep -q "^${key}:" "$file"; then
        sed -i -E "s|^${key}:.*$|${key}: ${value}|" "$file"
    else
        sed -i "/^${before_key}:/i ${key}: ${value}" "$file"
    fi
}

build_yaml() {
    local seed="$1"
    local out="$2"
    local name="$3"
    sed -E "s/^seed:.*/seed: $seed/" "$LAG_S1_YAML" > "$out"
    set_yaml_scalar "$out" "n_epochs" "60"
    set_yaml_scalar "$out" "geom_d_sat" "0.8"
    set_yaml_scalar "$out" "initial_joint_noise" "0.05"
    set_yaml_scalar "$out" "default_pose_variant" "harder"
    set_yaml_scalar "$out" "cost_limit_per_ep" "20.0"
    sed -i -E "s/^wandb_run_name:.*/wandb_run_name: $name/" "$out"
}

extract_metrics() {
    local log="$1"
    python3 - "$log" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="ignore").read()

m_j = re.search(r"训练完成.*?best J = (-?\d+\.\d+)", text)
m_geom = re.search(r"训练完成.*?best[ _]geom(?:_hold)?_rate[ =]+(\d+\.\d+)", text)

holds = [float(x) for x in re.findall(r"geom_hold_rate=(\d+\.\d+)", text)]
spheres = [int(x) for x in re.findall(r"epoch_collision_sphere=(\d+)", text)]
tables = [int(x) for x in re.findall(r"epoch_collision_table=(\d+)", text)]
rollout_costs = [float(x) for x in re.findall(r"rollout_ep_cost=(-?[\d.]+)", text)]
lambdas = [float(x) for x in re.findall(r"lam: ([\d.e+-]+)", text)]

first_hold = next((i + 1 for i, h in enumerate(holds) if h > 0.0), None)
first_full = next((i + 1 for i, h in enumerate(holds) if h >= 1.0), None)
full_epochs = sum(1 for h in holds if h >= 1.0)
hold_last = holds[-1] if holds else 0.0
hold_last10_mean = sum(holds[-10:]) / min(10, len(holds)) if holds else 0.0
mean_lambda_late = sum(lambdas[-20:]) / min(20, len(lambdas)) if lambdas else 0.0
mean_cost_late = (
    sum(rollout_costs[-20:]) / min(20, len(rollout_costs))
    if rollout_costs else 0.0
)

print(f"  best_J={m_j.group(1) if m_j else '?'}")
print(f"  best_geom={m_geom.group(1) if m_geom else '?'}")
print(f"  first_hold_ep={first_hold}  first_full_ep={first_full}")
print(f"  full_hold_epochs={full_epochs}")
print(f"  hold_at_last={hold_last:.3f}  hold_last10_mean={hold_last10_mean:.3f}")
print(f"  cum_collision_sphere={sum(spheres)}  cum_table={sum(tables)}")
print(f"  mean_lambda_late20={mean_lambda_late:.4f}")
print(f"  mean_rollout_cost_late20={mean_cost_late:.2f}")
PY
}

run_one() {
    local name="$1"
    local cfg="$2"
    local log="$LOGDIR/${name}.log"
    local start_ts
    local start_human
    start_ts=$(date +%s)
    start_human=$(date '+%H:%M:%S')

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

    local end_ts
    local end_human
    local dur
    local status
    end_ts=$(date +%s)
    end_human=$(date '+%H:%M:%S')
    dur=$((end_ts - start_ts))
    status="OK"
    [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    echo "[$end_human] END $name dur=${dur}s status=$status" | tee -a "$SUMMARY"
    extract_metrics "$log" | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"
}

build_yaml 8 "$TMPCFG/lag_s8_cost20_60ep.yaml" "lag_s8_cost20_60ep"
build_yaml 2 "$TMPCFG/lag_s2_cost20_60ep.yaml" "lag_s2_cost20_60ep"

run_one "lag_s8_cost20_60ep" "$TMPCFG/lag_s8_cost20_60ep.yaml"
run_one "lag_s2_cost20_60ep" "$TMPCFG/lag_s2_cost20_60ep.yaml"

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF
============================================================
=== DONE at $(date '+%Y-%m-%d %H:%M:%S') ===
Total: $((TOTAL_DUR / 60)) min

Reference:
  cost50 s8: bestG=1.0 first_full=13 full=3/60 hold_last=0.000 hold_last10=0.000 cum_s=2033 cum_t=82 lambda_late~=0
  cost10 s8: bestG=1.0 first_full=24 full=7/60 hold_last=0.812 hold_last10=0.794 cum_s=225  cum_t=45 lambda_late~=1.2
  cost50 s2: bestG=1.0 first_full= 9 full=17/30 hold_last=1.000 cum_s=601  cum_t=3
  cost10 s2: bestG=1.0 first_full=41 full=2/60 hold_last=0.000 hold_last10=0.000 cum_s=745  cum_t=73

Decision rubric:
  s8 cost20 pass:
    hold_last10_mean > 0.5 AND cum_sphere < 800
  s2 cost20 pass:
    hold_last10_mean > 0.5 OR full_hold_epochs >= 5
    AND cum_sphere < 1500

Both pass -> use cost_limit=20 for final benchmark.
s8 pass, s2 fail -> cost20 still too tight; try cost30.
s8 fail, s2 pass -> cost20 too loose for unstable seed; accept seed variance or consider schedule.
Both fail -> fixed cost_limit is not enough; do not run final yet.
============================================================
EOF

echo "Done. cat $SUMMARY"
