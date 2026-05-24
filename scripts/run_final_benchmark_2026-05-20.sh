#!/bin/bash
# FINAL Stage 1 benchmark — SAC vs LagSAC, harder pose, 10 seeds, 60ep.
#
# Parameter set (commit-to-it, no more iteration):
#   geom_d_sat: 0.8                  (Mode 1 fix, verified)
#   default_pose_variant: harder
#   initial_joint_noise: 0.05
#   n_epochs: 60
#   rew_geom_soft_success: 0.0       (no reward shaping)
#   LagSAC only: cost_limit_per_ep: 10.0  (verified 9x safety gain on s8)
#
# Evaluation framework (paper-grade):
#   - Use best_hold.msh (NOT final_agent) for all per-seed metrics
#   - Report mean ± std over 10 seeds for: best_geom_hold_rate, cum_collision_sphere,
#     cum_collision_table, λ trajectory, best_J
#   - Headline comparison: LagSAC vs SAC on cum_collision_sphere mean (expect ~3-5x lower)
#
# Time budget:
#   LagSAC 60ep ≈ 21min × 10 = 210 min
#   SAC 60ep ≈ 17min × 10 = 170 min
#   + IsaacSim init overhead 20 × 30s = 10 min
#   = ~6.5 h
#
# Run order: interleave (lag_s0, sac_s0, lag_s1, sac_s1, ...) so partial completion
# still has paired data.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/final_bench_logs"
SUMMARY="/tmp/final_bench_summary.txt"
TMPCFG="/tmp/final_bench_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"
SAC_S1_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"

SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_EPOCHS=60
NEW_D_SAT=0.8
NEW_COST_LIMIT=10.0
HARDER_NOISE=0.05
TAG="final_bench_$(date +%Y%m%d)"
RUN_TO=30m
DRY_RUN="${DRY_RUN:-0}"

START_TIME=$(date +%s)

cat > "$SUMMARY" <<EOF
=== FINAL Stage 1 benchmark started $(date '+%Y-%m-%d %H:%M:%S') ===
SAC vs LagSAC, harder pose, 10 seeds × 60ep.

Params:
  geom_d_sat=${NEW_D_SAT}, n_epochs=${N_EPOCHS}, noise=${HARDER_NOISE}, harder pose
  LagSAC cost_limit_per_ep=${NEW_COST_LIMIT} (SAC unaffected)
  No reward shaping changes.

Eval framework: best_hold.msh checkpoint, mean ± std over 10 seeds.
Expected: ~6.5h total. timeout=${RUN_TO}/run.
DRY_RUN=$DRY_RUN

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
    local algo=$1 seed=$2 out=$3 name=$4
    local base
    [ "$algo" = "lag" ] && base="$LAG_S1_YAML" || base="$SAC_S1_YAML"

    sed -E "s/^seed:.*/seed: $seed/" "$base" > "$out"
    set_yaml_scalar "$out" "n_epochs" "$N_EPOCHS"
    set_yaml_scalar "$out" "geom_d_sat" "$NEW_D_SAT"
    set_yaml_scalar "$out" "initial_joint_noise" "$HARDER_NOISE"
    set_yaml_scalar "$out" "default_pose_variant" "harder"
    if [ "$algo" = "lag" ]; then
        set_yaml_scalar "$out" "cost_limit_per_ep" "$NEW_COST_LIMIT"
    fi
    sed -i -E "s/^wandb_run_name:.*/wandb_run_name: $name/" "$out"
}

extract_metrics() {
    local log="$1"
    python3 - "$log" <<'PY'
import re, sys
log = open(sys.argv[1], errors='ignore').read()
m_J = re.search(r'训练完成.*?best J = (-?\d+\.\d+)', log)
m_geom = re.search(r'训练完成.*?best[ _]geom(_hold)?_rate[ =]+(\d+\.\d+)', log)
m_score = re.search(r'best_score = (-?\d+\.\d+)', log)
holds = [float(x) for x in re.findall(r'geom_hold_rate=(\d+\.\d+)', log)]
spheres = [int(x) for x in re.findall(r'epoch_collision_sphere=(\d+)', log)]
tables = [int(x) for x in re.findall(r'epoch_collision_table=(\d+)', log)]
lambdas = [float(x) for x in re.findall(r'lam: ([\d.e+-]+)', log)]
first_hold = next((i+1 for i, h in enumerate(holds) if h > 0), None)
first_full = next((i+1 for i, h in enumerate(holds) if h >= 1.0), None)
full_epochs = sum(1 for h in holds if h >= 1.0)
hold_last = holds[-1] if holds else 0.0
hold_last10 = sum(holds[-10:])/min(10, len(holds)) if holds else 0
lam_late = sum(lambdas[-20:])/min(20, len(lambdas)) if lambdas else None
print(f"  best_J={m_J.group(1) if m_J else '?'}")
print(f"  best_geom={m_geom.group(2) if m_geom else '?'}")
print(f"  best_score={m_score.group(1) if m_score else '?'}")
print(f"  first_hold_ep={first_hold}  first_full_ep={first_full}  full_hold_epochs={full_epochs}")
print(f"  hold_at_last={hold_last:.3f}  hold_last10_mean={hold_last10:.3f}")
print(f"  cum_collision_sphere={sum(spheres)}  cum_table={sum(tables)}")
if lam_late is not None: print(f"  mean_lambda_late20={lam_late:.4f}")
PY
}

run_one() {
    local algo=$1 seed=$2
    local cfg="$TMPCFG/${algo}_s${seed}.yaml"
    local name="${algo}_final_s${seed}"
    build_yaml "$algo" "$seed" "$cfg" "$name"

    local log="$LOGDIR/${name}.log"
    local start_ts=$(date +%s) start_human=$(date '+%H:%M:%S')

    echo "[$start_human] START $name" | tee -a "$SUMMARY"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[$start_human] DRY-RUN" | tee -a "$SUMMARY"
        return 0
    fi

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s "$RUN_TO" python scripts/run_lagrangian_chain_local_from_yaml.py \
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

# Interleave lag + sac per seed
for seed in "${SEEDS[@]}"; do
    run_one "lag" "$seed"
    run_one "sac" "$seed"
done

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF

============================================================
=== FINAL benchmark DONE at $(date '+%Y-%m-%d %H:%M:%S') ===
Total: $((TOTAL_DUR / 60)) min ($((TOTAL_DUR / 3600))h $((TOTAL_DUR % 3600 / 60))m)
============================================================
EOF

# Auto-aggregate table at end
echo "============================================================" | tee -a "$SUMMARY"
echo "AGGREGATE TABLE (mean ± std over completed seeds)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
python3 - <<PY | tee -a "$SUMMARY"
import re, statistics
from pathlib import Path
log = Path('$SUMMARY').read_text()
runs = {}
cur = None
for line in log.split('\n'):
    m = re.match(r'\[\d\d:\d\d:\d\d\] END (\w+)_final_s(\d+)', line)
    if m:
        cur = f"{m.group(1)}_s{m.group(2)}"; runs[cur] = {}; continue
    if cur:
        for k, v in re.findall(r'(\w+)=(-?[\d.eE+-]+|None|\?)', line):
            try: runs[cur][k] = float(v)
            except: pass

def agg(algo, key):
    vals = [d[key] for n, d in runs.items() if n.startswith(algo) and key in d]
    if not vals: return "n/a"
    if len(vals) == 1: return f"{vals[0]:.2f} (n=1)"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f} (n={len(vals)})"

def passed(algo, key, threshold):
    vals = [d.get(key, 0) for n, d in runs.items() if n.startswith(algo)]
    return sum(1 for v in vals if v >= threshold), len(vals)

print(f"\nLagSAC (10 seeds):")
print(f"  best_geom:            {agg('lag', 'best_geom')}")
print(f"  best_J:               {agg('lag', 'best_J')}")
print(f"  full_hold_epochs:     {agg('lag', 'full_hold_epochs')}")
print(f"  cum_collision_sphere: {agg('lag', 'cum_collision_sphere')}")
print(f"  cum_collision_table:  {agg('lag', 'cum_collision_table')}")
print(f"  best_geom ≥ 0.5:      {passed('lag', 'best_geom', 0.5)[0]}/{passed('lag', 'best_geom', 0.5)[1]}")

print(f"\nSAC (10 seeds):")
print(f"  best_geom:            {agg('sac', 'best_geom')}")
print(f"  best_J:               {agg('sac', 'best_J')}")
print(f"  full_hold_epochs:     {agg('sac', 'full_hold_epochs')}")
print(f"  cum_collision_sphere: {agg('sac', 'cum_collision_sphere')}")
print(f"  cum_collision_table:  {agg('sac', 'cum_collision_table')}")
print(f"  best_geom ≥ 0.5:      {passed('sac', 'best_geom', 0.5)[0]}/{passed('sac', 'best_geom', 0.5)[1]}")

# Safety ratio
lag_s = [d.get('cum_collision_sphere', 0) for n, d in runs.items() if n.startswith('lag')]
sac_s = [d.get('cum_collision_sphere', 0) for n, d in runs.items() if n.startswith('sac')]
if lag_s and sac_s:
    lag_mean = statistics.mean(lag_s); sac_mean = statistics.mean(sac_s)
    print(f"\nSafety ratio: SAC sphere / LagSAC sphere = {sac_mean/max(1,lag_mean):.2f}x")
PY

echo ""
echo "============================================================"
echo "Done. Paper-grade summary: cat /tmp/final_bench_summary.txt"
echo "Best-hold checkpoints saved per seed in:"
echo "  LagSAC: results/checkpoints_lag/$(date +%Y-%m-%d)/<HH-MM-SS>/best_hold.msh"
echo "  SAC:    results/checkpoints/$(date +%Y-%m-%d)/<HH-MM-SS>/best_hold.msh"
echo "============================================================"
