#!/bin/bash
# Short verification benchmark — 10 seeds × {SAC, LagSAC} × 30ep, ~3.5h.
#
# Goal: confirm参数能让 10 seeds × 2 algos 都收敛, AND LagSAC > SAC, 才正式跑 60ep final.
#
# Parameters (2 changes from tonight Block 1):
#   geom_d_sat: 0.30 → 0.8         (verified Mode 1 fix on LagSAC s2/s7/s8)
#   rew_geom_soft_success: 0 → 1.0 (NEW: conjunctive hold basin)
#   geom_soft_d_sigma: 0.05        (1.67× threshold 0.03)
#   geom_soft_radial_sigma: 0.05   (1.67× threshold 0.03)
#   geom_soft_axis_sigma: 1.0      (Stage 1 不严 axis)
#
# Why both at once: codex 提议 + math verified (+0.07/step net positive in success
# zone, vs 之前 -0.48/step which causes drift). Untested interaction is the risk
# this script tests.
#
# 通过标准 (all 3 must pass):
#   1. ≥ 8/10 seeds × BOTH algos reach best_geom ≥ 0.5
#   2. mean(LagSAC cum_collision_sphere) << mean(SAC cum_collision_sphere)
#   3. 30ep 内 ≥ 5 full_hold_epochs in successful seeds (avoid peak-drift)
#
# If passes → 60ep × 10 seeds final benchmark (~6.7h).
# If partial → diagnose specific failure mode, targeted fix.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/short_bench_logs"
SUMMARY="/tmp/short_bench_summary.txt"
TMPCFG="/tmp/short_bench_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"
SAC_S1_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"

SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_EPOCHS=30
NEW_D_SAT=0.8
NEW_SOFT_W=1.0
NEW_SOFT_SIGMA_D=0.05
NEW_SOFT_SIGMA_R=0.05
NEW_SOFT_SIGMA_AX=1.0
HARDER_NOISE=0.05
TAG="short_bench_sat0p8_soft_$(date +%Y%m%d)"
RUN_TO=15m
DRY_RUN="${DRY_RUN:-0}"

START_TIME=$(date +%s)

cat > "$SUMMARY" <<EOF
=== Short verification benchmark started $(date '+%Y-%m-%d %H:%M:%S') ===
Goal: 10 seeds × 2 algos × 30ep → pass criteria → commit to 60ep final.

Params (2 changes vs tonight Block 1):
  geom_d_sat: 0.30 → ${NEW_D_SAT}
  rew_geom_soft_success: 0 → ${NEW_SOFT_W}
  geom_soft_d_sigma=${NEW_SOFT_SIGMA_D}  geom_soft_radial_sigma=${NEW_SOFT_SIGMA_R}  geom_soft_axis_sigma=${NEW_SOFT_SIGMA_AX}

Unchanged: rew_geom_d=8, rew_geom_radial_tip=8, cost_limit=50, lr_lambda=0.005, noise=0.05, harder pose

Seeds: ${SEEDS[*]}  (10)
n_epochs=${N_EPOCHS}, ${#SEEDS[@]}×2=20 runs × ~10min ≈ 3.5h
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

extract_metrics() {
    local log="$1"
    python3 - "$log" <<'PY'
import re, sys
log = open(sys.argv[1], errors='ignore').read()
m_J = re.search(r'训练完成.*?best J = (-?\d+\.\d+)', log)
m_geom = re.search(r'训练完成.*?best[ _]geom(_hold)?_rate[ =]+(\d+\.\d+)', log)
holds = [float(x) for x in re.findall(r'geom_hold_rate=(\d+\.\d+)', log)]
d_errs = [float(x) for x in re.findall(r'd_err_mean=(\d+\.\d+)', log)]
spheres = [int(x) for x in re.findall(r'epoch_collision_sphere=(\d+)', log)]
tables = [int(x) for x in re.findall(r'epoch_collision_table=(\d+)', log)]

first_hold = next((i+1 for i, h in enumerate(holds) if h > 0), None)
first_full = next((i+1 for i, h in enumerate(holds) if h >= 1.0), None)
full_epochs = sum(1 for h in holds if h >= 1.0)
hold_last = holds[-1] if holds else 0.0
min_d = min(d_errs) if d_errs else None
cum_s = sum(spheres)
cum_t = sum(tables)

print(f"best_J={m_J.group(1) if m_J else '?'}")
print(f"best_geom={m_geom.group(2) if m_geom else 0.0}")
print(f"first_hold_ep={first_hold}")
print(f"first_full_ep={first_full}")
print(f"hold_at_last={hold_last:.3f}")
print(f"full_hold_epochs={full_epochs}")
print(f"min_d_err_mean={min_d:.3f}" if min_d else "min_d_err_mean=?")
print(f"cum_collision_sphere={cum_s}")
print(f"cum_collision_table={cum_t}")
PY
}

run_one() {
    local algo="$1" seed="$2"
    local base
    [ "$algo" = "lag" ] && base="$LAG_S1_YAML" || base="$SAC_S1_YAML"

    local cfg="$TMPCFG/${algo}_s${seed}.yaml"
    sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$base" > "$cfg"
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_s${seed}_short/" "$cfg"
    set_yaml_scalar "$cfg" "n_epochs" "$N_EPOCHS"
    set_yaml_scalar "$cfg" "default_pose_variant" "harder"
    set_yaml_scalar "$cfg" "initial_joint_noise" "$HARDER_NOISE"
    set_yaml_scalar "$cfg" "geom_d_sat" "$NEW_D_SAT"
    set_yaml_scalar "$cfg" "rew_geom_soft_success" "$NEW_SOFT_W"
    set_yaml_scalar "$cfg" "geom_soft_d_sigma" "$NEW_SOFT_SIGMA_D"
    set_yaml_scalar "$cfg" "geom_soft_radial_sigma" "$NEW_SOFT_SIGMA_R"
    set_yaml_scalar "$cfg" "geom_soft_axis_sigma" "$NEW_SOFT_SIGMA_AX"

    local name="${algo}_s${seed}_short"
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

    if [ "$DRY_RUN" = "1" ]; then
        echo "[$start_human] DRY-RUN (not launched)" | tee -a "$SUMMARY"
        return 0
    fi

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s "$RUN_TO" "${cmd[@]}" > "$log" 2>&1
    local rc=$?

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    local end_ts=$(date +%s) end_human=$(date '+%H:%M:%S')
    local dur=$((end_ts - start_ts))
    local status="OK"; [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    echo "[$end_human] END $name dur=${dur}s status=$status" | tee -a "$SUMMARY"
    extract_metrics "$log" | sed 's/^/  /' | tee -a "$SUMMARY"
}

# Interleave LagSAC + SAC per seed (so partial completion still has paired data)
for seed in "${SEEDS[@]}"; do
    run_one "lag" "$seed"
    run_one "sac" "$seed"
done

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF

============================================================
=== Short benchmark DONE at $(date '+%Y-%m-%d %H:%M:%S') ===
Total duration: $((TOTAL_DUR / 60)) min ($((TOTAL_DUR / 3600))h $((TOTAL_DUR % 3600 / 60))m)

Pass criteria check (look at extracted metrics above):
  1. ≥ 8/10 LagSAC AND ≥ 8/10 SAC have best_geom ≥ 0.5  → barrier+conjunctive both fixed
  2. mean(LagSAC cum_collision_sphere) << mean(SAC cum_collision_sphere)  → safety advantage
  3. successful seeds have full_hold_epochs ≥ 5  → stable hold (not peak-drift)

If all 3 pass → commit to 60ep × 10 seeds final benchmark (~6.7h)
If partial → diagnose specific failure (probably one of):
  - soft_success σ wrong → try σ_d/r=0.03 (= threshold)
  - sat=0.8 still too aggressive for some seeds → try sat=0.6
  - 30ep just too short → 45-60ep
============================================================
EOF

# Auto-summary table at end
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "AUTO summary table (parse from extracted metrics)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
python3 - <<PY | tee -a "$SUMMARY"
import re
from pathlib import Path
log = Path('$SUMMARY').read_text()
# Parse END blocks
runs = {}
cur_name = None
for line in log.split('\n'):
    m = re.match(r'\[(\d\d:\d\d:\d\d)\] END (\w+)_s(\d+)_short', line)
    if m:
        cur_name = f"{m.group(2)}_s{m.group(3)}"
        runs[cur_name] = {}
        continue
    if cur_name:
        m2 = re.match(r'\s+(\w+)=(-?[\d.]+|None|\?)', line)
        if m2:
            k, v = m2.group(1), m2.group(2)
            try: runs[cur_name][k] = float(v)
            except: runs[cur_name][k] = v
print(f"{'run':<8} {'bestJ':>9} {'bestG':>6} {'first_hold':>10} {'full_eps':>8} {'min_d':>6} {'cum_S':>7} {'cum_T':>6}")
print('-' * 80)
def fmt(d, k, w=9):
    v = d.get(k, '?')
    if isinstance(v, float): return f"{v:>{w}.3f}" if abs(v)<100 else f"{v:>{w}.1f}"
    return f"{str(v):>{w}}"
lag_geoms, sac_geoms, lag_cs, sac_cs = [], [], [], []
for name, d in sorted(runs.items()):
    algo = name.split('_')[0]
    bg = d.get('best_geom', 0)
    cs = d.get('cum_collision_sphere', 0)
    if algo == 'lag': lag_geoms.append(bg); lag_cs.append(cs)
    else: sac_geoms.append(bg); sac_cs.append(cs)
    print(f"{name:<8} {fmt(d,'best_J'):>9} {fmt(d,'best_geom',6)} {fmt(d,'first_hold_ep',10)} {fmt(d,'full_hold_epochs',8)} {fmt(d,'min_d_err_mean',6)} {fmt(d,'cum_collision_sphere',7)} {fmt(d,'cum_collision_table',6)}")
print()
lag_pass = sum(1 for g in lag_geoms if isinstance(g, float) and g >= 0.5)
sac_pass = sum(1 for g in sac_geoms if isinstance(g, float) and g >= 0.5)
print(f"LagSAC ≥0.5 best_geom: {lag_pass}/{len(lag_geoms)}    SAC ≥0.5: {sac_pass}/{len(sac_geoms)}")
if lag_cs and sac_cs:
    print(f"mean cum_collision_sphere: LagSAC={sum(lag_cs)/len(lag_cs):.0f}  SAC={sum(sac_cs)/len(sac_cs):.0f}  (ratio={sum(sac_cs)/max(1,sum(lag_cs)):.2f}x SAC higher)")
PY

echo ""
echo "Done. tail /tmp/short_bench_summary.txt | less"
