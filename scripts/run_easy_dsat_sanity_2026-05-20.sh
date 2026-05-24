#!/bin/bash
# Easy pose geom_d_sat sanity check — short paired ablation.
#
# Goal:
#   Test whether the harder-pose fix `geom_d_sat=0.8` hurts easy/default pose.
#
# Design:
#   easy/default pose, seeds 0 1 2
#   dsat in {0.3, 0.8}
#   algos in {LagSAC, SAC}
#   n_epochs=30
#
# Other params follow the proposed new easy benchmark setup:
#   initial_joint_noise=0.05
#   LagSAC cost_limit_per_ep=10.0
#   rew_geom_soft_success=0.0
#   all task reward weights otherwise from canonical Stage 1 YAMLs.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_easy_dsat_sanity_2026-05-20.sh
#   bash scripts/run_easy_dsat_sanity_2026-05-20.sh
#   tail -F /tmp/easy_dsat_sanity_summary.txt

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/easy_dsat_sanity_logs"
SUMMARY="/tmp/easy_dsat_sanity_summary.txt"
TMPCFG="/tmp/easy_dsat_sanity_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"
SAC_S1_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"

SEEDS=(0 1 2)
DSATS=(0.3 0.8)
N_EPOCHS=30
EASY_NOISE=0.05
LAG_COST_LIMIT=10.0
TAG="easy_dsat_sanity_$(date +%Y%m%d)"
RUN_TO=18m
DRY_RUN="${DRY_RUN:-0}"

START_TIME=$(date +%s)

cat > "$SUMMARY" <<EOF
=== Easy pose geom_d_sat sanity started $(date '+%Y-%m-%d %H:%M:%S') ===
PROJECT_ROOT=$PROJECT_ROOT
LOGDIR=$LOGDIR
TAG=$TAG
DRY_RUN=$DRY_RUN

Design:
  pose=easy
  seeds=${SEEDS[*]}
  geom_d_sat=${DSATS[*]}
  algos=LagSAC SAC
  n_epochs=$N_EPOCHS
  initial_joint_noise=$EASY_NOISE
  LagSAC cost_limit_per_ep=$LAG_COST_LIMIT
  rew_geom_soft_success=0.0
  timeout/run=$RUN_TO

Decision rule:
  If dsat=0.8 has comparable or better best_geom/best_J without higher collision
  on this 3-seed paired check, use dsat=0.8 for the full easy benchmark.
  If dsat=0.8 raises collision or lowers success, keep dsat=0.3 for easy pose.

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
text = open(sys.argv[1], errors="ignore").read()
m_J = re.search(r"训练完成.*?best J = (-?\d+\.\d+)", text)
m_geom = re.search(r"训练完成.*?best[ _]geom(?:_hold)?_rate[ =]+(\d+\.\d+)", text)
m_score = re.search(r"best_score = (-?\d+\.\d+)", text)
holds = [float(x) for x in re.findall(r"geom_hold_rate=(\d+\.\d+)", text)]
d_errs = [float(x) for x in re.findall(r"d_err_mean=(\d+\.\d+)", text)]
spheres = [int(x) for x in re.findall(r"epoch_collision_sphere=(\d+)", text)]
tables = [int(x) for x in re.findall(r"epoch_collision_table=(\d+)", text)]
lambdas = [float(x) for x in re.findall(r"lam: ([\d.e+-]+)", text)]

first_hold = next((i + 1 for i, h in enumerate(holds) if h > 0.0), None)
first_full = next((i + 1 for i, h in enumerate(holds) if h >= 1.0), None)
full_epochs = sum(1 for h in holds if h >= 1.0)
hold_last = holds[-1] if holds else 0.0
hold_last10 = sum(holds[-10:]) / min(10, len(holds)) if holds else 0.0
min_d = min(d_errs) if d_errs else None

print(f"best_J={m_J.group(1) if m_J else '?'}")
print(f"best_geom={m_geom.group(1) if m_geom else '0.0'}")
print(f"best_score={m_score.group(1) if m_score else '?'}")
print(f"first_hold_ep={first_hold}")
print(f"first_full_ep={first_full}")
print(f"full_hold_epochs={full_epochs}")
print(f"hold_at_last={hold_last:.3f}")
print(f"hold_last10_mean={hold_last10:.3f}")
print(f"min_d_err_mean={min_d:.4f}" if min_d is not None else "min_d_err_mean=?")
print(f"cum_collision_sphere={sum(spheres)}")
print(f"cum_collision_table={sum(tables)}")
if lambdas:
    print(f"mean_lambda_late10={sum(lambdas[-10:]) / min(10, len(lambdas)):.4f}")
PY
}

build_yaml() {
    local algo="$1" seed="$2" dsat="$3" out="$4" name="$5"
    local base
    [ "$algo" = "lag" ] && base="$LAG_S1_YAML" || base="$SAC_S1_YAML"

    sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$base" > "$out"
    set_yaml_scalar "$out" "n_epochs" "$N_EPOCHS"
    set_yaml_scalar "$out" "default_pose_variant" "easy"
    set_yaml_scalar "$out" "initial_joint_noise" "$EASY_NOISE"
    set_yaml_scalar "$out" "geom_d_sat" "$dsat"
    set_yaml_scalar "$out" "rew_geom_soft_success" "0.0"
    if [ "$algo" = "lag" ]; then
        set_yaml_scalar "$out" "cost_limit_per_ep" "$LAG_COST_LIMIT"
    fi
    set_yaml_scalar "$out" "wandb_run_name" "$name"
    set_yaml_scalar "$out" "wandb_group" "$TAG"
}

run_one() {
    local algo="$1" dsat="$2" seed="$3"
    local dsat_tag="${dsat/./p}"
    local name="${algo}_easy_dsat${dsat_tag}_s${seed}"
    local cfg="$TMPCFG/${name}.yaml"
    local log="$LOGDIR/${name}.log"
    build_yaml "$algo" "$seed" "$dsat" "$cfg" "$name"

    local start_ts start_human rc end_ts end_human dur status
    start_ts=$(date +%s)
    start_human=$(date '+%H:%M:%S')

    local cmd=(
        python scripts/run_lagrangian_chain_local_from_yaml.py
        --start_stage 1 --stop_stage 1
        --stage1_cfg "$cfg"
        --tag "$TAG"
        --skip_snapshot
    )

    echo "[$start_human] START $name" | tee -a "$SUMMARY"
    echo "  cfg=$cfg" >> "$SUMMARY"
    echo "  cmd=${cmd[*]}" >> "$SUMMARY"

    if [ "$DRY_RUN" = "1" ]; then
        grep -E "^(seed|n_epochs|default_pose_variant|initial_joint_noise|geom_d_sat|cost_limit_per_ep|rew_geom_soft_success|wandb_run_name|wandb_group):" "$cfg" | sed 's/^/  /' | tee -a "$SUMMARY"
        echo "" >> "$SUMMARY"
        return 0
    fi

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    timeout -k 60s "$RUN_TO" "${cmd[@]}" > "$log" 2>&1
    rc=$?

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    end_ts=$(date +%s)
    end_human=$(date '+%H:%M:%S')
    dur=$((end_ts - start_ts))
    status="OK"
    [ "$rc" -ne 0 ] && status="FAIL(rc=$rc)"

    echo "[$end_human] END   $name dur=${dur}s status=$status" | tee -a "$SUMMARY"
    extract_metrics "$log" | sed 's/^/  /' | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
}

# Interleave by seed, dsat, algo for paired partial results.
for seed in "${SEEDS[@]}"; do
    for dsat in "${DSATS[@]}"; do
        run_one "lag" "$dsat" "$seed"
        run_one "sac" "$dsat" "$seed"
    done
done

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF

============================================================
=== Easy pose geom_d_sat sanity DONE $(date '+%Y-%m-%d %H:%M:%S') ===
Total duration: $((TOTAL_DUR / 60)) min
============================================================
EOF

echo "" | tee -a "$SUMMARY"
echo "Aggregate:" | tee -a "$SUMMARY"
python3 - <<PY | tee -a "$SUMMARY"
import re, statistics
from pathlib import Path

summary = Path("$SUMMARY").read_text(errors="ignore")
runs = {}
cur = None
for line in summary.splitlines():
    m = re.match(r"\[\d\d:\d\d:\d\d\] END\s+(\w+)_easy_dsat(\dp\d)_s(\d+)", line)
    if m:
        algo, dsat, seed = m.group(1), m.group(2).replace("p", "."), int(m.group(3))
        cur = (algo, dsat, seed)
        runs[cur] = {}
        continue
    if cur:
        m = re.match(r"\s+(\w+)=(-?[\d.]+|None|\?)", line)
        if m:
            key, val = m.group(1), m.group(2)
            try:
                runs[cur][key] = float(val)
            except ValueError:
                runs[cur][key] = val

def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3g} (n=1)"
    return f"{statistics.mean(vals):.3g} +/- {statistics.stdev(vals):.3g} (n={len(vals)})"

for algo in ("lag", "sac"):
    print(f"\\n{algo.upper()}")
    for dsat in ("0.3", "0.8"):
        group = [d for (a, s, _), d in runs.items() if a == algo and s == dsat]
        pass_count = sum(1 for d in group if isinstance(d.get("best_geom"), (int, float)) and d["best_geom"] >= 0.5)
        print(f"  dsat={dsat}: runs={len(group)} pass(best_geom>=0.5)={pass_count}/{len(group)}")
        for key in ("best_J", "best_geom", "best_score", "full_hold_epochs", "cum_collision_sphere", "cum_collision_table", "min_d_err_mean"):
            print(f"    {key}: {mean_std([d.get(key) for d in group])}")
PY
