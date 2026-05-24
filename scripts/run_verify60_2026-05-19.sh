#!/bin/bash
# 60ep verification — 是否单纯 epoch 不够? 不动 reward, 只测时间.
#
# Targets:
#   SAC s2 (was: 30ep hold=0, d crossed but no hold)    → 60ep 能否 hold?
#   LagSAC s8 (was: 30ep peak hold=1 一闪, ep30 掉 0.31) → 60ep 能否 stable?
#
# Config: sat=0.8 (verified), no soft_success, no other changes.
# 跑 ~40 min (SAC ~17 + LagSAC ~21 + init overhead).

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

LOGDIR="/tmp/verify60_logs"
SUMMARY="/tmp/verify60_summary.txt"
TMPCFG="/tmp/verify60_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"
SAC_S1_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"

TAG="verify_sat0p8_60ep_$(date +%Y%m%d)"
START_TIME=$(date +%s)

cat > "$SUMMARY" <<EOF
=== 60ep verification started $(date '+%Y-%m-%d %H:%M:%S') ===
Targets: SAC s2 + LagSAC s8
Config: geom_d_sat=0.8, noise=0.05, harder pose, 60ep
NO reward changes (no soft_success, etc).

EOF

build_yaml() {
    local base=$1 seed=$2 out=$3 name=$4
    sed -E "s/^seed:.*/seed: $seed/" "$base" > "$out"
    sed -i -E 's/^n_epochs:.*/n_epochs: 60/' "$out"
    if grep -q '^geom_d_sat:' "$out"; then
        sed -i -E 's/^geom_d_sat:.*/geom_d_sat: 0.8/' "$out"
    else
        sed -i '/^wandb_project:/i geom_d_sat: 0.8' "$out"
    fi
    if grep -q '^initial_joint_noise:' "$out"; then
        sed -i -E 's/^initial_joint_noise:.*/initial_joint_noise: 0.05/' "$out"
    else
        sed -i '/^wandb_project:/i initial_joint_noise: 0.05' "$out"
    fi
    if grep -q '^default_pose_variant:' "$out"; then
        sed -i -E 's/^default_pose_variant:.*/default_pose_variant: harder/' "$out"
    else
        sed -i '/^wandb_project:/i default_pose_variant: harder' "$out"
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
holds = [float(x) for x in re.findall(r'geom_hold_rate=(\d+\.\d+)', log)]
d_errs = [float(x) for x in re.findall(r'd_err_mean=(\d+\.\d+)', log)]
spheres = [int(x) for x in re.findall(r'epoch_collision_sphere=(\d+)', log)]
tables = [int(x) for x in re.findall(r'epoch_collision_table=(\d+)', log)]
first_hold = next((i+1 for i, h in enumerate(holds) if h > 0), None)
first_full = next((i+1 for i, h in enumerate(holds) if h >= 1.0), None)
full_epochs = sum(1 for h in holds if h >= 1.0)
hold_last = holds[-1] if holds else 0.0
print(f"  best_J={m_J.group(1) if m_J else '?'}")
print(f"  best_geom={m_geom.group(2) if m_geom else '?'}")
print(f"  first_hold_ep={first_hold}")
print(f"  first_full_ep={first_full}")
print(f"  full_hold_epochs={full_epochs}")
print(f"  hold_at_last={hold_last:.3f}")
print(f"  min_d_err_mean={min(d_errs):.3f}" if d_errs else "  min_d_err_mean=?")
print(f"  cum_collision_sphere={sum(spheres)}")
print(f"  cum_collision_table={sum(tables)}")
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

# Build yamls
build_yaml "$SAC_S1_YAML" 2 "$TMPCFG/sac_s2_sat0p8_60ep.yaml" "sac_s2_sat0p8_60ep"
build_yaml "$LAG_S1_YAML" 8 "$TMPCFG/lag_s8_sat0p8_60ep.yaml" "lag_s8_sat0p8_60ep"

# Run sequentially
run_one "sac_s2_sat0p8_60ep" "$TMPCFG/sac_s2_sat0p8_60ep.yaml"
run_one "lag_s8_sat0p8_60ep" "$TMPCFG/lag_s8_sat0p8_60ep.yaml"

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

cat >> "$SUMMARY" <<EOF
============================================================
=== DONE at $(date '+%Y-%m-%d %H:%M:%S') ===
Total: $((TOTAL_DUR / 60)) min

Reference (30ep results, sat=0.8):
  SAC s2:    bestG=0 hold@30=0 full=0/30   ← did 60ep fix this?
  LagSAC s8: bestG=1 hold@30=0.31 full=2/30 ← did 60ep stabilize?

Decision:
  Both stable (full_epochs ≥ 10) → 60ep IS the fix, run 10-seed final benchmark
  Either still stuck → reward IS the issue, can't fix with epochs alone
============================================================
EOF

echo ""
echo "Done. Summary: cat $SUMMARY"
