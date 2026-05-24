#!/bin/bash
# Tonight overnight benchmark — Stage 1 harder pose multi-seed + best-LagSAC chain.
#
# Plan (~10h budget):
#   Block 1: Stage 1 harder 60ep × 10 seeds × {SAC, LagSAC} = 20 runs (~6.3h)
#   Block 2: Pick best LagSAC seed by (geom_rate, best_J), then chain:
#            2a: LagSAC S2 (120ep) → S3 (100ep) warm-started from that seed's S1 ckpt
#            2b: SAC chain at SAME seed for apples-to-apples comparison
#   Block 3: Record Stage 3 final videos for both chains (~10min)
#
# Usage:
#   ./scripts/run_tonight_2026-05-18.sh
#   DRY_RUN=1 ./scripts/run_tonight_2026-05-18.sh    # print plan only, no launches
#   nohup ./scripts/run_tonight_2026-05-18.sh > /tmp/tonight_main.log 2>&1 & disown
#
# Outputs:
#   /tmp/tonight_logs/<run_name>.log     — full stdout/stderr per run
#   /tmp/tonight_summary.txt             — append-only progress, ends with DONE marker
#   results/checkpoints*/2026-05-*/      — training ckpts
#   results/videos/tonight_*/            — recorded videos

set -uo pipefail
# Deliberately NOT set -e: individual run failures must not kill the whole script.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe_rl

# ──────────────────── Paths & config ────────────────────
LOGDIR="/tmp/tonight_logs"
SUMMARY="/tmp/tonight_summary.txt"
TMPCFG="/tmp/tonight_cfg"
mkdir -p "$LOGDIR" "$TMPCFG"

DRY_RUN="${DRY_RUN:-0}"
DATE_TAG=$(date +%Y%m%d)
DATESTR=$(date +%Y-%m-%d)
TONIGHT_DATESTR_NEXT=$(date -d "tomorrow" +%Y-%m-%d)

LAG_S1_YAML="$PROJECT_ROOT/conf/experiment/lag_stage1_prepos_clearance_b_route.yaml"
SAC_S1_YAML="$PROJECT_ROOT/conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml"
LAG_S2_YAML="$PROJECT_ROOT/conf/experiment/lag_stage2_preaxis_b_route.yaml"
SAC_S2_YAML="$PROJECT_ROOT/conf/experiment/sac_stage2_preaxis_b_route.yaml"
LAG_S3_YAML="$PROJECT_ROOT/conf/experiment/lag_stage3_b_route_calib.yaml"
SAC_S3_YAML="$PROJECT_ROOT/conf/experiment/sac_stage3_b_route_calib.yaml"

SEEDS=(0 1 2 3 4 5 6 7 8 9)
TAG_S1="tonight_s1_harder_${DATE_TAG}"
TAG_LAG_CHAIN="tonight_lag_chain_${DATE_TAG}"
TAG_SAC_CHAIN="tonight_sac_chain_${DATE_TAG}"
HARDER_EPOCHS="${HARDER_EPOCHS:-60}"
# The new hand-tuned harder pose has clean reset at noise=0.0/0.05. At noise=0.1,
# a small fraction of vector envs can start in sphere-proxy overlap, so the
# default tonight benchmark uses 0.05. Override with HARDER_NOISE=0.1 if needed.
HARDER_NOISE="${HARDER_NOISE:-0.05}"
VIDEO_NOISE="${VIDEO_NOISE:-0.0}"

START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$SUMMARY" <<EOF
=== Tonight benchmark started at $START_HUMAN ===
PROJECT_ROOT=$PROJECT_ROOT
LOGDIR=$LOGDIR
DATE_TAG=$DATE_TAG

Plan:
  Block 1: Stage 1 HARDER 60ep × 10 seeds × {LagSAC, SAC} = 20 runs (~6.3h)
  Block 2: Pick best LagSAC seed → chain S2 (120ep) → S3 (100ep) × {LagSAC, SAC} (~3h)
  Block 3: Record Stage 3 final videos × 2 (~10min)
  Total expected: ~9.7h. Budget: 10.5h.

  SEEDS=${SEEDS[*]}
  HARDER_EPOCHS=$HARDER_EPOCHS
  HARDER_NOISE=$HARDER_NOISE
  VIDEO_NOISE=$VIDEO_NOISE
  TAG_S1=$TAG_S1
  DRY_RUN=$DRY_RUN

EOF

# ──────────────────── Helpers ────────────────────

set_yaml_scalar() {
    local file="$1" key="$2" value="$3" before_key="${4:-wandb_project}"
    if grep -q "^${key}:" "$file"; then
        sed -i -E "s|^${key}:.*$|${key}: ${value}|" "$file"
    else
        sed -i "/^${before_key}:/i ${key}: ${value}" "$file"
    fi
}

# Build per-seed Stage 1 yaml with harder pose + seed override + run_name suffix.
make_s1_harder_yaml() {
    local base="$1" seed="$2" out="$3"
    sed -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$base" > "$out"
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_s${seed}_hard/" "$out"
    set_yaml_scalar "$out" "n_epochs" "$HARDER_EPOCHS"
    set_yaml_scalar "$out" "default_pose_variant" "harder"
    set_yaml_scalar "$out" "initial_joint_noise" "$HARDER_NOISE"
}

# Build chain yaml (Stage 2 or Stage 3): override load_agent + seed + run_name
# suffix. Keep reset distribution aligned with the harder-pose Stage 1 source
# policy, otherwise the chain silently switches back to easy HOME pose.
make_chain_yaml() {
    local base="$1" out="$2" load_path="$3" seed="$4" suffix="$5"
    # Override load_agent path (escape any slashes via | delimiter)
    sed -E "s|^load_agent:[[:space:]]+.+$|load_agent: $load_path|" "$base" > "$out"
    sed -i -E "s/^seed:[[:space:]]+[0-9-]+.*$/seed: $seed/" "$out"
    sed -i -E "s/^(wandb_run_name:[[:space:]]+.+)$/\1_${suffix}/" "$out"
    set_yaml_scalar "$out" "default_pose_variant" "harder"
    set_yaml_scalar "$out" "initial_joint_noise" "$HARDER_NOISE"
    # Keep safety geometry consistent across S1/S2/S3 tonight. Some older S3
    # configs still use 0.060 arm proxies; S1/S2 B-route uses 0.065.
    set_yaml_scalar "$out" "proxy_arm_radius" "0.065"
    set_yaml_scalar "$out" "proxy_ee_radius" "0.04"
}

# Run a single experiment with timeout + cleanup, log to file, append summary line.
# Args: name yaml tag stage timeout
run_experiment() {
    local name="$1" yaml="$2" tag="$3" stage="$4" to="$5"
    local log="$LOGDIR/${name}.log"
    local start_ts=$(date +%s) start_human=$(date '+%H:%M:%S')

    local cmd=(
        python scripts/run_lagrangian_chain_local_from_yaml.py
        --start_stage "$stage" --stop_stage "$stage"
        "--stage${stage}_cfg" "$yaml"
        --tag "$tag"
        --skip_snapshot
    )

    echo "[$start_human] START $name (yaml=$yaml stage=$stage to=$to)" | tee -a "$SUMMARY"
    echo "          CMD: ${cmd[*]}" >> "$SUMMARY"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[$start_human] DRY-RUN $name (not launched)" | tee -a "$SUMMARY"
        echo "" >> "$SUMMARY"
        return 0
    fi

    # Pre-cleanup: kill any zombie IsaacSim/python from previous run.
    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null
    sleep 2

    # timeout -k: SIGTERM then SIGKILL after 60s (IsaacSim PhysX ignores plain SIGTERM).
    timeout -k 60s "$to" "${cmd[@]}" > "$log" 2>&1
    local rc=$?

    pkill -9 -f train_sac_lagrangian.py 2>/dev/null
    pkill -9 -f "train_sac.py" 2>/dev/null
    pkill -9 -f run_lagrangian_chain_local_from_yaml.py 2>/dev/null

    local end_ts=$(date +%s) end_human=$(date '+%H:%M:%S')
    local dur=$((end_ts - start_ts))
    local status="OK"; [ $rc -ne 0 ] && status="FAIL (rc=$rc)"

    local best_J best_geom final_lambda
    best_J=$(grep -E "训练完成.*best J" "$log" 2>/dev/null | head -1 | grep -oE "best J = -?[0-9.]+" | awk '{print $4}')
    best_geom=$(grep -E "训练完成.*best[ _]geom(_hold)?_rate" "$log" 2>/dev/null | head -1 | grep -oE "best[ _]geom(_hold)?_rate[ =]+[0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    final_lambda=$(grep -E "训练完成.*final λ" "$log" 2>/dev/null | head -1 | grep -oE "final λ = [0-9.]+" | awk '{print $4}')

    echo "[$end_human] END   $name dur=${dur}s status=$status bestJ=${best_J:-?} bestGeom=${best_geom:-?} λ=${final_lambda:-?}" | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
    return $rc
}

# Find newest ckpt dir for an algo across today + tomorrow (handles midnight rollover).
# Args: algo ("lag" | "sac")
newest_ckpt_dir() {
    local algo="$1" subdir
    [ "$algo" = "lag" ] && subdir="checkpoints_lag" || subdir="checkpoints"
    ls -td "$PROJECT_ROOT/results/$subdir/$DATESTR"/* "$PROJECT_ROOT/results/$subdir/$TONIGHT_DATESTR_NEXT"/* 2>/dev/null | head -1
}

parse_best_J() {
    grep -E "训练完成.*best J" "$1" 2>/dev/null | head -1 | grep -oE "best J = -?[0-9.]+" | awk '{print $4}'
}
parse_best_geom() {
    grep -E "训练完成.*best[ _]geom(_hold)?_rate" "$1" 2>/dev/null | head -1 | grep -oE "best[ _]geom(_hold)?_rate[ =]+[0-9.]+" | grep -oE "[0-9.]+$" | head -1
}

# ──────────────────── Block 1: Stage 1 × 10 seeds × 2 algos ────────────────────
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Block 1: Stage 1 HARDER 60ep × 10 seeds × {LagSAC, SAC}" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

declare -A LAG_LOG=() LAG_CKPT=() SAC_LOG=() SAC_CKPT=()

for seed in "${SEEDS[@]}"; do
    cfg="$TMPCFG/lag_harder_s${seed}.yaml"
    make_s1_harder_yaml "$LAG_S1_YAML" "$seed" "$cfg"
    LAG_LOG[$seed]="$LOGDIR/lag_harder_s${seed}.log"
    if run_experiment "lag_harder_s${seed}" "$cfg" "$TAG_S1" 1 25m; then
        LAG_CKPT[$seed]="$(newest_ckpt_dir lag)"
    else
        LAG_CKPT[$seed]=""
    fi

    cfg="$TMPCFG/sac_harder_s${seed}.yaml"
    make_s1_harder_yaml "$SAC_S1_YAML" "$seed" "$cfg"
    SAC_LOG[$seed]="$LOGDIR/sac_harder_s${seed}.log"
    if run_experiment "sac_harder_s${seed}" "$cfg" "$TAG_S1" 1 25m; then
        SAC_CKPT[$seed]="$(newest_ckpt_dir sac)"
    else
        SAC_CKPT[$seed]=""
    fi
done

# ──────────────────── Block 2 selection: best LagSAC seed ────────────────────
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Block 2 selection: best LagSAC seed by (geom_rate, best_J)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

BEST_SEED="" BEST_KEY=""
for seed in "${SEEDS[@]}"; do
    log="${LAG_LOG[$seed]}"
    bj=$(parse_best_J "$log"); bg=$(parse_best_geom "$log")
    if [ -z "${LAG_CKPT[$seed]}" ] || [ ! -f "${LAG_CKPT[$seed]}/best_hold.msh" ]; then
        echo "  seed=$seed lag missing best_hold checkpoint; excluded" | tee -a "$SUMMARY"
        continue
    fi
    if [ -z "${SAC_CKPT[$seed]}" ] || [ ! -f "${SAC_CKPT[$seed]}/best_hold.msh" ]; then
        echo "  seed=$seed sac missing same-seed best_hold checkpoint; excluded" | tee -a "$SUMMARY"
        continue
    fi
    [ -z "$bj" ] && bj="-999"
    [ -z "$bg" ] && bg="0"
    # Composite: geom_rate dominates (×1000), best_J breaks ties (higher = less negative).
    key=$(python3 -c "print(${bg}*1000 + ${bj})")
    echo "  seed=$seed lag bestJ=$bj bestGeom=$bg key=$key" | tee -a "$SUMMARY"
    if [ -z "$BEST_KEY" ] || python3 -c "import sys; sys.exit(0 if $key > $BEST_KEY else 1)"; then
        BEST_KEY=$key
        BEST_SEED=$seed
    fi
done

if [ -z "$BEST_SEED" ]; then
    echo "ERROR: no valid LagSAC Stage 1 checkpoint found; skipping S2/S3 chains." | tee -a "$SUMMARY"
    LAG_S1_CKPT=""
    SAC_S1_CKPT=""
else
    LAG_S1_CKPT="${LAG_CKPT[$BEST_SEED]:-}/best_hold.msh"
    SAC_S1_CKPT="${SAC_CKPT[$BEST_SEED]:-}/best_hold.msh"
fi
echo "  >>> best LagSAC seed = $BEST_SEED (key=$BEST_KEY)" | tee -a "$SUMMARY"
echo "  LagSAC S1 warm-start: $LAG_S1_CKPT" | tee -a "$SUMMARY"
echo "  SAC    S1 warm-start: $SAC_S1_CKPT" | tee -a "$SUMMARY"

# ──────────────────── Block 2a: LagSAC chain S2 → S3 ────────────────────
LAG_S3_CKPT=""
if [ -n "$BEST_SEED" ] && { [ "$DRY_RUN" = "1" ] || [ -f "$LAG_S1_CKPT" ]; }; then
    echo "" | tee -a "$SUMMARY"
    echo "============================================================" | tee -a "$SUMMARY"
    echo "Block 2a: LagSAC chain S2 (120ep) → S3 (100ep) from seed=$BEST_SEED" | tee -a "$SUMMARY"
    echo "============================================================" | tee -a "$SUMMARY"

    lag_s2_cfg="$TMPCFG/lag_chain_s2.yaml"
    make_chain_yaml "$LAG_S2_YAML" "$lag_s2_cfg" "$LAG_S1_CKPT" "$BEST_SEED" "tonight"
    if run_experiment "lag_chain_s2_s${BEST_SEED}" "$lag_s2_cfg" "$TAG_LAG_CHAIN" 2 60m; then
        LAG_S2_DIR="$(newest_ckpt_dir lag)"
        LAG_S2_CKPT="$LAG_S2_DIR/best_hold.msh"
    else
        LAG_S2_CKPT=""
    fi
    echo "  LagSAC S2 ckpt: ${LAG_S2_CKPT:-N/A}" | tee -a "$SUMMARY"

    if [ "$DRY_RUN" = "1" ] || [ -f "$LAG_S2_CKPT" ]; then
        lag_s3_cfg="$TMPCFG/lag_chain_s3.yaml"
        make_chain_yaml "$LAG_S3_YAML" "$lag_s3_cfg" "$LAG_S2_CKPT" "$BEST_SEED" "tonight"
        if run_experiment "lag_chain_s3_s${BEST_SEED}" "$lag_s3_cfg" "$TAG_LAG_CHAIN" 3 75m; then
            LAG_S3_DIR="$(newest_ckpt_dir lag)"
            LAG_S3_CKPT="$LAG_S3_DIR/best_hold.msh"
        else
            LAG_S3_CKPT=""
        fi
        echo "  LagSAC S3 best_hold: $LAG_S3_CKPT" | tee -a "$SUMMARY"
    else
        echo "  SKIP LagSAC S3: S2 ckpt missing" | tee -a "$SUMMARY"
    fi
else
    echo "" | tee -a "$SUMMARY"
    echo "WARNING: LagSAC S1 best_hold.msh missing — skip LagSAC chain." | tee -a "$SUMMARY"
fi

# ──────────────────── Block 2b: SAC chain S2 → S3 (same seed) ────────────────────
SAC_S3_CKPT=""
if [ -n "$BEST_SEED" ] && { [ "$DRY_RUN" = "1" ] || [ -f "$SAC_S1_CKPT" ]; }; then
    echo "" | tee -a "$SUMMARY"
    echo "============================================================" | tee -a "$SUMMARY"
    echo "Block 2b: SAC chain S2 (120ep) → S3 (100ep) from seed=$BEST_SEED" | tee -a "$SUMMARY"
    echo "============================================================" | tee -a "$SUMMARY"

    sac_s2_cfg="$TMPCFG/sac_chain_s2.yaml"
    make_chain_yaml "$SAC_S2_YAML" "$sac_s2_cfg" "$SAC_S1_CKPT" "$BEST_SEED" "tonight"
    if run_experiment "sac_chain_s2_s${BEST_SEED}" "$sac_s2_cfg" "$TAG_SAC_CHAIN" 2 60m; then
        SAC_S2_DIR="$(newest_ckpt_dir sac)"
        SAC_S2_CKPT="$SAC_S2_DIR/best_hold.msh"
    else
        SAC_S2_CKPT=""
    fi
    echo "  SAC S2 ckpt: ${SAC_S2_CKPT:-N/A}" | tee -a "$SUMMARY"

    if [ "$DRY_RUN" = "1" ] || [ -f "$SAC_S2_CKPT" ]; then
        sac_s3_cfg="$TMPCFG/sac_chain_s3.yaml"
        make_chain_yaml "$SAC_S3_YAML" "$sac_s3_cfg" "$SAC_S2_CKPT" "$BEST_SEED" "tonight"
        if run_experiment "sac_chain_s3_s${BEST_SEED}" "$sac_s3_cfg" "$TAG_SAC_CHAIN" 3 75m; then
            SAC_S3_DIR="$(newest_ckpt_dir sac)"
            SAC_S3_CKPT="$SAC_S3_DIR/best_hold.msh"
        else
            SAC_S3_CKPT=""
        fi
        echo "  SAC S3 best_hold: $SAC_S3_CKPT" | tee -a "$SUMMARY"
    else
        echo "  SKIP SAC S3: S2 ckpt missing" | tee -a "$SUMMARY"
    fi
else
    echo "" | tee -a "$SUMMARY"
    echo "WARNING: SAC S1 best_hold.msh missing — skip SAC chain." | tee -a "$SUMMARY"
fi

# ──────────────────── Block 3: Record Stage 3 final videos ────────────────────
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "Block 3: Record Stage 3 final videos" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

VIDEO_OUT="$PROJECT_ROOT/results/videos/tonight_${DATE_TAG}"
mkdir -p "$VIDEO_OUT"

record_video() {
    local name="$1" ckpt="$2" tag="$3"
    # Fallback: if best_hold.msh missing (Stage 3 never reached success), use final_agent.msh.
    if [ ! -f "$ckpt" ]; then
        local fallback="${ckpt%/*}/final_agent.msh"
        if [ -f "$fallback" ]; then
            echo "  $name: best_hold.msh missing → fallback to final_agent.msh" | tee -a "$SUMMARY"
            ckpt="$fallback"
        else
            echo "  SKIP $name: no usable ckpt ($ckpt nor $fallback)" | tee -a "$SUMMARY"
            return
        fi
    fi
    local start_human=$(date '+%H:%M:%S')
    echo "  [$start_human] RECORD $name ckpt=$ckpt" | tee -a "$SUMMARY"
    pkill -9 -f isaac-sim 2>/dev/null; sleep 2
    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY-RUN: would record $name" | tee -a "$SUMMARY"
        return
    fi
    timeout -k 30s 15m python scripts/record_geom_video.py \
        --agent_path "$ckpt" \
        --geom_stage insert \
        --default_pose_variant harder \
        --initial_joint_noise "$VIDEO_NOISE" \
        --exclude_ee_from_physx_self_collision \
        --n_episodes 2 \
        --output_dir "$VIDEO_OUT" \
        --tag "$tag" \
        > "$LOGDIR/video_${name}.log" 2>&1
    local rc=$?
    pkill -9 -f isaac-sim 2>/dev/null
    local end_human=$(date '+%H:%M:%S')
    echo "  [$end_human] DONE   $name rc=$rc" | tee -a "$SUMMARY"
}

record_video "lag_s3_final" "$LAG_S3_CKPT" "lag_chain_s${BEST_SEED}"
record_video "sac_s3_final" "$SAC_S3_CKPT" "sac_chain_s${BEST_SEED}"

# ──────────────────── Done ────────────────────
END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

cat >> "$SUMMARY" <<EOF

============================================================
=== DONE at $END_HUMAN ===
Total duration: $((TOTAL_DUR / 60)) min ($((TOTAL_DUR / 3600))h $((TOTAL_DUR % 3600 / 60))m)
Best LagSAC seed: $BEST_SEED
LagSAC S3 best_hold: ${LAG_S3_CKPT:-N/A}
SAC    S3 best_hold: ${SAC_S3_CKPT:-N/A}
Videos: $VIDEO_OUT
Logs:   $LOGDIR
============================================================
EOF

echo "Tonight benchmark complete. Summary: $SUMMARY"
