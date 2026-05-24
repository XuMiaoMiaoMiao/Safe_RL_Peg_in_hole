# Archived Failed/Superseded Experiments — conf/experiment/

This directory contains 18 yaml configurations that were active during the 2026-05-12 → 2026-05-17 design iteration, but are now **superseded** by the B-route Lagrangian configurations in `conf/experiment/`.

Kept here (not deleted) for:
1. Git history continuity / experiment reproducibility
2. Reference if reviewers ask "why didn't you try X" — odds are it's here, with documented failure mode in [[feedback_bimanual_sac_failures]] memory entry
3. Lineage tracking for the B-route design

## Stage 1 LagSAC failed iterations (8 files)
- `lag_stage1_prepos_active_*.yaml` — early-2026-05-16 "active Lagrangian" attempts (v1 fixed-AB, v2 verify, v3 overlay, etc.) before B-route discovery
- `lag_stage1_prepos_active_clearance05_lam1_*.yaml` — clearance margin 0.5 with init λ=1, both with and without collision-reward-penalty hybrid; superseded by B-route v4 with margin 0.02
- `lag_stage1_prepos_active_clearance_nocollpen.yaml` / `_active_nocollpen.yaml` — pure no-cliff variants; rejected because absorbing termination still needed for sim stability
- `lag_stage1_prepos_clearance_clean_sphere_lag.yaml` — early "clean sphere" attempt, superseded by current `lag_stage1_prepos_clearance_b_route.yaml`
- `lag_stage1_prepos_clearance_conservative.yaml` — v1 conservative (lr_λ=0.001, λ_init=0.10) — documented in MEMORY as 36-epoch λ-freeze failure
- `lag_stage1_prepos_overnight_overlay.yaml` — overnight overlay variant, superseded

## Stage 2 LagSAC failed iterations (2 files)
- `lag_stage2_preaxis_active_clearance05_lam1_from_s1.yaml` — paired with active stage1 above
- `lag_stage2_preaxis_overnight_overlay.yaml` — overnight overlay variant

## Stage 3 LagSAC failed iterations (5 files)
- `lag_stage3_insert_clearance_constraint_keepshape.yaml` — early B-route precursor
- `lag_stage3_insert_local_tuned.yaml` — manual tuning attempt
- `lag_stage3_insert_overnight_overlay.yaml` — overnight overlay
- `lag_stage3_insert_pen_clean_scale1000.yaml` — penetration cost with scale=1000; rejected (see MEMORY [[feedback_bimanual_lagsac_cost_signal_trap]])
- `lag_stage3_insert_pen_hybrid_keepshape.yaml` — penetration + keepshape hybrid; rejected
- `lag_stage3_insert_pen_hybrid_scale1000.yaml` — penetration hybrid scale=1000; rejected

## What replaced these (in `../experiment/` active dir)
- Stage 1: `lag_stage1_prepos_clearance_b_route.yaml` (verified 2026-05-18 overnight)
- Stage 2: `lag_stage2_preaxis_local_tuned.yaml` (active for chain runs)
- Stage 3: `lag_stage3_b_route_calib.yaml` (cost=100) + `lag_stage3_b_route_calib_strict.yaml` (cost=50)
- SAC baselines: `sac_stage1_prepos_clearance_clean_sphere.yaml`, `sac_stage2_preaxis.yaml`, `sac_stage3_b_route_calib.yaml`

## How to restore (if needed)
```bash
# To revive any single config:
git mv conf/archive/failed_experiments/<name>.yaml conf/experiment/
# OR
cp conf/archive/failed_experiments/<name>.yaml conf/experiment/<name>.yaml
```

Archived 2026-05-18 by session cleanup pass.
