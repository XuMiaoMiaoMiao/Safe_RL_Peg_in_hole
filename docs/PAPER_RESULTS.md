# Paper Results — SAC vs LagSAC Bimanual Peg-in-Hole

**Project**: Course-level bimanual peg-in-hole comparison
**Reference**: Günster, Liu, Peters, Tateo 2024 (D-ATACOM, arxiv 2409.12045)
**Date**: 2026-05-18

## Task
- KUKA iiwa7 dual-arm in IsaacSim, peg-in-hole insertion
- 14-DoF action (2× 7 joints), velocity control
- 41-D obs (joint_pos + joint_vel + relative geometry)
- 3-stage curriculum: prepos (Stage 1g) → preaxis (Stage 2g) → insert (Stage 3g)

## Safety Architecture (B-route)
Three independent safety signals, all treated as a single soft constraint:
1. **Arm-arm sphere proxy** (17 spheres per side, joint + midpoint + EE) → cost-only (no termination)
2. **Arm-arm PhysX contact** (14 invisible capsule colliders on iiwa links) → hard absorbing + cliff
3. **Arm/EE-table geometric clearance** (`z - radius - table_z`) → hard absorbing + cliff + cost

CMDP cost = `max(arm_arm_clearance_violation, table_clearance_violation)` (D-ATACOM-style positive constraint, no clip).

## Results Summary

### Stage 1 prepos (main comparison)

#### Default pose (4 seeds, 30 epochs)
| Metric | LagSAC | SAC | Winner |
|---|---:|---:|:-:|
| best_J | -99.4 ± 14.6 | -116.2 ± 32.7 | **Lag (+16%)** |
| best_geom_rate | 0.78 ± 0.40* | 0.81 ± 0.06 | tie |
| cum_collision_total | 1751 ± 442 | 4517 ± 1294 | **Lag (2.6× safer)** |
| cum_collision_table | 42 | 94 | **Lag** |
| constraint_sat_rate | 0.66 ± 0.30 | 0.43 ± 0.29 | **Lag (+53%)** |
| Sample efficiency (ep→0.9) | 23.3 | not reached | **Lag** |

*lag_default_s3 outlier (geom=0.19), other 3 seeds = 0.94-1.00

#### Harder pose (5 seeds, 60 epochs)
Note: "Harder" = 90° axis misalignment + EE near hole at reset (not crossed forearm as originally targeted).

| Metric | LagSAC | SAC | Winner |
|---|---:|---:|:-:|
| best_J | -23.4 ± 9.5 | -22.6 ± 3.0 | tie |
| best_geom_rate | 0.95 ± 0.08 | 0.99 ± 0.03 | tie |
| cum_collision_total | 1707 ± 1280 | 2428 ± 1707 | **Lag (1.4× safer)** |
| cum_collision_table | 166 | 145 | tie |
| Sample efficiency (ep→0.9) | 47.0 | 47.6 | tie |
| Final λ | 0.000 (saturated) | n/a | — |

### Stage 3 insertion
**Pending 100-ep results.** Earlier 50-ep runs (LagSAC cost=100/50, SAC) all hit best_geom=0 by epoch 50 — *expected*, because the SAC v8 reference recipe also takes until epoch 74 to first reach success (run-20260512_210113-j86aecgp). 100-ep runs (launched 2026-05-18 09:25) are budgeted to reproduce v8 breakthrough.

**Key caveat for Stage 3:** Once peg enters hole, both EEs are kinematically locked relative to each other (peg/hole physically constrain the spheres), so arm-arm clearance cost ≡ 0 throughout. λ decays to ~0 → Lagrangian effectively inactive. Stage 3 therefore does **not** support a Lagrangian-vs-SAC safety comparison. Its purpose is to verify that B-route safety semantics **do not break downstream insertion learning** — i.e., LagSAC achieves comparable J and success rate as SAC v8 baseline.

## Headline Narrative (for paper)

LagSAC with B-route safety achieves both higher returns and substantially fewer safety violations than vanilla SAC on Stage 1 prepos (J = -99.4 ± 14.6 vs -116.2 ± 32.7, cumulative violations 2.6× lower, 4 seeds). Under a harder initial pose, both methods converge to comparable returns and 95-99% success rate, but LagSAC retains a 1.4× safety advantage during training, with its Lagrange multiplier decaying to ≈0 once task gradient dominates (consistent with D-ATACOM's analysis of dual variable behavior under successful primal optimization). On Stage 3 insertion, the Lagrangian becomes effectively inactive because peg/hole kinematic coupling drives arm-arm clearance cost to zero; the comparison there primarily verifies that the safety constraints do not interfere with downstream task learning.

## Reproduction Artifacts

### Active configs (`conf/experiment/`)
- `lag_stage1_prepos_clearance_b_route.yaml` — LagSAC Stage 1 baseline (paper main)
- `sac_stage1_prepos_clearance_clean_sphere.yaml` — SAC Stage 1 baseline (paper main)
- `lag_stage3_b_route_calib.yaml` / `_strict.yaml` — Stage 3 LagSAC (cost=100/50)
- `sac_stage3_b_route_calib.yaml` — Stage 3 SAC baseline
- `sac_stage3_insert.yaml` — original v8 reference (74-ep breakthrough source)

### Checkpoints saved (`results/checkpoints*/2026-05-18/`)
- Per-seed best_hold + final per Block P1-A and P1-B
- Stage 3 100-ep results pending

### W&B groups
- `overnight_s1_default_20260518` — Block P1-A (8 runs)
- `overnight_s1_harder_20260518` — Block P1-B (10 runs)
- `s3_100ep_20260518_0925` — Block P2-C (3 runs)

### Analysis scripts
- `scripts/analyze_overnight.py` — aggregate metrics across runs
- `/tmp/overnight_full_report.md` — d-ATACOM-style full table

## Known issues / future work
1. Single-seed Stage 3 (3 runs only, no error bars on Stage 3 metrics)
2. `lag_default_s3` outlier needs replication with different seed to confirm non-systematic
3. Harder pose ended up testing axis-misalignment, not crossed-forearm geometric blocking
4. PhysX hang at Epoch 17 occurred once during initial overnight run; fixed with `timeout -k` SIGKILL guard
