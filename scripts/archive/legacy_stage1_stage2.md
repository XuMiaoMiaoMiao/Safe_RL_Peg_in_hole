# Legacy Stage 1/2 SAC Notes (pre-geom_stage)

This is a historical snapshot of the old Stage 1 / Stage 2 SAC path before the
current `geom_stage` main line. The top-level `results/S*.msh` files referenced
here were cleaned from the workspace on 2026-05-12. Current usable checkpoints
live under `results/checkpoints/saved/`.

Do not use this file as the current training guide. The current guide is the
repo-root `README.md`.

## Context

The legacy path used the old spherical `pos_err` / `axis_err` reward path:

- Stage 1: position-only preinsert, `success_axis_threshold=inf`.
- Stage 2: preinsert position plus axis alignment, with a hard success cliff.
- Stage 3 v4 later tried V-shape depth + Phase A/B/C schedule; that code has
  been removed from main and can be restored from `scripts/archive/v4_baseline.py`.

The current main line is instead:

- Stage 1g: `geom_stage=prepos`
- Stage 2g: `geom_stage=preaxis`
- Stage 3g: `geom_stage=insert`

## Legacy Stage 1 Summary

Verified around 2026-05-08 with Davide init + sphere-proxy collision + hold-N
absorbing disabled.

| Parameter | Value |
|---|---|
| `lr_actor` | `1e-4` |
| `target_entropy` | `-7` |
| `alpha_max` | `0.05` |
| `n_epochs` | `50` |
| `n_steps_per_epoch` | `1024` |
| `num_envs` | `16` |
| `--use_axis_resid_obs` | enabled |
| `preinsert_success_pos_threshold` | `0.10m` |
| `success_axis_threshold` | `inf` |
| `rew_axis` | `0.0` |
| `terminal_hold_bonus` | `0` |

5-seed sweep result:

| seed | best_J | best_hold_rate | peak_max_hold | sphere_absorb |
|---|---:|---:|---:|---:|
| 0 | 53.01 | 1.000 | 58.8 | 39 |
| 42 | 7.45 | 0.875 | 40.6 | 60 |
| 999 | 4.96 | 0.812 | 20.5 | 54 |
| 123 | -4.38 | 0.688 | 16.6 | 78 |
| 7 | -12.14 | 0.375 | 15.1 | 73 |

Historical deploy checkpoint:

```text
results/S1_B_davideinit_seed0_best_agent.msh
```

This file is no longer present after cleanup. The current Stage 1g checkpoint is:

```text
results/checkpoints/saved/Stage1_prepos_for_stage2_S1g_refine_seed0_best_hold.msh
```

Historical training command:

```bash
python scripts/train_sac.py \
  --num_envs 16 --n_epochs 50 --n_steps_per_epoch 1024 --n_steps_per_fit 16 --n_eval_episodes 16 \
  --use_axis_resid_obs \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold inf \
  --rew_axis 0.0 --rew_home 0.0005 \
  --lr_actor 1e-4 --alpha_max 0.05 --target_entropy -7 \
  --terminal_hold_bonus 0 --hold_success_steps 10 \
  --seed 0 \
  --wandb_run_name S1_B_davideinit_seed0
```

## Legacy Stage 2 Summary

Verified around 2026-05-09. The important lesson was that
`success_axis_threshold` acted as a reward-function switch, not just a metric:
too-strict thresholds made the cliff bonus almost never fire.

| Parameter | Value |
|---|---|
| warm-start | legacy Stage 1 best agent |
| `--actor_only_warmstart` | enabled |
| `--critic_warmup_transitions` | `50000` |
| `success_axis_threshold` | `0.40` |
| `axis_gate_radius` | `0.40` |
| `rew_axis` | `1.0` |
| `rew_pos_success` | `1.0` |
| `rew_success` | `4.0` |
| `lr_actor` | `5e-5` |
| `alpha_max` | `0.05` |
| `target_entropy` | `-10` |
| `n_epochs` | `200` |

Comparison of legacy Stage 2 experiments:

| | exp 1 (`axis_th=0.20`) | exp 2 (`axis_th=0.30`) | exp 3 (`axis_th=0.40`) |
|---|---:|---:|---:|
| best epoch | 85 | 94 | 194 |
| best_J | 26.7 | 22.0 | 189 |
| hold@train_th | 0 | 0.25 | 1.00 |
| max_hold_mean | 0 | 5.5 | 105.7 |
| axis_in_pos_mean | 0.63 | 0.63 | 0.41 |
| axis_in_pos_min | 0.27 | 0.27 | 0.29 |

Historical deploy checkpoint:

```text
results/S2_oneshot_ep194_seed0_best_agent.msh
```

This file is no longer present after cleanup. The current Stage 2g checkpoint is:

```text
results/checkpoints/saved/Stage2_preaxis_for_stage3_2026-05-11_11-47-34_best_hold.msh
```

Historical training command:

```bash
python scripts/train_sac.py \
  --num_envs 16 --n_epochs 200 --n_steps_per_epoch 1024 --n_steps_per_fit 16 --n_eval_episodes 16 \
  --use_axis_resid_obs \
  --load_agent results/S1_B_davideinit_seed0_best_agent.msh --actor_only_warmstart --critic_warmup_transitions 50000 \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold 0.40 --axis_gate_radius 0.40 \
  --rew_axis 1.0 --rew_pos_success 1.0 --rew_success 4.0 --rew_home 0.0005 \
  --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --lr_actor 5e-5 --alpha_max 0.05 --target_entropy -10 \
  --terminal_hold_bonus 0 --hold_success_steps 10 \
  --seed 0 \
  --wandb_run_name S2_oneshot_axis040_rewx2_lowlr_seed0
```

## Lessons Kept In Main

- Hold-N absorbing should stay off for dense training; success is an eval metric.
- Cliff reward trains the distribution center, not the floor.
- `n_steps_per_epoch` must match `horizon * num_envs` for the geom insert path.
- Peg/hole feasibility must be checked with geometry (`penetration_max`), because
  the Stage 3 training path excludes EE PhysX self-collision to avoid false
  hard-absorbing contacts.
