# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 中做 peg-in-hole 的 SAC / LagSAC 训练。

当前主线是 **paper-grade SAC vs LagSAC benchmark**：两个算法使用同一个环境、同一个 task reward、同一个 hard termination 规则，区别只在 LagSAC 使用 cost critic 和 Lagrangian multiplier。

旧版长 README 已归档到 [docs/README_legacy_20260512.md](docs/README_legacy_20260512.md) 和 [docs/README_pre_rewrite_20260518.md](docs/README_pre_rewrite_20260518.md)。

## Current Status

日期：2026-05-18

当前 benchmark 使用 **B-route CMDP semantics**：

- task reward 和 safety cost 分离。
- sphere proxy / table clearance 是连续 cost signal，不再直接 terminate。
- PhysX arm contact 和 table collision 是 hard absorbing guard。
- hard absorbing 触发同一类 collision penalty，用来避免物理不稳定和 suicide policy。
- SAC 和 LagSAC 都记录同一组 safety/cost metrics，方便画论文风格对比图。

Stage 1 是当前主要对比实验。Stage 3 仍在做 100 epoch calibration；不要用 50 epoch 判断 Stage 3 是否失败，因为历史 SAC v8 第一次成功插入通常在 epoch 70 以后出现。

## Quick Start

进入环境：

```bash
cd ~/bimanual_peghole
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safe_rl
```

跑 Stage 1 LagSAC：

```bash
python scripts/run_lagrangian_chain_local_from_yaml.py \
  --start_stage 1 --stop_stage 1 \
  --stage1_cfg conf/experiment/lag_stage1_prepos_clearance_b_route.yaml \
  --tag manual_lagsac_s1
```

跑 Stage 1 SAC baseline：

```bash
python scripts/run_lagrangian_chain_local_from_yaml.py \
  --start_stage 1 --stop_stage 1 \
  --stage1_cfg conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml \
  --tag manual_sac_s1
```

跑 Stage 3 100 epoch calibration：

```bash
bash scripts/run_stage3_100ep.sh
```

生成论文风格结果图：

```bash
python scripts/analyze_overnight_paper.py
python scripts/plot_paper_benchmark.py
```

输出：

- `/tmp/overnight_report.md`
- `results/plots/overnight_paper/stage1_default_paper.svg`
- `results/plots/overnight_paper/stage1_harder_paper.svg`
- `results/plots/overnight_paper/stage1_paper_curves.csv`

## B-Route Semantics

| Component | Meaning | Used By Reward | Used By Cost | Terminates |
|---|---|---:|---:|---:|
| task geometry reward | prepos / preaxis / insert shaping | yes | no | no |
| arm-arm sphere clearance | D-ATACOM-style proxy violation | no | yes | no |
| table clearance | table collision / clearance violation | no | yes | yes when hard table collision is enabled |
| PhysX arm contact | real physical arm contact guard | hard cliff only | counted as safety event | yes |

The intended benchmark interpretation:

```text
SAC:
  maximize task reward
  monitor cost and violations

LagSAC:
  maximize task reward under E[cost] <= cost_limit
  update lambda with rollout episode cost
```

For paper plots, table contact and arm collision are both treated as safety violations. This is deliberate: a policy that avoids arm-arm contact by hitting the table is not safe.

## Canonical Configs

| Experiment | Config | Script | Epochs | Purpose |
|---|---|---|---:|---|
| Stage 1 LagSAC | [lag_stage1_prepos_clearance_b_route.yaml](conf/experiment/lag_stage1_prepos_clearance_b_route.yaml) | `scripts/train_sac_lagrangian.py` | 30 | main constrained baseline |
| Stage 1 SAC | [sac_stage1_prepos_clearance_clean_sphere.yaml](conf/experiment/sac_stage1_prepos_clearance_clean_sphere.yaml) | `scripts/train_sac.py` | 30 | native SAC baseline, same env semantics |
| Stage 3 SAC | [sac_stage3_b_route_calib.yaml](conf/experiment/sac_stage3_b_route_calib.yaml) | `scripts/train_sac.py` | 100 | insertion calibration |
| Stage 3 LagSAC cost 100 | [lag_stage3_b_route_calib.yaml](conf/experiment/lag_stage3_b_route_calib.yaml) | `scripts/train_sac_lagrangian.py` | 100 | relaxed constraint calibration |
| Stage 3 LagSAC cost 50 | [lag_stage3_b_route_calib_strict.yaml](conf/experiment/lag_stage3_b_route_calib_strict.yaml) | `scripts/train_sac_lagrangian.py` | 100 | stricter constraint calibration |

Important config flags:

```yaml
cost_signal: clearance
sphere_collision_terminates: false
physx_collision_terminates: true
enable_physx_arm_collision: true
enable_table_collision: true
table_collision_terminates: true
keep_collision_reward_penalty: true
```

Stage 3 additionally requires:

```yaml
exclude_ee_from_physx_self_collision: true
load_agent: results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh
```

## Checkpoints

Canonical warm-start chain:

| Stage | Checkpoint |
|---|---|
| Stage 1g prepos | `results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh` |
| Stage 2g preaxis | `results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh` |
| Stage 3g insert SAC v8 | `results/checkpoints/saved/Stage3g_insert_h200_from_h100_h150_stage2_2026-05-12_21-01-11/best_hold.msh` |

Use `best_hold.msh` for warm-start and visualization. `final_agent.msh` can drift after the best policy has already appeared.

## Current Results

Stage 1 default pose, 5 seeds:

| Algorithm | best J | hold success | cumulative safety violations | table violations |
|---|---:|---:|---:|---:|
| LagSAC | `-99.35 +/- 14.57` | `0.781 +/- 0.397` | `1751 +/- 443` | `42.5 +/- 44.9` |
| SAC | `-116.21 +/- 32.72` | `0.609 +/- 0.409` | `4517 +/- 1294` | `94.2 +/- 73.2` |

Interpretation: default pose is the cleaner paper result. LagSAC improves return and reduces total safety violations by about 61 percent.

Stage 1 harder pose, 5 seeds:

| Algorithm | best J | hold success | cumulative safety violations | table violations |
|---|---:|---:|---:|---:|
| LagSAC | `-23.36 +/- 9.48` | `0.950 +/- 0.082` | `1707 +/- 1280` | `166 +/- 103` |
| SAC | `-22.61 +/- 2.95` | `0.988 +/- 0.028` | `2428 +/- 1708` | `146 +/- 79` |

Interpretation: harder pose is not yet a clean paper comparison. Both algorithms mostly solve the task; LagSAC lowers total violations but table violations are not clearly better. Treat it as pose-design/debug data until the harder initial pose is finalized.

Stage 3:

- 50 epochs is too short for conclusion.
- Historical SAC v8 starts inserting around epoch 74 and reaches clean insertion around epoch 80.
- Use 100 epochs for SAC and LagSAC Stage 3 calibration.
- Current B-route Stage 3 cost may stay near zero because the Stage 2 warm-start already keeps arms/table safe. In that case Stage 3 mainly validates task learnability, not the safety advantage of LagSAC.

## Paper-Style Metrics

The plot script intentionally matches the structure of CMDP benchmark papers:

| Paper-style name | This repo metric | Notes |
|---|---|---|
| Discounted Return | `J` | gamma-discounted eval return |
| Maximum Violation | proxy from max cost / collision event rate | exact per-episode max violation was not logged in older runs |
| Episodic Sum of Cost | `eval_ep_cost` | sum of clearance/table cost over eval episodes |
| Task Success | `geom_hold_rate` | task-specific hold success rate |

The exact metric mapping is written into [scripts/plot_paper_benchmark.py](scripts/plot_paper_benchmark.py). If new runs log a true per-episode maximum violation, replace the current proxy there.

## Visualization

Visualize Stage 1 harder initial pose:

```bash
python scripts/visualize_policy.py \
  --agent_path results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh \
  --geom_stage prepos \
  --default_pose_variant harder \
  --initial_joint_noise 0.0 \
  --freeze_mode step --freeze_after_step 1 \
  --freeze_seconds 30 \
  --num_envs 4 --n_episodes 1 \
  --geom_d_target_neg=-0.08 \
  --geom_d_target_pos 0.03 \
  --geom_d_sat 0.3 --geom_radial_sat 1.5 \
  --geom_d_th 0.03 --geom_r_tip_th 0.03 \
  --rew_geom_d 8.0 --rew_geom_radial_tip 8.0
```

Visualize the full Stage 1 policy rollout:

```bash
python scripts/visualize_policy.py \
  --agent_path results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh \
  --geom_stage prepos \
  --default_pose_variant harder \
  --initial_joint_noise 0.0 \
  --freeze_mode episode_end \
  --freeze_seconds 20 \
  --num_envs 4 --n_episodes 4 \
  --geom_d_target_neg=-0.08 \
  --geom_d_target_pos 0.03 \
  --geom_d_sat 0.3 --geom_radial_sat 1.5 \
  --geom_d_th 0.03 --geom_r_tip_th 0.03 \
  --rew_geom_d 8.0 --rew_geom_radial_tip 8.0
```

Visualize collision/table diagnostics:

```bash
python scripts/visualize_policy.py \
  --agent_path results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh \
  --geom_stage prepos \
  --default_pose_variant harder \
  --initial_joint_noise 0.0 \
  --enable_physx_arm_collision \
  --enable_table_collision \
  --table_collision_terminates \
  --table_z 0.0 \
  --table_clearance_hard 0.0 \
  --table_clearance_cost_margin 0.03 \
  --debug_show_sphere_proxy \
  --num_envs 4 --n_episodes 1
```

The table contact is computed against the table plane; the yellow surface in IsaacSim is the visual reference. Sphere proxies are easier to inspect than the raw table collision plane.

## Analysis Scripts

Parse overnight logs and regenerate the report:

```bash
python scripts/analyze_overnight_paper.py
```

Plot paper-style algorithm comparisons:

```bash
python scripts/plot_paper_benchmark.py
```

Monitor a running overnight or Stage 3 job:

```bash
tail -f /tmp/overnight_summary.txt
ls -lh /tmp/overnight_logs
pgrep -af "train_sac|run_lagrangian"
```

If a run hangs inside IsaacSim/PhysX, use runners with:

```bash
timeout -k 60s <limit> <command>
```

Plain `timeout` only sends SIGTERM. IsaacSim can ignore SIGTERM while stuck in a C++/CUDA call.

## Guardrails

Do not change these while a run is active:

- `envs/dual_arm_peg_hole_env.py`
- `envs/dual_arm_peg_hole_cost_env.py`
- `scripts/train_sac.py`
- `scripts/train_sac_lagrangian.py`
- `scripts/run_lagrangian_chain_local_from_yaml.py`
- the YAML file currently used by the active run

General invariants:

- `n_steps_per_epoch = horizon * num_envs`.
- Stage 1 uses horizon 100 and `n_steps_per_epoch: 1600`.
- Stage 3 uses horizon 200 and `n_steps_per_epoch: 3200`.
- Stage 3 insertion requires `exclude_ee_from_physx_self_collision: true`.
- Judge Stage 3 only after at least 80 epochs, preferably 100.
- For paper comparison, use YAML configs instead of hand-written long CLI commands.

## Repository Map

```text
conf/experiment/             canonical experiment YAMLs
envs/                        IsaacSim environments and B-route semantics
scripts/train_sac.py         native SAC
scripts/train_sac_lagrangian.py
                              LagSAC / constrained SAC
scripts/run_lagrangian_chain_local_from_yaml.py
                              YAML-to-training runner
scripts/run_stage3_100ep.sh  Stage 3 100 epoch calibration runner
scripts/analyze_overnight_paper.py
                              log parser for benchmark report
scripts/plot_paper_benchmark.py
                              paper-style comparison plots
scripts/visualize_policy.py  policy and pose visualization
results/checkpoints/saved/   canonical checkpoints
results/plots/overnight_paper/
                              generated figures
docs/README_legacy_20260512.md
                              old long README
```
