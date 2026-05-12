# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 中做 peg-in-hole 的 SAC 训练 (mushroom-rl 2.0).

## ⭐ 当前主线 (2026-05-12 status)

走 **geom_stage** 路径 (S1g/S2g/S3g 几何同源 reward). 旧 Stage 1/2 球形 pos_err
路径 (`_compute_normal_reward`) 保留供 Lagrangian baseline 用 (`DualArmPegHoleCostEnv`),
旧 Stage 3 v4 (V-shape depth + Phase A/B/C schedule) 已删除, 可从
`scripts/archive/v4_baseline.py` 还原 (含完整 reward 方法 + CLI args 快照).

```
Stage 1g (prepos):   ckpt = results/checkpoints/saved/Stage1_prepos_for_stage2_S1g_refine_seed0_best_hold.msh
Stage 2g (preaxis):  ckpt = results/checkpoints/saved/Stage2_preaxis_for_stage3_2026-05-11_11-47-34_best_hold.msh
Stage 3g (insert):   ckpt = results/checkpoints/saved/SAC_v8_from_stage2_less_scrape_best_hold.msh
                            (SAC v8 from preaxis, clean insertion verified)
```

Stage 3g v8 SAC 用 5-channel reward (advance + dwell + pen + bad_entry + linear)
做 task ordering, 完整命令见 "Stage 3g 训练命令". v1-v7 一系列 reward 实验**失败**,
都不要重复 (见 memory `feedback_bimanual_sac_failures.md`).

下一步: Lagrangian SAC 对比 (`scripts/train_sac_lagrangian.py`, 用同 reward 但
`--rew_geom_bad_entry 0 --cost_signal penetration` 走 constraint 路径).

## ⚠️ 关键 invariants (任何 SAC/Lagrangian 训练都要遵守)

1. `n_steps_per_epoch = horizon × num_envs` (= 200 × 16 = **3200**). 不能用旧
   1024, 否则 VectorCore 每次 reset 全 envs, 训练永远只见 episode 前 64 步.
   (silent bug, 见 memory `feedback_mushroom_vectorcore_truncation.md`)
2. `--exclude_ee_from_physx_self_collision` 必传. 副作用是 PhysX 物理层也不阻止
   peg 穿过 hole walls. 物理 feasibility 信号 = 几何 `penetration_max` 量.
3. `insert_mask` 含 `penetration_max < geom_pen_th` 项 (默认 1mm). 不含此项
   ckpt selection 会选穿模假成功.
4. SAC seed=1 verified (seed=0 七次失败). multi-seed sweep 从 seed=1 起.

---

## 当前阶段链

当前 README 只保留 **geom_stage 主线**. 旧 Stage 1 / Stage 2 球形
`pos_err` + axis cliff 训练记录已移到
`scripts/archive/legacy_stage1_stage2.md`; 旧 Stage 3 v4 reward 还原源在
`scripts/archive/v4_baseline.py`.

主线三段 checkpoint:

| Stage | 目标 | checkpoint |
|---|---|---|
| Stage 1g `prepos` | peg 到 hole 入口前的 preinsert 位置 | `results/checkpoints/saved/Stage1_prepos_for_stage2_S1g_refine_seed0_best_hold.msh` |
| Stage 2g `preaxis` | preinsert 位置 + peg/hole 轴对齐 | `results/checkpoints/saved/Stage2_preaxis_for_stage3_2026-05-11_11-47-34_best_hold.msh` |
| Stage 3g `insert` | 真实 clean insertion | `results/checkpoints/saved/SAC_v8_from_stage2_less_scrape_best_hold.msh` |

当前 setup (post-Davide / collision-aware) 的关键事实:

1. `networks.py` 用 Davide 建议的低 gain Xavier init.
2. 双臂 sphere-proxy 自碰撞 hard absorbing 在 env 主线中保留.
3. hold-N absorbing 默认关闭 (`terminal_hold_bonus=0`); success 只作 eval / ckpt selection.
4. peg/hole collision proxy 挂在 EE link 下, 不是独立刚体. 训练时必须用几何
   `penetration_max` 判断物理可行性.

---

## Stage 3 (insertion, **已训练完成 — SAC v8 2026-05-12**)

### 当前状态

**Stage 3g (geom_stage="insert") SAC v8 clean insertion 训练成功**:
- ckpt: `results/checkpoints/saved/SAC_v8_from_stage2_less_scrape_best_hold.msh`
- 原始路径: `results/checkpoints/2026-05-12/08-58-55/best_hold.msh`
- final: insert_step_rate=0.554, hold_rate=1.000, entry pen=0.17mm mean, final pen=0mm
- best_score=125.75, max_run_mean=110.8/200 steps

### 目标

peg 真正进入 hole `d > geom_insert_d_ins` (默认 2.5cm), 同时 radial_max 与
penetration 都达标. 不要求插到底.

### Sign convention 与几何

```text
axial_tip = (peg_tip - hole_entry) · hole_axis     # raw
d         = -axial_tip                              # depth, d>0 = inserted
  d < 0  → preinsert (默认 -0.05 起步)
  d = 0  → entry plane
  d > 0  → 插入 hole

radial_vec = (peg_tip - hole_entry) - axial_tip · hole_axis
radial_err = ||radial_vec||                         # tip 单点
radial_max = max over peg samples of ||sample - hole_axis line||
e_axis     = 1 + cos(peg_axis, hole_axis)           # peg/hole 反向时 = 0
```

`peg_sample_offsets = (0.0, -0.02, -0.04, -0.06)`. peg 总长 7cm, 不要超 -0.06.

历史 `d` sign 验证脚本归档在 `scripts/archive/sanity_eval_stage3.py` (post-v4-removal
不可直接运行, 但 docstring 仍然记录验证流程).

### Stage 3g 训练命令 (SAC v8 verified recipe, 2026-05-12)

**这是当前主线**. 走 geom_stage 路径:
- `--geom_stage insert` 启用 geom reward + 41D obs
- 5 个 reward channel 协同 (advance + dwell + pen + bad_entry + linear radial/axis)
- `--n_steps_per_epoch 3200` 必传, 不能用旧 1024 (VectorCore truncation bug)
- `seed=1` 是 verified working, seed=0 反复失败

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac.py \
  --geom_stage insert \
  --load_agent results/checkpoints/saved/Stage2_preaxis_for_stage3_2026-05-11_11-47-34_best_hold.msh \
  --actor_only_warmstart \
  --critic_warmup_transitions 30000 \
  --exclude_ee_from_physx_self_collision \
  --geom_d_target_neg -0.08 --geom_d_target_pos 0.03 \
  --geom_d_target_ramp_start 0 --geom_d_target_ramp_end 20 \
  --geom_progress_floor 0.0 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.015 --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --geom_insert_d_ins 0.025 --geom_insert_r_max_th 0.025 \
  --geom_pen_th 0.001 \
  --rew_geom_d 0.0 --rew_geom_radial_tip 0 --rew_geom_radial_max 5.0 --rew_geom_axis 1.0 \
  --rew_geom_progress 0.0 --rew_geom_advance 25.0 \
  --geom_gate_radial_sigma 0.025 --geom_gate_axis_sigma 0.30 \
  --rew_geom_penetration 15.0 --geom_gate_penetration_sigma 0.005 \
  --rew_geom_soft_success 1.25 \
  --geom_soft_d_sigma 0.020 --geom_soft_radial_sigma 0.010 \
  --geom_soft_axis_sigma 0.15 --geom_soft_penetration_sigma 0.0025 \
  --rew_geom_bad_entry 0.3 \
  --geom_bad_entry_radial_safe 0.014 \
  --geom_bad_entry_axis_safe 0.10 \
  --geom_bad_entry_pen_safe 0.00075 \
  --cost_signal collision \
  --horizon 200 --n_epochs 80 --n_steps_per_epoch 3200 \
  --num_envs 16 --n_eval_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.001 --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --lr_actor 1e-5 --alpha_max 0.015 --target_entropy -7 \
  --seed 1 \
  --wandb_run_name SAC_v8_from_stage2_seed1 --wandb_group sac_lagr
```

5 个 reward channel 职责:
1. `r_geom_radial_max + r_geom_axis` (linear, always-on): 拉对齐
2. `r_geom_advance` (PBRS Δφ, w=25): 主推进信号, 只在 aligned 时给奖
3. `r_geom_penetration` (linear -15·pen): 直接罚穿模
4. `r_geom_soft_success` (4-项 Gaussian dwell well, w=1.25): clean target 处持续 +1/step
5. `r_geom_bad_entry` (normalized w=0.3, capped): task ordering — d>0+misaligned 罚

详见 memory `feedback_bimanual_sac_v8_recipe.md`.

期望 final 数字:
- insert_step_rate (strict mask) > 0.5
- geom_hold_rate = 1.000
- entry penetration mean < 0.5mm
- final state: d≈+0.026, axis_err≈0, radial≈3mm, penetration=0

### Stage 3g eval / viz

eval:
```bash
python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
  --agent_path results/checkpoints/saved/SAC_v8_from_stage2_less_scrape_best_hold.msh \
  --geom_stage insert --horizon 200 --geom_eval_epoch 20 \
  --exclude_ee_from_physx_self_collision \
  --geom_d_target_neg -0.08 --geom_d_target_pos 0.03 \
  --geom_d_target_ramp_start 0 --geom_d_target_ramp_end 20 \
  --geom_d_th 0.020 --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --geom_insert_d_ins 0.025 --geom_insert_r_max_th 0.025 --geom_pen_th 0.001 \
  --rew_geom_radial_max 5.0 --rew_geom_axis 1.0 --rew_geom_advance 25.0 \
  --rew_geom_penetration 15.0 --geom_gate_penetration_sigma 0.005 \
  --rew_geom_soft_success 1.25 --geom_soft_penetration_sigma 0.0025 \
  --rew_geom_bad_entry 0.3 --geom_bad_entry_radial_safe 0.014 --geom_bad_entry_pen_safe 0.00075 \
  --rew_home 0.001 --home_weights 1,1,1,1,0.75,0.5,0.5
```

viz: 同 eval, 改 `scripts/visualize_policy.py` + 加 `--num_envs 4 --n_episodes 4 --freeze_seconds 20 --hold_steps 50`.
带 `--hold_steps 50` 让 freeze 触发在 agent 已稳定 dwell 时, 看真 dwell 状态.

### Best ckpt 选择 (geom_stage 模式)

`best_agent.msh` 按 J 选. `best_hold.msh` 路径在 geom 模式下选 `geom_hold_rate`
最高 (≥ hold_success_steps 步连续 active geom success mask), tie-break
`geom_max_run_mean`. 见 `compute_geom_metrics`.

### 关键 trap (memory `feedback_bimanual_reward_shaping.md` + `feedback_bimanual_sac_v8_recipe.md`)

1. ★ **success 不 absorbing** — Rule 1: 边界 Q-cliff + dwell 信号截断, 不加
2. ★ **insert success 用 `radial_max` + `penetration_max` 而非 tip** — tip 准
   但杆身斜的假成功
3. ★ **peg_sample_offsets 不要超 -0.06** (peg 7cm), 否则采到 EE coupler 段
4. ★ **必须传 `--exclude_ee_from_physx_self_collision`**, 否则 peg-hole 接触
   被双臂自碰撞 hard absorb 误杀

---

## 几何和观测

预插入目标:

```text
preinsert_target = hole_entry + preinsert_offset * hole_axis
pos_vec          = peg_tip - preinsert_target
pos_err          = ||pos_vec||
axis_err         = 1 + dot(peg_axis, hole_axis)
```

`axis_err` 越小越好. 理想反向对齐时 `dot=-1`, 所以 `axis_err=0`.

`geom_stage` 主线会自动启用 axis-resid obs, 并在末尾追加 7 维几何量:

```text
obs_dim = 41
obs = joint_pos[14] + joint_vel[14] + pos_vec[3] + axis_resid[3]
      + d[1] + radial_vec[3] + peg_in_hole[3]

axis_resid = peg_axis + hole_axis
axis_err   = ||axis_resid||^2 / 2 = 1 + dot(peg_axis, hole_axis)
d          = -((peg_tip - hole_entry) · hole_axis)
radial_vec = (peg_tip - hole_entry) - axial_dist * hole_axis
peg_in_hole = peg_tip - hole_entry
```

`axis_resid` 不能简单替换为 `cross(peg_axis, hole_axis)`. cross 在平行/反平行时
都为 0, 无法区分最差同向和最好反向.

## Reward 公式

当前主线是 `geom_stage` reward. 对 Stage 3g (`geom_stage=insert`) 的关键项:

```text
radial_max = max_i distance(sample_i, hole_axis_line)
penetration_max = peg 表面相对 hole 内壁的几何穿模量, clamp 到 [0, 4mm]

alignment_gate = exp(-(radial_max / sigma_r)^2
                     -(axis_err / sigma_a)^2
                     -(penetration_max / sigma_pen_gate)^2)

phi = alignment_gate * clamp((d - d_neg) / (d_pos - d_neg), 0, 1)
r_advance = w_advance * (phi_t - phi_{t-1})

r_dwell = w_soft * exp(-(d_err / sigma_d)^2
                       -(radial_max / sigma_radial)^2
                       -(axis_err / sigma_axis)^2
                       -(penetration_max / sigma_pen_soft)^2)

r_bad_entry = -w_bad * depth_norm * clamp(sum_i violation_i, 0, 3)
  depth_norm = clamp(d / d_target_pos, 0, 2)
  violation_i = clamp(metric_i / safe_i - 1, 0, 3)

r_pen = -w_pen * penetration_max
r_linear = -w_radial_max * radial_max - w_axis * axis_err
```

Stage 3g v8 配方中, 5 个 channel 的职责:

1. `r_linear`: always-on 拉 radial / axis 对齐.
2. `r_advance`: delta-progress bootstrap, 防 hover 占位赚钱.
3. `r_pen`: 直接罚穿模.
4. `r_dwell`: clean target 附近持续正奖励, 防 advance 后期梯度消失.
5. `r_bad_entry`: d>0 但 radial / axis / penetration 不达标时惩罚, 强制 task ordering.

Absorbing 规则:

```text
collision (PhysX 力 OR sphere-proxy 几何 < clearance_hard): absorbing=True
hold-N success: terminal_hold_bonus=0 时不 absorbing, 只写 eval 计数
success 本身不终止, 避免 Q-target 边界断崖
```

旧 `_compute_normal_reward` 仍保留, 供 `DualArmPegHoleCostEnv` / Lagrangian baseline
使用; 不是当前 SAC v8 主线.

## 保留结果

```text
# Canonical saved best_hold chain (current main line)
results/checkpoints/saved/Stage1_prepos_for_stage2_S1g_refine_seed0_best_hold.msh
results/checkpoints/saved/Stage2_preaxis_strict_refine_parent_2026-05-11_10-38-18_best_hold.msh
results/checkpoints/saved/Stage2_preaxis_for_stage3_2026-05-11_11-47-34_best_hold.msh
results/checkpoints/saved/SAC_v7c_clean_entry_best_hold.msh
results/checkpoints/saved/SAC_v8_from_stage2_less_scrape_best_hold.msh

# Restored dated dirs (best_hold mirrors saved/)
results/checkpoints/2026-05-10/22-16-55/
results/checkpoints/2026-05-11/10-38-18/
results/checkpoints/2026-05-11/11-47-34/
results/checkpoints/2026-05-12/00-46-20/
results/checkpoints/2026-05-12/08-58-55/
```

顶层 `results/*.msh` 已清空; 不再依赖会被训练覆盖的 flat checkpoint. 由于
`results/` 被 `.gitignore` 忽略, 这些 ckpt 是本地实验资产, 不会随 GitHub merge
自动传给别人.

## Archive / 非主线

- 旧 Stage 1/2 legacy 记录: `scripts/archive/legacy_stage1_stage2.md`
- 旧 Stage 3 v4 还原源: `scripts/archive/v4_baseline.py`
- reward 设计期诊断脚本: `scripts/archive/plot_geom_*.py`
- USD asset sanity: `scripts/archive/check_peghole_asset.py`

Hydra / Lagrangian 路径由同学负责, README 主线不展开这些配置. merge 前如要启用
`scripts/train_sac_lagrangian.py`, 需要单独检查 import / config 与当前 `algorithm/`
目录是否一致.

## 历史

- 2026-05-09: Legacy Stage 2 oneshot 配方完成 (200 ep, axis_th=0.40, rew x2, lr 5e-5,
  alpha 0.05, te -10). 比之前 axis_th=0.20 / 0.30 实验 J 高 7x, max_hold 从 0
  → 105.7 步, axis_in_pos_mean 从 0.63 → 0.41. 关键 learning: cliff reward
  训中心不训 floor; success_axis_threshold 是 reward 函数开关不是 metric;
  post-peak 塌方靠 lr↓ + alpha↓ 治. 详见
  `scripts/archive/legacy_stage1_stage2.md`.
- 2026-05-08: 引入 Davide init + sphere-proxy + hold-N off, Stage 1 5-seed
  sweep 完成. 删除 2026-05-04/05/06/07 的 pre-collision ckpt (S1_axisresid_*,
  S2_*, M2_*, S2p*, S3_warmstart_baseline 等), 不再作为 baseline 比较.
- 2026-05-04 之前: 早期 M1/M2 调参, 没有 sphere-proxy collision, 不可比.
