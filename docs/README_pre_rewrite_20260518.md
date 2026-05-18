# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 中做 peg-in-hole 的 SAC 训练 (mushroom-rl 2.0).

## ⭐ 当前主线 (2026-05-12 status)

走 **geom_stage** 路径 (S1g/S2g/S3g 几何同源 reward). 旧 Stage 1/2 球形 pos_err
路径 (`_compute_normal_reward`) 保留供 Lagrangian baseline 用 (`DualArmPegHoleCostEnv`),
旧 Stage 3 v4 (V-shape depth + Phase A/B/C schedule) 已删除, 可从
`scripts/archive/v4_baseline.py` 还原 (含完整 reward 方法 + CLI args 快照).

```
Stage 1g (prepos):   ckpt = results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh
Stage 2g (preaxis):  ckpt = results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh
Stage 3g (insert):   ckpt = results/checkpoints/saved/Stage3g_insert_h200_from_h100_h150_stage2_2026-05-12_21-01-11/best_hold.msh
                            (SAC v8 full-chain retrain from current h100/h150 Stage 1g+2g; clean insertion verified)
```

Stage 3g v8 SAC 用 5-channel reward (advance + dwell + pen + bad_entry + linear)
做 task ordering, 完整命令见 "Stage 3g 训练命令". v1-v7 一系列 reward 实验**失败**,
都不要重复 (见 memory `feedback_bimanual_sac_failures.md`).

下一步: Lagrangian SAC 对比 (`scripts/train_sac_lagrangian.py`, 用同 reward 但
`--rew_geom_bad_entry 0 --cost_signal penetration` 走 constraint 路径).

## ⚠️ 关键 invariants (任何 SAC/Lagrangian 训练都要遵守)

1. `n_steps_per_epoch = horizon × num_envs`. 当前 full-episode 主线:
   Stage 1g h100 → **1600**, Stage 2g h150 → **2400**, Stage 3g h200 → **3200**.
   不要再把旧 1024 当默认正式参数; 1024 只可用于复现旧 truncation / curriculum ablation.
   (silent bug 背景见 memory `feedback_mushroom_vectorcore_truncation.md`)
2. Stage 3g insertion / 任何会发生 peg-hole 接触的训练必须传
   `--exclude_ee_from_physx_self_collision`. 副作用是 PhysX 物理层也不阻止
   peg 穿过 hole walls, 因此物理 feasibility 信号 = 几何 `penetration_max` 量.
   Stage 1g prepos 不接触 hole, 本次 h100 full-episode 训练没有使用该开关.
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
| Stage 1g `prepos` | peg 到 hole 入口前的 preinsert 位置 | `results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh` |
| Stage 2g `preaxis` | preinsert 位置 + peg/hole 轴对齐 | `results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh` |
| Stage 3g `insert` | 真实 clean insertion | `results/checkpoints/saved/Stage3g_insert_h200_from_h100_h150_stage2_2026-05-12_21-01-11/best_hold.msh` |

Stage 3g 已用 2026-05-12 h100/full-episode Stage 1g + h150/full-episode
Stage 2g 新链路重新训练并通过 eval. 旧的
`SAC_v8_from_stage2_less_scrape_best_hold.msh` 保留为旧 preaxis chain 的复现参考.

当前 setup (post-Davide / collision-aware) 的关键事实:

1. `networks.py` 用 Davide 建议的低 gain Xavier init.
2. 双臂 sphere-proxy 自碰撞 hard absorbing 在 env 主线中保留.
3. hold-N absorbing 默认关闭 (`terminal_hold_bonus=0`); success 只作 eval / ckpt selection.
4. peg/hole collision proxy 挂在 EE link 下, 不是独立刚体. 训练时必须用几何
   `penetration_max` 判断物理可行性.

---

## Stage 1g (prepos, **已训练完成 — h100 full-episode cold-start 2026-05-12**)

### 当前状态

**Stage 1g (`geom_stage="prepos"`) h100 full-episode cold-start 成功**:
- canonical ckpt: `results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh`
- 原始路径: `results/checkpoints/2026-05-12/18-35-31/best_hold.msh`
- wandb run: `S1g_prepos_h100_full_ep_balanced_seed0`
- 训练 best: `geom_hold_rate=1.000`, `geom_max_run_mean=65.2/100`,
  `best_score=65.188`, `best_J=-85.082`
- 训练后期 actor 有漂移; **部署 / warm-start 用 `best_hold.msh`, 不用
  `final_agent.msh`**.

独立 eval (16 ep, deterministic, h100):
- `J(γ)=-86.690`, `R=-99.146`
- `geom_step_rate=0.621`, `geom_hold_rate=1.000`,
  `geom_max_run_mean=55.7/100`, `final_success_rate=1.000`
- `d_err_mean=0.0171m`, final `d_err=0.0083m`
- penetration 全程 clean: `max_mean=0.00mm`, `max_max=0.00mm`
- `preaxis=0.000`, `insert=0.000` 是预期行为: Stage 1g 不约束
  `radial_max` / `axis_err`, 只学 prepos.

### 目标

Stage 1g 只要求 peg tip 到达 hole 入口前的 preinsert 几何位置:

```text
d_target = -0.0800m
prepos mask = |d - d_target| < geom_d_th  AND  radial_tip < geom_r_tip_th
默认阈值: geom_d_th=0.030, geom_r_tip_th=0.030
```

此阶段**不要求** peg/hole 轴对齐, 也不要求整根 peg 的 `radial_max` 小.
这些由 Stage 2g (`preaxis`) 处理.

### Stage 1g 训练命令 (h100 full-episode cold-start)

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac.py \
  --geom_stage prepos \
  --geom_radial_sat 1.5 --geom_d_sat 0.30 \
  --rew_geom_d 8.0 --rew_geom_radial_tip 8.0 \
  --horizon 100 --n_epochs 120 --n_steps_per_epoch 1600 \
  --num_envs 16 --n_eval_episodes 16 \
  --terminal_hold_bonus 0 --hold_success_steps 10 \
  --rew_home 0.00075 --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --lr_actor 7e-5 --lr_critic 3e-4 --lr_alpha 3e-4 \
  --alpha_max 0.15 --target_entropy -7 \
  --seed 0 \
  --wandb_run_name S1g_prepos_h100_full_ep_balanced_seed0 \
  --wandb_group geom_repro
```

训练结束后保存模型:

```bash
mkdir -p results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31
cp results/checkpoints/2026-05-12/18-35-31/best_hold.msh \
   results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/
cp results/checkpoints/2026-05-12/18-35-31/best_agent.msh \
   results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/
cp results/checkpoints/2026-05-12/18-35-31/final_agent.msh \
   results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/
```

### Stage 1g eval

```bash
python scripts/eval_sac.py \
  --agent_path results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh \
  --geom_stage prepos \
  --geom_radial_sat 1.5 --geom_d_sat 0.30 \
  --rew_geom_d 8.0 --rew_geom_radial_tip 8.0 \
  --geom_d_th 0.030 --geom_r_tip_th 0.030 \
  --horizon 100 \
  --num_envs 16 --n_episodes 16 \
  --hold_success_steps 10 \
  --headless
```

期望独立 eval 数字:

```text
geom_hold_rate = 1.000
geom_step_rate ≈ 0.62
geom_max_run_mean ≈ 55-65 / 100
final_success_rate = 1.000
penetration max = 0.00mm
```

### Stage 1g visualization

```bash
python scripts/visualize_policy.py \
  --agent_path results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh \
  --geom_stage prepos \
  --geom_radial_sat 1.5 --geom_d_sat 0.30 \
  --rew_geom_d 8.0 --rew_geom_radial_tip 8.0 \
  --geom_d_th 0.030 --geom_r_tip_th 0.030 \
  --horizon 100 \
  --num_envs 4 --n_episodes 4 \
  --hold_steps 10 \
  --freeze_seconds 20
```

图像判断: peg tip 应稳定在 hole entrance 前方的 preinsert 位置附近; 轴向
`axis_err` 和整杆 `radial_max` 可以很大, 这是 Stage 1g 预期. Stage 2g
才负责轴对齐和 `radial_max`.

## Stage 2g (preaxis, **已训练完成 — h150 full-episode from h100 Stage 1g 2026-05-12**)

### 当前状态

**Stage 2g (`geom_stage="preaxis"`) 从新 Stage 1g h100 `best_hold` 一次性训练成功**:
- canonical ckpt: `results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh`
- 原始路径: `results/checkpoints/2026-05-12/19-50-00/best_hold.msh`
- warm-start 源: `results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh`
- wandb run: `S2g_preaxis_h150_from_h100_stage1_seed0`
- 训练 best: `geom_hold_rate=1.000`, `geom_max_run_mean=81.1/150`,
  `best_score=81.125`, `best_J=-127.832`
- 部署 / Stage 3g warm-start 用 `best_hold.msh`; `best_agent.msh` 只代表 best J.

独立 eval (16 ep, deterministic, h150):
- `J(γ)=-133.500`, `R=-170.333`
- `geom_step_rate=0.535`, `geom_hold_rate=1.000`,
  `geom_max_run_mean=76.6/150`, `final_success_rate=1.000`
- `d_err_mean=0.0113m`, `radial_max_min=0.0095m`, `axis_err_min=0.022`
- final: `d=-0.0757m`, `d_err=0.0043m`, `radial_max=0.0151m`
  (max `0.0198m`), `axis_err=0.0289` (max `0.0443`)
- penetration 全程 clean: `max_mean=0.00mm`, `max_max=0.00mm`
- `insert=0.000` 是预期行为: Stage 2g 只对齐 preinsert, 不插入.

### 目标

Stage 2g 在 Stage 1g 的 prepos 基础上, 加入整根 peg 的径向约束和 peg/hole
轴对齐:

```text
d_target = -0.0800m
prepos mask  = |d - d_target| < geom_d_th  AND  radial_tip < geom_r_tip_th
preaxis mask = prepos mask AND radial_max < geom_r_max_th AND axis_err < geom_axis_th

本次阈值:
geom_d_th=0.020, geom_r_tip_th=0.020, geom_r_max_th=0.025, geom_axis_th=0.300
```

此阶段仍然**不要求插入**. 末态应停在 hole 外侧 preinsert 附近, 但整根 peg
已经和 hole axis 对齐.

### Stage 2g 训练命令 (h150 full-episode, from Stage 1g best_hold)

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac.py \
  --geom_stage preaxis \
  --load_agent results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh \
  --actor_only_warmstart \
  --critic_warmup_transitions 50000 \
  --geom_d_target_neg -0.08 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.020 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --rew_geom_d 8.0 \
  --rew_geom_radial_tip 2.0 \
  --rew_geom_radial_max 5.0 \
  --rew_geom_axis 1.2 \
  --horizon 150 --n_epochs 120 --n_steps_per_epoch 2400 \
  --num_envs 16 --n_eval_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.00075 --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --lr_actor 3e-5 --lr_critic 3e-4 --lr_alpha 3e-4 \
  --alpha_max 0.05 --target_entropy -7 \
  --seed 0 \
  --wandb_run_name S2g_preaxis_h150_from_h100_stage1_seed0 \
  --wandb_group geom_repro
```

参数要点:
- `horizon 150`, `n_steps_per_epoch 2400` 是 full-episode (`150 × 16`).
- `--actor_only_warmstart` + `--critic_warmup_transitions 50000` 让 Stage 2g
  冷 critic 先适应新 reward, 避免 Stage 1g critic 语义污染.
- 不传 `--exclude_ee_from_physx_self_collision`: preaxis 仍在 hole 外, 没有
  peg-hole 接触.

训练结束后保存模型:

```bash
mkdir -p results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00
cp results/checkpoints/2026-05-12/19-50-00/best_hold.msh \
   results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/
cp results/checkpoints/2026-05-12/19-50-00/best_agent.msh \
   results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/
cp results/checkpoints/2026-05-12/19-50-00/final_agent.msh \
   results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/
```

### Stage 2g eval

```bash
python scripts/eval_sac.py \
  --agent_path results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh \
  --geom_stage preaxis \
  --geom_d_target_neg -0.08 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.020 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --rew_geom_d 8.0 \
  --rew_geom_radial_tip 2.0 \
  --rew_geom_radial_max 5.0 \
  --rew_geom_axis 1.2 \
  --horizon 150 \
  --num_envs 16 --n_episodes 16 \
  --hold_success_steps 10 \
  --headless
```

期望独立 eval 数字:

```text
geom_hold_rate = 1.000
geom_step_rate ≈ 0.53
geom_max_run_mean ≈ 75-80 / 150
final_success_rate = 1.000
final radial_max ≈ 1.5cm
final axis_err ≈ 0.03
penetration max = 0.00mm
insert_step_rate = 0.000
```

### Stage 2g visualization

默认 `freeze_mode=first_hold` 会在第一次连续 10 步满足 preaxis mask 时冻结,
看到的是刚达标的 entry 状态. 想看最终稳定对齐状态, 用固定 step:

```bash
python scripts/visualize_policy.py \
  --agent_path results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh \
  --geom_stage preaxis \
  --geom_d_target_neg -0.08 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.020 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --rew_geom_d 8.0 \
  --rew_geom_radial_tip 2.0 \
  --rew_geom_radial_max 5.0 \
  --rew_geom_axis 1.2 \
  --horizon 150 \
  --num_envs 4 --n_episodes 4 \
  --hold_steps 10 \
  --freeze_mode step \
  --freeze_after_step 140 \
  --freeze_seconds 30
```

图像判断: peg 应在 hole 外侧 preinsert 位置附近, 不插入; 但 peg/hole 轴应明显
对齐, `radial_max` 小, `axis_err` 小. 若只想看刚达标瞬间, 去掉
`--freeze_mode step --freeze_after_step 140` 即可回到 first-hold 行为.

## Stage 3 (insertion, **已训练完成 — h200 full-chain SAC v8 2026-05-12**)

### 当前状态

**Stage 3g (`geom_stage="insert"`) 从新 Stage 2g h150 `best_hold` 一次性训练成功**:
- canonical ckpt: `results/checkpoints/saved/Stage3g_insert_h200_from_h100_h150_stage2_2026-05-12_21-01-11/best_hold.msh`
- 原始路径: `results/checkpoints/2026-05-12/21-01-11/best_hold.msh`
- warm-start 源: `results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh`
- wandb run: `SAC_v8_from_h100_h150_stage2_seed1`
- 训练 best: `geom_hold_rate=1.000`, `geom_max_run_mean=113.7/200`,
  `best_score=113.688`, `best_J=-57.228`
- 本次 run 结束时 `best_agent.msh` / `best_hold.msh` / `final_agent.msh`
  md5 相同; 仍以 `best_hold.msh` 作为部署 / 后续 warm-start 入口.

独立 eval (16 ep, deterministic, h200):
- `J(γ)=-53.489`, `R=23.489`
- `geom_step_rate=0.577`, `geom_hold_rate=1.000`,
  `geom_max_run_mean=115.5/200`, `final_success_rate=1.000`
- entry: `d=+0.0258m`, `radial_max=0.0059m`, `axis_err=0.0100`,
  `penetration=0.04mm mean / 0.65mm max`
- final: `d=+0.0343m`, `radial_max=0.0029m`, `axis_err=0.0014`,
  `penetration=0.00mm`
- active mask 内 penetration max `0.65mm < 1.0mm`; clean insertion verified.

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

### Stage 3g 训练命令 (h200 full-chain SAC v8, from current Stage 2g)

**这是当前主线**. 走 geom_stage 路径:
- `--geom_stage insert` 启用 geom reward + 41D obs
- 5 个 reward channel 协同 (advance + dwell + pen + bad_entry + linear radial/axis)
- `--n_steps_per_epoch 3200` 必传, 不能用旧 1024 (VectorCore truncation bug)
- `seed=1` 是 verified working, seed=0 反复失败

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac.py \
  --geom_stage insert \
  --load_agent results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh \
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
  --rew_geom_d 1.0 --rew_geom_radial_tip 0 --rew_geom_radial_max 5.0 --rew_geom_axis 1.0 \
  --rew_geom_progress 0.0 --rew_geom_advance 25.0 \
  --geom_gate_radial_sigma 0.025 --geom_gate_axis_sigma 0.30 \
  --rew_geom_penetration 20.0 --geom_gate_penetration_sigma 0.005 \
  --rew_geom_soft_success 1.25 \
  --geom_soft_d_sigma 0.018 --geom_soft_radial_sigma 0.009 \
  --geom_soft_axis_sigma 0.15 --geom_soft_penetration_sigma 0.0025 \
  --geom_d_gate_mode off \
  --rew_geom_bad_entry 0.30 \
  --geom_bad_entry_radial_safe 0.014 \
  --geom_bad_entry_axis_safe 0.10 \
  --geom_bad_entry_pen_safe 0.00075 \
  --cost_signal collision \
  --horizon 200 --n_epochs 80 --n_steps_per_epoch 3200 \
  --num_envs 16 --n_eval_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.001 --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --lr_actor 1e-5 --lr_critic 3e-4 --lr_alpha 3e-4 \
  --alpha_max 0.015 --target_entropy -7 \
  --seed 1 \
  --wandb_run_name SAC_v8_from_h100_h150_stage2_seed1 \
  --wandb_group geom_repro
```

5 个 reward channel 职责:
1. `r_geom_radial_max + r_geom_axis` (linear, always-on): 拉对齐
2. `r_geom_advance` (PBRS Δφ, w=25): 主推进信号, 只在 aligned 时给奖
3. `r_geom_penetration` (linear -20·pen): 直接罚穿模
4. `r_geom_soft_success` (4-项 Gaussian dwell well, w=1.25): clean target 处持续 +1/step
5. `r_geom_bad_entry` (normalized w=0.3, capped): task ordering — d>0+misaligned 罚

详见 memory `feedback_bimanual_sac_v8_recipe.md`.

本次 eval 验收数字:
- insert_step_rate / `geom_step_rate` = 0.577
- `geom_hold_rate = 1.000`, `geom_max_run_mean=115.5/200`
- entry penetration mean = 0.04mm, active-mask max = 0.65mm
- final state: `d=+0.0343m`, `axis_err=0.0014`, `radial_max=0.0029m`,
  penetration=0.00mm

### Stage 3g eval / viz

eval:
```bash
python scripts/eval_sac.py \
  --agent_path results/checkpoints/saved/Stage3g_insert_h200_from_h100_h150_stage2_2026-05-12_21-01-11/best_hold.msh \
  --geom_stage insert \
  --geom_eval_epoch 20 \
  --exclude_ee_from_physx_self_collision \
  --geom_d_target_neg -0.08 --geom_d_target_pos 0.03 \
  --geom_d_target_ramp_start 0 --geom_d_target_ramp_end 20 \
  --geom_progress_floor 0.0 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.015 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --geom_insert_d_ins 0.025 --geom_insert_r_max_th 0.025 \
  --geom_pen_th 0.001 \
  --rew_geom_d 1.0 \
  --rew_geom_radial_tip 0 \
  --rew_geom_radial_max 5.0 \
  --rew_geom_axis 1.0 \
  --rew_geom_progress 0.0 \
  --rew_geom_advance 25.0 \
  --geom_gate_radial_sigma 0.025 \
  --geom_gate_axis_sigma 0.30 \
  --rew_geom_penetration 20.0 \
  --geom_gate_penetration_sigma 0.005 \
  --rew_geom_soft_success 1.25 \
  --geom_soft_d_sigma 0.018 \
  --geom_soft_radial_sigma 0.009 \
  --geom_soft_axis_sigma 0.15 \
  --geom_soft_penetration_sigma 0.0025 \
  --geom_d_gate_mode off \
  --rew_geom_bad_entry 0.30 \
  --geom_bad_entry_radial_safe 0.014 \
  --geom_bad_entry_axis_safe 0.10 \
  --geom_bad_entry_pen_safe 0.00075 \
  --cost_signal collision \
  --horizon 200 \
  --num_envs 16 --n_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.001 --home_weights 1,1,1,1,0.75,0.5,0.5 \
  --headless
```

visualization:
```bash
python scripts/visualize_policy.py \
  --agent_path results/checkpoints/saved/Stage3g_insert_h200_from_h100_h150_stage2_2026-05-12_21-01-11/best_hold.msh \
  --geom_stage insert \
  --geom_eval_epoch 20 \
  --exclude_ee_from_physx_self_collision \
  --geom_d_target_neg -0.08 --geom_d_target_pos 0.03 \
  --geom_d_target_ramp_start 0 --geom_d_target_ramp_end 20 \
  --geom_progress_floor 0.0 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.015 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --geom_insert_d_ins 0.025 --geom_insert_r_max_th 0.025 \
  --geom_pen_th 0.001 \
  --rew_geom_d 1.0 \
  --rew_geom_radial_tip 0 \
  --rew_geom_radial_max 5.0 \
  --rew_geom_axis 1.0 \
  --rew_geom_progress 0.0 \
  --rew_geom_advance 25.0 \
  --geom_gate_radial_sigma 0.025 \
  --geom_gate_axis_sigma 0.30 \
  --rew_geom_penetration 20.0 \
  --geom_gate_penetration_sigma 0.005 \
  --rew_geom_soft_success 1.25 \
  --geom_soft_d_sigma 0.018 \
  --geom_soft_radial_sigma 0.009 \
  --geom_soft_axis_sigma 0.15 \
  --geom_soft_penetration_sigma 0.0025 \
  --geom_d_gate_mode off \
  --rew_geom_bad_entry 0.30 \
  --geom_bad_entry_radial_safe 0.014 \
  --geom_bad_entry_axis_safe 0.10 \
  --geom_bad_entry_pen_safe 0.00075 \
  --cost_signal collision \
  --horizon 200 \
  --num_envs 4 --n_episodes 4 \
  --hold_steps 10 \
  --freeze_mode first_hold \
  --freeze_seconds 20 \
  --rew_home 0.001 --home_weights 1,1,1,1,0.75,0.5,0.5
```

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
# Canonical saved best_hold chain (current h100/h150/h200 full-chain SAC)
results/checkpoints/saved/Stage1g_h100_full_ep_cold_2026-05-12_18-35-31/best_hold.msh
results/checkpoints/saved/Stage2g_preaxis_h150_from_h100_stage1_2026-05-12_19-50-00/best_hold.msh
results/checkpoints/saved/Stage3g_insert_h200_from_h100_h150_stage2_2026-05-12_21-01-11/best_hold.msh

# Older saved best_hold ckpts kept for reproducibility / ablation reference
results/checkpoints/saved/Stage1_prepos_for_stage2_S1g_refine_seed0_best_hold.msh
results/checkpoints/saved/Stage2_preaxis_strict_refine_parent_2026-05-11_10-38-18_best_hold.msh
results/checkpoints/saved/Stage2_preaxis_for_stage3_2026-05-11_11-47-34_best_hold.msh
results/checkpoints/saved/SAC_v8_from_stage2_less_scrape_best_hold.msh
results/checkpoints/saved/SAC_v7c_clean_entry_best_hold.msh

# Restored dated dirs (best_hold mirrors saved/)
results/checkpoints/2026-05-10/22-16-55/
results/checkpoints/2026-05-11/10-38-18/
results/checkpoints/2026-05-11/11-47-34/
results/checkpoints/2026-05-12/00-46-20/
results/checkpoints/2026-05-12/08-58-55/
results/checkpoints/2026-05-12/18-35-31/
results/checkpoints/2026-05-12/19-50-00/
results/checkpoints/2026-05-12/21-01-11/
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
## 已知 stale code / config (不要直接信)

进 Stage 3 前最好先 cleanup 这几处 (或至少知道避开):

- 已全部更新

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

---

# bimanual_peghole_lagrangianSAC

## 基本说明

双臂 KUKA iiwa 在 IsaacSim 中做 peg-in-hole 的 Lagrangian SAC 安全约束训练 (mushroom-rl 2.0).
基于 `bimanual_peghole` SAC 主线, 将碰撞约束从 reward 剥离为独立 cost 信号, 用 Lagrange 乘子
λ 动态控制安全预算.

标准 SAC 把碰撞编码为巨大负奖励 (`r_min/(1-γ) ≈ -200`), reward critic 和安全信号混在一起难以分别调参.
Lagrangian SAC 把碰撞从 reward 里剥出来, 用独立 cost critic Q_C 学习, 用 Lagrange 乘子 λ 动态
控制安全预算, reward critic 只专注任务进展.

### 与 SAC 的核心差异

| 方面 | SAC (`train_sac.py`) | Lagrangian SAC (`train_sac_lagrangian.py`) |
|------|----------------------|--------------------------------------------|
| Replay buffer | 标准 `(s,a,r,s',absorb,last)` | `ConstrainedReplayMemory` 多存一列 cost `c` |
| Critic 数量 | 2 个 reward critic | 4 个: reward critic ×2 + cost critic ×2 |
| 碰撞处理 | reward = `r_min/(1-γ) ≈ -200`, episode 终止 | reward = shaped (不加大负奖励), cost = 1.0, episode 仍终止 |
| Actor loss | `(α·logπ − Q_R).mean()` | `(α·logπ − Q_R + λ·Q_C).mean()` |
| 环境类 | `DualArmPegHoleEnv` | `DualArmPegHoleCostEnv` |

### cost_limit 标定方法

`--cost_limit` 是训练最关键的超参. 设 0 会让 λ 一上来就爆; 设太高约束形同虚设.

推荐流程:

1. 用现有 SAC checkpoint 跑一次短 eval (64 ep 即可):
   ```bash
   python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
     --agent_path results/<sac_checkpoint>.msh [... 同训练的 env 参数 ...]
   ```
2. 读输出的 `absorb_sphere` 计数, 算 per-step collision rate:
   ```
   collision_rate = absorb_sphere_per_epoch / n_steps_per_epoch
   ```
3. 取 0.5× 作为起步预算: `--cost_limit = 0.5 × collision_rate`
4. 训练中看 wandb `cost_violation = cost_rate − cost_limit`:
   - 长期正 → λ 持续上升, 正常收紧
   - 长期负 → cost_limit 过松, 考虑降低
   - λ 冲到 `lambda_max` 且 `cost_violation` 仍正 → cost_limit 太严或任务太难

### 关键超参数说明

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `--cost_limit` | (必填) | `0.5 × baseline_rate` | per-step cost 预算; 见上方标定流程 |
| `--lr_lambda` | `1e-3` | `1e-4 ~ 1e-3` | 比 `lr_actor` 低 1–10×; 太高 λ 震荡, 太低约束收紧慢 |
| `--lambda_max` | `100.0` | `50 ~ 200` | λ 上限; 频繁冲顶说明 cost_limit 太严或任务太难 |
| `--init_log_lambda` | `0.0` | `0.0 ~ 2.0` | λ 初值 = `exp(init_log_lambda)`; 从 SAC ckpt warmstart 可适当调高 |
| `--gamma_cost` | `None` (= env γ) | `None` 或 `0.95~0.99` | None 复用 env γ=0.99; 设小一点让约束更短视、收紧更快 |

### 训练命令

从 SAC checkpoint warm-start (actor-only, 推荐起点):

```bash
cd ~/bimanual_peghole && conda activate safe_rl

python scripts/train_sac_lagrangian.py \
  --num_envs 16 --n_epochs 200 --n_steps_per_epoch 1024 --n_steps_per_fit 16 --n_eval_episodes 16 \
  --use_axis_resid_obs \
  --load_agent results/<sac_checkpoint>.msh --actor_only_warmstart \
  --critic_warmup_transitions 50000 \
  --preinsert_success_pos_threshold 0.10 --success_axis_threshold <axis_th> \
  --rew_axis <rew_axis> --rew_pos_success 1.0 --rew_success <rew_success> --rew_home 0.0005 \
  --lr_actor 5e-5 --lr_alpha 3e-4 --alpha_max 0.05 --target_entropy -10 \
  --cost_limit <0.5×baseline_collision_rate> \
  --lr_lambda 1e-3 --lambda_max 100 --init_log_lambda 0.0 \
  --seed 0 \
  --wandb_run_name <run_name>
```

训练结束**立即**备份 ckpt (顶层文件会被下次训练覆盖):

```bash
cp results/best_agent_lag.msh  results/<run_name>_best_agent.msh
cp results/best_hold_lag.msh   results/<run_name>_best_hold.msh
cp results/final_agent_lag.msh results/<run_name>_final.msh
```

### Eval 时看什么

| 指标 | 健康表现 | 异常信号 |
|------|----------|----------|
| `cost_rate` | 逐渐收敛到 ≤ `cost_limit` | 长期 >> cost_limit → λ 失控或任务太难 |
| `cost_violation` | 先正后收敛到 ≤ 0 | 持续正 + λ 冲顶 → cost_limit 太严 |
| `lambda` (λ) | 平稳增长后趋于稳定 | 爆到 `lambda_max` → 上调 `lambda_max` 或放松 `cost_limit` |
| `J` | 不低于 SAC baseline 太多 | 崩到 baseline 一半以下 → `lr_lambda` 太高或 `cost_limit` 太严 |
| `hold_success_rate` | 与 SAC baseline 可比 | 大幅下降 → λ 过大压制了 actor |

> `best_agent_lag.msh` 是最高 J 的 ckpt, **不保证满足 cost_limit** (高 J 可能在 λ
> 收紧之前就出现). 部署前在 wandb `cost_rate` 时间线上核查对应 epoch 是否达标.

### Warmstart 路径

**SAC → Lagrangian SAC** (跨算法, 必须 `--actor_only_warmstart`):

```bash
--load_agent results/<sac_ckpt>.msh --actor_only_warmstart
# critic / cost critic / α / λ / replay 全部冷启动; --keep_replay 此时被忽略
```

**Lagrangian SAC → Lagrangian SAC** (同算法全量, 保留旧 critic 和 replay):

```bash
--load_agent results/<lag_ckpt>.msh
# 加 --keep_replay 可保留旧 replay buffer (reward/cost 函数未变时才合理)
```

**Lagrangian SAC → Lagrangian SAC (actor-only)** (reward 或 cost 函数有改动时):

```bash
--load_agent results/<lag_ckpt>.msh --actor_only_warmstart
```

---

## Stage 1 训练

### 训练命令 (Hydra, 冷启动)

```bash
python scripts/train_hydra.py experiment@train=phase1_lagrangian
```

无需 SAC checkpoint，直接冷启动。`conf/experiment/phase1_lagrangian.yaml` 已包含所有超参，
`--load_agent` 未设置即走 `_cold_create_sac_lag()`。

完整超参见 `conf/experiment/phase1_lagrangian.yaml`，关键值：

| 参数 | 值 | 备注 |
|------|-----|------|
| `lr_actor` | 1e-4 | 冷启动保守值 |
| `alpha_max` | 0.05 | 防 entropy 项压制精度信号 |
| `target_entropy` | -7 | = -act_dim/2, 冷启动 14-DoF |
| `cost_limit` | 0.02 | TODO: 用 SAC baseline eval 标定 |
| `lr_lambda` | 1e-3 | |
| `lambda_max` | 100.0 | |
| `init_log_lambda` | 0.0 | → λ_init = 1.0 |
| `gamma_cost` | null (= env γ = 0.99) | |
| `clearance_hard` | 0.0 | sphere-proxy 开启 |
| `n_epochs` | 100 | |

### λ 的作用与行为分析

**λ 的作用：**

Actor loss = `α·logπ − Q_R + λ·Q_C`

λ 是 Lagrange 乘子，量化"当前约束有多紧"：
- λ 大 → actor 被迫远离高 Q_C 的动作，主动压 cost
- λ → 0 → actor loss 退化为纯 SAC，cost critic 对 actor 零影响

λ 更新规则 (`lagrangian_sac.py:292`)：

```
log_λ += lr_λ × violation
violation = Q_C × (1 − γ_c) − cost_limit
```

violation > 0（超预算）→ λ 上升；violation < 0（有余量）→ λ 下降。

**λ 为何衰减至下限（4.5e-5 = e^{-10}）：**

冷启动时分两个阶段：

- **epoch 1–4（Q_C 低估期）**：cost critic 未收敛，Q_C ≈ 0，导致 `Q_C×(1-γ_c) < cost_limit`，
  violation 看起来为负 → λ 持续下降。此时 eval 实测 cost_rate 已超标（epoch 3: 0.030，
  epoch 4: 0.038），但 λ 更新用的是 replay buffer 中的 Q_C 估计，冷启动阶段低估导致 λ 反常下降。

- **epoch 5+ （约束真正满足期）**：policy 突然学会到达 preinsert 区域（epoch 5: hold_rate=0.938），
  真实 cost_rate 归零。violation 恒为 `-cost_limit = -0.02`，λ 持续下降直到触碰
  lower clamp `log_λ = -10`（`lagrangian_sac.py:300`），停在 λ ≈ 4.5e-5。

**λ ≈ 0 的影响：**

训练从 epoch 5 起等价于标准 SAC，Lagrangian 约束机制完全休眠：

```
actor loss ≈ α·logπ − Q_R + 4.5e-5 · Q_C  ≈  α·logπ − Q_R
```

**是否需要 debug：** 否，λ 行为是正确的数学结果（约束满足、cost_limit 有余量则 λ
应下降）。崩溃（epoch 16/20）发生时 λ 已为 0，与 Lagrangian 机制无关。

**真实崩溃原因：** UTD=16 配合小 replay buffer（epoch 10 时仅约 20K transitions）
导致 Q 值过估计，actor 梯度爆炸。标准 SAC policy collapse，不是 Lagrangian 问题。

**冷启动 Q_C 低估的缓解方法（供参考）：**
- 增大 `--critic_warmup_transitions`（如 50K），让 cost critic 先收敛再放开 λ 更新
- `init_log_lambda` 不要设太高（当前 0.0 = λ_init=1.0 已属于偏高，容易被早期负 violation 快速压下）

### 实验记录 (run-20260510_120554-7bkfr82a, 2026-05-10, 训练中)

seed=42，冷启动，100 epoch（log 已记录至 epoch 56，训练仍在进行）。

**训练曲线概述：**

| epoch 区间 | J 范围 | hold_rate | cost_rate | λ | 事件 |
|---|---:|---:|---:|---:|---|
| 1–4 | -26.9 → -8.2 | 0–0.125 | 0.016–0.038 | 0.36→0.017 | 学习初期，cost_rate 超标但 Q_C 低估，λ 反常下降 |
| 5–10 | 4.8 → 112.0 | 0.938→1.000 | 0.000 | 0.006→≈0 | policy 突破，hold_rate 跳至 1.0，λ 降至下限 |
| 11–15 | 86.6 → 110.9 | 1.000 | 0.000 | ≈4.5e-5 | 平台期，max_hold_mean 稳定在 96–133 步 |
| 16 | **−22.0** | 0.250 | 0.000 | ≈4.5e-5 | **第一次 policy collapse**（UTD=16 + 小 replay buffer Q 过估） |
| 17–19 | 18.2 → 76.6 | 0.938→1.000 | 0.000 | ≈4.5e-5 | 部分恢复 |
| 20–21 | **−54.7 → −51.8** | 0.000 | 0.000 | ≈4.5e-5 | **第二次 policy collapse，更严重** |
| 22–32 | 2.1 → 107.4 | 0.750→1.000 | 0.000–0.005 | ≈4.5e-5 | 完全恢复，逐步爬回平台 |
| 33–44 | 115.8 → **117.7** | 1.000 | 0.000 | ≈4.5e-5 | **超越崩溃前 peak**，epoch 44 创新高 best_J=117.67 |
| 45–56 | 109.6 → 116.3 | 1.000 | 0.000 | ≈4.5e-5 | 稳定平台期（进行中） |

**关键指标（截至 epoch 56）：**

| 指标 | 值 | epoch |
|------|-----|-------|
| best_J | **117.67** | **44** |
| best_hold_rate | 1.000 | 8 |
| best_hold_max_hold_mean | **136.5 步** | **44** |
| cost_rate @ peak | 0.000 | — |
| λ @ peak | ≈4.5e-5 | — |
| 第一次崩溃 | J=−22.0 | 16 |
| 第二次崩溃 | J=−54.7 | 20 |
| 超越崩溃前 peak | epoch 33（J=115.8 > 前 peak 112.0）| 33 |

**checkpoint 位置：**

```text
results/checkpoints_lag/2026-05-10/12-02-15/best_agent.msh   # best_J=117.67, epoch 44 (持续更新)
results/checkpoints_lag/2026-05-10/12-02-15/best_hold.msh    # best hold_rate=1.0
results/checkpoints_lag/2026-05-10/12-02-15/final_agent.msh  # 训练结束后写入
```

**结论与注意事项：**

- cost_rate 从 epoch 5 起持续为 0，满足 `cost_limit=0.02` 的安全约束
- λ 降至 lower bound（≈4.5e-5），Lagrangian 机制从 epoch 5 起等价于纯 SAC
- policy collapse 出现两次（epoch 16/20），但**均完全恢复，且 epoch 33 起超越崩溃前 peak**——与 SAC Phase 1 不同，这里 replay buffer 中的 bad transitions 被稀释后 policy 重新收敛
- 部署选 `best_agent.msh`（当前 epoch 44，训练结束后可能继续更新），不要用 `final_agent.msh`
- 可视化命令：

```bash
python scripts/train_hydra.py --multirun experiment@train=record_checkpoint \
    train.checkpoint_dir=results/checkpoints_lag/2026-05-10/12-02-15 \
    train.tag=lag_phase1
```

---

## Stage 2 训练

### Troubleshooting

#### Mushroom-rl `last` 语义: 不要把 fit dataset 的 `last.sum()` 当真实 episode 数

`VectorizedDataset.flatten()` 会在交给 `agent.fit()` 前把当前数据块最后一行强制标成
`last=True`:

```python
last_padded = self._data.last
last_padded[-1, :] = True
```

这个 `last` 更像 trajectory segment / fit chunk boundary, 不总是环境真实 episode
结束。当前默认 `n_steps_per_fit = num_envs = 16`, 每次 fit 只有 1 个 vector-step,
因此 fit dataset 里的 `last` 往往整行都是 True, 即使真实 episode 还没到 `horizon`
也没碰撞终止。

这不会破坏 SAC / Lagrangian SAC 主体训练:

- SAC reward critic target 用 `absorbing` 决定是否 bootstrap, 标准 SAC 直接丢弃 replay
  里的 `last`
- Lagrangian SAC reward critic / cost critic target 也用 `absorbing`
- replay 是 1-step transition; `last` 不参与 1-step Bellman target

真正要小心的是 `lambda_update_mode == "episode_rate"`。不要用:

```python
cost.sum() / dataset.last.sum()
```

或 replay random batch 里的:

```python
cost.sum() / last.sum()
```

来估计每集平均碰撞次数。前者会被 fit chunk boundary 污染, 后者来自随机 transition
采样, `last.sum()` 也不是完整 episode 数。

当前默认配置使用 `lambda_update_mode: max_recent_replay`, `cost_limit` 是 per-step
collision rate 预算, λ 更新走 `cost.mean()` / `recent_cost.mean()` 语义, 不依赖
`last.sum()`。

如果切到 `lambda_update_mode: episode_rate`, 当前实现会走独立路径:

```python
agent.update_lambda_from_episode_statistics(
    cost_episode_rate=c["cost_episode_sum_mean"],
    source="eval_episode_rate",
)
```

也就是每个 epoch 的完整 eval episodes 结束后, 用
`compute_cost_metrics()["cost_episode_sum_mean"] = sum(cost) / n_eval_episodes`
更新 λ。cost critic 仍继续 off-policy 用 replay 学; 只有 λ 的 dual update 不再使用
replay random batch 或 flattened dataset 的 `last`。
