# bimanual_peghole

双臂 KUKA iiwa 在 IsaacSim 里做 peg-in-hole 的 RL 控制。最终目标是用
**Lagrangian SAC** 处理装配约束；当前阶段：

- **M1' / M2 (当前)** — 普通 SAC + **stage flag 化的 preinsert 任务**。
  peg/hole 仍然是视觉-only (无 CollisionAPI)，左末端挂 peg、右末端挂
  hole；目标是把 `peg_tip` 拉到
  `preinsert_target = hole_entry + 5cm · hole_axis`，姿态对齐由 `axis_dot`
  obs + `axis_err` reward 项控制。

  同一个 env、同一个 32 维 obs、同一条 reward 骨架，stage 仅由
  `--rew_axis` 和 `--success_axis_threshold` 切换；M1'→M2a→M2b 之间
  `--load_agent` warm-start，不需要 cold start。peg/hole frame 用解析式
  (`EE_pose ⊗ const_offset`)，不依赖 XFormPrim 的 Fabric flush，headless
  训练永不 stale。

历史阶段：phase 1 的 reaching 任务 (左右 EE 各自到固定胸前点) 和早期
**31 维 strict pos-only M1** 已退役，主线不再保留兼容；如需回看见 git
历史 (`git log --grep=reaching` / `git log --grep="31.dim"`)。

## 环境

```bash
conda env create -f environment.yml   # 创建 safe_rl
conda activate safe_rl
```

依赖：

- `mushroom-rl` (dev 分支)、`torch==2.7.0`、`numpy==1.26.0`、`wandb`
  由 `environment.yml` 安装
- IsaacSim 仍需在目标机器上可用；若环境里没有，补装 `pip install isaacsim`
- 机器人 USD 资产随仓库提供：
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa.usd` — 原始 iiwa (env 不再直接用)
  - `assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda` — **当前唯一支持**,
    在原始 iiwa 上挂了视觉 peg/hole 与 `peg_tip` / `hole_entry` 参考帧
  - 加载无 peg/hole 的旧 USD 会被 `_verify_peghole_prims_exist` 直接 raise

## 目录结构

```
envs/
  dual_arm_peg_hole_env.py   # IsaacSim 子类: 14 DoF velocity, 32 维 obs (含
                             #   axis_dot), 解析式 peg/hole frame, stage flag
                             #   化 reward (rew_axis / success_axis_threshold)
  __init__.py                # 导出 DualArmPegHoleEnv / AGENT_OBS_DIM /
                             #   DEFAULT_PREINSERT_OFFSET

networks.py                  # SAC actor / critic MLP (input → 256 → 256 → out)

scripts/
  train_sac.py               # SAC 训练 (VectorCore, 默认 num_envs=16);
                             #   --load_agent 续训, 默认清 replay
  eval_sac.py                # 加载 best_agent.msh 评估
  visualize_targets.py       # 不训练, 看 peg(红)/hole(绿)/preinsert(黄) marker
  visualize_policy.py        # 跑训好的 policy, hold-N 满足时冻结画面;
                             #   M2 时需传 --success_axis_threshold
  _eval_utils.py             # deterministic policy + hold-N success 指标
  archive/                   # 一次性诊断脚本归档 (主线不依赖, 见
                             #   scripts/archive/README.md). diagnose_m1_axis /
                             #   diagnose_m2b_clearance / check_peghole_asset
                             #   — 改 USD / sphere proxy / 重新验证 PhysX
                             #   失明时再跑.

assets/
  usd/
    dual_arm_iiwa/
      dual_arm_iiwa.usd               # 原始机器人 (历史保留)
      dual_arm_iiwa_with_peghole.usda # 当前唯一加载: robot + 视觉 peg/hole
      build_peghole_usd.py            # 重新生成 composed USDA
      configuration/*.usd             # 机器人分层资产

results/                     # 训练产物 (best_agent.msh / SAC logs / wandb)
environment.yml
```

## Stage flag 设计

```
                          rew_axis    success_axis_threshold
M1'  pos-only             0.0         inf                       (32 维 baseline)
M2a  pos + 粗轴对齐        2.0         0.5                       (≈ ±60° 锥)
M2b  pos + 紧轴对齐        2.0         0.2                       (≈ ±37° 锥)
```

`success_axis_threshold = inf` 时 success_mask 退化成 pos-only，`-w_axis * axis_err`
在 `rew_axis = 0` 时数学上为 0，所以 M1' 和老 strict-pos-only 在 reward 量级上等价。
M2a/M2b 通过 `--load_agent` 续训上一个 stage 的 checkpoint。

### M2c — sphere-proxy clearance (实际 curriculum)

M2b 阶段一次性诊断 (`scripts/archive/diagnose_m2b_clearance.py`) 验证 PhysX
collision 力检测 @1N 都抓不到, 真实 sphere-proxy clearance (减半径) 92.5% 的
success 帧是穿插状态. M2c 加几何 clearance reward / success gate (软),
**不加 CollisionAPI, 当前不开 hard absorbing**.

**当前已落地版本 (M2c safe-3cm)**:

```
              rew_clearance  clearance_soft  clearance_hard  备注
training:     0.5            0.00            -inf (off)      从 M2a warm-start, 短跑 ~5-10 epoch 即停
eval gate:    —              0.03            —               静态 eval / visualize / 视觉验证用
checkpoint:   results/best_agent_M2c_safe3cm_static.msh
```

实际经验:
- 长训 (>10 epoch) 会破坏 M2a 已有 reach (pos_err 涨回 ~1m, hold_success 跌回 0).
  best checkpoint 通常在 **epoch 1-3** 出现, 必须依赖 clearance-aware best 保存.
- `clearance_hard` 暂不开. 当前问题不是不安全 (sphere min ~ -2cm, 偶发), 而是
  训练易漂; hard absorbing 的 r_min/(1-γ) cliff 会放大不稳定. 等 reward shaping
  稳定了再加.
- `--keep_replay` 当前未尝试; 主要问题不是 replay 标签污染, 而是继续更新
  actor/critic 时 reach 信号被 clearance penalty 压垮.
- **5cm 不可作为 success gate**: 静态 M2a 的 ≥5cm per-step rate 已经 78%, 但
  连续 hold 10 步全部 ≥5cm 的比例只有 ~0.2 — gate 太严, success 过稀疏.
  3cm 是当前现实拐点 (`hold_success_rate_with_clearance ≈ 0.86`).

Penalty 公式 (支持 `clearance_soft <= 0`):
```
gap     = relu(clearance_soft - min_clearance)            # 单位 m, >=0
penalty = (gap / clearance_penalty_scale) ** 2            # normalized 无量纲
reward -= rew_clearance * penalty
```
`clearance_penalty_scale = 0.05m` 固定, 让 penalty 量级与 soft 阈值数值解耦.

Sphere proxy: 每侧 19 球 = 8 关节 (`arm_link_0..7`) + 7 段中点 + 4 EE 部件
(coupler / hande_link / 2 finger), 默认半径 arm=6cm / EE=3cm, 可用
`--proxy_arm_radius` / `--proxy_ee_radius` 覆盖.
clearance_ij = ||c_L_i - c_R_j|| - r_L_i - r_R_j.
不进 obs (32 维不变), 只走 reward / success / absorbing. 见 `_compute_min_clearance()`.

**已知 limitation (sphere proxy 精度)**: sphere proxy 是粗 safety proxy, 不是
mesh-level clearance. 当前覆盖在 EE / finger 末端和某些 arm link 弯曲段的视觉
mesh 处偏粗 (visualize_targets.py --show_proxies 可见). 如果怀疑 arm 球过大,
先用 `--show_proxies --proxy_arm_radius 0.05 --proxy_ee_radius 0.03` 对同一个
checkpoint 做视觉和 eval 对比, 再决定是否用新半径继续训练. M2c safe-3cm 在默认
proxy 下完成数值 + 视觉验证, 没观察到 policy 利用 proxy 漏洞做穿插; 但未来追
5cm 以上 / 真实 mesh-level safety 时应该重做 proxy 模型 (EE 部件多球 / capsule
端点).

**reward soft 与 success gate 已拆开** (env `clearance_success_threshold` 字段):
- `clearance_soft`: reward penalty 阈值, 给 policy 梯度
- `clearance_success_threshold`: success_mask 阈值, 控制 hold dwell 是否触发
- 默认 `clearance_success_threshold = clearance_soft` (向后兼容)
- 追 5cm 时典型用法: `--clearance_soft 0.05 --clearance_success_threshold 0.03`,
  reward 推到 5cm, success gate 留 3cm 不让 hold 稀疏.

M2c safe-3cm 当前用同一阈值 (默认), 没启用拆分.

`is_absorbing` 仍保留 hard absorbing 分支 (`clearance < clearance_hard` →
r_min/(1-γ) absorbing), 与 collision 同语义. M2c safe-3cm 版本不启用 (hard=-inf).

### M2d — approach / 共轴预插入 (不插入)

M2c safe-3cm 解决的是"安全到孔附近": success 仍是 `pos_err < 8~10cm`
这种球形容差, `axis_err` 只约束方向, 不保证 peg_tip 在 hole 中线上. 视觉
freeze 里见到过 `radial_err ≈ 4~6cm` 的合法 M2c dwell 帧, 这离 M3 需要的
mm 级插入起点太远.

M2d 在 M2c 和 M3 之间补一层低速 approach / servo-in:

```
axial_dist = dot(peg_tip - hole_entry, hole_axis)
axial_off  = abs(axial_dist - preinsert_offset)   # 目标仍是 hole_entry + 5cm*axis
radial_err = ||(peg_tip-hole_entry) - axial_dist*hole_axis||

success_M2d = axis_ok ∧ axial_off < axial_th ∧ radial_err < radial_th
              ∧ clearance >= clearance_success_threshold
```

M2d **不使用旧的 pos_err 球形 gate** 作为 success. `pos_err` 仍保留在 obs 和
日志里做诊断. M2d checkpoint selection 使用 **final-window stability**:
每个 episode 最后 `--final_window_steps` 步全部满足
`axis∧axial∧radial∧clearance` 才算 `final_window_success_rate_m2d`.
`final_window_in_thresh_rate_m2d` 和最后窗口全步的 radial/axial/axis/clearance
均值用于诊断与 tie-break. 旧的 `hold_success_rate_m2d/max_hold` 只说明是否
曾经路过 gate, 不再作为 M2d 保存 best 的主目标.

M2d 推荐打开 `--use_m2d_obs`, obs 从 32 维变 37 维:

```
base 32 + axial_dist[1] + axial_off[1] + radial_vec_hole2[2] + radial_err[1]
```

其中 `radial_vec_hole2` 是 radial_vec 在 hole 横截面基底 (right-EE X/Z)
下的 2D 坐标, 让 policy 直接看到"往哪边修". obs 维度改变后不能从 32 维
M2c checkpoint warm-start; M2c safe-3cm 作为行为基线保留, M2d 从 cold start
训练 37 维 agent.

建议起步:
- `action_scale=0.4` 起步: cold start 先保证能回到目标附近; 低速 servo 可在
  M2d 收敛后再降到 0.1.
- `rew_pos=0.2`: 保留弱 reach anchor, 但不让球形 pos reward 主导.
- 已验证的 curriculum 起步是 `0.30 -> 0.12 -> 0.08 -> 0.04`.
  直接 4cm cold start 之前 30 epoch 没有 M2d hold success; 从 30cm 起步能先
  学会 axial stop, 再逐步压 radial.
- M2d 不建议使用 `--terminal_hold_bonus`: 它会在进 gate 后截断 episode,
  让训练看不到后续是否漂走. `train_sac.py` 在 M2d active 且
  `terminal_hold_bonus>0` 时默认 raise, 除非显式传
  `--allow_m2d_terminal_bonus`.

## 运行

```bash
# M1': 32 维 baseline, axis 项关闭 (rew_axis 默认 0, axis_th 默认 inf)
python scripts/train_sac.py --no_wandb --n_epochs 100 \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50
cp results/best_agent.msh results/best_agent_M1p_32dim_pos10cm.msh

# M2a: 从 M1' warm-start, 加 axis reward (粗对齐)
# 默认会清空 replay buffer (旧 reward 数据不能带过来), 加 --keep_replay 可保留
python scripts/train_sac.py --no_wandb --n_epochs 150 \
    --load_agent results/best_agent_M1p_32dim_pos10cm.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 2.0 --success_axis_threshold 0.5
cp results/best_agent.msh results/best_agent_M2a_axis05.msh

# M2b: 从 M2a 收紧到 ±37° 锥
python scripts/train_sac.py --no_wandb --n_epochs 100 \
    --load_agent results/best_agent_M2a_axis05.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 2.0 --success_axis_threshold 0.2

# M2c safe-3cm 训练 (从 M2a warm-start, 短跑早停):
# 经验上 best checkpoint 出现在 epoch 1-3, 跑 5-10 epoch 足够; 长训会破坏 reach.
# train 用 clearance_soft=0.0 (sphere just-touching 起步), 不开 hard absorbing.
python scripts/train_sac.py --no_wandb --n_epochs 10 \
    --load_agent results/best_agent_M2a_axis05.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 2.0 --success_axis_threshold 0.5 \
    --rew_clearance 0.5 --clearance_soft 0.00
cp results/best_agent.msh results/best_agent_M2c_safe3cm_static.msh

# M2c eval — reward soft 仍用 0, success gate 用 3cm 验收 dwell.
python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
    --agent_path results/best_agent_M2c_safe3cm_static.msh \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 2.0 --success_axis_threshold 0.5 \
    --rew_clearance 0.5 --clearance_soft 0.00 --clearance_success_threshold 0.03

# M2c 视觉验证 (3cm gate)
python scripts/visualize_policy.py \
    --agent_path results/best_agent_M2c_safe3cm_static.msh \
    --preinsert_success_pos_threshold 0.10 \
    --rew_axis 2.0 --success_axis_threshold 0.5 \
    --rew_clearance 0.5 --clearance_soft 0.00 --clearance_success_threshold 0.03

# M2d.0 approach: 37 维 cold start, 共轴预插入 (30cm loose gate).
python scripts/train_sac.py --no_wandb --n_epochs 30 \
    --use_m2d_obs \
    --rew_pos 0.5 --action_scale 0.4 \
    --rew_axis 2.0 --success_axis_threshold 0.5 \
    --rew_axial_off 10.0 --axial_success_threshold 0.30 \
    --rew_radial 3.0 --radial_success_threshold 0.30 \
    --rew_clearance 0.5 --clearance_soft 0.00 --clearance_success_threshold 0.03 \
    --final_window_steps 30
cp results/best_agent.msh results/best_agent_M2d_approach30cm.msh

# 后续 warm-start 按 12cm / 8cm / 4cm 收紧. 每档先看 logger 的
# final_window_success / final_window_radial / final_window_axial, 再用
# visualize_policy freeze 抽查.

# M2d eval / visual freeze: 优先看 final-window 稳停, hold/max_hold 只做诊断.
python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
    --agent_path results/best_agent_M2d_approach30cm.msh \
    --use_m2d_obs \
    --rew_pos 0.5 --action_scale 0.4 \
    --rew_axis 2.0 --success_axis_threshold 0.5 \
    --rew_axial_off 10.0 --axial_success_threshold 0.30 \
    --rew_radial 3.0 --radial_success_threshold 0.30 \
    --rew_clearance 0.5 --clearance_soft 0.00 --clearance_success_threshold 0.03 \
    --final_window_steps 30

python scripts/visualize_policy.py \
    --agent_path results/best_agent_M2d_approach30cm.msh \
    --use_m2d_obs \
    --rew_pos 0.5 --action_scale 0.4 \
    --rew_axis 2.0 --success_axis_threshold 0.5 \
    --rew_axial_off 10.0 --axial_success_threshold 0.30 \
    --rew_radial 3.0 --radial_success_threshold 0.30 \
    --rew_clearance 0.5 --clearance_soft 0.00 --clearance_success_threshold 0.03

# 可选: 对同一个 checkpoint 只改 sphere proxy 半径做可视化/评估敏感性检查.
# 不传时默认 arm=0.06m / EE=0.03m; 下例把 arm 半径改成 5cm.
python scripts/visualize_policy.py \
    --agent_path results/best_agent_M2d_approach8cm.msh \
    --use_m2d_obs --show_proxies \
    --proxy_arm_radius 0.05 --proxy_ee_radius 0.03 \
    --rew_pos 0.1 --action_scale 0.3 \
    --rew_axis 2.0 --success_axis_threshold 0.35 \
    --rew_axial_off 8.0 --axial_success_threshold 0.08 \
    --rew_radial 16.0 --radial_success_threshold 0.08 \
    --rew_clearance 0.5 --clearance_soft 0.00 --clearance_success_threshold 0.03

# M1'/M2a/M2b eval (无 clearance gate, 沿用旧命令)
python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \
    --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \
    --rew_axis 2.0 --success_axis_threshold 0.5

# 可视化 marker (不训练)
python scripts/visualize_targets.py
python scripts/visualize_targets.py --preinsert_offset 0.08 --duration 20
python scripts/visualize_targets.py --n_resets 30   # 看 reset 分布

# 可视化 policy 在 hold-N 时冻结 (M2 必须传 --success_axis_threshold!)
python scripts/visualize_policy.py \
    --preinsert_success_pos_threshold 0.10 \
    --rew_axis 2.0 --success_axis_threshold 0.5
```

> 改 USD / sphere proxy / 重新验证 PhysX 失明时, 见
> `scripts/archive/README.md` (一次性诊断脚本).

### 产物位置

- `results/best_agent.msh` — best checkpoint, 选择标准随 stage 变化:
  - M1' (terminal_hold_bonus 启用, 无 M2c/M2d gate): best J
  - M2a/M2b/M2c (含 clearance gate): hold-score with clearance, 第一个 hold
    success 出现前允许 J fallback
  - M2c safe-3cm: hold-score with clearance only (J 不再 fallback, 避免远离/
    避碰策略刷高 J 但任务失败)
  - M2d (axial/radial gate 启用): M2d hold-score (axis∧axial∧radial∧clearance),
    无 J fallback. **注意**: 当前是 episode 内 max consecutive run 口径, 不是
    末段稳定度, 容易选到"路过 gate"解; freeze 验证仍是必要步骤. 见 train_sac
    里 `m2c_checkpoint_selection` / `_m2d_active` 分支.
- `results/SAC/` — mushroom-rl Logger 输出
- `results/wandb/` — wandb run 目录

## 任务设定

- **动作**: `a ∈ [-1,1]^14` → joint velocity `rad/s`, 系数 `action_scale=0.4`,
  控制周期 `0.1s` (`timestep=0.02 × n_intermediate=5`)
- **观测 (32 维; M2d 用 `--use_m2d_obs` 扩到 37 维)**:
  ```
  joint_pos[14] + joint_vel[14] + pos_vec[3] + axis_dot[1]
  pos_vec  = peg_tip - preinsert_target          # env-local
  axis_dot = dot(peg_axis, hole_axis) ∈ [-1,+1]  # -1 = 完美轴反平行
  ```
  M2d 追加:
  ```
  axial_dist[1] + axial_off[1] + radial_vec_hole2[2] + radial_err[1]
  ```
  `radial_vec_hole2` 给出 hole 横截面里的纠偏方向, 不只是一个 norm.
- **Peg / Hole frame** (env-local): 解析式
  ```
  peg_tip   = LeftEE_pos  + R(LeftEE_quat)  · (-0.0055, -0.0175, 0.125)
  hole_entry= RightEE_pos + R(RightEE_quat) · (-0.0055, -0.015,  0.125)
  hole_axis =                R(RightEE_quat) · (0, -1, 0)
  peg_axis  =                R(LeftEE_quat)  · (0, +1, 0)
  ```
  常量来自 `build_peghole_usd.py` 的 `PART_X / PART_Z + R_x(+90°)` 推导。
  这绕过 XFormPrim → Fabric flush 链路, headless / `render=False` 也保证 fresh。
- **Reward (统一骨架)**:
  ```
  - w_pos     · pos_err                          # ||peg_tip - preinsert_target||
  - w_axis    · axis_err                         # 1 + dot(peg_axis, hole_axis), 0 = ideal
  - w_joint_limit · joint_limit_norm             # 软极限, 进 margin 后才计
  - w_action  · ||raw a||²                       # pre-scale action, 与 action_scale 解耦
  + w_success · 1[success]                       # per-step dwell bonus, 不终止
  success = (pos_err < pos_th) ∧ (axis_err < axis_th)
  ```
  `rew_axis = 0` 时 axis 项消失 (M1'); `success_axis_threshold = inf` 时
  success 退化为 pos-only。collision (`arm_L` vs `arm_R` 自碰撞) 是唯一硬
  absorbing, reward 盖成 `r_min/(1-γ) ≈ -200`。
- **Eval success**: episode 内出现长度 `≥ hold_success_steps (default 10 ≈ 1s)`
  的连续 in-threshold 段。in-threshold 与 reward 用同一个 success_mask, 所以
  M2 的 hold-N 同时要求 pos 和 axis 都进。

## 已知约束 / 坑

- `num_envs=1` 触发 IsaacSim cloner `*` pattern bug，**至少用 2**。
- velocity 控制要求 `kp=0` (env `__init__` 里强制置零)，否则 reset 写的
  `pos_target` 会把关节钉住。
- `setup()` / `__init__` 末尾各 `world.step()` 一次，让 BODY_POS / BODY_ROT view
  同步物理状态 (否则 reset 后第一步 obs stale)。
- `_simulation_pre_step` 注入 `G(q)` 重力补偿 effort，让 agent 不需要从零学
  这个非线性映射；与 velocity drive 加性叠加。
- 不要装 `mushroom_rl.rl_utils.preprocessors.StandardizationPreprocessor`，
  Welford 在 vectorized env 下 std 会衰减到 0 → obs 被 clip 成 ±10 垃圾。
- `train_sac.py` / `eval_sac.py` 的 eval episode 数要与 `num_envs` 对齐
  (默认直接取 `num_envs`)。
- `success` 默认不做 absorbing — 只给每步 dwell bonus，避免边界 hugging 的
  Q-target 断崖。M1'/M2a/M2b 如要切 hold-N absorbing, 给
  `--terminal_hold_bonus > 0`; M2d 不建议开, 训练脚本默认禁止。
- **stage 切换 (M1'→M2a→M2b) 默认清空 replay buffer**: 旧 transitions 的
  reward 标签按旧 reward 算的, 留下来会拖 critic. 显式 `--keep_replay`
  可保留 (一般只在调试时用)。
- **31 维老 M1 checkpoint 不能 warm-start 到 32 维 env**: actor 输入层
  shape 不匹配, `Agent.load` 后 forward 直接抛错. 主线已退役;
  `results/best_agent_M1_31dim_pos10cm.msh` 仅供
  `scripts/archive/diagnose_m1_axis.py` 一次性诊断使用。
- peg/hole 是**视觉 only**：不产生接触力，也不会触发 `arm_L`/`arm_R`
  collision group。M3 才会给 peg 加 `CollisionAPI` 并设计 collision group。

## 后续阶段 (规划)

- **M2c+ (可选, 追 5cm 安全 margin)** — `clearance_soft` 与
  `clearance_success_threshold` 已拆开 (env 内字段独立, default 沿用 soft 向后
  兼容). 想追 5cm 时可以从 M2c safe-3cm warm-start, 跑短训:
  `--rew_clearance 0.5 --clearance_soft 0.05 --clearance_success_threshold 0.03`.
  reward 推到 5cm, gate 仍 3cm 保 hold 不稀疏. 也可考虑 hard absorbing
  (clearance_hard 接负值, 配 r_min cliff), 但需要先把 reward shaping 训稳定.
- **M3** — 在 M2d 已经把 `axial_off/radial_err` 拉到厘米级后, 再做真插入:
  `axial_dist < 0` 推进 + analytic illegal-insertion penalty。peg/hole
  CollisionAPI 暂不作为当前必需项; 若以后加入, 必须先单独验证它不会改变现有
  articulation 动力学。
- **M4** — Lagrangian SAC: 把 collision force / 接触力作为 cost, 由对偶变量
  自适应惩罚; 此时 cost 才有真正的物理意义。
