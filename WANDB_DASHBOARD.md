# LagSAC W&B Dashboard 配置

把 wandb workspace 整理成 3 张主图 + 1 个 summary table. 复制 panel 到你的 wandb workspace 里, 5 分钟设置好.

未来的 run (跑过 `define_metric` 改动后的训练脚本) 会自动:
- Use `epoch` as the W&B step metric.
- Hide: `legacy_eval_*`, `warmstart_*`, `geom_raw_epoch`, `geom_actor_epoch`, `geom_d_target_eff`, `env_steps`, `eval_ep_len`, `geom_n_ep_with_entry`, `alpha`
- Best/peak summary (MAX): `best_J`, `best_score`, `best_geom_rate`, `best_geom_max_run_mean`, `best_hold_rate`
- Last-value summary: `lambda`, `log_lambda`, `rollout_ep_cost`, `eval_ep_cost`, `*_violation`, `rollout_ep_n`, `final_lambda`, `epoch_absorb`, `eval_step_cost`

## 推荐 workspace 布局

### Panel 1: Task Progress (任务质量)
- Plot: line chart, x-axis = `epoch`
- Y-axis lines:
  - `best_J` (主 metric)
  - `best_score` (LagSAC 收敛后稳定值)
  - `best_geom_rate` (右 y-axis or separate panel)
- Reference lines:
  - SAC ref: best_J ≈ -85 (Stage 1) / -127.83 (Stage 2) / -57.23 (Stage 3)
  - SAC ref: best_score ≈ 65 (Stage 1) / 81 (Stage 2) / 113 (Stage 3)

### Panel 2: Safety / Cost (约束满足)
- Plot: line chart, x-axis = `epoch`
- Y-axis lines:
  - `rollout_ep_cost` (训练期 mean episode cost)
  - `eval_ep_cost` (eval 期 mean episode cost)
- Horizontal reference: `cost_limit_per_ep` (从 config 取; binary collision 常用 0.10, clearance05 使用 0.5)
- 看健康: rollout cost 上升 (cold-start) → 下降到 limit 以下

### Panel 3: Lagrangian Dual (λ 自适应)
- Plot: line chart, x-axis = `epoch`
- Y-axis (log scale 推荐):
  - `lambda` (主信号)
  - `log_lambda` (raw 空间, debug 用)
- Reference lines:
  - λ floor: 4.54e-5 (= e^-10, lambda_min=0 时)
  - C-soft 阈值: log_λ = -9.9
- 看健康: λ 早期上升响应 cost → policy 学避碰 → λ 缓慢衰减

### Panel 4 (optional): Collision Counters
- Plot: bar chart, x-axis = `epoch`
- Y-axis:
  - `epoch_absorb` (总碰撞次数 per epoch)
  - `epoch_absorb_sphere` (sphere proxy 碰撞)
  - `epoch_absorb_physx` (PhysX 实际接触)
- 看健康: 早期数十次 → epoch 20+ 降到 0

## Summary Table (跨 run 对比用)

Wandb table view, columns:

| 字段 | summary | 解读 |
|---|---|---|
| `wandb_run_name` | - | run 标识 |
| `best_J` | max | 任务最佳 J |
| `best_score` | max | 任务最佳 score (Stage 1: 65, Stage 2: 81, Stage 3: 113 是 SAC ref) |
| `best_geom_rate` | max | 任务成功率 |
| `final_lambda` | last | λ 是否回 floor / 卡 max |
| `eval_ep_cost` | last | 最终 eval cost (应 ≤ cost_limit) |
| `eval_step_cost` | last | step-level cost rate |
| `rollout_ep_cost` | last | rollout 训练期 cost |
| `rollout_ep_n` | last | λ 更新使用的完成 episode 数 |
| `lambda_update_source` | last/debug | 确认 λ 更新路径是否是期望模式 |

## Stage 3 (penetration cost) 时增加 panel

Stage 3 + `cost_signal=penetration` 时, **额外加 panel 5**:

### Panel 5: Penetration Quality (Stage 3 only)
- `geom_pen_in_active_mean` (active mask 内平均穿模, 应 ≈ 0)
- `geom_pen_in_active_max` (worst-case 穿模)
- `geom_entry_penetration_mean` (入口处穿模)
- `geom_final_penetration_mean` (终点穿模)
- `geom_clean_step_rate` (clean step 比例)

这是 Stage 3 paper contribution 主图: "LagSAC 用 constraint 替换 reward shaping, penetration 比 SAC v8 更低".

## 跨 run 对比图 (ablation 用)

`Group by` workspace 用 `wandb_group` 字段 (yaml 里设的). 不同的 group:
- `lag_geom_diag_frozen_lambda_*`: frozen λ baseline (= SAC)
- `lag_geom_stage1_active_verify_*`: active λ floor 验证
- `lag_geom_stage1_active_overlay_*`: d-atacom 风格 safety overlay
- `lag_geom_stage1_active_clearance05_*`: clearance cost 替换 collision reward penalty
- (TODO) `lag_geom_stage*_active_constraint`: constraint binding (penetration cost)

在每个 group 内, run 自动叠加在同一 panel 上, 方便对比.

## 如何应用 (一次性 5 分钟)

1. 打开 wandb workspace: https://wandb.ai/miaoxu010522-lund/bimanual_peghole
2. 进任一 run (e.g., active_overlay 这次)
3. 顶部 "Workspace" → "New section" → 命名 `Task Progress`
4. 在 section 里 "Add panel" → Line plot → 选 `best_J`, `best_score`, `best_geom_rate`
5. 重复 step 3-4, 创建 `Safety`, `Lagrangian`, `Collisions` sections
6. 顶部 "Save view as..." → 命名 `LagSAC main`
7. 以后跑 LagSAC run 都自动进这个 view (workspace 是 project-level, 跨 run 共享)

## 当前 Stage 1 clearance05 的关键 panel 截图位置

跑完后, 这三张图是 paper figure 1 的素材:
- 左: `best_score / best_geom_rate vs epoch` + SAC ref lines (65.2 / 1.0)
- 中: `rollout_ep_cost vs epoch` + horizontal cost_limit line at 0.5
- 右: `lambda vs epoch` (log scale optional)

故事: "去掉 SAC-style collision reward penalty 后, clearance-cost LagSAC 早期碰撞较多; λ 上升到约 4 后, epoch 37 左右把碰撞压到 0, best_geom_rate 达到 1.0, best_score 高于 SAC S1g reference."
