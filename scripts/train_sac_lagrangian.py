"""Lagrangian SAC 训练 — 双臂 peg-in-hole, mushroom-rl + VectorCore.

跟 train_sac.py 的差异:
- 算法换成 algorithm.lagrangian_sac.SACLagrangian (在 PROJECT_ROOT/algorithm/ 下).
- env._create_info_dictionary 必须返回 {"cost": tensor} (已在
  envs/dual_arm_peg_hole_cost_env.py 实现).
- 多了 --cost_limit_per_ep / --lr_lambda / --lambda_max / --init_log_lambda /
  --gamma_cost flags.
- 每 epoch eval 多算 eval_step_cost / eval_ep_cost, wandb 加 lambda / rollout_ep_cost.

Warmstart 注意:
- 从 SAC checkpoint warm-start: **必须 --actor_only_warmstart**, 否则报错
  (SAC 没有 cost critic / lambda, 全量 load 会缺字段).
- 从 SACLagrangian checkpoint warm-start: 可全量 (默认) 或 actor-only.

Cost signal 调标 (当前仅用 rollout_episode_rate 模式):
- env 的 cost 由 DualArmPegHoleCostEnv.cost() 选择:
    cost_signal="collision"    → PhysX OR sphere-proxy collision indicator (0/1)
    cost_signal="penetration"  → geom penetration_max 连续值 [0, 4mm]
- cost_limit_per_ep (--cost_limit_per_ep) 始终是每集平均 cost sum (per-episode):
    collision 模式: ≈ 每集碰撞次数 (e.g., 0.10 = 平均 0.1 次碰撞/集)
    penetration 模式: episode 内 penetration_max 累积量 (米·步)
- λ 更新信号来自 rollout EpisodeCostTracker, 不依赖 replay batch 或 eval.

整体架构：
  envs/dual_arm_peg_hole_cost_env.py   ← cost 信号来源 (DualArmPegHoleCostEnv)
           ↓  _create_info_dictionary() → {"cost": tensor}
  algorithm/lagrangian_sac.py          ← ConstrainedReplayMemory + SACLagrangian
           ↓  fit() 读 dataset.info.data["cost"]
  scripts/train_sac_lagrangian.py      ← 训练入口，接所有 CLI 参数
  scripts/_eval_utils.py               ← compute_cost_metrics() 评估 cost 指标

"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks import ActorNetwork, CriticNetwork
from scripts._eval_utils import (
    compute_geom_metrics,
    compute_cost_metrics,
    compute_hold_metrics,
    deterministic_policy,
    parse_home_weights,
    resolve_eval_episode_count,
)
from scripts.rollout_cost_tracker import (
    StepCostBridge,
    CostEnvWrapper,
    EpisodeCostTracker,
)


def _actor_net(approximator):
    return approximator.model.network


def _copy_actor_net_with_optional_partial(dst_net, src_net, *, allow_partial):
    """Copy ActorNetwork weights, optionally partial-copying h1 input columns."""
    src_in = src_net._h1.weight.shape[1]
    dst_in = dst_net._h1.weight.shape[1]
    if src_in != dst_in and not allow_partial:
        raise ValueError(
            "actor obs 维度不匹配: "
            f"checkpoint obs={src_in}D, env obs={dst_in}D. "
            "geom_stage 默认不允许 34D→41D partial warm-start; "
            "请先训练同 obs 维度 checkpoint，或显式传 --allow_partial_geom_warmstart 做 ablation."
        )
    with torch.no_grad():
        if src_in == dst_in:
            dst_net._h1.weight.copy_(src_net._h1.weight.to(dst_net._h1.weight.device))
        else:
            n = min(src_in, dst_in)
            dst_net._h1.weight[:, :n].copy_(
                src_net._h1.weight[:, :n].to(dst_net._h1.weight.device)
            )
        dst_net._h1.bias.copy_(src_net._h1.bias.to(dst_net._h1.bias.device))
        for name in ("_h2", "_out"):
            dst_layer = getattr(dst_net, name)
            src_layer = getattr(src_net, name)
            if dst_layer.weight.shape != src_layer.weight.shape:
                raise ValueError(
                    f"actor layer {name}.weight shape mismatch: "
                    f"{tuple(src_layer.weight.shape)} -> {tuple(dst_layer.weight.shape)}"
                )
            dst_layer.weight.copy_(src_layer.weight.to(dst_layer.weight.device))
            dst_layer.bias.copy_(src_layer.bias.to(dst_layer.bias.device))
    return "exact" if src_in == dst_in else f"partial_h1_{src_in}D_to_{dst_in}D"


def warmstart_actor_with_optional_partial(agent, old_agent, *, allow_partial):
    mu_mode = _copy_actor_net_with_optional_partial(
        _actor_net(agent.policy._mu_approximator),
        _actor_net(old_agent.policy._mu_approximator),
        allow_partial=allow_partial,
    )
    sigma_mode = _copy_actor_net_with_optional_partial(
        _actor_net(agent.policy._sigma_approximator),
        _actor_net(old_agent.policy._sigma_approximator),
        allow_partial=allow_partial,
    )
    return mu_mode if mu_mode == sigma_mode else f"mu={mu_mode}, sigma={sigma_mode}"


# ─────────────────────────────────────────────────────────────────────────────

INITIAL_REPLAY_SIZE = 10_000
MAX_REPLAY_SIZE = 500_000
BATCH_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=None,
                   help="episode horizon (env-steps). 默认走 env 默认值. "
                        "geom insert 通常建议 200.")
    p.add_argument("--render", action="store_true", help="打开 IsaacSim 窗口")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_epochs", type=int, default=400)
    p.add_argument("--n_steps_per_epoch", type=int, default=1024,
                   help="每个 epoch 收集的总 env-step 数 (不是 vector-step)")
    p.add_argument("--n_steps_per_fit", type=int, default=None,
                   help="两次 fit 之间收集的总 env-step 数 (默认 = num_envs, 即 1 个 vector-step)")
    p.add_argument("--utd", type=int, default=None,
                   help="每次 fit 块对应的总梯度步数. 默认自动取 n_steps_per_fit, 使 true UTD≈1")
    p.add_argument("--lr_actor", type=float, default=3e-4,
                   help="cold-start 联合任务推荐降到 1e-4 (避免 noisy critic 拉坏 actor), "
                        "warm-start 用 default 即可.")
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--alpha_max", type=float, default=0.1,
                   help="alpha 上限. cold-start 联合任务推荐 0.2 (探索), warm-start 0.1 稳态. "
                        "Lagrangian SAC 从已收敛 actor warmstart 时, Stage 2 经验建议保守用 0.05 "
                        "(λ 额外给 actor 施压, alpha 过高易引发塌方).")
    p.add_argument("--target_entropy", type=float, default=None,
                   help="目标 entropy. 默认自动取 -act_dim (SAC 标准). "
                        "14-DoF cold-start 联合任务可考虑 -7 (= -act_dim/2), "
                        "让 SAC 倾向稍集中, alpha 不顶 cap.")
    p.add_argument("--critic_warmup_transitions", type=int, default=None,
                   help="actor / α / λ 开始更新前需要的 replay 容量 (env-steps). 默认 = "
                        "INITIAL_REPLAY_SIZE (10K). actor-only warmstart 场景建议 "
                        "10240~15360 (约 5~10 epoch): critic 需要少量预热, 但时间太长会导致 "
                        "actor 解冻时梯度爆炸. 避免设 50000+. 必须 >= INITIAL_REPLAY_SIZE.")
    p.add_argument("--n_eval_episodes", type=int, default=None,
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除")

    # ---- Lagrangian 专属 ----------------------------------------------------
    p.add_argument("--cost_limit_per_ep", type=float, required=True,
                   help="每集平均 cost sum 预算 (rollout_episode_rate 模式). "
                        "collision 模式: ≈ 每集碰撞次数 (e.g., 0.10 = 0.1 次碰/集). "
                        "penetration 模式: episode 内 penetration_max 累积量 (米·步). "
                        "建议用 SAC baseline eval 实测 eval_ep_cost 后取 0.5× 标定.")
    p.add_argument("--lr_lambda", type=float, default=1e-3,
                   help="Lagrange 乘子学习率. 1e-3~1e-4, 通常比 lr_actor 低.")
    p.add_argument("--lambda_max", type=float, default=100.0,
                   help="λ clamp 上限. λ 冲到几百会把 actor 锁死.")
    p.add_argument("--lambda_min", type=float, default=0.0,
                   help="λ clamp 下限 (默认 0 = 不限). warmstart 场景建议 0.05~0.1, "
                        "防止 cost=0 时 λ 指数衰减到零后安全信号完全消失.")
    p.add_argument("--init_log_lambda", type=float, default=0.0,
                   help="log_λ 初值, 默认 0 → λ_init=1.")
    p.add_argument("--gamma_cost", type=float, default=None,
                   help="cost MDP 折扣. 默认 = env γ. 设 1.0 = average-cost "
                        "(注意此时 cost_limit_per_ep 直接是 Q_C 量纲).")
    p.add_argument("--lambda_update_mode", type=str, default="rollout_episode_rate",
                   choices=("recent_cost_rate", "episode_rate", "rollout_episode_rate", "q_cost"),
                   help="λ 更新信号来源. "
                        "rollout_episode_rate (推荐): cost_limit_per_ep = 每集平均 cost sum; "
                        "使用当前 policy rollout 中完成的真实 episode cost 统计, "
                        "在 core.learn() 结束后 drain EpisodeCostTracker 更新 λ; "
                        "不读 replay/eval, 不依赖 Mushroom flattened last. "
                        "episode_rate (eval-based, 有滞后): cost_limit_per_ep = 每集平均 cost sum; "
                        "λ 只在 eval 后用 eval_ep_cost 更新. "
                        "recent_cost_rate: fit() 中仅用当前 rollout 采样块的 "
                        "per-step mean(cost), 保留作消融; 不读 replay. "
                        "q_cost: 用 replay batch 的 Q_C(s,a) 更新 λ, cost_limit_per_ep "
                        "会按 horizon/gamma_cost 换算到 Q_C 量纲.")
    p.add_argument("--damp_scale", type=float, default=0.0,
                   help="q_cost 模式的 λ 阻尼强度. 0=关闭.")
    p.add_argument("--min_lambda_update_episodes", type=int, default=None,
                   help="rollout_episode_rate 模式: 触发 λ 更新所需的最少 episode 数. "
                        "默认自动取 num_envs (一个完整 env 轮次). 若一个 epoch 内完成的 "
                        "episode 数不足, 该 epoch 跳过 λ 更新, 下个 epoch 继续累积.")
    p.add_argument("--actor_grad_clip", type=float, default=None,
                   help="actor 参数梯度 L2 norm 上限. 默认不裁剪. warmstart 后 "
                        "critic warmup 结束时第一次 actor 更新易梯度爆炸, 建议 1.0.")
    # ------------------------------------------------------------------------

    # ---- env 参数 -----------------------------------------------------------
    # Reward weights are passed through to DualArmPegHoleEnv; the CMDP adapter
    # reuses the parent task reward and only moves safety pressure into cost().
    p.add_argument("--rew_pos", type=float, default=None,
                   help="pos_err 惩罚系数. 默认 1.0.")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="axis_err 惩罚系数. 默认 0.5 (axis_err∈[0,2], 折合满量程≈pos项).")
    p.add_argument("--rew_axis_progress", type=float, default=None,
                   help="LEGACY ignored: old standalone Lagrangian Stage-2 reward field. "
                        "当前 reward 复用父类 env, 不使用该项.")
    p.add_argument("--rew_success", type=float, default=None,
                   help="成功 per-step bonus. 默认 2.0.")
    p.add_argument("--rew_pos_success", type=float, default=None,
                   help="pos-only success bonus. 默认父类 0.0; Stage-2 warm-start 可设 1.0.")
    p.add_argument("--rew_joint_limit", type=float, default=None,
                   help="父类 joint-limit soft penalty 权重. 默认 0.02.")
    p.add_argument("--rew_action", type=float, default=None,
                   help="动作 L2 正则系数. 默认 0.005.")
    p.add_argument("--rew_home", type=float, default=None,
                   help="Home 偏差正则系数 (均匀权重). 默认 0.001.")
    p.add_argument("--home_weights", type=parse_home_weights, default=None,
                   help="home regularizer 逐关节权重. 接受 7 维单臂或 14 维完整权重.")
    # env 几何 / 物理
    p.add_argument("--initial_joint_noise", type=float, default=None)
    p.add_argument("--default_pose_variant", type=str, default=None,
                   choices=["easy", "harder"],
                   help="Reset pose variant. 'easy' keeps historical HOME_JOINT_POS; "
                        "'harder' uses crossed-forearm reset pose.")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None)
    p.add_argument("--preinsert_offset", type=float, default=None)
    p.add_argument("--success_axis_threshold", type=float, default=None)
    p.add_argument("--axis_gate_radius", type=float, default=None,
                   help="axis 惩罚距离门控半径. 默认 inf = 不门控.")
    p.add_argument("--joint_limit_margin_frac", type=float, default=None,
                   help="joint-limit penalty 起算 margin fraction. 默认 0.8.")
    p.add_argument("--clearance_hard", type=float, default=None)
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="eval 指标: 连续 N 步在阈内算 hold success. 不影响训练 reward.")

    # warmstart
    p.add_argument("--load_agent", type=str, default=None,
                   help="warm-start checkpoint. 从 SAC checkpoint 加载必须配 "
                        "--actor_only_warmstart, 否则报错 (SAC 没有 cost critic / λ).")
    p.add_argument("--keep_replay", action="store_true")
    p.add_argument("--actor_only_warmstart", action="store_true",
                   help="仅继承 actor (mu/sigma) 权重. SACLagrangian 从 SAC checkpoint "
                        "warmstart 必开此项. SACLagrangian → SACLagrangian 也建议开 "
                        "(reward 函数 / cost 信号若变, 旧 critic 语义错).")
    p.add_argument("--allow_partial_geom_warmstart", action="store_true",
                   help="允许 actor 第一层从旧 32/34D obs checkpoint partial-copy 到 41D geom obs. "
                        "默认禁止, 避免旧球形 reward manifold 污染 geom 路径; 仅 ablation 使用.")
    p.add_argument("--proxy_arm_radius", type=float, default=None)
    p.add_argument("--proxy_ee_radius", type=float, default=None)
    p.add_argument("--enable_table_collision",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Enable table safety: geometry-plane table clearance enters cost, "
                        "and a runtime PhysX table collider provides contact/absorbing diagnostics.")
    p.add_argument("--table_collision_terminates",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="If table safety is enabled, terminate on PhysX arm/EE-vs-table contact.")
    p.add_argument("--table_z", type=float, default=None,
                   help="World/env-local table top z for proxy clearance. Default 0.0.")
    p.add_argument("--table_clearance_hard", type=float, default=None,
                   help="Hard absorbing threshold for arm/EE table clearance. Default 0.0m.")
    p.add_argument("--table_clearance_cost_margin", type=float, default=None,
                   help="Margin for table clearance cost. Default 0.03m.")
    p.add_argument("--exclude_ee_from_physx_self_collision", action="store_true",
                   help="Stage 3 peg/hole 真实 collider 用: PhysX arm_L vs arm_R "
                        "self-collision 分组排除左右 EE link, 避免正常 peg-hole "
                        "接触被 hard absorbing 误杀. EE 区域仍由 sphere-proxy 兜底.")

    # obs
    p.add_argument("--use_axis_resid_obs", action="store_true",
                   help="agent obs 32 → 34 维: axis_dot[1] 替换成 axis_resid[3] = "
                        "peg_axis + hole_axis (world frame). 模长 ∈ [0, 2], "
                        "0 = 完美反对齐 (preinsert), 2 = 同向 (home pose). "
                        "全程光滑无奇异, 且 axis_err = ||resid||²/2 = 1+dot 与 reward 同语义, "
                        "success_axis_threshold 仍然用旧的 1+dot 量纲 (0.2 ≈ ±37° 锥). "
                        "**注意**: obs 维度变, 32 维 checkpoint 不能 warm-start 到 34 维.")

    # ──────────────── Geometric preinsert (Stage 1g/2g/3g) ────────────────
    p.add_argument("--geom_stage", type=str, default=None,
                   choices=("prepos", "preaxis", "insert"),
                   help="启用几何同源 reward + 41D obs. "
                        "prepos=Stage1g, preaxis=Stage2g, insert=Stage3g.")
    p.add_argument("--geom_d_target_neg", type=float, default=None)
    p.add_argument("--geom_d_target_pos", type=float, default=None)
    p.add_argument("--geom_d_target_ramp_start", type=int, default=None)
    p.add_argument("--geom_d_target_ramp_end", type=int, default=None)
    p.add_argument("--rew_geom_d", type=float, default=None)
    p.add_argument("--rew_geom_radial_tip", type=float, default=None)
    p.add_argument("--rew_geom_radial_max", type=float, default=None)
    p.add_argument("--rew_geom_axis", type=float, default=None)
    p.add_argument("--geom_d_sat", type=float, default=None)
    p.add_argument("--geom_radial_sat", type=float, default=None)
    p.add_argument("--rew_geom_soft_success", type=float, default=None)
    p.add_argument("--geom_soft_d_sigma", type=float, default=None)
    p.add_argument("--geom_soft_radial_sigma", type=float, default=None)
    p.add_argument("--geom_soft_axis_sigma", type=float, default=None)
    p.add_argument("--geom_soft_penetration_sigma", type=float, default=None)
    p.add_argument("--geom_d_th", type=float, default=None)
    p.add_argument("--geom_r_tip_th", type=float, default=None)
    p.add_argument("--geom_r_max_th", type=float, default=None)
    p.add_argument("--geom_axis_th", type=float, default=None)
    p.add_argument("--geom_insert_d_ins", type=float, default=None)
    p.add_argument("--geom_insert_r_max_th", type=float, default=None)
    p.add_argument("--geom_pen_th", type=float, default=None)
    p.add_argument("--rew_geom_progress", type=float, default=None)
    p.add_argument("--geom_gate_radial_sigma", type=float, default=None)
    p.add_argument("--geom_gate_axis_sigma", type=float, default=None)
    p.add_argument("--rew_geom_penetration", type=float, default=None,
                   help="soft penetration penalty -w·penetration_max. "
                        "cost_signal=collision 时保留在 reward 里; "
                        "cost_signal=penetration 时可用 "
                        "--drop_penetration_reward_for_cost 清零该组件.")
    p.add_argument("--geom_gate_penetration_sigma", type=float, default=None)
    p.add_argument("--cost_signal", type=str, default=None,
                   choices=("collision", "penetration", "clearance"),
                   help="CMDP cost 信号: collision=机器人碰撞0/1; penetration=连续 penetration_max; "
                        "clearance=双臂 proxy clearance margin cost.")
    p.add_argument("--clearance_cost_margin", type=float, default=None)
    p.add_argument("--clearance_cost_clip_max", type=float, default=None,
                   help="Optional upper clip for clearance cost. Default None = "
                        "D-ATACOM-style max((margin-clearance)/margin, 0) with "
                        "no upper clip; pass 1.0 to reproduce the old [0,1] cost.")
    p.add_argument("--cost_scale", type=float, default=None,
                   help="Multiplicative gain on info['cost'] / env.cost(). "
                        "Default None = env default (1.0). For cost_signal=penetration "
                        "use 1000 (m→mm) or 1/geom_pen_th to make cost dimensionless, "
                        "otherwise λ·Q_C is 5 orders of magnitude below Q_R and the "
                        "Lagrangian gradient cannot move the policy.")
    p.add_argument("--keep_collision_reward_penalty",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop_penetration_reward_for_cost",
                   action=argparse.BooleanOptionalAction, default=True)
    # Current benchmark semantics: sphere proxy + table clearance proxy are cost-only;
    # collision absorbing and reward cliff use PhysX real-contact masks only.
    p.add_argument("--sphere_collision_terminates",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Deprecated compatibility flag. Ignored: sphere proxy is "
                        "cost-only and does not trigger absorbing.")
    p.add_argument("--physx_collision_terminates",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="True (default) keeps PhysX real-contact as episode terminal. "
                        "Recommended True even in clean CMDP runs as physics-stability guard.")
    p.add_argument("--enable_physx_arm_collision",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Apply PhysX CollisionAPI to iiwa arm links at runtime via "
                        "capsule colliders sized from URDF. Without this, "
                        "epoch_absorb_physx is always 0 because the robot USD has "
                        "no CollisionAPI on arm links. Default False = legacy.")
    p.add_argument("--geom_progress_floor", type=float, default=None)
    p.add_argument("--rew_geom_advance", type=float, default=None)
    p.add_argument("--geom_d_gate_mode", type=str, default=None,
                   choices=("off", "alignment"))
    p.add_argument("--rew_geom_bad_entry", type=float, default=None)
    p.add_argument("--geom_bad_entry_radial_safe", type=float, default=None)
    p.add_argument("--geom_bad_entry_axis_safe", type=float, default=None)
    p.add_argument("--geom_bad_entry_pen_safe", type=float, default=None)
    p.add_argument("--terminal_hold_bonus", type=float, default=0.0,
                   help="兼容配置参数. Lagrangian 路径固定不启用 hold-N cliff, "
                        "该值应保持 0.0.")

    # wandb
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()



def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.n_steps_per_fit is None:
        args.n_steps_per_fit = args.num_envs
    if args.utd is None:
        args.utd = args.n_steps_per_fit
    if args.utd < 1:
        raise ValueError("--utd 必须 >= 1")
    if args.n_steps_per_fit < args.num_envs:
        raise ValueError(
            f"n_steps_per_fit ({args.n_steps_per_fit}) 不能小于 num_envs ({args.num_envs})"
        )
    if args.n_steps_per_epoch < args.n_steps_per_fit:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 不能小于 "
            f"n_steps_per_fit ({args.n_steps_per_fit})"
        )
    if args.n_steps_per_fit % args.num_envs != 0:
        raise ValueError(
            f"n_steps_per_fit ({args.n_steps_per_fit}) 必须能被 num_envs 整除"
        )
    if args.n_steps_per_epoch % args.n_steps_per_fit != 0:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 必须能被 n_steps_per_fit 整除"
        )
    if args.critic_warmup_transitions is None:
        args.critic_warmup_transitions = INITIAL_REPLAY_SIZE
    if args.critic_warmup_transitions < INITIAL_REPLAY_SIZE:
        raise ValueError(
            f"--critic_warmup_transitions ({args.critic_warmup_transitions}) 必须 >= "
            f"INITIAL_REPLAY_SIZE ({INITIAL_REPLAY_SIZE})"
        )
    if args.cost_limit_per_ep < 0.0:
        raise ValueError(f"--cost_limit_per_ep ({args.cost_limit_per_ep}) 必须 >= 0")
    if args.cost_limit_per_ep == 0.0:
        print("[WARN] --cost_limit_per_ep=0 会让 λ 一上来就爆, 一般先用 baseline eval 标定后取保守预算.")
    # Default: one full generation of parallel envs (each env completes ≥1 episode).
    if args.min_lambda_update_episodes is None:
        args.min_lambda_update_episodes = args.num_envs
    args.n_eval_episodes = resolve_eval_episode_count(
        args.n_eval_episodes, args.num_envs, "--n_eval_episodes"
    )

    from envs import DualArmPegHoleCostEnv
    env_kwargs = dict(num_envs=args.num_envs, headless=not args.render)
    if args.horizon is not None:
        env_kwargs["horizon"] = args.horizon
    for key in (
        # Parent task reward:
        "rew_pos", "rew_axis", "rew_success", "rew_pos_success",
        "rew_joint_limit", "rew_action", "rew_home", "home_weights",
        "axis_gate_radius", "joint_limit_margin_frac",
        # 几何 / 物理
        "initial_joint_noise", "default_pose_variant",
        "preinsert_success_pos_threshold",
        "preinsert_offset", "success_axis_threshold",
        "clearance_hard", "proxy_arm_radius", "proxy_ee_radius",
        "table_z", "table_clearance_hard", "table_clearance_cost_margin",
        "clearance_cost_margin", "clearance_cost_clip_max", "cost_scale",
        "keep_collision_reward_penalty", "drop_penetration_reward_for_cost",
        "sphere_collision_terminates", "physx_collision_terminates",
        "enable_physx_arm_collision",
        # Geometric preinsert kwargs:
        "geom_stage", "geom_d_target_neg", "geom_d_target_pos",
        "geom_d_target_ramp_start", "geom_d_target_ramp_end",
        "rew_geom_d", "rew_geom_radial_tip", "rew_geom_radial_max",
        "rew_geom_axis", "geom_d_sat", "geom_radial_sat",
        "rew_geom_soft_success", "geom_soft_d_sigma",
        "geom_soft_radial_sigma", "geom_soft_axis_sigma",
        "geom_soft_penetration_sigma",
        "geom_d_th", "geom_r_tip_th", "geom_r_max_th", "geom_axis_th",
        "geom_insert_d_ins", "geom_insert_r_max_th", "geom_pen_th",
        "rew_geom_progress", "geom_gate_radial_sigma", "geom_gate_axis_sigma",
        "rew_geom_penetration", "geom_gate_penetration_sigma", "cost_signal",
        "geom_progress_floor", "rew_geom_advance",
        "geom_d_gate_mode", "rew_geom_bad_entry",
        "geom_bad_entry_radial_safe", "geom_bad_entry_axis_safe",
        "geom_bad_entry_pen_safe",
    ):
        value = getattr(args, key, None)
        if value is not None:
            env_kwargs[key] = value
    # Keep 41D geom obs pos_vec target aligned with the depth target unless the
    # user explicitly overrides preinsert_offset.
    if args.geom_stage is not None and args.preinsert_offset is None:
        d_target_neg = (
            args.geom_d_target_neg
            if args.geom_d_target_neg is not None
            else -0.08
        )
        env_kwargs["preinsert_offset"] = abs(float(d_target_neg))
    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    if args.exclude_ee_from_physx_self_collision:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True
    env_kwargs["enable_table_collision"] = bool(args.enable_table_collision)
    env_kwargs["table_collision_terminates"] = bool(args.table_collision_terminates)
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    mdp = DualArmPegHoleCostEnv(**env_kwargs)
    mdp.seed(args.seed)

    # IsaacSim 启动后才能导入 mushroom_rl / algo
    from mushroom_rl.core import Agent, VectorCore, Logger, Dataset
    from algorithm import SACLagrangian

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]
    target_entropy = args.target_entropy
    if target_entropy is None:
        target_entropy = -float(act_dim)

    def _cold_create_sac_lag():
        actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                            output_shape=(act_dim,))
        actor_optimizer = {"class": optim.Adam, "params": {"lr": args.lr_actor}}
        critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                             output_shape=(1,), action_dim=act_dim,
                             optimizer={"class": optim.Adam, "params": {"lr": args.lr_critic}},
                             loss=F.mse_loss)
        cost_critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                                  output_shape=(1,), action_dim=act_dim,
                                  optimizer={"class": optim.Adam, "params": {"lr": args.lr_critic}},
                                  loss=F.mse_loss)
        return SACLagrangian(
            mdp_info=mdp.info,
            actor_mu_params=actor_params,
            actor_sigma_params=actor_params,
            actor_optimizer=actor_optimizer,
            critic_params=critic_params,
            cost_critic_params=cost_critic_params,
            batch_size=BATCH_SIZE,
            initial_replay_size=INITIAL_REPLAY_SIZE,
            max_replay_size=MAX_REPLAY_SIZE,
            warmup_transitions=args.critic_warmup_transitions,
            tau=0.005,
            lr_alpha=args.lr_alpha,
            use_log_alpha_loss=True,
            target_entropy=target_entropy,
            cost_limit=args.cost_limit_per_ep,
            lr_lambda=args.lr_lambda,
            lambda_max=args.lambda_max,
            lambda_min=args.lambda_min,
            init_log_lambda=args.init_log_lambda,
            gamma_cost=args.gamma_cost,
            lambda_update_mode=args.lambda_update_mode,
            actor_grad_clip=args.actor_grad_clip,
            damp_scale=args.damp_scale,
        )

    if args.load_agent is not None:
        load_path = Path(args.load_agent)
        if not load_path.is_file():
            raise FileNotFoundError(f"--load_agent 路径不存在: {load_path}")
        old_agent = Agent.load(str(load_path))
        old_class = type(old_agent).__name__

        if args.actor_only_warmstart:
            agent = _cold_create_sac_lag()
            mode = warmstart_actor_with_optional_partial(
                agent,
                old_agent,
                allow_partial=(
                    args.allow_partial_geom_warmstart and mdp._geom_stage is not None
                ),
            )
            print(f"[WARM-START actor-only] from {old_class} @ {load_path}; "
                  f"actor_copy={mode}; critic / cost critic / α / λ / replay 全部冷启动.")
            if args.keep_replay:
                print("[WARM-START actor-only] --keep_replay 已忽略.")
            del old_agent
        else:
            if old_class != "SACLagrangian":
                raise RuntimeError(
                    f"--load_agent 是 {old_class}, 全量 warmstart 只能用 SACLagrangian "
                    "checkpoint. 从 SAC checkpoint warmstart 必须加 --actor_only_warmstart."
                )
            agent = old_agent
            print(f"[WARM-START full] 整体加载 SACLagrangian from {load_path}")
            if not args.keep_replay:
                agent._replay_memory.reset()
                print("[WARM-START full] replay buffer 已清空.")
            else:
                print("[WARM-START full] 保留旧 replay buffer.")
    else:
        agent = _cold_create_sac_lag()

    def clamp_alpha(_dataset=None):
        with torch.no_grad():
            agent._log_alpha.clamp_(max=math.log(args.alpha_max))

    # ── Rollout episode cost tracker setup ────────────────────────────────────
    # rollout_episode_rate mode needs two extra pieces:
    #   CostEnvWrapper  — intercepts mdp.step_all() to cache (cost, mask) into
    #                     the bridge before VectorCore fires callback_step.
    #   EpisodeCostTracker — used as callback_step; reads bridge; accumulates
    #                     per-env episode cost sums; pushes to buffer on episode end.
    # All other modes leave _tracker = None and use the real mdp directly.
    if args.lambda_update_mode == "rollout_episode_rate":
        _bridge = StepCostBridge()
        _tracker = EpisodeCostTracker(
            num_envs=args.num_envs,
            bridge=_bridge,
            min_episodes=args.min_lambda_update_episodes,
        )
        _env_for_core = CostEnvWrapper(mdp, _bridge)
    else:
        _bridge = None
        _tracker = None
        _env_for_core = mdp
    # ──────────────────────────────────────────────────────────────────────────

    core = VectorCore(
        agent, _env_for_core,
        callbacks_fit=[clamp_alpha],
        # callback_step=None → VectorCore replaces it with a no-op lambda internally.
        callback_step=_tracker,
    )

    from datetime import datetime
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = results_dir / "checkpoints_lag" / run_ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_J_path = ckpt_dir / "best_agent.msh"
    best_hold_path = ckpt_dir / "best_hold.msh"
    final_path = ckpt_dir / "final_agent.msh"
    best_J_path_flat = results_dir / "best_agent_lag.msh"
    best_hold_path_flat = results_dir / "best_hold_lag.msh"
    final_path_flat = results_dir / "final_agent_lag.msh"
    best_hold_path_flat.unlink(missing_ok=True)
    logger = Logger("SACLagrangian", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"checkpoint 目录: {ckpt_dir}")
    mdp.set_geom_epoch(0)
    if mdp._geom_stage is not None:
        obs_mode = f"geom_{mdp._geom_stage} (axis_resid+hole_geom)"
    else:
        obs_mode = "axis_resid" if mdp._use_axis_resid_obs else "base"
    logger.info(f"obs_dim={obs_dim} ({obs_mode})  "
                f"act_dim={act_dim}  horizon={mdp.info.horizon}")
    logger.info(f"action_scale={mdp._action_scale:.3f}")
    logger.info(
        "physx_self_collision_group="
        + ("arm_links_only" if mdp._exclude_ee_from_physx_self_collision
           else "arm_links_plus_ee")
    )
    axis_th_str = (
        "inf" if math.isinf(mdp._success_axis_threshold)
        else f"{mdp._success_axis_threshold:.3f}"
    )
    axis_gate_str = (
        "inf" if math.isinf(mdp._axis_gate_radius)
        else f"{mdp._axis_gate_radius:.3f}m"
    )
    logger.info(f"preinsert_pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
                f"axis_th={axis_th_str}  cost_signal={mdp._cost_signal}")
    if mdp._geom_stage is None:
        logger.info(
            f"task reward (normal): w_pos={mdp._w_pos:.3f}  "
            f"w_axis={mdp._w_axis:.3f}  axis_gate_radius={axis_gate_str}  "
            f"w_pos_success={mdp._w_pos_success:.3f}  "
            f"w_success={mdp._w_success:.3f}  "
            f"w_joint_limit={mdp._w_joint_limit:.4f}  "
            f"w_action={mdp._w_action:.4f}  w_home={mdp._w_home:.4f}"
        )
    else:
        logger.info(
            f"geom_stage={mdp._geom_stage}  "
            f"d_target_neg={mdp._geom_d_target_neg:+.3f}  "
            f"d_target_pos={mdp._geom_d_target_pos:+.3f}  "
            f"d_target_eff={mdp._geom_d_target_eff:+.3f}"
        )
        logger.info(
            f"geom reward: w_d={mdp._w_geom_d:.2f}  "
            f"w_rad_tip={mdp._w_geom_radial_tip:.2f}  "
            f"w_rad_max={mdp._w_geom_radial_max:.2f}  "
            f"w_axis={mdp._w_geom_axis:.2f}  "
            f"w_progress={mdp._w_geom_progress:.2f}  "
            f"w_advance={mdp._w_geom_advance:.2f}  "
            f"w_bad_entry={mdp._w_geom_bad_entry:.2f}  "
            f"w_penetration={mdp._w_geom_penetration:.2f}  "
            "r_geom_penetration="
            + (
                "0"
                if (
                    mdp._cost_signal == "penetration"
                    and getattr(mdp, "_drop_penetration_reward_for_cost", False)
                )
                else "kept"
            )
        )
        logger.info(
            f"geom thresholds: d_th={mdp._geom_d_th:.3f}  "
            f"r_tip_th={mdp._geom_r_tip_th:.3f}  "
            f"r_max_th={mdp._geom_r_max_th:.3f}  "
            f"axis_th={mdp._geom_axis_th:.3f}  "
            f"insert_d_ins={mdp._geom_insert_d_ins:+.3f}  "
            f"insert_r_max_th={mdp._geom_insert_r_max_th:.3f}  "
            f"pen_th={mdp._geom_pen_th*1000:.1f}mm"
        )
    if args.load_agent is not None:
        logger.info(f"warm-start: {args.load_agent}")
    logger.info(f"target_entropy={target_entropy:.3f}  "
                f"lr_actor={args.lr_actor:.1e}  lr_critic={args.lr_critic:.1e}  "
                f"lr_alpha={args.lr_alpha:.1e}  alpha_max={args.alpha_max:.3f}")
    gamma_cost_resolved = (args.gamma_cost if args.gamma_cost is not None
                           else mdp.info.gamma)
    grad_clip_str = f"{args.actor_grad_clip:.2f}" if args.actor_grad_clip else "off"
    logger.info(f"[Lagrangian] cost_limit_per_ep={args.cost_limit_per_ep:.4f}  "
                f"cost_signal={mdp._cost_signal}  "
                f"clearance_clip_max={mdp._clearance_cost_clip_max}  "
                f"cost_scale={mdp._cost_scale:g}  "
                f"lr_lambda={args.lr_lambda:.1e}  "
                f"lambda_max={args.lambda_max:.1f}  lambda_min={args.lambda_min:.4f}  "
                f"init_log_lambda={args.init_log_lambda:.3f}  "
                f"gamma_cost={gamma_cost_resolved:.3f}  "
                f"lambda_update_mode={args.lambda_update_mode}  "
                f"damp_scale={args.damp_scale:.3f}  "
                f"actor_grad_clip={grad_clip_str}")
    if args.lambda_update_mode == "q_cost":
        logger.info(f"[q_cost] q_cost_limit={agent._q_cost_limit:.6f} "
                    f"(from cost_limit_per_ep={args.cost_limit_per_ep:.6f})")
    if args.lambda_update_mode == "rollout_episode_rate":
        logger.info(
            f"[rollout_episode_rate] EpisodeCostTracker 已启用  "
            f"min_lambda_update_episodes={args.min_lambda_update_episodes}  "
            "λ 在每个 epoch 的 core.learn() 结束后由 drain() 更新; "
            "eval 期间 tracker 暂停 (active=False) 避免 deterministic policy 数据污染."
        )
    critic_only_steps = max(0, args.critic_warmup_transitions - INITIAL_REPLAY_SIZE)
    critic_only_epochs = math.ceil(critic_only_steps / args.n_steps_per_epoch)
    if critic_only_steps > 0:
        logger.info(
            f"critic_warmup_transitions={args.critic_warmup_transitions} env-steps "
            f"(replay-fill {INITIAL_REPLAY_SIZE} + critic-only {critic_only_steps} "
            f"≈ {critic_only_epochs:.1f} epoch, actor/α/λ 此期间冻结)"
        )
    if mdp._geom_stage == "insert" and critic_only_epochs > 0:
        logger.info(
            f"geom insert schedule offset: critic_only_epochs={critic_only_epochs} "
            f"(actor-relative epoch = max(0, raw_epoch - {critic_only_epochs}))"
        )
    logger.info(f"n_steps_per_epoch={args.n_steps_per_epoch}  "
                f"n_steps_per_fit={args.n_steps_per_fit}  num_envs={args.num_envs}")

    mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)
    obs, _ = mdp.reset_all(mask)
    pos_err, axis_err, in_thresh_mask = mdp._compute_task_errors(obs)
    logger.info("reset stats: "
                f"in_thresh_rate={float(in_thresh_mask.float().mean()):.3f}  "
                f"pos_err_mean={float(pos_err.mean()):.4f}m  "
                f"pos_err_min={float(pos_err.min()):.4f}m  "
                f"pos_err_max={float(pos_err.max()):.4f}m  "
                f"axis_err_mean={float(axis_err.mean()):.4f}  "
                f"axis_err_max={float(axis_err.max()):.4f}")
    if mdp._geom_stage is not None and mdp._cached_d is not None:
        logger.info(
            "geom reset stats: "
            f"d_mean={float(mdp._cached_d.mean()):+.4f}m  "
            f"radial_max_mean={float(mdp._cached_radial_max.mean()):.4f}m  "
            f"penetration_max_mean={float(mdp._cached_penetration_max.mean())*1000:.2f}mm"
        )

    wandb_run = None
    if not args.no_wandb:
        import wandb
        plot_group = args.wandb_group or "ungrouped"
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config={**vars(args), "algo": "SACLagrangian",
                    # Plain config key for W&B Reports "Group by".
                    # wandb.init(group=...) is run metadata and is not always
                    # exposed in the Group by selector; plot_group is.
                    "plot_group": plot_group,
                    "target_entropy_resolved": target_entropy,
                    "gamma_cost_resolved": gamma_cost_resolved,
                    "cost_signal_resolved": mdp._cost_signal,
                    "clearance_cost_clip_max_resolved": mdp._clearance_cost_clip_max,
                    "cost_scale_resolved": mdp._cost_scale,
                    "geom_stage_resolved": mdp._geom_stage,
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

        # ── Wandb dashboard organization (2026-05-17 cleanup) ───────────────
        # Goal: dashboard shows ONLY paper-grade benchmark metrics by default.
        # Everything else is hidden (still uploaded for offline analysis, but
        # not cluttering the default workspace panel grid).
        #
        # Paper benchmark categories (kept visible):
        #   Task return:     J, R, best_J
        #   Task quality:    geom_hold_rate, geom_max_run_mean, geom_step_rate,
        #                    geom_final_success_rate, best_geom_*
        #   Safety events:   epoch_absorb_{sphere,physx,table},
        #                    epoch_collision_{sphere,physx,table}
        #   Cost signal:     eval_ep_cost, rollout_ep_cost, cost_limit_per_ep
        #   Lagrangian:      lambda, eval_ep_violation, rollout_ep_violation
        #
        # All `geom_entry_*`, `geom_final_*`, `geom_pen_*`, `geom_radial_*`,
        # `geom_axis_*`, `geom_d_*`, sub-mask rates, etc. — useful for
        # post-hoc debugging but noise on training-time dashboard. Hidden.
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        # Hide everything noisy / derived / debug-only.
        for pat in (
            # Existing hidden list (legacy + warmstart + meta diagnostics)
            "legacy_eval_*", "warmstart_*",
            "geom_raw_epoch", "geom_actor_epoch", "geom_d_target_eff",
            "env_steps", "eval_ep_len", "geom_n_ep_with_entry", "alpha",
            # 2026-05-17 cleanup: hide geom diagnostics noise
            "geom_d_target_mean", "geom_d_err_mean", "geom_d_err_min",
            "geom_radial_tip_mean", "geom_radial_tip_min",
            "geom_radial_max_mean", "geom_radial_max_min",
            "geom_axis_err_mean", "geom_axis_err_min",
            "geom_prepos_step_rate", "geom_preaxis_step_rate", "geom_insert_step_rate",
            "geom_ep_success_rate_mean",
            # Entry-time diagnostics (useful for post-hoc, not training plot)
            "geom_entry_d_mean", "geom_entry_d_err_mean",
            "geom_entry_radial_max_mean", "geom_entry_radial_max_max",
            "geom_entry_axis_err_mean", "geom_entry_axis_err_max",
            "geom_entry_penetration_mean", "geom_entry_penetration_max",
            # Final-state diagnostics (useful for post-hoc, not training plot)
            "geom_final_d_mean", "geom_final_d_err_mean",
            "geom_final_radial_max_mean", "geom_final_radial_max_max",
            "geom_final_axis_err_mean", "geom_final_axis_err_max",
            "geom_final_penetration_mean", "geom_final_penetration_max",
            # Penetration analytics (only relevant for penetration-cost runs)
            "geom_pen_max_mean", "geom_pen_max_max", "geom_clean_step_rate",
            "geom_pen_in_active_mean", "geom_pen_in_active_max",
            # Cost / λ internals (derived or string)
            "log_lambda", "lambda_update_source", "rollout_ep_n",
            "eval_step_cost", "epoch_absorb", "epoch_collision",
            # Score / metric-rate duplicates of best_geom_*
            "best_score", "best_metric_rate", "best_metric_max_run_mean",
            "best_hold_rate", "best_hold_max_hold_mean",
        ):
            wandb.define_metric(pat, hidden=True)
        # ── Summary aggregations (run table column ordering) ────────────────
        # Best/peak metrics: MAX summary (track peak across epochs).
        for m in ("best_J", "best_geom_rate", "best_geom_max_run_mean"):
            wandb.define_metric(m, summary="max")
        # Per-epoch metrics: LAST summary (final epoch value).
        for m in (
            # Task return
            "J", "R",
            # Task quality (current epoch)
            "geom_hold_rate", "geom_max_run_mean", "geom_step_rate",
            "geom_final_success_rate",
            # Safety (last epoch)
            "epoch_absorb_total", "epoch_collision_total",
            "epoch_absorb_sphere", "epoch_absorb_physx", "epoch_absorb_table",
            "epoch_collision_sphere", "epoch_collision_physx", "epoch_collision_table",
            # Cost / λ (last epoch)
            "eval_ep_cost", "rollout_ep_cost",
            "eval_ep_max_violation", "rollout_ep_max_violation",
            "eval_ep_violation", "rollout_ep_violation",
            "cost_limit_per_ep", "lambda", "final_lambda",
        ):
            wandb.define_metric(m, summary="last")

    empty_dataset = Dataset.generate(mdp.info, agent.info, n_steps=1, n_envs=args.num_envs)

    if args.load_agent is not None:
        logger.info("=" * 60)
        logger.info("[EVAL @ epoch 0] warm-start actor BEFORE 任何 fit / 任何 warmup")
        with deterministic_policy(agent):
            ds0 = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        m0 = compute_hold_metrics(ds0, mdp, args.hold_success_steps)
        mg0 = (
            compute_geom_metrics(ds0, mdp, args.hold_success_steps)
            if mdp._geom_stage is not None else None
        )
        c0 = compute_cost_metrics(ds0, args.n_eval_episodes)
        if mg0 is not None:
            logger.info(
                f"  geom_hold_rate={mg0['geom_hold_rate']:.3f}  "
                f"geom_step_rate={mg0['geom_step_rate']:.3f}  "
                f"d_err_mean={mg0['geom_d_err_mean']:.4f}m  "
                f"radial_max_mean={mg0['geom_radial_max_mean']:.4f}m  "
                f"axis_err_mean={mg0['geom_axis_err_mean']:.4f}  "
                f"pen_mean={mg0['geom_pen_max_mean']*1000:.2f}mm"
            )
        else:
            logger.info(f"  pos_success_rate={m0['pos_success_rate']:.3f}  "
                        f"pos_err_mean={m0['pos_err_mean']:.4f}m  "
                        f"axis_err_mean={m0['axis_err_mean']:.4f}  "
                        f"hold_success_rate={m0['hold_success_rate']:.3f}")
        logger.info(f"  eval_step_cost={c0['cost_rate']:.4f}  "
                    f"eval_ep_cost={c0['cost_episode_sum_mean']:.3f}  "
                    f"cost_limit_per_ep={args.cost_limit_per_ep:.4f}")
        logger.info("=" * 60)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": 0,
                "warmstart_pos_success_rate": m0["pos_success_rate"],
                "warmstart_pos_err_mean": m0["pos_err_mean"],
                "warmstart_axis_err_mean": m0["axis_err_mean"],
                "warmstart_hold_success_rate": m0["hold_success_rate"],
                "warmstart_eval_step_cost": c0["cost_rate"],
                "warmstart_eval_ep_cost": c0["cost_episode_sum_mean"],
                **({
                    "warmstart_geom_hold_rate": mg0["geom_hold_rate"],
                    "warmstart_geom_step_rate": mg0["geom_step_rate"],
                    "warmstart_geom_d_err_mean": mg0["geom_d_err_mean"],
                    "warmstart_geom_radial_max_mean": mg0["geom_radial_max_mean"],
                    "warmstart_geom_pen_max_mean": mg0["geom_pen_max_mean"],
                } if mg0 is not None else {}),
            }, step=0)

    warmup_vector_steps = math.ceil(INITIAL_REPLAY_SIZE / args.num_envs)
    logger.info(f"填充 replay: {INITIAL_REPLAY_SIZE} env-steps "
                f"(约 {warmup_vector_steps} vector-steps × {args.num_envs} envs)")
    core.learn(n_steps=INITIAL_REPLAY_SIZE, n_steps_per_fit=INITIAL_REPLAY_SIZE)
    # Discard episodes accumulated during the initial replay fill.
    # They were generated by the untrained policy and must not seed the first
    # real λ update. Calling drain() without processing the result throws them away.
    if _tracker is not None:
        _tracker.drain()
        _tracker.drain_max()

    fits_per_epoch = args.n_steps_per_epoch // args.n_steps_per_fit
    vector_steps_per_fit = args.n_steps_per_fit / args.num_envs
    vector_steps_per_epoch = args.n_steps_per_epoch / args.num_envs
    effective_utd = args.utd / args.n_steps_per_fit
    logger.info(f"utd={args.utd}  collect-fits/epoch={fits_per_epoch}  "
                f"total-fits/epoch={fits_per_epoch * args.utd}  "
                f"true_UTD={effective_utd:.3f}  "
                f"env-steps/epoch={args.n_steps_per_epoch}  "
                f"vector-steps/fit≈{vector_steps_per_fit:.1f}  "
                f"vector-steps/epoch≈{vector_steps_per_epoch:.1f}")

    best_J = -np.inf
    best_score = -np.inf
    best_hold_rate = -1.0
    best_hold_score = -1.0
    total_env_steps = INITIAL_REPLAY_SIZE

    def _cmdp_absorb_counts():
        logging_state = mdp.get_logging_state()
        return (
            int(logging_state["absorb_count"]),
            int(logging_state["absorb_count_physx"]),
            int(logging_state["absorb_count_sphere"]),
            int(logging_state.get("absorb_count_table", 0)),
            int(logging_state.get("collision_count", 0)),
            int(logging_state.get("collision_count_physx", 0)),
            int(logging_state.get("collision_count_sphere", 0)),
            int(logging_state.get("collision_count_table", 0)),
        )

    (
        absorb_prev, absorb_physx_prev, absorb_sphere_prev, absorb_table_prev,
        coll_prev, coll_physx_prev, coll_sphere_prev, coll_table_prev,
    ) = _cmdp_absorb_counts()

    for epoch in range(args.n_epochs):
        actor_epoch = max(0, epoch - critic_only_epochs)
        mdp.set_geom_epoch(actor_epoch)

        # Reset per-env cost accumulators so partial episodes from the previous
        # epoch's rollout boundary don't bleed into this epoch's first completed
        # episode. The deque of fully completed episodes is NOT cleared here —
        # it persists only if ready() was False last epoch (uncommon).
        if _tracker is not None:
            _tracker.reset_accum()

        core.learn(
            n_steps=args.n_steps_per_epoch,
            n_steps_per_fit=args.n_steps_per_fit,
            quiet=True,
        )
        clamp_alpha()

        # ── Rollout episode λ update (rollout_episode_rate mode) ──────────────
        # Immediately after rollout, drain the EpisodeCostTracker and update λ
        # before replay-only extra UTD fits. This lets the current epoch's safety
        # feedback influence the bulk of the actor updates instead of waiting
        # until the next epoch.
        # drain() returns mean(episode_cost_sums) over episodes that completed
        # during core.learn() this epoch — the on-policy, episode-normalized signal.
        # This is the λ-update data stream: completed episode cost → rolling window
        # → λ dual ascent. It runs from training rollout data, not eval.
        #
        # If fewer than min_lambda_update_episodes completed this epoch (e.g.,
        # the horizon is long relative to n_steps_per_epoch), we skip the update
        # and leave the buffer intact for accumulation into the next epoch.
        _rollout_ep_mean = float("nan")
        _rollout_ep_max = float("nan")
        _rollout_n_ep = 0
        if _tracker is not None:
            if _tracker.ready():
                _rollout_ep_mean, _rollout_n_ep = _tracker.drain()
                # Companion drain for "Maximum Violation per Episode".
                # Lockstep with drain(): same episodes, drained together each epoch.
                _rollout_ep_max, _ = _tracker.drain_max()
                agent.update_lambda_from_rollout_episodes(_rollout_ep_mean, _rollout_n_ep)
            else:
                # Report current buffer size without draining (accumulates to next epoch).
                _rollout_n_ep = _tracker.n_episodes
        # ──────────────────────────────────────────────────────────────────────

        for _ in range(fits_per_epoch * (args.utd - 1)):
            agent.fit(empty_dataset)
            clamp_alpha()
        total_env_steps += args.n_steps_per_epoch

        (
            absorb_now, absorb_physx_now, absorb_sphere_now, absorb_table_now,
            coll_now, coll_physx_now, coll_sphere_now, coll_table_now,
        ) = _cmdp_absorb_counts()
        absorb_epoch = absorb_now - absorb_prev
        absorb_physx_epoch = absorb_physx_now - absorb_physx_prev
        absorb_sphere_epoch = absorb_sphere_now - absorb_sphere_prev
        absorb_table_epoch = absorb_table_now - absorb_table_prev
        coll_epoch = coll_now - coll_prev
        coll_physx_epoch = coll_physx_now - coll_physx_prev
        coll_sphere_epoch = coll_sphere_now - coll_sphere_prev
        coll_table_epoch = coll_table_now - coll_table_prev

        # Disable tracker during eval: eval runs the deterministic policy, and
        # those episode costs must NOT enter the rollout buffer (different policy,
        # potentially different cost distribution, would bias λ's signal).
        if _tracker is not None:
            _tracker.active = False
        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        # Re-enable tracker before next epoch's core.learn().
        if _tracker is not None:
            _tracker.active = True

        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / args.n_eval_episodes
        m = compute_hold_metrics(dataset, mdp, args.hold_success_steps)
        mg = (
            compute_geom_metrics(dataset, mdp, args.hold_success_steps)
            if mdp._geom_stage is not None else None
        )
        c = compute_cost_metrics(dataset, args.n_eval_episodes)
        lambda_update_mode = getattr(agent, "_lambda_update_mode", args.lambda_update_mode)
        # episode_rate: update λ from eval episodes (deterministic policy, one epoch lag).
        # rollout_episode_rate: λ was already updated above from the rollout tracker;
        #   do NOT call update_lambda_from_episode_statistics() here.
        if lambda_update_mode == "episode_rate":
            agent.update_lambda_from_episode_statistics(
                cost_episode_rate=c["cost_episode_sum_mean"],
                source="eval_episode_rate",
            )

        improved_J = J > best_J
        if improved_J:
            best_J = J
            agent.save(str(best_J_path))
            agent.save(str(best_J_path_flat))
        if mg is not None:
            track_rate = mg["geom_hold_rate"]
            track_score = mg["geom_max_run_mean"]
        else:
            track_rate = m["hold_success_rate"]
            track_score = m["max_hold_mean"]

        score = track_rate * track_score
        improved_score = track_rate > 0 and score > best_score
        if improved_score:
            best_score = score

        hold_rate = track_rate
        max_hold = track_score
        improved_hold = (
            hold_rate > best_hold_rate
            or (hold_rate == best_hold_rate and max_hold > best_hold_score)
        )
        if improved_hold and hold_rate > 0:
            best_hold_rate = hold_rate
            best_hold_score = max_hold
            agent.save(str(best_hold_path))
            agent.save(str(best_hold_path_flat))

        (
            absorb_prev, absorb_physx_prev, absorb_sphere_prev, absorb_table_prev,
            coll_prev, coll_physx_prev, coll_sphere_prev, coll_table_prev,
        ) = _cmdp_absorb_counts()

        lambda_val = float(agent._log_lambda.exp().item())
        rollout_ep_cost = float(getattr(agent, "_rollout_ep_cost", float("nan")))
        rollout_ep_violation = float(getattr(agent, "_rollout_ep_violation", float("nan")))
        # eval-side violation: deterministic policy, one epoch behind rollout signal.
        eval_ep_violation = c['cost_episode_sum_mean'] - args.cost_limit_per_ep

        logger.epoch_info(
            epoch + 1, J=J, R=R, best_J=best_J,
            **{("best_geom" if mg is not None else "best_hold"):
               best_hold_rate if best_hold_rate >= 0 else 0.0},
            best_score=best_score,
            eval_step_cost=c['cost_rate'],
            lam=lambda_val,
            epoch_absorb=absorb_epoch,
        )
        if mg is not None:
            logger.info(
                f"geom schedule @ raw_epoch={epoch} actor_epoch={actor_epoch}: "
                f"d_target_eff={mdp._geom_d_target_eff:+.4f}m  "
                f"stage={mdp._geom_stage}"
            )
            logger.info(
                f"geom eval ({mdp._geom_stage} active mask): "
                f"geom_step_rate={mg['geom_step_rate']:.3f}  "
                f"geom_hold_rate={mg['geom_hold_rate']:.3f} "
                f"(>= {args.hold_success_steps} consec steps)  "
                f"geom_max_run_mean={mg['geom_max_run_mean']:.1f}  "
                f"final_success_rate={mg['geom_final_success_rate']:.3f}  "
                f"d_err_mean={mg['geom_d_err_mean']:.4f}m  "
                f"d_err_min={mg['geom_d_err_min']:.4f}m  "
                f"radial_max_min={mg['geom_radial_max_min']:.4f}m  "
                f"axis_err_min={mg['geom_axis_err_min']:.3f}"
            )
            logger.info(
                f"geom masks: prepos_step_rate={mg['geom_prepos_step_rate']:.3f}  "
                f"preaxis_step_rate={mg['geom_preaxis_step_rate']:.3f}  "
                f"insert_step_rate={mg['geom_insert_step_rate']:.3f}"
            )
            logger.info(
                f"geom entry (n_ep={mg['geom_n_ep_with_entry']}): "
                f"d={mg['geom_entry_d_mean']:+.4f}m  "
                f"d_err={mg['geom_entry_d_err_mean']:.4f}m  "
                f"rm_mean={mg['geom_entry_radial_max_mean']:.4f}m "
                f"(max {mg['geom_entry_radial_max_max']:.4f}m)  "
                f"ax_mean={mg['geom_entry_axis_err_mean']:.4f} "
                f"(max {mg['geom_entry_axis_err_max']:.4f})  "
                f"pen_mean={mg['geom_entry_penetration_mean']*1000:.2f}mm "
                f"(max {mg['geom_entry_penetration_max']*1000:.2f}mm)"
            )
            logger.info(
                f"geom penetration: "
                f"max_mean={mg['geom_pen_max_mean']*1000:.2f}mm "
                f"max_max={mg['geom_pen_max_max']*1000:.2f}mm "
                f"clean_step_rate={mg['geom_clean_step_rate']:.3f} "
                f"(in active mask: mean={mg['geom_pen_in_active_mean']*1000:.2f}mm "
                f"max={mg['geom_pen_in_active_max']*1000:.2f}mm)"
            )
            logger.info("eval stats: [legacy 球形 pos/axis 指标 skipped — geom 模式 ckpt 选择走 geom_*]")
        else:
            logger.info("eval stats: "
                        f"hold_success_rate={m['hold_success_rate']:.3f}  "
                        f"max_hold_mean={m['max_hold_mean']:.1f}  "
                        f"eval_ep_len={ep_len:.1f}  "
                        f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
                        f"pos_success_rate={m['pos_success_rate']:.3f}  "
                        f"pos_err_mean={m['pos_err_mean']:.4f}m  "
                        f"axis_err_mean={m['axis_err_mean']:.4f}")
            if m["pos_in_thresh_count"] > 0:
                logger.info("  ↳ pos_in_thresh diagnostics: "
                            f"count={m['pos_in_thresh_count']}  "
                            f"axis_err_mean={m['axis_err_in_pos_thresh_mean']:.4f}  "
                            f"axis_err_min={m['axis_err_in_pos_thresh_min']:.4f}")
            else:
                logger.info("  ↳ pos_in_thresh diagnostics: count=0  axis_err=n/a")
        logger.info(f"  ↳ eval_step_cost={c['cost_rate']:.4f}  "
                    f"eval_ep_cost={c['cost_episode_sum_mean']:.3f}  "
                    f"eval_ep_max_violation={c['cost_episode_max_mean']:.4f}  "
                    f"eval_ep_violation={eval_ep_violation:+.4f}  "
                    f"rollout_ep_cost={rollout_ep_cost:.3f}  "
                    f"rollout_ep_max_violation={_rollout_ep_max:.4f}  "
                    f"rollout_ep_violation={rollout_ep_violation:+.4f}  "
                    f"rollout_ep_n={_rollout_n_ep}  "
                    f"λ={lambda_val:.3f}  "
                    f"epoch_absorb_total={absorb_epoch}  "
                    f"epoch_absorb_sphere={absorb_sphere_epoch}  "
                    f"epoch_absorb_physx={absorb_physx_epoch}  "
                    f"epoch_absorb_table={absorb_table_epoch}  "
                    f"epoch_collision_total={coll_epoch}  "
                    f"epoch_collision_sphere={coll_sphere_epoch}  "
                    f"epoch_collision_physx={coll_physx_epoch}  "
                    f"epoch_collision_table={coll_table_epoch}")

        if wandb_run is not None:
            _legacy = (lambda k: f"legacy_{k}") if mg is not None else (lambda k: k)
            best_rate_value = best_hold_rate if best_hold_rate >= 0 else 0.0
            best_run_value = best_hold_score if best_hold_score >= 0 else 0.0
            best_metric_fields = (
                {
                    "best_metric_rate": best_rate_value,
                    "best_metric_max_run_mean": best_run_value,
                    "best_geom_rate": best_rate_value,
                    "best_geom_max_run_mean": best_run_value,
                }
                if mg is not None else
                {
                    "best_metric_rate": best_rate_value,
                    "best_metric_max_run_mean": best_run_value,
                    "best_hold_rate": best_rate_value,
                    "best_hold_max_hold_mean": best_run_value,
                }
            )
            wandb_run.log({
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "best_score": best_score,
                **best_metric_fields,
                "eval_ep_len": ep_len,
                _legacy("eval_success_rate"): m["hold_success_rate"],
                _legacy("eval_max_hold_mean"): m["max_hold_mean"],
                _legacy("eval_in_thresh_rate"): m["in_thresh_rate"],
                _legacy("eval_final_in_thresh_rate"): m["final_in_thresh_rate"],
                _legacy("eval_pos_success_rate"): m["pos_success_rate"],
                _legacy("eval_pos_err_mean"): m["pos_err_mean"],
                _legacy("eval_axis_err_mean"): m["axis_err_mean"],
                _legacy("eval_pos_in_thresh_count"): m["pos_in_thresh_count"],
                _legacy("eval_axis_err_in_pos_thresh_mean"): m["axis_err_in_pos_thresh_mean"],
                _legacy("eval_axis_err_in_pos_thresh_min"): m["axis_err_in_pos_thresh_min"],
                **({
                    "geom_raw_epoch": epoch,
                    "geom_actor_epoch": actor_epoch,
                    "geom_d_target_eff": mdp._geom_d_target_eff,
                    "geom_step_rate": mg["geom_step_rate"],
                    "geom_hold_rate": mg["geom_hold_rate"],
                    "geom_max_run_mean": mg["geom_max_run_mean"],
                    "geom_ep_success_rate_mean": mg["geom_ep_success_rate_mean"],
                    "geom_final_success_rate": mg["geom_final_success_rate"],
                    "geom_prepos_step_rate": mg["geom_prepos_step_rate"],
                    "geom_preaxis_step_rate": mg["geom_preaxis_step_rate"],
                    "geom_insert_step_rate": mg["geom_insert_step_rate"],
                    "geom_d_target_mean": mg["geom_d_target_mean"],
                    "geom_d_err_mean": mg["geom_d_err_mean"],
                    "geom_d_err_min": mg["geom_d_err_min"],
                    "geom_radial_tip_mean": mg["geom_radial_tip_mean"],
                    "geom_radial_tip_min": mg["geom_radial_tip_min"],
                    "geom_radial_max_mean": mg["geom_radial_max_mean"],
                    "geom_radial_max_min": mg["geom_radial_max_min"],
                    "geom_axis_err_mean": mg["geom_axis_err_mean"],
                    "geom_axis_err_min": mg["geom_axis_err_min"],
                    "geom_n_ep_with_entry": mg["geom_n_ep_with_entry"],
                    "geom_entry_d_mean": mg["geom_entry_d_mean"],
                    "geom_entry_d_err_mean": mg["geom_entry_d_err_mean"],
                    "geom_entry_radial_max_mean": mg["geom_entry_radial_max_mean"],
                    "geom_entry_radial_max_max": mg["geom_entry_radial_max_max"],
                    "geom_entry_axis_err_mean": mg["geom_entry_axis_err_mean"],
                    "geom_entry_axis_err_max": mg["geom_entry_axis_err_max"],
                    "geom_entry_penetration_mean": mg["geom_entry_penetration_mean"],
                    "geom_entry_penetration_max": mg["geom_entry_penetration_max"],
                    "geom_final_d_mean": mg["geom_final_d_mean"],
                    "geom_final_d_err_mean": mg["geom_final_d_err_mean"],
                    "geom_final_radial_max_mean": mg["geom_final_radial_max_mean"],
                    "geom_final_radial_max_max": mg["geom_final_radial_max_max"],
                    "geom_final_axis_err_mean": mg["geom_final_axis_err_mean"],
                    "geom_final_axis_err_max": mg["geom_final_axis_err_max"],
                    "geom_final_penetration_mean": mg["geom_final_penetration_mean"],
                    "geom_final_penetration_max": mg["geom_final_penetration_max"],
                    "geom_pen_max_mean": mg["geom_pen_max_mean"],
                    "geom_pen_max_max": mg["geom_pen_max_max"],
                    "geom_clean_step_rate": mg["geom_clean_step_rate"],
                    "geom_pen_in_active_mean": mg["geom_pen_in_active_mean"],
                    "geom_pen_in_active_max": mg["geom_pen_in_active_max"],
                } if mg is not None else {}),
                "alpha": agent._alpha.item(),
                # Lagrangian safety
                "lambda": lambda_val,
                "log_lambda": float(agent._log_lambda.item()),
                "lambda_update_source": getattr(
                    agent, "_lambda_update_source", "unknown"),
                # rollout signal (on-policy, stochastic): drives λ update
                "rollout_ep_cost": rollout_ep_cost,
                "rollout_ep_violation": rollout_ep_violation,
                "rollout_ep_n": _rollout_n_ep,
                # Maximum Violation per Episode: mean over episodes of max_t cost_t,
                # from training rollout (and eval, one epoch lag).
                "rollout_ep_max_violation": _rollout_ep_max,
                # eval signal (deterministic policy, one epoch lag)
                "eval_step_cost": c["cost_rate"],
                "eval_ep_cost": c["cost_episode_sum_mean"],
                "eval_ep_max_violation": c["cost_episode_max_mean"],
                "eval_ep_violation": eval_ep_violation,
                "cost_limit_per_ep": args.cost_limit_per_ep,
                # absorb 计数 (epoch 级别) — collisions that actually terminated
                "epoch_absorb": absorb_epoch,
                "epoch_absorb_total": absorb_epoch,
                "epoch_absorb_physx": absorb_physx_epoch,
                "epoch_absorb_sphere": absorb_sphere_epoch,
                "epoch_absorb_table": absorb_table_epoch,
                # collision event 计数:
                # total/physx = real PhysX contact events;
                # sphere = cost-proxy violations; table/physx = PhysX contact.
                "epoch_collision": coll_epoch,
                "epoch_collision_total": coll_epoch,
                "epoch_collision_physx": coll_physx_epoch,
                "epoch_collision_sphere": coll_sphere_epoch,
                "epoch_collision_table": coll_table_epoch,
            }, step=epoch + 1)

    agent.save(str(final_path))
    agent.save(str(final_path_flat))

    if best_hold_rate < 0:
        best_hold_display = "n/a"
    else:
        best_hold_display = f"{best_hold_rate:.3f} (max_hold_mean={best_hold_score:.1f})"
    best_metric_name = "best_geom_rate" if mdp._geom_stage is not None else "best_hold_rate"
    logger.info(
        f"训练完成. best J = {best_J:.3f}  "
        f"{best_metric_name} = {best_hold_display}  "
        f"final λ = {float(agent._log_lambda.exp().item()):.3f}"
    )
    logger.info(f"checkpoint 写入: {ckpt_dir}/ 下的 "
                f"{best_J_path.name} / {best_hold_path.name} / {final_path.name}. "
                "**eval 时三个都跑一遍**, 注意 best_J 不一定满足 cost_limit_per_ep, "
                "需手动看 wandb cost_rate 选 safe 子集.")

    if wandb_run is not None:
        # Duplicate the grouping field into summary as a W&B UI fallback.
        # Some Reports/Workspace dropdowns discover summary keys faster than
        # config keys added to old runs through the Public API.
        wandb_run.summary["plot_group"] = args.wandb_group or "ungrouped"
        best_rate_value = best_hold_rate if best_hold_rate >= 0 else 0.0
        best_run_value = best_hold_score if best_hold_score >= 0 else 0.0
        wandb_run.summary["best_J"] = best_J
        wandb_run.summary["best_score"] = best_score
        wandb_run.summary["best_metric_rate"] = best_rate_value
        wandb_run.summary["best_metric_max_run_mean"] = best_run_value
        if mdp._geom_stage is not None:
            wandb_run.summary["best_geom_rate"] = best_rate_value
            wandb_run.summary["best_geom_max_run_mean"] = best_run_value
        else:
            wandb_run.summary["best_hold_rate"] = best_rate_value
            wandb_run.summary["best_hold_max_hold_mean"] = best_run_value
        wandb_run.summary["final_lambda"] = float(agent._log_lambda.exp().item())
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
