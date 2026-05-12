"""SAC 训练 — 双臂 peg-in-hole preinsert (stage flag 化, mushroom-rl + VectorCore).

obs 默认 32 维 base (joint_pos+joint_vel+pos_vec+axis_dot); 推荐用
--use_axis_resid_obs 切到 34 维 (axis_resid 替换 axis_dot, 当前正式 Stage 1/2
都是 34 维). 同一个 env / 同一条 reward 骨架, stage 用 reward 权重 +
success_axis_threshold 切换:

    Stage 1 = pos-only         --rew_axis 0.0  --success_axis_threshold inf
    Stage 2 = pos + axis 对齐  --rew_axis 0.5  --success_axis_threshold 0.50
                              --axis_gate_radius 0.40 --rew_pos_success 1.0

Stage 2 用 --load_agent path/to/Stage1_checkpoint.msh + --actor_only_warmstart
续训: 只继承 Stage 1 actor 权重, critic/alpha/replay 冷启动. 旧 critic 按旧
reward 学, 用到 Stage 2 reward 上 Q 语义已错, 会把 actor 拽离 Stage 1 manifold.
另外 --critic_warmup_transitions 50000 让冷 critic 先单独学 ~40 epoch.

正式训练命令见 README.md "Stage 1 训练命令" / "Stage 2 训练命令" 段.

注意: num_envs=1 触发 IsaacSim cloner 的 `*` pattern 失败 → 至少 2.
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
    compute_hold_metrics,
    deterministic_policy,
    parse_home_weights,
    resolve_eval_episode_count,
    warmstart_actor_with_partial_copy,
)


INITIAL_REPLAY_SIZE = 10_000
MAX_REPLAY_SIZE = 500_000
BATCH_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=None,
                   help="episode horizon (env-steps). 默认走 env 默认值 (Stage 1/2 = 150). "
                        "Stage 3 推荐 200 (reach + insert + dwell ≥10).")
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
                   help="cold-start 联合任务推荐降到 1e-4 (避免 noisy critic 拉坏 actor),"
                        "warm-start 用 default 即可.")
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--alpha_max", type=float, default=0.1,
                   help="alpha 上限. cold-start 联合任务推荐 0.2 (探索), warm-start 0.1 稳态.")
    p.add_argument("--target_entropy", type=float, default=None,
                   help="目标 entropy. 默认自动取 -act_dim (SAC 标准). "
                        "14-DoF cold-start 联合任务可考虑 -7 (= -act_dim/2), "
                        "让 SAC 倾向稍集中, alpha 不顶 cap.")
    p.add_argument("--critic_warmup_transitions", type=int, default=None,
                   help="actor 开始更新前需要的 replay 容量 (env-steps). 默认 = "
                        "INITIAL_REPLAY_SIZE (10K), 即 critic 仅领先 actor 1 步, "
                        "等价无 critic-only warmup. **actor-only warmstart 推荐设大** "
                        "(e.g. 50000 ≈ 40 epoch), 让冷启动的 critic 先用 warm-start "
                        "actor 收集的 trajectories 单独把 Q 学到合理量级, 再放开 actor, "
                        "避免随机 critic 把转移过来的 actor 推离 Stage 1 manifold. "
                        "必须 >= INITIAL_REPLAY_SIZE.")
    p.add_argument("--n_eval_episodes", type=int, default=None,
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 的 reset 关节噪声")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None,
                   help="覆盖 env 的 preinsert 位置成功阈值 (env 默认 0.10m, 即当前 "
                        "Stage 1/Stage 2 curriculum). 如果显式想跑老 5cm, 传 0.05.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert offset (默认 0.05m)")
    p.add_argument("--rew_action", type=float, default=None,
                   help="覆盖 env 的动作 L2 惩罚权重")
    p.add_argument("--rew_home", type=float, default=None,
                   help="home regularizer 权重 (joint-range 归一化的 ||q - q_home||²). "
                        "默认 0 关闭. 当前 Stage 1/Stage 2 建议 0.0005 当极弱 tie-breaker.")
    p.add_argument("--home_weights", type=parse_home_weights, default=None,
                   help="home regularizer 的逐关节权重. 接受 7 维单臂权重(自动复制到左右臂)"
                        "或 14 维完整权重, 逗号/空格分隔. 默认全 1. 例如: "
                        "'1,1,1,1,0.5,0.25,0.25'.")
    p.add_argument("--rew_success", type=float, default=None,
                   help="覆盖 env 的 per-step full_success (pos∧axis) bonus (默认 2.0)")
    p.add_argument("--rew_pos_success", type=float, default=None,
                   help="pos-only success bonus (env 默认 0.0). Stage 2 当前正式值 1.0: "
                        "维持 Stage 1 已学的'进 pos 阈值给 bonus'信号, 避免 Stage 2 把 axis 加上后"
                        "Stage 1 成功状态突然失去 bonus 造成 reward 断崖.")
    p.add_argument("--axis_gate_radius", type=float, default=None,
                   help="axis 惩罚的距离门控半径 (m). env 默认 inf = 不门控. "
                        "Stage 2 推荐 0.40: pos_err >= 0.40m 时 axis 项=0, 在 "
                        "[pos_th, 0.40m] 区间线性 ramp, 进 pos_th 后 gate 满.")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重 (默认 0.0 = Stage 1 pos-only). "
                        "Stage 2 当前正式值 0.5 启用轴对齐惩罚.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="覆盖 env 的 axis_err success 阈值 (默认 inf = Stage 1 不检查 axis). "
                        "Stage 2 训练阈值 0.50, 严格 eval 用 0.20/0.30. 接受 'inf' 字符串.")
    # ──────────────── Stage 2 actor-relative curriculum ────────────────
    p.add_argument("--stage2_curriculum", action="store_true",
                   help="启用 Stage 2 2A→2B schedule: critic warmup 期间不走 schedule; "
                        "actor-relative epoch < start 时 axis success 仍为 inf, "
                        "w_axis 用 start; start→end 期间 w_axis 线性 ramp 到目标值; "
                        "epoch>=start 后 success_axis_threshold 切到目标值.")
    p.add_argument("--stage2_axis_ramp_start", type=int, default=30,
                   help="Stage 2 curriculum 的 actor-relative ramp 起点. 默认 30.")
    p.add_argument("--stage2_axis_ramp_end", type=int, default=60,
                   help="Stage 2 curriculum 的 actor-relative ramp 终点. 默认 60.")
    p.add_argument("--stage2_axis_weight_start", type=float, default=0.25,
                   help="Stage 2A 的弱 axis shaping 权重. 默认 0.25.")
    p.add_argument("--stage2_axis_weight_end", type=float, default=None,
                   help="Stage 2B/C 的最终 axis 权重. 默认使用 --rew_axis; 若未传 --rew_axis 则用 1.0.")
    p.add_argument("--stage2_axis_success_threshold", type=float, default=None,
                   help="Stage 2B/C 的最终 axis success 阈值. 默认使用 --success_axis_threshold; "
                        "若未传则用 0.40. Stage 2A 固定使用 inf.")
    p.add_argument("--load_agent", type=str, default=None,
                   help="warm-start 路径: 从该 checkpoint 加载 agent (actor/critic/"
                        "optimizer state). obs 维度必须与 checkpoint 匹配 — 32 维 base "
                        "和 34 维 axis_resid 之间不能 warm-start.")
    p.add_argument("--keep_replay", action="store_true",
                   help="warm-start 时保留旧 replay buffer. 默认会清空 — 因为 stage "
                        "切换 (Stage 1→Stage 2) reward 函数变了, 旧 transitions 的 "
                        "reward 标签按旧 reward 算, 留着会拖 critic.")
    p.add_argument("--actor_only_warmstart", action="store_true",
                   help="warm-start 时只继承 actor (mu/sigma 网络) 权重, critic / "
                        "alpha / optimizers / replay buffer 全部冷启动. stage 切换"
                        "(Stage 1→Stage 2 reward shape 变化) 时强烈建议打开 — 否则旧 critic "
                        "按旧 reward 学的 Q 语义会拖坏 actor (Stage 2 一上来 actor 就被拽离 "
                        "Stage 1 learned manifold). 此 flag 打开时 --keep_replay 自动失效.")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "0 = 关闭 (baseline). >0 启用 absorbing termination.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="eval success 定义 + env 终止阈值: 连续 N 步都在阈值内. "
                        "N=10 ≈ 1s hold (per-step dt≈0.1s).")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="覆盖 env 的 sphere-proxy 自碰撞兜底阈值 (m). 默认 0.0 = 球壳一接触即"
                        "触发 hard absorbing. 关闭时写 --clearance_hard=-inf, 只信 PhysX 力检测.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="覆盖 env 的 arm sphere proxy 半径 (默认 0.06m).")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="覆盖 env 的 EE sphere proxy 半径 (默认 0.04m).")
    p.add_argument("--exclude_ee_from_physx_self_collision", action="store_true",
                   help="Stage 3 peg/hole 真实 collider 用: PhysX arm_L vs arm_R "
                        "self-collision 分组排除左右 EE link, 避免正常 peg-hole "
                        "接触被 hard absorbing 误杀. EE 区域仍由 sphere-proxy 兜底.")
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
                   help="启用几何同源 preinsert reward + 41D obs. "
                        "prepos=Stage1g(d+tip radial), preaxis=Stage2g(+radial_max+axis), "
                        "insert=Stage3g(moving d_target + radial_max + axis).")
    p.add_argument("--allow_partial_geom_warmstart", action="store_true",
                   help="默认禁止旧 34D ckpt partial-copy 到 geom 41D, 防把旧球形 reward manifold 带入新路径. "
                        "确实要做 ablation 才打开.")
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
    p.add_argument("--geom_d_th", type=float, default=None)
    p.add_argument("--geom_r_tip_th", type=float, default=None)
    p.add_argument("--geom_r_max_th", type=float, default=None)
    p.add_argument("--geom_axis_th", type=float, default=None)
    p.add_argument("--geom_insert_d_ins", type=float, default=None)
    p.add_argument("--geom_insert_r_max_th", type=float, default=None)
    p.add_argument("--geom_pen_th", type=float, default=None,
                   help="insert success mask penetration threshold (m). "
                        "默认 env=0.001; strict eval 可用 0.0005.")
    p.add_argument("--geom_soft_penetration_sigma", type=float, default=None,
                   help="把 penetration 加进 soft_success Gaussian well 的 σ (m). "
                        "默认 None=不加. 设 0.001 让 dwell well 只在干净状态给奖. "
                        "配 --rew_geom_soft_success>0 启用 dwell anchor.")
    # Alignment-gated progress reward (codex 2026-05-11):
    # r_progress = w · clamp(d - d_target_neg, 0, range) · exp(-(rm/σ_r)² - (axis/σ_a)²)
    # 默认 0 (关闭, 跟旧 additive 行为兼容); insert 推荐 8-10.
    p.add_argument("--rew_geom_progress", type=float, default=None,
                   help="alignment-gated progress reward weight (0 = off, default). "
                        "推荐 insert 配方: 8-10")
    p.add_argument("--geom_gate_radial_sigma", type=float, default=None,
                   help="alignment gate σ_r for radial_max (default 0.025)")
    p.add_argument("--geom_gate_axis_sigma", type=float, default=None,
                   help="alignment gate σ_a for axis_err (default 0.30)")
    # Penetration-aware reward (SAC vs Lagrangian SAC 对比所需)
    p.add_argument("--rew_geom_penetration", type=float, default=None,
                   help="penetration soft penalty -w·penetration_max (默认 0=off). "
                        "推荐 SAC: 10-20; Lagrangian SAC: 0 (用 cost 信号代替)")
    p.add_argument("--geom_gate_penetration_sigma", type=float, default=None,
                   help="alignment_gate 包含 penetration 项的 σ (m), 例如 0.002=2mm. "
                        "默认 None=不进 gate. 设了让 gated progress 在穿模时趋 0.")
    p.add_argument("--cost_signal", type=str, default=None,
                   choices=["collision", "penetration"],
                   help="info['cost'] 信号: 'collision' (0/1 老语义) 或 'penetration' "
                        "(连续 [0,4mm], 给 Lagrangian SAC 当 constraint cost)")
    p.add_argument("--geom_progress_floor", type=float, default=None,
                   help="progress reward 起点 (m). 默认 0.0 = 只奖励 d>0 真插入. "
                        "旧行为 (奖励'接近 entrance') 用 d_target_neg=-0.08")
    p.add_argument("--rew_geom_advance", type=float, default=None,
                   help="delta-progress (PBRS) reward weight (默认 0=off). "
                        "r_advance = w · (phi_t - phi_{t-1}) where phi = clean_gate × "
                        "clamp((d - d_neg)/(d_pos - d_neg), 0, 1). 推荐 25.0. "
                        "用法: 通常 rew_geom_progress=0 + rew_geom_advance=25 把 state-progress "
                        "切换成 delta-progress, 治 hover-at-position 漏洞.")
    # Task-ordering 修复 (codex 2026-05-11 v6)
    p.add_argument("--geom_d_gate_mode", type=str, default=None,
                   choices=["off", "alignment"],
                   help="r_d 是否乘 alignment_gate. 'off' (默认) = r_d 跟 alignment 解耦; "
                        "'alignment' = r_d 只在 aligned state 给奖励, 强制 task ordering.")
    p.add_argument("--rew_geom_bad_entry", type=float, default=None,
                   help="bad entry penalty weight (默认 0=off). "
                        "公式 (normalized v6.1): -w · depth_norm · Σ(relu(metric/safe - 1)). "
                        "depth_norm = d/d_target_pos clamp [0,2]. "
                        "1cm 超 1cm 阈值 + d 在 target → -1.0/step (w=1). 推荐 0.5-1.5.")
    p.add_argument("--geom_bad_entry_radial_safe", type=float, default=None,
                   help="bad entry radial 安全阈值 (默认 0.010m=10mm)")
    p.add_argument("--geom_bad_entry_axis_safe", type=float, default=None,
                   help="bad entry axis 安全阈值 (默认 0.10)")
    p.add_argument("--geom_bad_entry_pen_safe", type=float, default=None,
                   help="bad entry penetration 安全阈值 (默认 0.0005m=0.5mm)")
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
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
            f"n_steps_per_fit ({args.n_steps_per_fit}) 必须能被 num_envs ({args.num_envs}) 整除"
        )
    if args.n_steps_per_epoch % args.n_steps_per_fit != 0:
        raise ValueError(
            f"n_steps_per_epoch ({args.n_steps_per_epoch}) 必须能被 "
            f"n_steps_per_fit ({args.n_steps_per_fit}) 整除"
        )
    if args.critic_warmup_transitions is None:
        args.critic_warmup_transitions = INITIAL_REPLAY_SIZE
    if args.critic_warmup_transitions < INITIAL_REPLAY_SIZE:
        raise ValueError(
            f"--critic_warmup_transitions ({args.critic_warmup_transitions}) 必须 >= "
            f"INITIAL_REPLAY_SIZE ({INITIAL_REPLAY_SIZE}); 否则 replay 还没 initialized "
            "actor gate 已经打开, 设置无意义."
        )
    args.n_eval_episodes = resolve_eval_episode_count(
        args.n_eval_episodes, args.num_envs, "--n_eval_episodes"
    )

    from envs import DualArmPegHoleEnv
    env_kwargs = dict(num_envs=args.num_envs, headless=not args.render)
    if args.horizon is not None:
        env_kwargs["horizon"] = args.horizon
    for key in ("initial_joint_noise", "preinsert_success_pos_threshold",
                "preinsert_offset", "rew_action", "rew_success", "rew_pos_success",
                "rew_axis", "rew_home", "home_weights", "axis_gate_radius",
                "success_axis_threshold", "terminal_hold_bonus",
                "clearance_hard", "proxy_arm_radius", "proxy_ee_radius",
                # Geometric preinsert kwargs:
                "geom_stage", "geom_d_target_neg", "geom_d_target_pos",
                "geom_d_target_ramp_start", "geom_d_target_ramp_end",
                "rew_geom_d", "rew_geom_radial_tip", "rew_geom_radial_max",
                "rew_geom_axis", "geom_d_sat", "geom_radial_sat",
                "rew_geom_soft_success", "geom_soft_d_sigma",
                "geom_soft_radial_sigma", "geom_soft_axis_sigma",
                "geom_d_th", "geom_r_tip_th", "geom_r_max_th", "geom_axis_th",
                "geom_insert_d_ins", "geom_insert_r_max_th", "geom_pen_th",
                "geom_soft_penetration_sigma",
                "rew_geom_progress", "geom_gate_radial_sigma", "geom_gate_axis_sigma",
                "rew_geom_penetration", "geom_gate_penetration_sigma", "cost_signal",
                "geom_progress_floor", "rew_geom_advance",
                "geom_d_gate_mode", "rew_geom_bad_entry",
                "geom_bad_entry_radial_safe", "geom_bad_entry_axis_safe",
                "geom_bad_entry_pen_safe"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    # Geom reward 的轴向目标 d_target_neg=-preinsert_offset. 41D obs 里仍保留
    # pos_vec=(peg_tip-preinsert_target), 所以 geom 模式下默认把旧 pos_vec 的
    # target 同步到 geom_d_target_neg, 避免 reward 目标 -8cm 但 obs 目标 -5cm
    # 的 silent mismatch. 用户显式传 --preinsert_offset 时尊重用户设置.
    if args.geom_stage is not None and args.preinsert_offset is None:
        d_target_neg = (
            args.geom_d_target_neg
            if args.geom_d_target_neg is not None
            else -0.08
        )
        env_kwargs["preinsert_offset"] = abs(float(d_target_neg))
    # bool flags: 直接读 args, 不能用 None 哨兵 (action="store_true" 默认 False)
    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    if args.exclude_ee_from_physx_self_collision:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    mdp = DualArmPegHoleEnv(**env_kwargs)
    mdp.seed(args.seed)

    # IsaacSim 启动后才能导入 mushroom_rl (避免 carb 冲突)
    from mushroom_rl.algorithms.actor_critic import SAC
    from mushroom_rl.core import Agent, VectorCore, Logger, Dataset

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]
    target_entropy = args.target_entropy
    if target_entropy is None:
        target_entropy = -float(act_dim)

    def _cold_create_sac():
        actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                            output_shape=(act_dim,))
        actor_optimizer = {"class": optim.Adam, "params": {"lr": args.lr_actor}}
        critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                             output_shape=(1,), action_dim=act_dim,
                             optimizer={"class": optim.Adam, "params": {"lr": args.lr_critic}},
                             loss=F.mse_loss)
        return SAC(
            mdp_info=mdp.info,
            actor_mu_params=actor_params,
            actor_sigma_params=actor_params,
            actor_optimizer=actor_optimizer,
            critic_params=critic_params,
            batch_size=BATCH_SIZE,
            initial_replay_size=INITIAL_REPLAY_SIZE,
            max_replay_size=MAX_REPLAY_SIZE,
            warmup_transitions=args.critic_warmup_transitions,
            tau=0.005,
            lr_alpha=args.lr_alpha,
            use_log_alpha_loss=True,
            target_entropy=target_entropy,
        )

    if args.load_agent is not None:
        load_path = Path(args.load_agent)
        if not load_path.is_file():
            raise FileNotFoundError(f"--load_agent 路径不存在: {load_path}")

        if args.actor_only_warmstart:
            # Cold-create 一个新 SAC (匹配当前 reward / env), 然后用共享 helper
            # 把旧 actor (mu / sigma) 拷过来. critic / alpha / optimizers / replay
            # 全部新, 避免旧 critic 用过时 Q 语义拽 actor.
            # 单一来源 helper, 避免 train/eval/viz 间漂移.
            agent = _cold_create_sac()
            if mdp._geom_stage is not None and not args.allow_partial_geom_warmstart:
                old_agent_probe = Agent.load(str(load_path))
                old_h1_in = (
                    old_agent_probe.policy._mu_approximator.model.network
                    ._h1.weight.shape[1]
                )
                new_h1_in = (
                    agent.policy._mu_approximator.model.network
                    ._h1.weight.shape[1]
                )
                del old_agent_probe
                if old_h1_in != new_h1_in:
                    raise ValueError(
                        "geom_stage 默认要求完整 actor warm-start (obs 维度一致). "
                        f"当前 ckpt obs={old_h1_in}D, env obs={new_h1_in}D. "
                        "请先用 geom_stage=prepos cold-start 生成 41D ckpt, 再 warm-start; "
                        "若只是做 ablation, 显式加 --allow_partial_geom_warmstart."
                    )
            mode = warmstart_actor_with_partial_copy(agent, load_path)
            print(f"[WARM-START actor-only] {mode} from {load_path}; "
                  "critic / alpha / replay 全部冷启动.")
            if args.keep_replay:
                print("[WARM-START actor-only] --keep_replay 已忽略 (replay 强制冷启).")
        else:
            old_agent = Agent.load(str(load_path))
            # 全量 warm-start: 继承 actor + critic + optimizer + alpha (+ optionally replay).
            # obs 维度必须与 checkpoint 完全一致, 否则 forward 抛 shape 错.
            agent = old_agent
            print(f"[WARM-START full] 整体加载 agent from {load_path}")
            if not args.keep_replay:
                agent._replay_memory.reset()
                print("[WARM-START full] replay buffer 已清空 — 重新走 INITIAL_REPLAY_SIZE 填充. "
                      "若要保留旧 buffer, 加 --keep_replay.")
            else:
                print("[WARM-START full] 保留旧 replay buffer (--keep_replay).")
    else:
        agent = _cold_create_sac()
    def clamp_alpha(_dataset=None):
        with torch.no_grad():
            agent._log_alpha.clamp_(max=math.log(args.alpha_max))

    core = VectorCore(agent, mdp, callbacks_fit=[clamp_alpha])

    from datetime import datetime
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = results_dir / "checkpoints" / run_ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # 三种 checkpoint, 各自独立追踪, 互不覆盖:
    #   best_J.msh        - 最高 discounted return J (=best_agent.msh 的别名, 向后兼容)
    #   best_hold.msh     - stage 相关指标的最佳快照, 语义随 stage 变:
    #                       Stage 1/2: 最高 hold_success_rate (pos<pos_th ∧ axis<axis_th
    #                                  连续 hold_n_steps 步), tie-break max_hold_mean
    #                       geom:      最高 geom_hold_rate (active geom success mask
    #                                  连续 hold_n_steps 步), tie-break geom_max_run_mean
    #   final_agent.msh   - 训练结束时无条件保存最后一个 epoch 的 actor
    # 防止 "终态稳定 hold=1.0 但 J 没破 best_J 上限" 这种情况下 best_agent 锁在
    # 早期低 hold 的快照, 独立 eval 评的不是真实最终策略.
    # ckpt_dir: 按时间归档, 各次运行互不覆盖.
    # results_dir: 直接保存一份, 方便 eval 脚本直接用.
    best_J_path = ckpt_dir / "best_agent.msh"
    best_hold_path = ckpt_dir / "best_hold.msh"
    final_path = ckpt_dir / "final_agent.msh"
    best_J_path_flat = results_dir / "best_agent.msh"
    best_hold_path_flat = results_dir / "best_hold.msh"
    final_path_flat = results_dir / "final_agent.msh"
    logger = Logger("SAC", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"checkpoint 目录: {ckpt_dir}")
    if mdp._geom_stage is not None:
        obs_mode = f"geom_{mdp._geom_stage} (axis_resid+hole_geom)"
    elif mdp._use_axis_resid_obs:
        obs_mode = "axis_resid"
    else:
        obs_mode = "base"

    stage2_curriculum_active = bool(
        args.stage2_curriculum and mdp._geom_stage is None
    )
    if args.stage2_curriculum and mdp._geom_stage is not None:
        logger.info("[WARN] --stage2_curriculum ignored because geom_stage is enabled.")
    if args.stage2_curriculum and args.stage2_axis_ramp_end <= args.stage2_axis_ramp_start:
        raise ValueError(
            f"--stage2_axis_ramp_end ({args.stage2_axis_ramp_end}) must be > "
            f"--stage2_axis_ramp_start ({args.stage2_axis_ramp_start})"
        )
    stage2_axis_weight_end = (
        float(args.stage2_axis_weight_end)
        if args.stage2_axis_weight_end is not None
        else (float(args.rew_axis) if args.rew_axis is not None else 1.0)
    )
    stage2_axis_success_threshold = (
        float(args.stage2_axis_success_threshold)
        if args.stage2_axis_success_threshold is not None
        else (
            float(args.success_axis_threshold)
            if args.success_axis_threshold is not None
            else 0.40
        )
    )

    def apply_stage2_curriculum(actor_epoch):
        """Apply Stage 2 2A→2B schedule in-place on the env.

        actor_epoch is actor-relative: critic-only warmup is excluded.
        Stage 2A keeps success axis threshold at inf so the Stage 1 pos anchor
        is not invalidated while weak local axis shaping starts working.
        """
        if not stage2_curriculum_active:
            return None
        s, e = args.stage2_axis_ramp_start, args.stage2_axis_ramp_end
        if actor_epoch < s:
            t = 0.0
            axis_th = float("inf")
        elif actor_epoch < e:
            t = (actor_epoch - s) / max(e - s, 1)
            axis_th = stage2_axis_success_threshold
        else:
            t = 1.0
            axis_th = stage2_axis_success_threshold
        w_axis = (
            args.stage2_axis_weight_start
            + t * (stage2_axis_weight_end - args.stage2_axis_weight_start)
        )
        mdp._w_axis = float(w_axis)
        mdp._success_axis_threshold = float(axis_th)
        return t, float(w_axis), float(axis_th)

    # Make epoch-0 warm-start eval and replay-fill use the initial schedule
    # settings. For geom insert, replay fill starts at d_target_neg; the target
    # only ramps after actor-relative epoch starts moving.
    apply_stage2_curriculum(0)
    mdp.set_geom_epoch(0)

    logger.info(f"obs_dim={obs_dim} ({obs_mode})  "
                f"act_dim={act_dim}  horizon={mdp.info.horizon}")
    if mdp._geom_stage is not None:
        logger.info(
            f"geom_stage={mdp._geom_stage}  "
            f"d_target_neg={mdp._geom_d_target_neg:+.3f}  "
            f"d_target_pos={mdp._geom_d_target_pos:+.3f}  "
            f"d_target_eff={mdp._geom_d_target_eff:+.3f}"
        )
        logger.info(
            f"geom weights: w_d={mdp._w_geom_d:.2f}  "
            f"w_rad_tip={mdp._w_geom_radial_tip:.2f}  "
            f"w_rad_max={mdp._w_geom_radial_max:.2f}  "
            f"w_axis={mdp._w_geom_axis:.2f}  "
            f"w_soft_success={mdp._w_geom_soft_success:.2f}"
        )
        logger.info(
            f"geom thresholds: d_th={mdp._geom_d_th:.3f}  "
            f"r_tip_th={mdp._geom_r_tip_th:.3f}  "
            f"r_max_th={mdp._geom_r_max_th:.3f}  "
            f"axis_th={mdp._geom_axis_th:.3f}  "
            f"insert_d_ins={mdp._geom_insert_d_ins:+.3f}  "
            f"insert_r_max_th={mdp._geom_insert_r_max_th:.3f}  "
            f"pen_th={mdp._geom_pen_th*1000:.1f}mm  "
            f"d_sat={mdp._geom_d_sat:.3f}  radial_sat={mdp._geom_radial_sat:.3f}"
        )
        if mdp._geom_stage == "insert":
            logger.info(
                f"geom insert schedule (actor-relative epoch): "
                f"d_target ramp [{mdp._geom_d_target_ramp_start},"
                f"{mdp._geom_d_target_ramp_end}) "
                f"{mdp._geom_d_target_neg:+.3f}→{mdp._geom_d_target_pos:+.3f}"
            )
    if stage2_curriculum_active:
        logger.info(
            "stage2 curriculum (actor-relative epoch): "
            f"2A [0,{args.stage2_axis_ramp_start}) "
            f"w_axis={args.stage2_axis_weight_start:.3f}, axis_th=inf → "
            f"2B [{args.stage2_axis_ramp_start},{args.stage2_axis_ramp_end}) "
            f"w_axis ramp {args.stage2_axis_weight_start:.3f}→{stage2_axis_weight_end:.3f}, "
            f"axis_th={stage2_axis_success_threshold:.3f} → "
            f"2C [{args.stage2_axis_ramp_end},∞) "
            f"w_axis={stage2_axis_weight_end:.3f}, axis_th={stage2_axis_success_threshold:.3f}"
        )
    logger.info(f"action_scale={mdp._action_scale:.3f}")
    logger.info(
        "physx_self_collision_group="
        + ("arm_links_only" if mdp._exclude_ee_from_physx_self_collision
           else "arm_links_plus_ee")
    )
    logger.info(f"preinsert_pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
                f"axis_th={mdp._success_axis_threshold:.3f}  "
                f"w_pos={mdp._w_pos:.3f}  w_axis={mdp._w_axis:.3f}  "
                f"w_pos_success={mdp._w_pos_success:.3f}  "
                f"w_success={mdp._w_success:.3f}  "
                f"axis_gate_radius={mdp._axis_gate_radius:.3f}m  "
                f"w_home={mdp._w_home:.4f}  "
                f"preinsert_offset={mdp._preinsert_offset:.3f}m")
    if mdp._w_home > 0:
        logger.info(
            "home_weights="
            + ",".join(f"{float(w):.3g}" for w in mdp._home_weights.detach().cpu())
        )
    if args.load_agent is not None:
        logger.info(f"warm-start: {args.load_agent}")
    logger.info(f"target_entropy={target_entropy:.3f}  "
                f"lr_actor={args.lr_actor:.1e}  lr_critic={args.lr_critic:.1e}  "
                f"lr_alpha={args.lr_alpha:.1e}  alpha_max={args.alpha_max:.3f}")
    critic_only_steps = args.critic_warmup_transitions - INITIAL_REPLAY_SIZE
    if critic_only_steps > 0:
        critic_only_epochs = critic_only_steps / args.n_steps_per_epoch
        logger.info(
            f"critic_warmup_transitions={args.critic_warmup_transitions} env-steps "
            f"(replay-fill {INITIAL_REPLAY_SIZE} + critic-only {critic_only_steps} ≈ "
            f"{critic_only_epochs:.1f} epoch, actor 此期间冻结)"
        )
    else:
        logger.info(
            f"critic_warmup_transitions={args.critic_warmup_transitions} env-steps "
            "(无 critic-only 期, actor 与 critic 几乎同时开启)"
        )
    logger.info(f"n_steps_per_epoch={args.n_steps_per_epoch} env-steps  "
                f"n_steps_per_fit={args.n_steps_per_fit} env-steps  "
                f"num_envs={args.num_envs}")

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

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            group=args.wandb_group,
            config={**vars(args), "algo": "SAC",
                    "target_entropy_resolved": target_entropy,
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

    empty_dataset = Dataset.generate(mdp.info, agent.info, n_steps=1, n_envs=args.num_envs)

    # Epoch-0 eval: warm-start actor 在 *任何训练之前* 的表现. 用来确认 actor
    # 权重转移真的生效 (actor-only warmstart 后 pos_err 应该接近 Stage 1 收敛水平).
    # 如果这里 pos_success_rate 已经接近 0, 后面再讨论训练失败就没意义了 —
    # actor 转移本身就没成功.
    if args.load_agent is not None:
        logger.info("=" * 60)
        logger.info("[EVAL @ epoch 0] warm-start actor BEFORE 任何 fit / 任何 warmup")
        with deterministic_policy(agent):
            ds0 = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        m0 = compute_hold_metrics(ds0, mdp, args.hold_success_steps)
        logger.info(f"  pos_success_rate={m0['pos_success_rate']:.3f}  "
                    f"pos_err_mean={m0['pos_err_mean']:.4f}m  "
                    f"axis_err_mean={m0['axis_err_mean']:.4f}  "
                    f"hold_success_rate={m0['hold_success_rate']:.3f}")
        logger.info(f"  conditional (pos_in_thresh count={m0['pos_in_thresh_count']}):  "
                    f"axis_err_in_pos_th_mean={m0['axis_err_in_pos_thresh_mean']:.4f}  "
                    f"axis_err_in_pos_th_min={m0['axis_err_in_pos_thresh_min']:.4f}")
        logger.info("=" * 60)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": 0,
                "warmstart_pos_success_rate": m0["pos_success_rate"],
                "warmstart_pos_err_mean": m0["pos_err_mean"],
                "warmstart_axis_err_mean": m0["axis_err_mean"],
                "warmstart_hold_success_rate": m0["hold_success_rate"],
                "warmstart_axis_err_in_pos_thresh_mean":
                    m0["axis_err_in_pos_thresh_mean"]
                    if m0["pos_in_thresh_count"] > 0 else 0.0,
            }, step=0)

    warmup_vector_steps = math.ceil(INITIAL_REPLAY_SIZE / args.num_envs)
    logger.info("填充 replay buffer: "
                f"{INITIAL_REPLAY_SIZE} env-steps "
                f"(约 {warmup_vector_steps} vector-steps × {args.num_envs} envs)")
    core.learn(n_steps=INITIAL_REPLAY_SIZE, n_steps_per_fit=INITIAL_REPLAY_SIZE)

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
    # best_score 初始化 0.0 (而非 -inf): track_rate × track_score 必 ≥ 0,
    # geom insert 前期 geom_hold_rate=0 时不会触发 "improved", 但 log/wandb summary
    # 显示 0.0 比 -inf 干净. 不影响 ckpt 选择 (improved 仍要求 score > best_score).
    best_score = 0.0
    # best_hold 用 (hold_success_rate, max_hold_mean) 二级 key:
    # hold_success_rate 是首要指标 (0-1, 大多数 epoch 是 0), max_hold_mean 当 tie-breaker
    # (0-N, hold rate 相同时偏好平均 hold 更长的). 两者都要 > 之前才更新.
    best_hold_rate = -1.0  # 用 -1 而非 0, 让首次出现 hold_rate=0.0 时不会触发"改进"
    best_hold_score = -1.0  # tie-breaker: max_hold_mean
    total_env_steps = INITIAL_REPLAY_SIZE
    absorb_prev = mdp._absorb_count
    absorb_physx_prev = mdp._absorb_count_physx
    absorb_sphere_prev = mdp._absorb_count_sphere
    # Actor-relative epoch offset: critic warmup 期间 SAC 的 actor 不更新,
    # 把这部分 epoch 排除在 schedule 之外, 让 ramp 的边界跟 actor 实际开始
    # 学习的时刻对齐. critic_warmup_transitions == INITIAL_REPLAY_SIZE 时 offset=0.
    # Stage 2 curriculum 和 geom insert schedule 共用这套 actor-relative 计数.
    critic_only_steps = max(0, args.critic_warmup_transitions - INITIAL_REPLAY_SIZE)
    critic_only_epochs = math.ceil(critic_only_steps / args.n_steps_per_epoch)
    if stage2_curriculum_active and critic_only_epochs > 0:
        logger.info(
            f"stage2 curriculum offset: critic_only_epochs={critic_only_epochs} "
            f"(actor-relative epoch = max(0, raw_epoch - {critic_only_epochs}))"
        )
    if mdp._geom_stage == "insert" and critic_only_epochs > 0:
        logger.info(
            f"geom insert schedule offset: critic_only_epochs={critic_only_epochs} "
            f"(actor-relative epoch = max(0, raw_epoch - {critic_only_epochs}))"
        )
    for epoch in range(args.n_epochs):
        # Reward schedules walk using *actor-relative* epoch — critic warmup
        # 期间一律 actor_epoch=0, 不让 actor 还冻结时 schedule 偷跑.
        actor_epoch = max(0, epoch - critic_only_epochs)
        stage2_state = apply_stage2_curriculum(actor_epoch)
        mdp.set_geom_epoch(actor_epoch)
        core.learn(
            n_steps=args.n_steps_per_epoch,
            n_steps_per_fit=args.n_steps_per_fit,
            quiet=True,
        )
        clamp_alpha()
        for _ in range(fits_per_epoch * (args.utd - 1)):
            agent.fit(empty_dataset)
            clamp_alpha()
        total_env_steps += args.n_steps_per_epoch

        absorb_epoch = mdp._absorb_count - absorb_prev
        absorb_physx_epoch = mdp._absorb_count_physx - absorb_physx_prev
        absorb_sphere_epoch = mdp._absorb_count_sphere - absorb_sphere_prev

        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / args.n_eval_episodes
        m = compute_hold_metrics(dataset, mdp, args.hold_success_steps)
        # Geom metric: geom_step_rate / geom_hold_rate / d_err / radial_max /
        # penetration. 严格判定从 dataset.info.data 读, 与 reward / mask 同源.
        mg = (compute_geom_metrics(dataset, mdp, args.hold_success_steps)
              if mdp._geom_stage is not None else None)

        improved_J = J > best_J
        if improved_J:
            best_J = J
            agent.save(str(best_J_path))
            agent.save(str(best_J_path_flat))

        # best_hold ckpt 选择 + best_score 跟踪:
        #   Stage 1/2 (Lagrangian baseline): rate=hold_success_rate
        #              (pos<pos_th ∧ axis<axis_th 连续 N 步), score=max_hold_mean
        #   Geom:      rate=geom_hold_rate (active geom success mask 连续 N 步)
        #              score=geom_max_run_mean
        # 同一文件路径 best_hold.msh, 不同 stage 语义不同; 用日志区分.
        if mdp._geom_stage is not None:
            track_rate = mg['geom_hold_rate']
            track_score = mg['geom_max_run_mean']
        else:
            track_rate = m['hold_success_rate']
            track_score = m['max_hold_mean']
        score = track_rate * track_score
        # Stage 2 curriculum 的 2A 阶段 axis_th=inf, 此时 hold_success_rate
        # 只是 pos-only 成功, 不能和 2B/2C 的 pos+axis 成功竞争 best_hold.
        # 否则 2A 很容易用高 pos-only hold 锁住 best_hold.msh, 后面真正
        # axis-threshold 生效后的 checkpoint 反而覆盖不了.
        hold_ckpt_eligible = not (
            stage2_curriculum_active
            and actor_epoch < args.stage2_axis_ramp_start
        )
        improved_score = hold_ckpt_eligible and track_rate > 0 and score > best_score
        if improved_score:
            best_score = score
        improved_hold = (
            track_rate > best_hold_rate
            or (track_rate == best_hold_rate and track_score > best_hold_score)
        )
        # 只有 track_rate > 0 才写 best_hold.msh, 避免早期 0/0 持平也写一遍
        if hold_ckpt_eligible and improved_hold and track_rate > 0:
            best_hold_rate = track_rate
            best_hold_score = track_score
            agent.save(str(best_hold_path))
            agent.save(str(best_hold_path_flat))

        absorb_prev = mdp._absorb_count
        absorb_physx_prev = mdp._absorb_count_physx
        absorb_sphere_prev = mdp._absorb_count_sphere

        # epoch_info 字段命名按 stage 语义切换:
        # geom 路径: best_geom (active geom success mask hold rate)
        # 其他 (Lagrangian baseline): best_hold (pos+axis hold rate)
        logger.epoch_info(
            epoch + 1, J=J, R=R, best_J=best_J,
            **{("best_geom" if mdp._geom_stage is not None else "best_hold"):
               best_hold_rate if best_hold_rate >= 0 else 0.0},
            best_score=best_score,
            absorb_epoch=absorb_epoch,
            absorb_physx=absorb_physx_epoch,
            absorb_sphere=absorb_sphere_epoch,
        )
        if mdp._geom_stage is not None:
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
                f"d_target_mean={mg['geom_d_target_mean']:+.4f}m  "
                f"d_err_mean={mg['geom_d_err_mean']:.4f}m  "
                f"d_err_min={mg['geom_d_err_min']:.4f}m  "
                f"radial_tip_min={mg['geom_radial_tip_min']:.4f}m  "
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
                f"geom final: "
                f"d={mg['geom_final_d_mean']:+.4f}m  "
                f"d_err={mg['geom_final_d_err_mean']:.4f}m  "
                f"rm_mean={mg['geom_final_radial_max_mean']:.4f}m "
                f"(max {mg['geom_final_radial_max_max']:.4f}m)  "
                f"ax_mean={mg['geom_final_axis_err_mean']:.4f} "
                f"(max {mg['geom_final_axis_err_max']:.4f})  "
                f"pen_mean={mg['geom_final_penetration_mean']*1000:.2f}mm "
                f"(max {mg['geom_final_penetration_max']*1000:.2f}mm)"
            )
            logger.info(
                f"geom penetration: "
                f"max_mean={mg['geom_pen_max_mean']*1000:.2f}mm "
                f"max_max={mg['geom_pen_max_max']*1000:.2f}mm "
                f"clean_step_rate={mg['geom_clean_step_rate']:.3f} "
                f"(in active mask: mean={mg['geom_pen_in_active_mean']*1000:.2f}mm "
                f"max={mg['geom_pen_in_active_max']*1000:.2f}mm)"
            )
        if stage2_curriculum_active:
            _, w_axis_eff, axis_th_eff = stage2_state
            axis_th_str = "inf" if math.isinf(axis_th_eff) else f"{axis_th_eff:.3f}"
            logger.info(
                f"stage2 curriculum @ raw_epoch={epoch} actor_epoch={actor_epoch}: "
                f"w_axis_eff={w_axis_eff:.3f}  "
                f"axis_th_eff={axis_th_str}  "
                f"axis_gate_radius={mdp._axis_gate_radius:.3f}m"
            )
        if mdp._geom_stage is not None:
            # geom 模式下 m 是旧球形 pos<pos_th ∧ axis<axis_th 指标, 跟 geom_*
            # 不是同一套几何, 并列打印会误判 ckpt 状态 (旧 hold_success_rate=1.0
            # 不等于 geom_hold_rate=1.0). console 跳过 legacy 块; 调试需要可读
            # wandb 上仍保留的 m_* 字段或 compute_hold_metrics 返回值.
            logger.info("eval stats: [legacy 球形 pos/axis 指标 skipped — geom 模式 ckpt 选择走 geom_*]")
        else:
            logger.info("eval stats: "
                        f"hold_success_rate={m['hold_success_rate']:.3f} "
                        f"(>= {args.hold_success_steps} consecutive steps)  "
                        f"max_hold_mean={m['max_hold_mean']:.1f}  "
                        f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
                        f"final_in_thresh_rate={m['final_in_thresh_rate']:.3f}  "
                        f"pos_success_rate={m['pos_success_rate']:.3f}  "
                        f"pos_err_mean={m['pos_err_mean']:.4f}m  "
                        f"axis_err_mean={m['axis_err_mean']:.4f}  "
                        f"axis_gate_mean={m['axis_gate_mean']:.3f}  "
                        f"gated_axis_pen={m['gated_axis_penalty_mean']:.3f}")
            # 条件指标: 关键证据是 'pos_in_thresh 时 axis_err 是否下降'.
            # pos_in_thresh_count=0 时 NaN — 用 'n/a' 显示, 避免误读成 0.
            if m['pos_in_thresh_count'] > 0:
                cond_str = (f"axis_err_in_pos_th_mean={m['axis_err_in_pos_thresh_mean']:.4f}  "
                            f"axis_err_in_pos_th_min={m['axis_err_in_pos_thresh_min']:.4f}  "
                            f"axis_gate_in_pos_th_mean={m['axis_gate_in_pos_thresh_mean']:.3f}")
            else:
                cond_str = "axis_err_in_pos_th=n/a (pos_in_thresh_count=0)"
            logger.info(f"  ↳ pos_in_thresh_count={m['pos_in_thresh_count']}  {cond_str}")
        if wandb_run is not None:
            # geom 模式下 m 是旧球形 pos/axis 指标, 与 geom_* 不可比. wandb key 加
            # legacy_ 前缀, 让 dashboard 上 geom run 没有裸的 eval_success_rate 等
            # 字段, 旧 (非-geom) run 不受影响. _legacy(...) 在 geom 模式下加前缀.
            _legacy = (lambda k: f"legacy_{k}") if mdp._geom_stage is not None else (lambda k: k)
            wandb_run.log({
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "best_score": best_score,
                "best_hold_rate":
                    best_hold_rate if best_hold_rate >= 0 else 0.0,
                "best_hold_max_hold_mean":
                    best_hold_score if best_hold_score >= 0 else 0.0,
                "eval_ep_len": ep_len,
                _legacy("eval_success_rate"): m["hold_success_rate"],
                _legacy("eval_max_hold_mean"): m["max_hold_mean"],
                _legacy("eval_in_thresh_rate"): m["in_thresh_rate"],
                _legacy("eval_final_in_thresh_rate"): m["final_in_thresh_rate"],
                _legacy("eval_pos_success_rate"): m["pos_success_rate"],
                _legacy("eval_pos_err_mean"): m["pos_err_mean"],
                _legacy("eval_axis_err_mean"): m["axis_err_mean"],
                _legacy("eval_axis_gate_mean"): m["axis_gate_mean"],
                _legacy("eval_gated_axis_penalty_mean"): m["gated_axis_penalty_mean"],
                **({"stage2_curriculum_epoch": actor_epoch,
                    "stage2_w_axis_eff": mdp._w_axis,
                    "stage2_axis_threshold_eff": mdp._success_axis_threshold}
                   if stage2_curriculum_active else {}),
                **({"geom_raw_epoch": epoch,
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
                    "geom_n_ep_with_entry":       mg["geom_n_ep_with_entry"],
                    "geom_entry_d_mean":          mg["geom_entry_d_mean"],
                    "geom_entry_d_err_mean":      mg["geom_entry_d_err_mean"],
                    "geom_entry_radial_max_mean": mg["geom_entry_radial_max_mean"],
                    "geom_entry_radial_max_max":  mg["geom_entry_radial_max_max"],
                    "geom_entry_axis_err_mean":   mg["geom_entry_axis_err_mean"],
                    "geom_entry_axis_err_max":    mg["geom_entry_axis_err_max"],
                    "geom_final_d_mean":          mg["geom_final_d_mean"],
                    "geom_final_d_err_mean":      mg["geom_final_d_err_mean"],
                    "geom_final_radial_max_mean": mg["geom_final_radial_max_mean"],
                    "geom_final_radial_max_max":  mg["geom_final_radial_max_max"],
                    "geom_final_axis_err_mean":   mg["geom_final_axis_err_mean"],
                    "geom_final_axis_err_max":    mg["geom_final_axis_err_max"],
                    "geom_pen_max_mean":          mg["geom_pen_max_mean"],
                    "geom_pen_max_max":           mg["geom_pen_max_max"],
                    "geom_clean_step_rate":       mg["geom_clean_step_rate"],
                    "geom_pen_in_active_mean":    mg["geom_pen_in_active_mean"],
                    "geom_pen_in_active_max":     mg["geom_pen_in_active_max"],
                    "geom_entry_penetration_mean": mg["geom_entry_penetration_mean"],
                    "geom_entry_penetration_max":  mg["geom_entry_penetration_max"],
                    "geom_final_penetration_mean": mg["geom_final_penetration_mean"],
                    "geom_final_penetration_max":  mg["geom_final_penetration_max"]}
                   if mdp._geom_stage is not None else {}),
                _legacy("eval_pos_in_thresh_count"): m["pos_in_thresh_count"],
                _legacy("eval_axis_err_in_pos_thresh_mean"):
                    m["axis_err_in_pos_thresh_mean"]
                    if m["pos_in_thresh_count"] > 0 else float("nan"),
                _legacy("eval_axis_gate_in_pos_thresh_mean"):
                    m["axis_gate_in_pos_thresh_mean"]
                    if m["pos_in_thresh_count"] > 0 else float("nan"),
                "alpha": agent._alpha.item(),
                "absorb_per_epoch": absorb_epoch,
                "absorb_physx_per_epoch": absorb_physx_epoch,
                "absorb_sphere_per_epoch": absorb_sphere_epoch,
            }, step=epoch + 1)

    # 训练结束时无条件保存最后一个 epoch 的 actor — 这是 "稳态终态" 的 ground truth,
    # 独立 eval 应当至少评一次它, 跟 best_J / best_hold 对照, 避免 "best_J 锁早期低 hold
    # 快照, 终态 hold=1.0 但因 J 未破上限被永远丢失" 这类 silent failure.
    agent.save(str(final_path))
    agent.save(str(final_path_flat))

    hold_metric_name = (
        "geom_hold_rate" if mdp._geom_stage is not None
        else "hold_success_rate"
    )
    score_metric_name = (
        "geom_max_run_mean" if mdp._geom_stage is not None
        else "max_hold_mean"
    )
    if best_hold_rate < 0:
        best_hold_display = f"n/a (no {hold_metric_name} success)"
    else:
        best_hold_display = (f"{best_hold_rate:.3f} "
                             f"({score_metric_name}={best_hold_score:.1f})")
    logger.info(
        f"训练完成. best J = {best_J:.3f}  "
        f"best {hold_metric_name} = {best_hold_display}  "
        f"best_score = {best_score:.3f}"
    )
    logger.info(f"checkpoint 写入: {ckpt_dir}/ 下的 "
                f"{best_J_path.name} (best J) / "
                f"{best_hold_path.name} (best hold) / {final_path.name} (final). "
                "**eval 时务必对三个都跑一遍**, best_J 不一定 = 最稳策略.")

    if wandb_run is not None:
        wandb_run.summary["best_J"] = best_J
        wandb_run.summary["best_score"] = best_score
        # Stage 1/2 (Lagrangian baseline): best_hold_rate = hold_success_rate
        # Geom: best_hold_rate = geom_hold_rate (active geom mask)
        # 同 key 名字, 不同 stage 不同语义; wandb run config.algo + config.geom_stage 区分.
        wandb_run.summary["best_hold_rate"] = (
            best_hold_rate if best_hold_rate >= 0 else 0.0
        )
        wandb_run.summary["best_hold_max_hold_mean"] = (
            best_hold_score if best_hold_score >= 0 else 0.0
        )
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
