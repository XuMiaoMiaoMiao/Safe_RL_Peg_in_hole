"""SAC 训练 — 双臂 peg-in-hole preinsert (stage flag 化, mushroom-rl + VectorCore).

obs 32 维 (joint_pos+joint_vel+pos_vec+axis_dot), 同一个 env / 同一个 obs / 同一条
reward 骨架. stage 用 reward 权重 + success_axis_threshold 切换:

    M1' = pos-only           --rew_axis 0.0  --success_axis_threshold inf
    M2a = pos + 粗轴对齐      --rew_axis 2.0  --success_axis_threshold 0.5
    M2b = pos + 紧轴对齐      --rew_axis 2.0  --success_axis_threshold 0.2

M2a/M2b 用 --load_agent path/to/M1p_checkpoint.msh 续训, 不用 cold start.

运行:
    conda activate safe_rl
    # M1': 建立 32 维 baseline (相当于 pos-only)
    python scripts/train_sac.py --no_wandb \\
        --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50

    # M2a: 从 M1' warm-start, 加 axis reward
    python scripts/train_sac.py --no_wandb \\
        --load_agent results/best_agent_M1p_32dim_pos10cm.msh \\
        --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \\
        --rew_axis 2.0 --success_axis_threshold 0.5

    # M2b: 从 M2a 收紧
    python scripts/train_sac.py --no_wandb \\
        --load_agent results/best_agent_M2a_axis05.msh \\
        --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \\
        --rew_axis 2.0 --success_axis_threshold 0.2

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
    compute_hold_metrics,
    deterministic_policy,
    resolve_eval_episode_count,
    summarize_approach,
    summarize_clearance,
)


INITIAL_REPLAY_SIZE = 10_000
MAX_REPLAY_SIZE = 500_000
BATCH_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--render", action="store_true", help="打开 IsaacSim 窗口")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_epochs", type=int, default=400)
    p.add_argument("--n_steps_per_epoch", type=int, default=1024,
                   help="每个 epoch 收集的总 env-step 数 (不是 vector-step)")
    p.add_argument("--n_steps_per_fit", type=int, default=None,
                   help="两次 fit 之间收集的总 env-step 数 (默认 = num_envs, 即 1 个 vector-step)")
    p.add_argument("--utd", type=int, default=None,
                   help="每次 fit 块对应的总梯度步数. 默认自动取 n_steps_per_fit, 使 true UTD≈1")
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--alpha_max", type=float, default=0.2,
                   help="alpha 上限, 抑制高维动作下 entropy 奖励压过任务 reward")
    p.add_argument("--target_entropy", type=float, default=None,
                   help="目标 entropy. 默认自动取 -act_dim (SAC 标准设置)")
    p.add_argument("--n_eval_episodes", type=int, default=None,
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 的 reset 关节噪声")
    p.add_argument("--action_scale", type=float, default=None,
                   help="覆盖 env 的动作缩放. M2d 37维 cold start 起步建议 0.4; "
                        "收敛后再降到 0.1 做低速 servo.")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None,
                   help="覆盖 env 的 preinsert 位置成功阈值 (env 默认 0.10m, 即当前 "
                        "M1'/M2 curriculum). 如果显式想跑老 5cm, 传 0.05.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert offset (默认 0.05m)")
    p.add_argument("--rew_pos", type=float, default=None,
                   help="覆盖 env 的球形 pos_err reward 权重. M2d approach 传 0, "
                        "避免旧 pos 球形 reward 与 axial/radial 对准目标拉扯.")
    p.add_argument("--rew_action", type=float, default=None,
                   help="覆盖 env 的动作 L2 惩罚权重")
    p.add_argument("--rew_success", type=float, default=None,
                   help="覆盖 env 的 per-step success bonus (默认 2.0)")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重 (默认 0.0 = M1' pos-only). "
                        "M2a/M2b 设 2.0 启用轴对齐惩罚.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="覆盖 env 的 axis_err success 阈值 (默认 inf = M1' 不检查 axis). "
                        "M2a 用 0.5, M2b 用 0.2. 接受 'inf' 字符串.")
    # M2c sphere-proxy clearance — default 全关 (env 内 default = -inf / 0).
    p.add_argument("--rew_clearance", type=float, default=None,
                   help="M2c clearance penalty 权重. 默认 0 = M2c 不启用. "
                        "M2c 起步建议 2.0 (与 rew_axis 同量级).")
    p.add_argument("--clearance_soft", type=float, default=None,
                   help="M2c soft threshold (m): clearance < soft 时 reward 罚, "
                        "且 success_mask 要求 clearance >= soft. 默认 -inf = 不启用. "
                        "curriculum 起步常用 0.0 / 0.02 / 0.05; 支持 <0 (深度穿插训练).")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="M2c hard threshold (m): clearance < hard 触发 absorbing "
                        "(同 collision r_min/(1-γ)). 默认 -inf = 不启用. "
                        "curriculum 起步常用 -0.03 / 0.0 / 0.02.")
    p.add_argument("--clearance_penalty_scale", type=float, default=None,
                   help="M2c penalty 归一化 scale (m): penalty=(gap/scale)^2. "
                        "默认 0.05m, 一般不动.")
    p.add_argument("--clearance_success_threshold", type=float, default=None,
                   help="M2c success gate 阈值 (m). 默认沿用 clearance_soft (向后兼容). "
                        "显式拆开时: reward_soft 给梯度 (如 0.05), success_threshold "
                        "给 dwell gate (如 0.03), 让 5cm 训练目标不让 hold 稀疏.")
    p.add_argument("--log_clearance", action="store_true",
                   help="metric-only: 即使 M2c 关闭也每步算 clearance + 注入 "
                        "dataset.info, 让 logger 输出 clearance 分布. 不改 reward.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="M2c sphere proxy 的 arm/link 半径 (m). 默认 0.06. "
                        "改它会改变 clearance reward/success 定义; 做对比实验时显式传.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="M2c sphere proxy 的 EE/finger 半径 (m). 默认 0.03. "
                        "改它会改变 clearance reward/success 定义; 做对比实验时显式传.")
    # M2d approach — default 全关. 不改 obs 维度, 但 env 每步把 axial/radial
    # 几何量注入 dataset.info, 让 eval/logger 按共轴预插入指标算 hold-N.
    p.add_argument("--use_m2d_obs", action="store_true",
                   help="把 agent obs 从 32 维扩到 37 维: 追加 axial_dist / "
                        "axial_off / radial_vec_hole2 / radial_err. 这是 M2d "
                        "共轴预插入的推荐设置, 不能加载 32 维旧 checkpoint.")
    p.add_argument("--rew_axial_off", type=float, default=None,
                   help="M2d 轴向偏差 reward 权重: axial_off=|axial_dist-preinsert_offset|.")
    p.add_argument("--rew_radial", type=float, default=None,
                   help="M2d 横向偏差 reward 权重: radial_err 到 hole 中线的距离.")
    p.add_argument("--axial_success_threshold", type=float, default=None,
                   help="M2d 轴向 success 阈值 (m). 37维 cold start 建议先 0.30, "
                        "再按 0.12/0.08/0.04 收紧.")
    p.add_argument("--radial_success_threshold", type=float, default=None,
                   help="M2d 横向 success 阈值 (m). 37维 cold start 建议先 0.30, "
                        "再按 0.12/0.08/0.04 收紧.")
    p.add_argument("--log_axial_radial", action="store_true",
                   help="metric-only: 即使不启用 M2d gate/reward 也每步注入 "
                        "axial_off/radial_err 到 dataset.info.")
    p.add_argument("--load_agent", type=str, default=None,
                   help="warm-start 路径: 从该 checkpoint 加载 agent (actor/critic/"
                        "optimizer state). obs 维度必须匹配; 31 维 M1 老 checkpoint "
                        "不能加载到 32 维 env, 先重训 M1'.")
    p.add_argument("--keep_replay", action="store_true",
                   help="warm-start 时保留旧 replay buffer. 默认会清空 — 因为 stage "
                        "切换 (M1'→M2a, M2a→M2b) reward 函数变了, 旧 transitions 的 "
                        "reward 标签按旧 reward 算, 留着会拖 critic.")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "0 = 关闭 (baseline). >0 启用 absorbing termination.")
    p.add_argument("--allow_m2d_terminal_bonus", action="store_true",
                   help="默认 M2d active 时禁止 terminal_hold_bonus>0, 因为它会截断 "
                        "进 gate 后的漂移行为. 显式传本 flag 才允许该配置.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="eval success 定义 + env 终止阈值: 连续 N 步都在阈值内. "
                        "N=10 ≈ 1s hold (per-step dt≈0.1s).")
    p.add_argument("--final_window_steps", type=int, default=30,
                   help="M2d 稳停指标窗口长度: episode 最后 N 步全部满足 "
                        "axis∧axial∧radial∧clearance 才算 final-window success.")
    p.add_argument("--wandb_project", type=str, default="bimanual_peghole")
    p.add_argument("--wandb_run_name", type=str, default=None)
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
    args.n_eval_episodes = resolve_eval_episode_count(
        args.n_eval_episodes, args.num_envs, "--n_eval_episodes"
    )

    from envs import DualArmPegHoleEnv
    env_kwargs = dict(num_envs=args.num_envs, headless=not args.render)
    for key in ("action_scale", "initial_joint_noise",
                "preinsert_success_pos_threshold", "preinsert_offset",
                "rew_pos", "rew_action", "rew_success", "rew_axis",
                "success_axis_threshold", "terminal_hold_bonus",
                "rew_clearance", "clearance_soft", "clearance_hard",
                "clearance_penalty_scale", "clearance_success_threshold",
                "proxy_arm_radius", "proxy_ee_radius",
                "rew_axial_off", "rew_radial",
                "axial_success_threshold", "radial_success_threshold"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    if args.log_clearance:
        env_kwargs["log_clearance"] = True
    if args.log_axial_radial:
        env_kwargs["log_axial_radial"] = True
    if args.use_m2d_obs:
        env_kwargs["use_m2d_obs"] = True
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    mdp = DualArmPegHoleEnv(**env_kwargs)
    if (mdp._m2d_active and mdp._terminal_hold_bonus > 0.0
            and not args.allow_m2d_terminal_bonus):
        raise ValueError(
            "M2d active 时不允许 terminal_hold_bonus > 0: 这会让 episode 在进 gate "
            "后立即截断, 无法训练/评估是否稳停. 若确实要复现实验, 显式传 "
            "--allow_m2d_terminal_bonus."
        )
    mdp.seed(args.seed)

    # IsaacSim 启动后才能导入 mushroom_rl (避免 carb 冲突)
    from mushroom_rl.algorithms.actor_critic import SAC
    from mushroom_rl.core import Agent, VectorCore, Logger, Dataset

    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]
    target_entropy = args.target_entropy
    if target_entropy is None:
        target_entropy = -float(act_dim)

    if args.load_agent is not None:
        # Warm-start: 加载已有 agent 的 actor/critic/optimizer (+ replay buffer).
        # obs 维度必须匹配 (32 维); 加载 31 维老 checkpoint 会在 forward 时抛 shape 错.
        load_path = Path(args.load_agent)
        if not load_path.is_file():
            raise FileNotFoundError(f"--load_agent 路径不存在: {load_path}")
        agent = Agent.load(str(load_path))
        print(f"[WARM-START] 已加载 agent from {load_path}")
        # 默认清空 replay buffer: stage 切换 (M1'→M2a, M2a→M2b) reward 函数变了,
        # 旧 transitions 的 reward 标签按旧 reward 算的, 留下来会拖 critic. 仅在
        # 用户显式 --keep_replay 时保留.
        if not args.keep_replay:
            agent._replay_memory.reset()
            print("[WARM-START] replay buffer 已清空 — 重新走 INITIAL_REPLAY_SIZE 填充. "
                  "若要保留旧 buffer, 加 --keep_replay.")
        else:
            print("[WARM-START] 保留旧 replay buffer (--keep_replay).")
    else:
        actor_params = dict(network=ActorNetwork, input_shape=(obs_dim,),
                            output_shape=(act_dim,))
        actor_optimizer = {"class": optim.Adam, "params": {"lr": args.lr_actor}}
        critic_params = dict(network=CriticNetwork, input_shape=(obs_dim,),
                             output_shape=(1,), action_dim=act_dim,
                             optimizer={"class": optim.Adam, "params": {"lr": args.lr_critic}},
                             loss=F.mse_loss)

        agent = SAC(
            mdp_info=mdp.info,
            actor_mu_params=actor_params,
            actor_sigma_params=actor_params,
            actor_optimizer=actor_optimizer,
            critic_params=critic_params,
            batch_size=BATCH_SIZE,
            initial_replay_size=INITIAL_REPLAY_SIZE,
            max_replay_size=MAX_REPLAY_SIZE,
            warmup_transitions=INITIAL_REPLAY_SIZE,
            tau=0.005,
            lr_alpha=args.lr_alpha,
            use_log_alpha_loss=True,
            target_entropy=target_entropy,
        )
    def clamp_alpha(_dataset=None):
        with torch.no_grad():
            agent._log_alpha.clamp_(max=math.log(args.alpha_max))

    core = VectorCore(agent, mdp, callbacks_fit=[clamp_alpha])

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    best_path = results_dir / "best_agent.msh"
    if best_path.exists():
        best_path.unlink()
    logger = Logger("SAC", results_dir=str(results_dir))
    logger.strong_line()
    logger.info(f"清理旧 best checkpoint (本次 run 自建): {best_path}")
    logger.info(f"obs_dim={obs_dim}  act_dim={act_dim}  horizon={mdp.info.horizon}")
    if mdp._use_m2d_obs and args.load_agent is not None:
        logger.info("[M2d obs] 当前 obs_dim=37, 只能加载同为 37 维的 M2d checkpoint; "
                    "32 维 M2c checkpoint 会在网络 forward 时 shape mismatch.")
    logger.info(f"action_scale={mdp._action_scale:.3f}")
    logger.info(f"preinsert_pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
                f"axis_th={mdp._success_axis_threshold:.3f}  "
                f"w_pos={mdp._w_pos:.3f}  w_axis={mdp._w_axis:.3f}  "
                f"preinsert_offset={mdp._preinsert_offset:.3f}m")
    if (mdp._w_clearance > 0
            or math.isfinite(mdp._clearance_soft)
            or math.isfinite(mdp._clearance_hard)
            or math.isfinite(mdp._clearance_success_threshold)):
        logger.info(f"[M2c] w_clearance={mdp._w_clearance:.3f}  "
                    f"clearance_soft={mdp._clearance_soft:+.4f}m  "
                    f"clearance_hard={mdp._clearance_hard:+.4f}m  "
                    f"success_threshold={mdp._clearance_success_threshold:+.4f}m  "
                    f"penalty_scale={mdp._clearance_penalty_scale:.4f}m  "
                    f"proxy_arm={mdp._proxy_arm_radius:.3f}m  "
                    f"proxy_ee={mdp._proxy_ee_radius:.3f}m")
    if (mdp._w_axial_off > 0
            or mdp._w_radial > 0
            or math.isfinite(mdp._axial_success_threshold)
            or math.isfinite(mdp._radial_success_threshold)):
        logger.info(f"[M2d] w_axial_off={mdp._w_axial_off:.3f}  "
                    f"w_radial={mdp._w_radial:.3f}  "
                    f"axial_th={mdp._axial_success_threshold:.4f}m  "
                    f"radial_th={mdp._radial_success_threshold:.4f}m  "
                    "success = axis∧axial∧radial∧clearance (no pos sphere gate)")
    if args.load_agent is not None:
        logger.info(f"warm-start: {args.load_agent}")
    logger.info(f"target_entropy={target_entropy:.3f}  "
                f"lr_actor={args.lr_actor:.1e}  lr_critic={args.lr_critic:.1e}  "
                f"lr_alpha={args.lr_alpha:.1e}  alpha_max={args.alpha_max:.3f}")
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
    if mdp._last_axial_off is not None:
        logger.info("reset M2d geometry: "
                    f"axial_off_mean={float(mdp._last_axial_off.mean()):.4f}m  "
                    f"radial_err_mean={float(mdp._last_radial_err.mean()):.4f}m")

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config={**vars(args), "algo": "SAC",
                    "target_entropy_resolved": target_entropy,
                    "obs_dim": obs_dim, "act_dim": act_dim,
                    "horizon": mdp.info.horizon, "gamma": mdp.info.gamma},
            dir=str(results_dir),
        )
        logger.info(f"wandb run: {wandb_run.url}")

    # placeholder dataset — agent.fit() 需要 dataset 入参但 SAC 内部走 replay buffer,
    # 不会读 dataset 内容. 仅在 utd>1 时给额外 fit 步用.
    empty_dataset = Dataset.generate(mdp.info, agent.info, n_steps=1, n_envs=args.num_envs)

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
    best_score = -np.inf
    best_m2d_key = None
    m2c_checkpoint_selection = (
        mdp._w_clearance > 0.0
        or math.isfinite(mdp._clearance_soft)
        or math.isfinite(mdp._clearance_hard)
        or math.isfinite(mdp._clearance_success_threshold)
    )
    m2d_checkpoint_selection = (
        math.isfinite(mdp._axial_success_threshold)
        or math.isfinite(mdp._radial_success_threshold)
    )
    structured_checkpoint_selection = m2c_checkpoint_selection or m2d_checkpoint_selection
    use_J_for_best = mdp._terminal_hold_bonus > 0 and not structured_checkpoint_selection
    if m2d_checkpoint_selection:
        logger.info("best checkpoint selection: M2d final-window stability first "
                    f"(last {args.final_window_steps} steps all in-thresh; no J fallback)")
    elif m2c_checkpoint_selection:
        logger.info("best checkpoint selection: hold-score only "
                    "(clearance-aware when clearance_soft is finite; no J fallback)")
    elif use_J_for_best:
        logger.info("best checkpoint selection: best J "
                    "(terminal_hold_bonus enabled, M2c disabled)")
    else:
        logger.info("best checkpoint selection: hold-score, with J fallback until first hold success")
    total_env_steps = INITIAL_REPLAY_SIZE
    absorb_prev = mdp._absorb_count
    for epoch in range(args.n_epochs):
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

        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_eval_episodes, quiet=True)
        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        ep_len = len(dataset) / args.n_eval_episodes
        # M2c clearance-aware hold metric: 启用条件是 env._clearance_success_threshold
        # 有限值. metric 跟 env 的 success_mask 用同一阈值, 与 hold 触发逻辑严格一致.
        # 否则 (M1'/M2a/M2b) 不传 clearance_threshold, metric 退化为 pos∧axis only.
        # clearance 数据来自 env._create_info_dictionary 注入的 dataset.info.
        clearance_threshold_for_metric = (
            mdp._clearance_success_threshold
            if math.isfinite(mdp._clearance_success_threshold) else None
        )
        axial_threshold_for_metric = (
            mdp._axial_success_threshold
            if math.isfinite(mdp._axial_success_threshold) else None
        )
        radial_threshold_for_metric = (
            mdp._radial_success_threshold
            if math.isfinite(mdp._radial_success_threshold) else None
        )
        m = compute_hold_metrics(
            dataset, mdp, args.hold_success_steps,
            clearance_threshold=clearance_threshold_for_metric,
            axial_threshold=axial_threshold_for_metric,
            radial_threshold=radial_threshold_for_metric,
            final_window_steps=args.final_window_steps,
        )
        clr = summarize_clearance(dataset)
        app = summarize_approach(dataset, mdp)

        improved_J = J > best_J
        if improved_J:
            best_J = J
        # M2c 启用时用 clearance-aware hold metric 做 best_score 评判, 避免按
        # pos∧axis-only success 选 checkpoint 时挑到 cross-over 解.
        if "hold_success_rate_m2d" in m:
            primary_hold = m["hold_success_rate_m2d"]
            primary_max_hold = m["max_hold_m2d_mean"]
        elif "hold_success_rate_with_clearance" in m:
            primary_hold = m["hold_success_rate_with_clearance"]
            primary_max_hold = m["max_hold_full_mean"]
        else:
            primary_hold = m["hold_success_rate"]
            primary_max_hold = m["max_hold_mean"]
        score = primary_hold * primary_max_hold

        def _finite_cost(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return float("inf")
            return value if math.isfinite(value) else float("inf")

        def _finite_reward(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return -float("inf")
            return value if math.isfinite(value) else -float("inf")

        if "hold_success_rate_m2d" in m:
            # Primary: final-window stability. A model must keep the final N
            # steps inside axis/axial/radial/clearance gates to be eligible.
            # max_hold remains diagnostic only; it can be satisfied by passing
            # through the gate and drifting away.
            final_window_success = m.get("final_window_success_rate_m2d", 0.0)
            final_window_in_thresh = m.get("final_window_in_thresh_rate_m2d", 0.0)
            m2d_key = (
                float(final_window_success),
                float(final_window_in_thresh),
                -_finite_cost(m.get("final_window_radial_err_mean_m2d")),
                -_finite_cost(m.get("final_window_axial_off_mean_m2d")),
                -_finite_cost(m.get("final_window_axis_err_mean_m2d")),
                _finite_reward(m.get("final_window_clearance_mean_m2d")),
                float(primary_hold),
                float(min(primary_max_hold, args.hold_success_steps)),
            )
            improved_score = (
                final_window_success > 0
                and (best_m2d_key is None or m2d_key > best_m2d_key)
            )
            if improved_score:
                best_m2d_key = m2d_key
                best_score = float(final_window_success)
        else:
            improved_score = primary_hold > 0 and score > best_score
            if improved_score:
                best_score = score
        # M2c 不能按 best J fallback 保存: J 可能被远离/避碰策略刷高, 但任务成功为 0.
        # 旧阶段保留原选择: terminal-hold 阶段按 J; 其它阶段在第一次 hold success
        # 出现前允许 J fallback, 避免早期 baseline 没有 best_agent.msh.
        if structured_checkpoint_selection:
            save_now = improved_score
        elif use_J_for_best:
            save_now = improved_J
        else:
            save_now = improved_score or (best_score == -np.inf and improved_J)
        if save_now:
            agent.save(str(results_dir / "best_agent.msh"))
            if "hold_success_rate_m2d" in m:
                logger.info(
                    "[BEST M2d] saved checkpoint: "
                    f"final_window_success={m.get('final_window_success_rate_m2d', 0.0):.3f}  "
                    f"final_window_in_thresh={m.get('final_window_in_thresh_rate_m2d', 0.0):.3f}  "
                    f"final_window_radial="
                    f"{m.get('final_window_radial_err_mean_m2d', float('nan')):.4f}m  "
                    f"final_window_axial="
                    f"{m.get('final_window_axial_off_mean_m2d', float('nan')):.4f}m  "
                    f"final_window_clearance="
                    f"{m.get('final_window_clearance_mean_m2d', float('nan')):+.4f}m  "
                    f"hold={primary_hold:.3f}  max_hold={primary_max_hold:.1f}"
                )

        absorb_prev = mdp._absorb_count

        logger.epoch_info(epoch + 1, J=J, R=R, best_J=best_J, best_score=best_score,
                          absorb_epoch=absorb_epoch)
        logger.info("eval stats (pos∧axis): "
                    f"hold_success_rate={m['hold_success_rate']:.3f} "
                    f"(>= {args.hold_success_steps} consecutive steps)  "
                    f"max_hold_mean={m['max_hold_mean']:.1f}  "
                    f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
                    f"final_in_thresh_rate={m['final_in_thresh_rate']:.3f}  "
                    f"pos_err_mean={m['pos_err_mean']:.4f}m  "
                    f"axis_err_mean={m['axis_err_mean']:.4f}")
        if "hold_success_rate_with_clearance" in m:
            logger.info(
                f"eval stats (pos∧axis∧clearance, gate={m['clearance_threshold_used']:+.4f}m): "
                f"hold_success_rate_with_clearance={m['hold_success_rate_with_clearance']:.3f}  "
                f"max_hold_full_mean={m['max_hold_full_mean']:.1f}  "
                f"clearance_pass_rate={m['clearance_pass_rate']:.3f}"
            )
        if "hold_success_rate_m2d" in m:
            logger.info(
                "eval stats (M2d axis∧axial∧radial∧clearance): "
                f"hold_success_rate_m2d={m['hold_success_rate_m2d']:.3f}  "
                f"max_hold_m2d_mean={m['max_hold_m2d_mean']:.1f}  "
                f"m2d_in_thresh_rate={m['m2d_in_thresh_rate']:.3f}  "
                f"m2d_final_in_thresh_rate={m['m2d_final_in_thresh_rate']:.3f}  "
                f"axis_pass={m['m2d_axis_pass_rate']:.3f}  "
                f"axial_pass={m['m2d_axial_pass_rate']:.3f}  "
                f"radial_pass={m['m2d_radial_pass_rate']:.3f}  "
                f"axial_off_mean={m.get('axial_off_mean', float('nan')):.4f}m  "
                f"radial_err_mean={m.get('radial_err_mean', float('nan')):.4f}m"
            )
            logger.info(
                f"eval final-window stats (M2d, last {args.final_window_steps} steps): "
                f"success_rate={m.get('final_window_success_rate_m2d', 0.0):.3f}  "
                f"in_thresh_rate={m.get('final_window_in_thresh_rate_m2d', 0.0):.3f}  "
                f"axis={m.get('final_window_axis_err_mean_m2d', float('nan')):.4f}  "
                f"axial={m.get('final_window_axial_off_mean_m2d', float('nan')):.4f}m  "
                f"radial={m.get('final_window_radial_err_mean_m2d', float('nan')):.4f}m  "
                f"clearance={m.get('final_window_clearance_mean_m2d', float('nan')):+.4f}m"
            )
            logger.info(
                "eval terminal stats (M2d): "
                f"final_axis={m.get('m2d_final_axis_err_mean', float('nan')):.4f}  "
                f"final_axial={m.get('m2d_final_axial_off_mean', float('nan')):.4f}m  "
                f"final_radial={m.get('m2d_final_radial_err_mean', float('nan')):.4f}m  "
                f"final_clearance={m.get('m2d_final_clearance_mean', float('nan')):+.4f}m  "
                f"success_final_axis="
                f"{m.get('m2d_success_final_axis_err_mean', float('nan')):.4f}  "
                f"success_final_axial="
                f"{m.get('m2d_success_final_axial_off_mean', float('nan')):.4f}m  "
                f"success_final_radial="
                f"{m.get('m2d_success_final_radial_err_mean', float('nan')):.4f}m  "
                f"success_final_clearance="
                f"{m.get('m2d_success_final_clearance_mean', float('nan')):+.4f}m"
            )
        logger.info("clearance stats (sphere proxy, M2c+ instrumentation): "
                    f"mean={clr['clearance_mean']:+.4f}m  "
                    f"p10={clr['clearance_p10']:+.4f}m  "
                    f"min={clr['clearance_min']:+.4f}m  "
                    f"per-step rate ≥-2cm={clr['clearance_rate_at_-2cm']:.3f}  "
                    f"≥0={clr['clearance_rate_at_+0cm']:.3f}  "
                    f"≥+2cm={clr['clearance_rate_at_+2cm']:.3f}  "
                    f"≥+5cm={clr['clearance_rate_at_+5cm']:.3f}")
        if math.isfinite(app["axial_off_mean"]):
            logger.info("approach stats (M2d instrumentation): "
                        f"axial_dist_mean={app['axial_dist_mean']:+.4f}m  "
                        f"axial_off_mean={app['axial_off_mean']:.4f}m  "
                        f"axial_off_p90={app['axial_off_p90']:.4f}m  "
                        f"radial_err_mean={app['radial_err_mean']:.4f}m  "
                        f"radial_err_p90={app['radial_err_p90']:.4f}m")
        if wandb_run is not None:
            log_dict = {
                "epoch": epoch + 1, "env_steps": total_env_steps,
                "J": J, "R": R, "best_J": best_J, "best_score": best_score,
                "eval_ep_len": ep_len,
                "eval_success_rate": m["hold_success_rate"],
                "eval_max_hold_mean": m["max_hold_mean"],
                "eval_in_thresh_rate": m["in_thresh_rate"],
                "eval_final_in_thresh_rate": m["final_in_thresh_rate"],
                "eval_pos_err_mean": m["pos_err_mean"],
                "eval_axis_err_mean": m["axis_err_mean"],
                "alpha": agent._alpha.item(),
                "absorb_per_epoch": absorb_epoch,
            }
            log_dict.update({
                "eval_clearance_mean": clr["clearance_mean"],
                "eval_clearance_p10": clr["clearance_p10"],
                "eval_clearance_min": clr["clearance_min"],
                "eval_clearance_rate_neg2cm": clr["clearance_rate_at_-2cm"],
                "eval_clearance_rate_0cm": clr["clearance_rate_at_+0cm"],
                "eval_clearance_rate_2cm": clr["clearance_rate_at_+2cm"],
                "eval_clearance_rate_5cm": clr["clearance_rate_at_+5cm"],
            })
            if "hold_success_rate_with_clearance" in m:
                log_dict.update({
                    "eval_success_rate_with_clearance": m["hold_success_rate_with_clearance"],
                    "eval_max_hold_full_mean": m["max_hold_full_mean"],
                    "eval_clearance_pass_rate": m["clearance_pass_rate"],
                    "eval_clearance_threshold_used": m["clearance_threshold_used"],
                })
            if "hold_success_rate_m2d" in m:
                log_dict.update({
                    "eval_success_rate_m2d": m["hold_success_rate_m2d"],
                    "eval_max_hold_m2d_mean": m["max_hold_m2d_mean"],
                    "eval_m2d_in_thresh_rate": m["m2d_in_thresh_rate"],
                    "eval_m2d_final_in_thresh_rate": m["m2d_final_in_thresh_rate"],
                    "eval_final_window_success_rate_m2d": (
                        m.get("final_window_success_rate_m2d", float("nan"))
                    ),
                    "eval_final_window_in_thresh_rate_m2d": (
                        m.get("final_window_in_thresh_rate_m2d", float("nan"))
                    ),
                    "eval_final_window_axis_err_mean_m2d": (
                        m.get("final_window_axis_err_mean_m2d", float("nan"))
                    ),
                    "eval_final_window_axial_off_mean_m2d": (
                        m.get("final_window_axial_off_mean_m2d", float("nan"))
                    ),
                    "eval_final_window_radial_err_mean_m2d": (
                        m.get("final_window_radial_err_mean_m2d", float("nan"))
                    ),
                    "eval_final_window_clearance_mean_m2d": (
                        m.get("final_window_clearance_mean_m2d", float("nan"))
                    ),
                    "eval_m2d_axis_pass_rate": m["m2d_axis_pass_rate"],
                    "eval_m2d_axial_pass_rate": m["m2d_axial_pass_rate"],
                    "eval_m2d_radial_pass_rate": m["m2d_radial_pass_rate"],
                    "eval_axial_off_mean": m.get("axial_off_mean", float("nan")),
                    "eval_radial_err_mean": m.get("radial_err_mean", float("nan")),
                    "eval_m2d_final_axis_err_mean": (
                        m.get("m2d_final_axis_err_mean", float("nan"))
                    ),
                    "eval_m2d_final_axial_off_mean": (
                        m.get("m2d_final_axial_off_mean", float("nan"))
                    ),
                    "eval_m2d_final_radial_err_mean": (
                        m.get("m2d_final_radial_err_mean", float("nan"))
                    ),
                    "eval_m2d_final_clearance_mean": (
                        m.get("m2d_final_clearance_mean", float("nan"))
                    ),
                    "eval_m2d_success_final_axis_err_mean": (
                        m.get("m2d_success_final_axis_err_mean", float("nan"))
                    ),
                    "eval_m2d_success_final_axial_off_mean": (
                        m.get("m2d_success_final_axial_off_mean", float("nan"))
                    ),
                    "eval_m2d_success_final_radial_err_mean": (
                        m.get("m2d_success_final_radial_err_mean", float("nan"))
                    ),
                    "eval_m2d_success_final_clearance_mean": (
                        m.get("m2d_success_final_clearance_mean", float("nan"))
                    ),
                })
            if math.isfinite(app["axial_off_mean"]):
                log_dict.update({
                    "eval_axial_dist_mean": app["axial_dist_mean"],
                    "eval_axial_off_p90": app["axial_off_p90"],
                    "eval_radial_err_p90": app["radial_err_p90"],
                })
            wandb_run.log(log_dict, step=epoch + 1)

    logger.info(f"训练完成. best J = {best_J:.3f}  best_score = {best_score:.3f}")
    if wandb_run is not None:
        wandb_run.summary["best_J"] = best_J
        wandb_run.summary["best_score"] = best_score
        wandb_run.finish()
    mdp.stop()


if __name__ == "__main__":
    main()
