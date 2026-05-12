"""Stage 3 v4 reward dry-run: 加载 Stage 2 ckpt (34→41 partial copy),
deterministic rollout, 打印各 reward component 的 mean/std/min/max, 不做任何 fit.

目的: v4 reward 改动了 6 个分量 (depth V-shape, radial 全部 max, anchor 拆 pos+radial),
直接 200 ep 训练发现量级写错就是 5 小时浪费. 这个脚本 5 分钟跑出每项分量的实际数值,
人工确认:
    1. 各分量 sign 是否对 (depth V-shape 应该 d=d_target 时最大 ≈ +0.15)
    2. 各分量量级比例是否合理 (主导项跟次要项不要差 50x)
    3. Phase A/B/C schedule 切换时分量数值是否符合预期 (radial_anchor 在 Phase A = 0)

actor warm-start 走 train_sac.py 同款 partial copy (warmstart_actor_with_partial_copy):
Stage 2 (34D) → Stage 3 (41D), 第一层前 34 列拷, 新 7 列 zero-init. epoch 0 actor 输出
严格等价 Stage 2 actor — dry-run 直接验证训练 path 上的 actor 行为, 不是另一份 actor.

跑法:
    python scripts/archive/dryrun_stage3_v4.py \
        --agent_path results/S2_oneshot_ep194_seed0_best_agent.msh \
        --headless --exclude_ee

可选 --phase {A,B,C,all} 切换 schedule 阶段 (默认 all).

reward components (v4 设计, 见 envs/dual_arm_peg_hole_env.py:_compute_stage3_reward_components):
    r_depth          V-shape: +w_depth · (d_target - |d - d_target|), clamp(min=-cap)
    r_axis           -w_axis · axis_gate(pos_err) · axis_err
    r_radial_base    -w_radial · radial_max
    r_radial_close   -w_rad2_eff · sigmoid((d-d_g)/s_g) · radial_max
    r_pos_anchor     -w_pos_anchor_eff · pos_err           (Phase A 主)
    r_radial_anchor  -w_radial_anchor_eff · radial_max     (Phase B/C)
    r_insert         +w_insert_eff · 1[d>d_ins ∧ radial_max<r_ins ∧ axis<a_ins]
    r_joint_limit    -w_joint_limit · joint_limit_norm
    r_action         -w_action · ||raw_action||²
    r_home           -w_home · home_norm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Archived under scripts/archive/, so parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from networks import ActorNetwork, CriticNetwork
from scripts._eval_utils import (
    deterministic_policy,
    parse_home_weights,
    resolve_eval_episode_count,
    warmstart_actor_with_partial_copy,
)


# 跟 train_sac.py 同源 — SAC 构造常量, 别在这里偏离.
INITIAL_REPLAY_SIZE = 10_000
MAX_REPLAY_SIZE = 500_000
BATCH_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT
                               / "results/S2_oneshot_ep194_seed0_best_agent.msh"))
    p.add_argument("--n_episodes", type=int, default=None,
                   help="默认取 num_envs (16). 必须能被 num_envs 整除.")
    p.add_argument("--num_envs", type=int, default=16,
                   help="必须 >=2 (num_envs=1 触发 cloner bug).")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=0,
                   help="torch/np/mdp 全部 seed, 让 dry-run 可复现 (跟 train_sac 同款 seed=0 入口).")
    p.add_argument("--horizon", type=int, default=200,
                   help="env episode 长度. 默认 200, 跟 Stage 3 训练命令一致, 不是 env 默认 150.")
    p.add_argument("--phase", type=str, default="all",
                   choices=("A", "B", "C", "all"),
                   help="测试哪个 schedule 阶段. 默认 all (A, B 中点, C 各跑一次).")

    # Stage 3 v4 reward params (沿用 env 默认, 显式 CLI 让 dry-run 跟训练命令对齐)
    p.add_argument("--rew_depth", type=float, default=None)
    p.add_argument("--rew_radial", type=float, default=None)
    p.add_argument("--rew_radial_close", type=float, default=None)
    p.add_argument("--rew_insert", type=float, default=None)
    p.add_argument("--rew_pos_anchor", type=float, default=None)
    p.add_argument("--rew_radial_anchor", type=float, default=None)
    p.add_argument("--stage3_d_target", type=float, default=None)
    p.add_argument("--stage3_depth_penalty_cap", type=float, default=None)
    p.add_argument("--stage3_d_g", type=float, default=None)
    p.add_argument("--stage3_s_g", type=float, default=None)
    p.add_argument("--stage3_d_ins", type=float, default=None)
    p.add_argument("--stage3_r_ins", type=float, default=None)
    p.add_argument("--stage3_a_ins", type=float, default=None)
    p.add_argument("--stage3_rad2_ramp_start", type=int, default=None)
    p.add_argument("--stage3_rad2_ramp_end", type=int, default=None)

    # Stage 2 兼容参数 (warm-start ckpt 训练时的设置, axis/anchor 必须对齐, 否则
    # axis_gate / axis_err 量级跟 ckpt 学到的 actor 行为对不上).
    p.add_argument("--rew_axis", type=float, default=1.0)
    p.add_argument("--rew_home", type=float, default=0.0005)
    p.add_argument("--home_weights", type=parse_home_weights, default="1,1,1,1,0.75,0.5,0.5")
    p.add_argument("--axis_gate_radius", type=float, default=float("inf"))
    p.add_argument("--exclude_ee", action="store_true",
                   help="Stage 3 推荐传 (peg-hole 接触不被 hard absorb 误杀).")

    # SAC cold-create 参数 (dry-run 不 fit, 这些只影响 actor 网络初始化, 不影响行为).
    # 默认值跟 Stage 2 oneshot 配方对齐, 让 SAC 内部 bookkeeping 不报错.
    p.add_argument("--lr_actor", type=float, default=5e-5)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--target_entropy", type=float, default=-10.0)
    p.add_argument("--critic_warmup_transitions", type=int, default=INITIAL_REPLAY_SIZE)
    return p.parse_args()


PHASE_LABELS = {
    "A": ("Phase A: pos_anchor 主, radial_anchor=0, rad2=0, insert=0",
          lambda mdp: 0),
    "B": ("Phase B 中点: pos→radial 50/50, rad2 半 peak, insert=0",
          lambda mdp: (mdp._stage3_rad2_ramp_start + mdp._stage3_rad2_ramp_end) // 2),
    "C": ("Phase C: pos=0, radial_anchor=peak, rad2=peak, insert=peak",
          lambda mdp: mdp._stage3_rad2_ramp_end + 5),
}


def _summarize_component(name, arr):
    """打印一个 reward component 的统计量. arr shape: (n_steps,) flatten."""
    if arr.size == 0:
        return f"  {name:18s} <empty>"
    return (f"  {name:18s} "
            f"mean={arr.mean():+.5f}  std={arr.std():.5f}  "
            f"min={arr.min():+.5f}  max={arr.max():+.5f}  "
            f"|.| sum={np.abs(arr).sum():+.4f}")


def run_phase(mdp, agent, core, args, phase_key):
    label, epoch_fn = PHASE_LABELS[phase_key]
    epoch = epoch_fn(mdp)
    mdp.set_stage3_epoch(epoch)
    print()
    print("=" * 80)
    print(f"[PHASE {phase_key}] stage3_epoch={epoch}  {label}")
    print(f"  effective weights: "
          f"w_pos_anchor={mdp._stage3_w_pos_anchor_eff:.4f}  "
          f"w_radial_anchor={mdp._stage3_w_radial_anchor_eff:.4f}  "
          f"w_rad2={mdp._stage3_w_rad2_eff:.3f}  "
          f"w_insert={mdp._stage3_w_insert_eff:.3f}")
    print("=" * 80)

    # 收集 reward components: monkey-patch reward(), 在 env 算之前抓 components.
    # 用 _compute_stage3_reward_components 保证跟 training reward 同源.
    buf = {}
    original_reward = mdp.reward

    def hooked_reward(obs, action, next_obs, absorbing):
        if mdp._stage3:
            comps = mdp._compute_stage3_reward_components(next_obs)
            for k, v in comps.items():
                buf.setdefault(k, []).append(v.detach().cpu().numpy().reshape(-1))
        return original_reward(obs, action, next_obs, absorbing)

    mdp.reward = hooked_reward
    try:
        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_episodes, quiet=True)
    finally:
        mdp.reward = original_reward

    # 打印 per-component 统计 + total
    component_order = (
        "r_depth", "r_axis", "r_radial_base", "r_radial_close",
        "r_pos_anchor", "r_radial_anchor", "r_insert",
        "r_joint_limit", "r_action", "r_home",
    )
    n_samples = sum(a.size for a in buf['r_depth']) if 'r_depth' in buf else 0
    print(f"\n  reward components (n_step·n_env samples = {n_samples}):")
    total = None
    for name in component_order:
        if name not in buf:
            continue
        arr = np.concatenate(buf[name])
        print(_summarize_component(name, arr))
        total = arr if total is None else total + arr
    if total is not None:
        print(_summarize_component("TOTAL_reward", total))

    # 对比: J / R 看 deterministic rollout 的累计 reward
    J = float(torch.mean(dataset.discounted_return).item())
    R = float(torch.mean(dataset.undiscounted_return).item())
    print(f"\n  rollout summary: J={J:+.3f}  R={R:+.3f}  "
          f"horizon={mdp.info.horizon}  n_episodes={args.n_episodes}")


def cold_create_sac(mdp, args):
    """跟 train_sac.py 的 _cold_create_sac 同款构造. dry-run 不 fit, 但 SAC 需要
    完整 actor+critic+alpha 才能 init / 接 VectorCore. 默认值跟 Stage 2 oneshot
    对齐, 不影响 deterministic rollout 行为."""
    from mushroom_rl.algorithms.actor_critic import SAC
    obs_dim = mdp.info.observation_space.shape[0]
    act_dim = mdp.info.action_space.shape[0]
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
        target_entropy=args.target_entropy,
    )


def main():
    args = parse_args()
    # 跟 train_sac.py:200 同款 seeding 入口, 让 dry-run 可复现.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes")

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=args.headless,
        horizon=args.horizon,           # 跟 Stage 3 训练 (200) 对齐, env 默认 150
        stage3=True,
        use_axis_resid_obs=True,        # stage3 隐含, 显式让 actor obs dim 对齐
        rew_axis=args.rew_axis,
        rew_home=args.rew_home,
        home_weights=args.home_weights,
        axis_gate_radius=args.axis_gate_radius,
    )
    if args.exclude_ee:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True
    for key in ("rew_depth", "rew_radial", "rew_radial_close", "rew_insert",
                "rew_pos_anchor", "rew_radial_anchor",
                "stage3_d_target", "stage3_depth_penalty_cap",
                "stage3_d_g", "stage3_s_g",
                "stage3_d_ins", "stage3_r_ins", "stage3_a_ins",
                "stage3_rad2_ramp_start", "stage3_rad2_ramp_end"):
        v = getattr(args, key)
        if v is not None:
            env_kwargs[key] = v

    print(f"[DRYRUN env_kwargs] {env_kwargs}")
    mdp = DualArmPegHoleEnv(**env_kwargs)
    mdp.seed(args.seed)                  # 跟 train_sac.py:273 同款
    print(f"[DRYRUN] seed={args.seed}  horizon={mdp.info.horizon}")
    print(f"[DRYRUN] stage3 v4 reward — peak weights:")
    print(f"  w_depth={mdp._w_depth:.2f}  w_radial={mdp._w_radial:.2f}  "
          f"w_radial_close_peak={mdp._w_radial_close_peak:.2f}  "
          f"w_insert_peak={mdp._w_insert_peak:.2f}")
    print(f"  w_pos_anchor_peak={mdp._w_pos_anchor_peak:.3f}  "
          f"w_radial_anchor_peak={mdp._w_radial_anchor_peak:.3f}")
    print(f"[DRYRUN] depth V-shape: d_target={mdp._stage3_d_target:+.3f}m  "
          f"penalty_cap={mdp._stage3_depth_penalty_cap:.3f}  "
          f"d_g={mdp._stage3_d_g:+.3f}  s_g={mdp._stage3_s_g:.3f}")
    print(f"[DRYRUN] insert thresh: d_ins={mdp._stage3_d_ins:+.3f}  "
          f"r_ins={mdp._stage3_r_ins:.4f}  a_ins={mdp._stage3_a_ins:.2f}")
    print(f"[DRYRUN] schedule: ramp_start={mdp._stage3_rad2_ramp_start}  "
          f"ramp_end={mdp._stage3_rad2_ramp_end}")

    # IsaacSim 启动后才能导入 mushroom_rl
    from mushroom_rl.core import VectorCore

    # Cold-create 41D SAC, 然后从 Stage 2 ckpt (34D) partial copy actor.
    # 跟 train_sac.py 同款 helper 保证 dry-run 验证的就是训练 path 上的 actor.
    agent = cold_create_sac(mdp, args)
    load_path = Path(args.agent_path)
    if not load_path.is_file():
        raise FileNotFoundError(f"--agent_path 不存在: {load_path}")
    mode = warmstart_actor_with_partial_copy(agent, load_path)
    print(f"[DRYRUN warm-start] {mode} from {load_path}")

    core = VectorCore(agent, mdp)
    if args.phase == "all":
        for phase_key in ("A", "B", "C"):
            run_phase(mdp, agent, core, args, phase_key)
    else:
        run_phase(mdp, agent, core, args, args.phase)

    print()
    print("=" * 80)
    print("[DRYRUN] 检查清单:")
    print("  1. r_depth 在 Phase A: 平均应该是负的 (Stage 2 ckpt 在 d≈-0.05, V-shape 谷底).")
    print("     d_target=0.03 时, d=-0.05 → r_depth = 5 * (0.03 - 0.08) = -0.25")
    print("  2. r_radial_base: ≈ -1.0 * radial_max (Stage 2 init radial_max≈0.025 → ≈ -0.025)")
    print("  3. r_pos_anchor (Phase A): -peak * pos_err ≈ -0.15 * 0.11 = -0.0165")
    print("  4. r_radial_anchor (Phase A): 应该 = 0 (Phase A schedule 关闭).")
    print("  5. r_radial_close (Phase A): 应该 = 0 (rad2_eff=0).")
    print("  6. r_insert: Phase A/B = 0; Phase C 当前 ckpt 大概率仍 = 0 (radial_max>0.005).")
    print("  7. 各项量级比例: 不要某项 |mean| 比另一项 |mean| 差 50x.")
    print("     主导项 r_depth, r_radial_base, r_pos_anchor 应该在 0.01-0.3/step 量级.")
    print("=" * 80)


if __name__ == "__main__":
    main()
