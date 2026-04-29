"""评估训练好的 SAC agent (32 维 obs, stage flag 化 reward).

eval 时 --rew_axis / --success_axis_threshold 应与训练时保持一致, 否则
success / hold-N 触发条件不同, 数字没有可比性.

运行:
    conda activate safe_rl
    # M1' (pos-only)
    python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \\
        --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50

    # M2a/M2b (pos + axis)
    python scripts/eval_sac.py --headless --num_envs 16 --n_episodes 64 \\
        --preinsert_success_pos_threshold 0.10 --terminal_hold_bonus 50 \\
        --rew_axis 2.0 --success_axis_threshold 0.5
"""

import argparse
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import (
    compute_hold_metrics,
    deterministic_policy,
    resolve_eval_episode_count,
    summarize_approach,
    summarize_clearance,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--n_episodes", type=int, default=None,
                   help="评估 episode 数. 默认自动取 num_envs, 并要求能被 num_envs 整除")
    p.add_argument("--num_envs", type=int, default=16,
                   help="与训练保持一致 (16). num_envs=1 会触发 cloner bug.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--initial_joint_noise", type=float, default=None,
                   help="覆盖 env 默认 reset 关节噪声. 应传与 train 相同的值")
    p.add_argument("--action_scale", type=float, default=None,
                   help="覆盖 env 动作缩放. 应传与 train 相同的值.")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None,
                   help="覆盖 env 默认 preinsert 位置成功阈值 (env 默认 0.10m). "
                        "应传与 train 相同的值.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 默认 preinsert offset. 应传与 train 相同的值")
    p.add_argument("--rew_pos", type=float, default=None,
                   help="覆盖 env 的 pos_err reward 权重. **必须与 train 一致**.")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重. eval 会跑 env reward 算 J/R, 所以 "
                        "**必须与 train 一致** 才能与训练曲线比较 (hold metrics 只看 "
                        "success_axis_threshold, 不受这个权重影响).")
    p.add_argument("--rew_action", type=float, default=None,
                   help="覆盖 env 的动作 L2 惩罚权重. **必须与 train 一致**, 否则 J/R "
                        "数值与训练时不可比.")
    p.add_argument("--rew_success", type=float, default=None,
                   help="覆盖 env 的 per-step success bonus. **必须与 train 一致**.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="覆盖 env 默认 axis_err success 阈值. **必须与 train 时一致**, "
                        "否则 hold_success_rate / final_in_thresh_rate 数字不可比.")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "**train 时若启用了它, eval 也必须传同样的值**.")
    # M2c clearance — eval 必须与 train 一致 (success_mask + absorbing 触发条件
    # 都依赖). reward 数字也依赖 rew_clearance.
    p.add_argument("--rew_clearance", type=float, default=None,
                   help="M2c clearance penalty 权重. 与 train 一致.")
    p.add_argument("--clearance_soft", type=float, default=None,
                   help="M2c soft threshold (m). 与 train 一致 — 影响 success_mask.")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="M2c hard threshold (m). 与 train 一致 — 影响 absorbing.")
    p.add_argument("--clearance_penalty_scale", type=float, default=None,
                   help="M2c penalty 归一化 scale (m). 与 train 一致.")
    p.add_argument("--clearance_success_threshold", type=float, default=None,
                   help="M2c success gate 阈值 (m). 与 train 一致 — 影响 hold metric "
                        "和 success_mask, 默认沿用 clearance_soft.")
    p.add_argument("--log_clearance", action="store_true",
                   help="metric-only: 即使 M2c 关闭也每步算 clearance + 输出分布. "
                        "用于在 M2b checkpoint 上看 clearance, 不改 reward / hold metric.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="M2c sphere proxy 的 arm/link 半径 (m). 默认 0.06. "
                        "评估时应与训练或对比实验设置一致.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="M2c sphere proxy 的 EE/finger 半径 (m). 默认 0.03. "
                        "评估时应与训练或对比实验设置一致.")
    p.add_argument("--use_m2d_obs", action="store_true",
                   help="把 env obs 切到 37 维 M2d 几何观测. 评估 37 维 M2d checkpoint "
                        "时必须传; 评估 32 维 M2c checkpoint 时不要传.")
    p.add_argument("--rew_axial_off", type=float, default=None,
                   help="M2d 轴向偏差 reward 权重. 与 train 一致.")
    p.add_argument("--rew_radial", type=float, default=None,
                   help="M2d 横向偏差 reward 权重. 与 train 一致.")
    p.add_argument("--axial_success_threshold", type=float, default=None,
                   help="M2d 轴向 success 阈值 (m). 与 train 一致.")
    p.add_argument("--radial_success_threshold", type=float, default=None,
                   help="M2d 横向 success 阈值 (m). 与 train 一致.")
    p.add_argument("--log_axial_radial", action="store_true",
                   help="metric-only: 注入 axial_off/radial_err 到 dataset.info.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="验证 success 定义: episode 内至少出现连续 N 步都在阈值内.")
    p.add_argument("--final_window_steps", type=int, default=30,
                   help="M2d 稳停指标窗口长度: episode 最后 N 步全部满足 "
                        "axis∧axial∧radial∧clearance 才算 final-window success.")
    p.add_argument("--stochastic", action="store_true",
                   help="使用 SAC 采样策略评估. 默认使用 deterministic tanh(mu)")
    return p.parse_args()


def main():
    args = parse_args()
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes"
    )

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(num_envs=args.num_envs, headless=args.headless)
    for key in ("action_scale", "initial_joint_noise",
                "preinsert_success_pos_threshold", "preinsert_offset",
                "rew_pos", "rew_axis", "rew_action", "rew_success",
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
    print(f"[EVAL ENV] {env_kwargs}")

    mdp = DualArmPegHoleEnv(**env_kwargs)
    if mdp._m2d_active and mdp._terminal_hold_bonus > 0.0:
        print("[EVAL WARNING] M2d active 且 terminal_hold_bonus>0: episode 会在 hold "
              "后截断, final-window 稳停指标会受到终止时刻影响.")

    from mushroom_rl.core import Agent, VectorCore
    agent = Agent.load(args.agent_path)
    core = VectorCore(agent, mdp)

    if args.stochastic:
        dataset = core.evaluate(n_episodes=args.n_episodes,
                                render=not args.headless, quiet=False)
    else:
        with deterministic_policy(agent):
            dataset = core.evaluate(n_episodes=args.n_episodes,
                                    render=not args.headless, quiet=False)
    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    print(f"J(γ)={J:.3f}  R={R:.3f}")

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
    print(
        f"hold_success_rate={m['hold_success_rate']:.3f} "
        f"(>= {args.hold_success_steps} consecutive steps, pos∧axis only)  "
        f"max_hold_mean={m['max_hold_mean']:.1f}  "
        f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
        f"final_in_thresh_rate={m['final_in_thresh_rate']:.3f}  "
        f"pos_err_mean={m['pos_err_mean']:.4f}m  "
        f"axis_err_mean={m['axis_err_mean']:.4f}"
    )
    if "hold_success_rate_with_clearance" in m:
        print(
            f"hold_success_rate_with_clearance="
            f"{m['hold_success_rate_with_clearance']:.3f}  "
            f"(gate={m['clearance_threshold_used']:+.4f}m)  "
            f"max_hold_full_mean={m['max_hold_full_mean']:.1f}  "
            f"clearance_pass_rate={m['clearance_pass_rate']:.3f}"
        )
    if "hold_success_rate_m2d" in m:
        print(
            f"hold_success_rate_m2d={m['hold_success_rate_m2d']:.3f} "
            f"(axis∧axial∧radial∧clearance, "
            f"axial<{m.get('axial_threshold_used', float('nan')):.4f}m, "
            f"radial<{m.get('radial_threshold_used', float('nan')):.4f}m)  "
            f"max_hold_m2d_mean={m['max_hold_m2d_mean']:.1f}  "
            f"m2d_in_thresh_rate={m['m2d_in_thresh_rate']:.3f}  "
            f"final_m2d_in_thresh_rate={m['m2d_final_in_thresh_rate']:.3f}\n"
            f"m2d pass rates: axis={m['m2d_axis_pass_rate']:.3f}  "
            f"axial={m['m2d_axial_pass_rate']:.3f}  "
            f"radial={m['m2d_radial_pass_rate']:.3f}  "
            f"axial_off_mean={m.get('axial_off_mean', float('nan')):.4f}m  "
            f"radial_err_mean={m.get('radial_err_mean', float('nan')):.4f}m"
        )
        print(
            "m2d terminal geometry: "
            f"final_axis={m.get('m2d_final_axis_err_mean', float('nan')):.4f}  "
            f"final_axial={m.get('m2d_final_axial_off_mean', float('nan')):.4f}m  "
            f"final_radial={m.get('m2d_final_radial_err_mean', float('nan')):.4f}m  "
            f"final_clearance={m.get('m2d_final_clearance_mean', float('nan')):+.4f}m"
        )
        print(
            "m2d success-final geometry: "
            f"axis={m.get('m2d_success_final_axis_err_mean', float('nan')):.4f}  "
            f"axial={m.get('m2d_success_final_axial_off_mean', float('nan')):.4f}m  "
            f"radial={m.get('m2d_success_final_radial_err_mean', float('nan')):.4f}m  "
            f"clearance="
            f"{m.get('m2d_success_final_clearance_mean', float('nan')):+.4f}m"
        )
        print(
            f"m2d final-window stability (last {args.final_window_steps} steps, "
            "all steps counted): "
            f"success_rate={m.get('final_window_success_rate_m2d', 0.0):.3f}  "
            f"in_thresh_rate={m.get('final_window_in_thresh_rate_m2d', 0.0):.3f}  "
            f"axis={m.get('final_window_axis_err_mean_m2d', float('nan')):.4f}  "
            f"axial={m.get('final_window_axial_off_mean_m2d', float('nan')):.4f}m  "
            f"radial={m.get('final_window_radial_err_mean_m2d', float('nan')):.4f}m  "
            f"clearance={m.get('final_window_clearance_mean_m2d', float('nan')):+.4f}m"
        )
    print(
        f"clearance (sphere proxy): "
        f"mean={clr['clearance_mean']:+.4f}m  "
        f"p10={clr['clearance_p10']:+.4f}m  "
        f"min={clr['clearance_min']:+.4f}m  "
        f"  per-step rate "
        f"≥-2cm={clr['clearance_rate_at_-2cm']:.3f}  "
        f"≥0={clr['clearance_rate_at_+0cm']:.3f}  "
        f"≥+2cm={clr['clearance_rate_at_+2cm']:.3f}  "
        f"≥+5cm={clr['clearance_rate_at_+5cm']:.3f}"
    )
    if math.isfinite(app["axial_off_mean"]):
        print(
            f"approach (M2d geometry): "
            f"axial_dist_mean={app['axial_dist_mean']:+.4f}m  "
            f"axial_off_mean={app['axial_off_mean']:.4f}m  "
            f"axial_off_p90={app['axial_off_p90']:.4f}m  "
            f"radial_err_mean={app['radial_err_mean']:.4f}m  "
            f"radial_err_p90={app['radial_err_p90']:.4f}m"
        )

    mdp.stop()


if __name__ == "__main__":
    main()
