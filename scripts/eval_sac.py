"""评估训练好的 SAC agent.

Stage 1/2 (Lagrangian baseline): 32/34 维 obs.
Geom Stage 1g/2g/3g (主线): 41 维 obs + geom metrics.

eval 时 --rew_axis / --success_axis_threshold / --use_axis_resid_obs 等应与训练
时保持一致, 否则 success / hold-N 触发条件不同, 数字没有可比性.

Geom ckpt eval 必须传 --geom_stage {prepos,preaxis,insert} (env 切到 41D obs).

正式 eval 命令见 README.md "Eval 命令" 段.
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import (
    compute_geom_metrics,
    compute_hold_metrics,
    deterministic_policy,
    parse_home_weights,
    resolve_eval_episode_count,
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
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None,
                   help="覆盖 env 默认 preinsert 位置成功阈值 (env 默认 0.10m). "
                        "应传与 train 相同的值.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 默认 preinsert offset. 应传与 train 相同的值")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重. hold 指标不依赖 reward, 但 J/R 会依赖; "
                        "应与 train 一致. 默认 0 = Stage 1 行为.")
    p.add_argument("--rew_success", type=float, default=None,
                   help="覆盖 env 的 per-step full_success (pos∧axis) bonus. "
                        "评估 J/R 时应与 train 一致.")
    p.add_argument("--rew_pos_success", type=float, default=None,
                   help="覆盖 env 的 pos-only success bonus. 应与 train 一致.")
    p.add_argument("--axis_gate_radius", type=float, default=None,
                   help="覆盖 env 的 axis 距离门控半径. 应与 train 一致.")
    p.add_argument("--rew_home", type=float, default=None,
                   help="覆盖 env 的 home regularizer 权重. 评估 J/R 时应与 train 一致.")
    p.add_argument("--home_weights", type=parse_home_weights, default=None,
                   help="home regularizer 的逐关节权重. 应与 train 一致. "
                        "接受 7 维单臂或 14 维完整权重, 逗号/空格分隔.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="覆盖 env 默认 axis_err success 阈值. **必须与 train 时一致**, "
                        "否则 hold_success_rate / final_in_thresh_rate 数字不可比.")
    p.add_argument("--terminal_hold_bonus", type=float, default=None,
                   help="hold-N 步成功后的终结 bonus + episode 终止. "
                        "**train 时若启用了它, eval 也必须传同样的值**.")
    p.add_argument("--hold_success_steps", type=int, default=10,
                   help="验证 success 定义: episode 内至少出现连续 N 步都在阈值内.")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="覆盖 env 的 sphere-proxy 自碰撞兜底阈值. 应与 train 时一致, "
                        "否则碰撞触发率不同, success / J 数字不可比. 关闭写 --clearance_hard=-inf.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="覆盖 arm sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="覆盖 EE sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--exclude_ee_from_physx_self_collision", action="store_true",
                   help="应与 geom insert train 一致: PhysX self-collision 分组排除 EE link, "
                        "避免 peg-hole 正常接触被算作 hard absorbing.")
    p.add_argument("--use_axis_resid_obs", action="store_true",
                   help="34 维 obs (axis_resid 替换 axis_dot). **必须与 train 一致**, "
                        "否则 actor 输入维度对不上加载会失败.")
    p.add_argument("--horizon", type=int, default=None,
                   help="覆盖 env 默认 horizon. geom insert 训练用 200, 不传走 env 默认 150.")
    p.add_argument("--stochastic", action="store_true",
                   help="使用 SAC 采样策略评估. 默认使用 deterministic tanh(mu)")

    # Geometric preinsert eval. Must match the checkpoint stage so obs=41D.
    p.add_argument("--geom_stage", type=str, default=None,
                   choices=("prepos", "preaxis", "insert"),
                   help="启用 geom env (41D obs + geom metrics). Geom ckpt 必传.")
    p.add_argument("--geom_eval_epoch", type=int, default=None,
                   help="eval 前调用 set_geom_epoch(epoch). 默认: insert 用 ramp_end, "
                        "prepos/preaxis 用 0.")
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
    p.add_argument("--geom_pen_th", type=float, default=None)
    p.add_argument("--geom_soft_penetration_sigma", type=float, default=None)
    # Alignment-gated progress reward
    p.add_argument("--rew_geom_progress", type=float, default=None)
    p.add_argument("--geom_gate_radial_sigma", type=float, default=None)
    p.add_argument("--geom_gate_axis_sigma", type=float, default=None)
    # Penetration-aware reward
    p.add_argument("--rew_geom_penetration", type=float, default=None)
    p.add_argument("--geom_gate_penetration_sigma", type=float, default=None)
    p.add_argument("--cost_signal", type=str, default=None,
                   choices=["collision", "penetration"])
    p.add_argument("--geom_progress_floor", type=float, default=None)
    p.add_argument("--rew_geom_advance", type=float, default=None)
    p.add_argument("--geom_d_gate_mode", type=str, default=None,
                   choices=["off", "alignment"])
    p.add_argument("--rew_geom_bad_entry", type=float, default=None)
    p.add_argument("--geom_bad_entry_radial_safe", type=float, default=None)
    p.add_argument("--geom_bad_entry_axis_safe", type=float, default=None)
    p.add_argument("--geom_bad_entry_pen_safe", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes"
    )
    from envs import DualArmPegHoleEnv

    env_kwargs = dict(num_envs=args.num_envs, headless=args.headless)
    if args.horizon is not None:
        env_kwargs["horizon"] = args.horizon
    for key in ("initial_joint_noise", "preinsert_success_pos_threshold",
                "preinsert_offset", "rew_axis", "rew_success", "rew_pos_success",
                "rew_home", "home_weights", "axis_gate_radius",
                "success_axis_threshold", "terminal_hold_bonus",
                "clearance_hard", "proxy_arm_radius", "proxy_ee_radius",
                # Geom reward / threshold params.
                "geom_stage", "geom_d_target_neg", "geom_d_target_pos",
                "geom_d_target_ramp_start", "geom_d_target_ramp_end",
                "rew_geom_d", "rew_geom_radial_tip", "rew_geom_radial_max",
                "rew_geom_axis", "geom_d_sat", "geom_radial_sat",
                "rew_geom_soft_success", "geom_soft_d_sigma",
                "geom_soft_radial_sigma", "geom_soft_axis_sigma",
                "geom_d_th", "geom_r_tip_th", "geom_r_max_th",
                "geom_axis_th", "geom_insert_d_ins", "geom_insert_r_max_th",
                "geom_pen_th", "geom_soft_penetration_sigma",
                "rew_geom_progress", "geom_gate_radial_sigma",
                "geom_gate_axis_sigma",
                "rew_geom_penetration", "geom_gate_penetration_sigma", "cost_signal",
                "geom_progress_floor", "rew_geom_advance",
                "geom_d_gate_mode", "rew_geom_bad_entry",
                "geom_bad_entry_radial_safe", "geom_bad_entry_axis_safe",
                "geom_bad_entry_pen_safe"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    # geom_stage 下默认同步 preinsert_offset = abs(geom_d_target_neg), 满足 env 的
    # 一致性 fail-fast (env __init__ 现在强制 preinsert_offset == abs(geom_d_target_neg),
    # 否则 obs pos_vec target 和 reward d_target 是不同深度). 用户显式传不一致的值时
    # env 会 raise — CLI 不绕过 env invariant.
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
    env_kwargs["success_hold_steps"] = args.hold_success_steps
    print(f"[EVAL ENV] {env_kwargs}")

    mdp = DualArmPegHoleEnv(**env_kwargs)

    if args.geom_stage is not None:
        if args.geom_eval_epoch is not None:
            geom_eval_epoch = args.geom_eval_epoch
        elif mdp._geom_stage == "insert":
            geom_eval_epoch = mdp._geom_d_target_ramp_end
        else:
            geom_eval_epoch = 0
        mdp.set_geom_epoch(geom_eval_epoch)
        print(f"[EVAL geom] stage={mdp._geom_stage}, "
              f"set_geom_epoch({geom_eval_epoch}). "
              f"d_target_eff={mdp._geom_d_target_eff:+.4f}m  "
              f"thresh: d_th={mdp._geom_d_th:.3f}  "
              f"r_tip_th={mdp._geom_r_tip_th:.3f}  "
              f"r_max_th={mdp._geom_r_max_th:.3f}  "
              f"axis_th={mdp._geom_axis_th:.3f}  "
              f"insert_d_ins={mdp._geom_insert_d_ins:+.3f}  "
              f"insert_r_max_th={mdp._geom_insert_r_max_th:.3f}  "
              f"pen_th={mdp._geom_pen_th*1000:.1f}mm")

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

    m = compute_hold_metrics(dataset, mdp, args.hold_success_steps)
    if mdp._geom_stage is not None:
        # geom 模式下 m 是旧球形 pos<pos_th ∧ axis<axis_th 指标, 跟 geom_* 不是同一套
        # 几何, 并列打印 (hold_success_rate=1.0 vs geom_hold_rate=0.0x) 容易误判 ckpt
        # 真实性能. 跟 train_sac console 行为一致, geom 模式只看下面的 geom_* 块.
        print("[legacy 球形 pos/axis 指标 skipped — geom 模式 ckpt 评估走 geom_* 字段, 见下方]")
    else:
        print(
            f"hold_success_rate={m['hold_success_rate']:.3f} "
            f"(>= {args.hold_success_steps} consecutive steps)  "
            f"max_hold_mean={m['max_hold_mean']:.1f}  "
            f"in_thresh_rate={m['in_thresh_rate']:.3f}  "
            f"final_in_thresh_rate={m['final_in_thresh_rate']:.3f}  "
            f"pos_success_rate={m['pos_success_rate']:.3f}  "
            f"pos_err_mean={m['pos_err_mean']:.4f}m  "
            f"axis_err_mean={m['axis_err_mean']:.4f}  "
            f"axis_gate_mean={m['axis_gate_mean']:.3f}  "
            f"gated_axis_pen={m['gated_axis_penalty_mean']:.3f}"
        )
        if m['pos_in_thresh_count'] > 0:
            print(
                f"  ↳ pos_in_thresh_count={m['pos_in_thresh_count']}  "
                f"axis_err_in_pos_th_mean={m['axis_err_in_pos_thresh_mean']:.4f}  "
                f"axis_err_in_pos_th_min={m['axis_err_in_pos_thresh_min']:.4f}  "
                f"axis_gate_in_pos_th_mean={m['axis_gate_in_pos_thresh_mean']:.3f}"
            )
        else:
            print(f"  ↳ pos_in_thresh_count=0  axis_err_in_pos_th=n/a")

    if args.geom_stage is not None:
        mg = compute_geom_metrics(dataset, mdp, args.hold_success_steps)
        print(
            f"geom ({args.geom_stage} active mask): "
            f"geom_step_rate={mg['geom_step_rate']:.3f}  "
            f"geom_hold_rate={mg['geom_hold_rate']:.3f} "
            f"(>= {args.hold_success_steps} consec steps)  "
            f"geom_max_run_mean={mg['geom_max_run_mean']:.1f}  "
            f"final_success_rate={mg['geom_final_success_rate']:.3f}"
        )
        print(
            f"  ↳ d_target={mg['geom_d_target_mean']:+.4f}m  "
            f"d_err_mean={mg['geom_d_err_mean']:.4f}m  "
            f"d_err_min={mg['geom_d_err_min']:.4f}m  "
            f"radial_tip_min={mg['geom_radial_tip_min']:.4f}m  "
            f"radial_max_min={mg['geom_radial_max_min']:.4f}m  "
            f"axis_err_min={mg['geom_axis_err_min']:.3f}"
        )
        print(
            f"  ↳ mask rates: prepos={mg['geom_prepos_step_rate']:.3f}  "
            f"preaxis={mg['geom_preaxis_step_rate']:.3f}  "
            f"insert={mg['geom_insert_step_rate']:.3f}"
        )
        # entry/final diagnostics: 区分 "approach-then-dwell" vs "tilt-then-align"
        print(
            f"  ↳ entry @ first active mask "
            f"(n_ep={mg['geom_n_ep_with_entry']}): "
            f"d={mg['geom_entry_d_mean']:+.4f}m  "
            f"d_err={mg['geom_entry_d_err_mean']:.4f}m  "
            f"radial_max={mg['geom_entry_radial_max_mean']:.4f}m  "
            f"(max {mg['geom_entry_radial_max_max']:.4f}m)  "
            f"axis_err={mg['geom_entry_axis_err_mean']:.4f}  "
            f"(max {mg['geom_entry_axis_err_max']:.4f})  "
            f"penetration={mg['geom_entry_penetration_mean']*1000:.2f}mm  "
            f"(max {mg['geom_entry_penetration_max']*1000:.2f}mm)"
        )
        print(
            f"  ↳ final @ ep end: "
            f"d={mg['geom_final_d_mean']:+.4f}m  "
            f"d_err={mg['geom_final_d_err_mean']:.4f}m  "
            f"radial_max={mg['geom_final_radial_max_mean']:.4f}m  "
            f"(max {mg['geom_final_radial_max_max']:.4f}m)  "
            f"axis_err={mg['geom_final_axis_err_mean']:.4f}  "
            f"(max {mg['geom_final_axis_err_max']:.4f})  "
            f"penetration={mg['geom_final_penetration_mean']*1000:.2f}mm  "
            f"(max {mg['geom_final_penetration_max']*1000:.2f}mm)"
        )
        # penetration: 物理穿模量 (peg 表面 - hole 内壁), > 0 = 几何上不可能.
        # clamp 到 [0, wall_thickness=4mm], 远场假数据被 radial range check 过滤.
        # clean_step_rate 是 penetration<1e-4 的 step 比例 (= "trajectory 物理合理"的占比).
        print(
            f"  ↳ penetration: max_mean={mg['geom_pen_max_mean']*1000:.2f}mm  "
            f"max_max={mg['geom_pen_max_max']*1000:.2f}mm  "
            f"clean_step_rate={mg['geom_clean_step_rate']:.3f}  "
            f"(在 active mask 内: mean={mg['geom_pen_in_active_mean']*1000:.2f}mm  "
            f"max={mg['geom_pen_in_active_max']*1000:.2f}mm)"
        )

    mdp.stop()


if __name__ == "__main__":
    main()
