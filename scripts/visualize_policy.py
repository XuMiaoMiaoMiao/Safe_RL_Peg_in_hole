"""跑训好的 policy, 在指定 env 第一次达到 hold-N 时冻结 IsaacSim 画面.

eval_sac.py 是数值验证, 这个脚本是肉眼验证. mushroom 的 VectorCore.evaluate
是全自动的, absorb 一触发 episode 立刻 reset, 你根本看不到"agent 稳稳停在
preinsert"那一帧. 这里通过 monkey-patch is_absorbing, 让 env 在指定 env
计数器达到 hold-N 时调 _world.render() 反复刷, 不调 _world.step(), 物理就冻住了.

stage 注意: 冻结由 env 的 success counter 触发, 而 success_mask =
(pos<pos_th) ∧ (axis_err<axis_th). Stage 2 视觉验证时**必须**传 --rew_axis 和
--success_axis_threshold 与训练时一致 (Stage 2 推荐 0.50 或 0.30), 否则
axis_th=inf 会让 freeze 在"位置进阈但姿态没达标"时误触发, 看到的不是 Stage 2
真正的成功状态.

正式 Stage 1 / Stage 2 可视化命令见 README.md "可视化命令" 段.

注意:
- 必须 num_envs >= 2 (cloner bug).
- terminal_hold_bonus 强制设 0, 否则 env 自己会在 hold-N absorb 把 episode reset.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy, parse_home_weights


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--num_envs", type=int, default=2,
                   help="至少 2 (num_envs=1 触发 cloner bug)")
    p.add_argument("--viz_env_idx", type=int, default=0)
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=0.10,
                   help="跟 train 一致 (env/train/eval 默认 0.10m, Stage 1/Stage 2 curriculum). "
                        "传更紧的值会让 freeze 条件更严, 可能看不到 hold-N 触发.")
    p.add_argument("--initial_joint_noise", type=float, default=0.1,
                   help="跟 env/train 默认 0.1 一致 (Stage 1 训练时的 reset 噪声).")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert offset (默认 0.05m)")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重. visualize 不算 reward, 但保留 "
                        "CLI 一致性 (env 内部根据这个值打印的 axis 项形式不同).")
    p.add_argument("--rew_home", type=float, default=None,
                   help="覆盖 env 的 home regularizer 权重. visualize 不依赖 reward, "
                        "但保留与 train/eval CLI 一致性.")
    p.add_argument("--home_weights", type=parse_home_weights, default=None,
                   help="home regularizer 的逐关节权重. visualize 不依赖 reward, "
                        "但建议与 train/eval CLI 保持一致.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="**必须与 train 一致**. Stage 2 训练用 0.50, 严格视觉验证可用 0.30. 不传 = "
                        "默认 inf = Stage 1 行为, 会让 freeze 在 axis 还没对齐时误触发.")
    p.add_argument("--hold_steps", type=int, default=10,
                   help="连续 N 步在阈内即冻结. 默认 10 = 跟 train 的 hold_success_steps 一致")
    p.add_argument("--n_episodes", type=int, default=2,
                   help="VectorCore.evaluate 跑几集. 一般冻结发生在第一集就够看")
    p.add_argument("--freeze_seconds", type=float, default=15.0,
                   help="冻结多少秒, Ctrl-C 可提前退出")
    p.add_argument("--freeze_mode", type=str, default="first_hold",
                   choices=("first_hold", "episode_end", "step"),
                   help="画面冻结时机: first_hold=首次连续 hold_steps 成功时冻结(旧行为); "
                        "episode_end=目标 env episode 末尾冻结; "
                        "step=目标 env 到 --freeze_after_step 时冻结.")
    p.add_argument("--freeze_after_step", type=int, default=None,
                   help="--freeze_mode step 时使用: 在目标 env episode 的第 N 步冻结.")
    p.add_argument("--stochastic", action="store_true",
                   help="用 SAC 采样策略而不是 deterministic tanh(mu)")
    p.add_argument("--rew_pos_success", type=float, default=None,
                   help="覆盖 env 的 pos-only success bonus. 应与 train 一致.")
    p.add_argument("--axis_gate_radius", type=float, default=None,
                   help="覆盖 env 的 axis 距离门控半径. 应与 train 一致.")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="覆盖 env 的 sphere-proxy 自碰撞兜底阈值. 应与 train 一致, "
                        "否则可能出现 train 不撞 / viz 老 reset 的错觉. 关闭写 --clearance_hard=-inf.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="覆盖 arm sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="覆盖 EE sphere proxy 半径. 应与 train 一致.")
    p.add_argument("--exclude_ee_from_physx_self_collision", action="store_true",
                   help="geom insert 可视化用: PhysX self-collision 分组排除 EE link, "
                        "避免 peg-hole 正常接触被 hard absorbing 截断.")
    p.add_argument("--use_axis_resid_obs", action="store_true",
                   help="34 维 obs (axis_resid 替换 axis_dot). 应与 train 一致.")
    p.add_argument("--horizon", type=int, default=None,
                   help="覆盖 env 默认 horizon. geom insert 训练用 200, 不传走 env 默认 150.")

    # Geometric preinsert viz: freeze trigger uses active geom success mask.
    p.add_argument("--geom_stage", type=str, default=None,
                   choices=("prepos", "preaxis", "insert"),
                   help="启用 geom env (41D obs); freeze 触发改用 active geom success mask.")
    p.add_argument("--geom_eval_epoch", type=int, default=None,
                   help="viz 前调用 set_geom_epoch(epoch). 默认: insert 用 ramp_end, "
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
    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) 必须落在 [0, {args.num_envs - 1}]"
        )
    if args.freeze_mode == "step":
        if args.freeze_after_step is None:
            raise ValueError("--freeze_mode step 需要同时传 --freeze_after_step N")
        if args.freeze_after_step <= 0:
            raise ValueError("--freeze_after_step 必须 > 0")

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=False,
        preinsert_success_pos_threshold=args.preinsert_success_pos_threshold,
        initial_joint_noise=args.initial_joint_noise,
        success_hold_steps=args.hold_steps,
        terminal_hold_bonus=0.0,
    )
    if args.horizon is not None:
        env_kwargs["horizon"] = args.horizon
    if args.preinsert_offset is not None:
        env_kwargs["preinsert_offset"] = args.preinsert_offset
    if args.rew_axis is not None:
        env_kwargs["rew_axis"] = args.rew_axis
    if args.rew_home is not None:
        env_kwargs["rew_home"] = args.rew_home
    if args.home_weights is not None:
        env_kwargs["home_weights"] = args.home_weights
    if args.success_axis_threshold is not None:
        env_kwargs["success_axis_threshold"] = args.success_axis_threshold
    for key in ("rew_pos_success", "axis_gate_radius",
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
    mdp = DualArmPegHoleEnv(**env_kwargs)
    if args.geom_stage is not None:
        if args.geom_eval_epoch is not None:
            geom_eval_epoch = args.geom_eval_epoch
        elif mdp._geom_stage == "insert":
            geom_eval_epoch = mdp._geom_d_target_ramp_end
        else:
            geom_eval_epoch = 0
        mdp.set_geom_epoch(geom_eval_epoch)
        print(f"[VIZ GEOM] stage={mdp._geom_stage}  "
              f"d_target_eff={mdp._geom_d_target_eff:+.4f}m  "
              f"hold_steps={args.hold_steps}  freeze 触发: 连续 hold_steps 步 active geom mask=True")
        print(f"[VIZ GEOM] thresh: d_th={mdp._geom_d_th:.3f}  "
              f"r_tip_th={mdp._geom_r_tip_th:.3f}  "
              f"r_max_th={mdp._geom_r_max_th:.3f}  "
              f"axis_th={mdp._geom_axis_th:.3f}  "
              f"insert_d_ins={mdp._geom_insert_d_ins:+.3f}  "
              f"insert_r_max_th={mdp._geom_insert_r_max_th:.3f}  "
              f"pen_th={mdp._geom_pen_th*1000:.1f}mm")
    else:
        print(f"[VIZ STAGE 1/2] pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
              f"axis_th={mdp._success_axis_threshold:.3f}  "
              f"w_axis={mdp._w_axis:.3f}  "
              f"w_pos_success={mdp._w_pos_success:.3f}  "
              f"w_home={mdp._w_home:.4f}  "
              f"axis_gate_radius={mdp._axis_gate_radius:.3f}m")

    from mushroom_rl.core import Agent, VectorCore

    agent = Agent.load(args.agent_path)
    print(f"[VIZ] loaded agent from {args.agent_path}")
    eval_horizon = int(args.horizon if args.horizon is not None else mdp.info.horizon)
    if args.freeze_mode == "first_hold":
        print(f"[VIZ] watching env {args.viz_env_idx} for "
              f"{args.hold_steps} consecutive in_thresh steps")
    elif args.freeze_mode == "episode_end":
        print(f"[VIZ] watching env {args.viz_env_idx}; freeze at episode end "
              f"(horizon={eval_horizon})")
    else:
        print(f"[VIZ] watching env {args.viz_env_idx}; freeze at episode step "
              f"{args.freeze_after_step}")

    # Geom: per-env mask 连续计数器 (hooked is_absorbing 维护).
    # Stage 1/2 legacy: 沿用 env 的 _consecutive_inthresh.
    custom_consec = torch.zeros(mdp._n_envs, dtype=torch.int64, device=mdp._device)
    episode_steps = torch.zeros(mdp._n_envs, dtype=torch.int64, device=mdp._device)
    state = {"frozen": False, "step_idx": 0}
    original_is_absorbing = mdp.is_absorbing

    def freeze_on_success(obs):
        result = original_is_absorbing(obs)
        state["step_idx"] += 1
        episode_steps.add_(1)
        if args.geom_stage is not None and mdp._cached_d is not None:
            geom_prepos, geom_preaxis, geom_insert, geom_success = (
                mdp._compute_geom_success_masks()
            )
            custom_consec.copy_(
                torch.where(geom_success, custom_consec + 1, torch.zeros_like(custom_consec))
            )
            cnt = int(custom_consec[args.viz_env_idx].item())
            label = f"geom_{mdp._geom_stage}"
        else:
            cnt = int(mdp._consecutive_inthresh[args.viz_env_idx].item())
            label = "in_thresh"
        if cnt > 0 and state["step_idx"] % 5 == 0:
            if args.geom_stage is not None and mdp._cached_d is not None:
                print(f"  step {state['step_idx']}  env {args.viz_env_idx}  "
                      f"{label} count={cnt}  "
                      f"d={float(mdp._cached_d[args.viz_env_idx]):+.4f}m  "
                      f"d_target={mdp._geom_d_target_eff:+.4f}m  "
                      f"radial_tip={float(mdp._cached_radial_err_tip[args.viz_env_idx]):.4f}m  "
                      f"radial_max={float(mdp._cached_radial_max[args.viz_env_idx]):.4f}m  "
                      f"axis_err={float(mdp._cached_axis_err[args.viz_env_idx]):.4f}")
            else:
                pos_err = mdp._last_pos_err
                axis_err = mdp._last_axis_err
                print(f"  step {state['step_idx']}  env {args.viz_env_idx}  "
                      f"{label} count={cnt}  "
                      f"pos_err={float(pos_err[args.viz_env_idx]):.4f}m  "
                      f"axis_err={float(axis_err[args.viz_env_idx]):.4f}")
        target_ep_step = int(episode_steps[args.viz_env_idx].item())
        if args.freeze_mode == "first_hold":
            should_freeze = cnt >= args.hold_steps
            freeze_reason = f"hold-{args.hold_steps}"
        elif args.freeze_mode == "episode_end":
            should_freeze = target_ep_step >= eval_horizon
            freeze_reason = f"episode-end step {target_ep_step}/{eval_horizon}"
        else:
            should_freeze = target_ep_step >= int(args.freeze_after_step)
            freeze_reason = f"episode-step {target_ep_step}"

        if not state["frozen"] and should_freeze:
            if args.geom_stage is not None and mdp._cached_d is not None:
                print(
                    f"\n[FREEZE] env {args.viz_env_idx} reached {freeze_reason} "
                    f"(geom-{mdp._geom_stage}) "
                    f"at step {state['step_idx']}\n"
                    f"  d          = {float(mdp._cached_d[args.viz_env_idx]):+.4f}m\n"
                    f"  d_target   = {mdp._geom_d_target_eff:+.4f}m\n"
                    f"  radial_tip = {float(mdp._cached_radial_err_tip[args.viz_env_idx]):.4f}m  "
                    f"(r_tip_th={mdp._geom_r_tip_th:.3f})\n"
                    f"  radial_max = {float(mdp._cached_radial_max[args.viz_env_idx]):.4f}m  "
                    f"(r_max_th={mdp._geom_r_max_th:.3f}, insert_r_max_th={mdp._geom_insert_r_max_th:.3f})\n"
                    f"  axis_err   = {float(mdp._cached_axis_err[args.viz_env_idx]):.4f}  "
                    f"(axis_th={mdp._geom_axis_th:.3f})\n"
                    f"  penetration= {float(mdp._cached_penetration_max[args.viz_env_idx])*1000:.2f}mm  "
                    f"(pen_th={mdp._geom_pen_th*1000:.1f}mm)\n"
                    f"  freezing for {args.freeze_seconds:.1f}s, Ctrl-C to exit early"
                )
            else:
                pos_err = mdp._last_pos_err
                axis_err = mdp._last_axis_err
                print(
                    f"\n[FREEZE] env {args.viz_env_idx} reached {freeze_reason} "
                    f"at step {state['step_idx']}\n"
                    f"  pos_err  = {float(pos_err[args.viz_env_idx]):.4f}m\n"
                    f"  axis_err = {float(axis_err[args.viz_env_idx]):.4f}  "
                    f"(0 = perfect; success_axis_threshold = {mdp._success_axis_threshold:.3f})\n"
                    f"  freezing for {args.freeze_seconds:.1f}s, Ctrl-C to exit early"
                )
            state["frozen"] = True
            end_t = time.monotonic() + args.freeze_seconds
            try:
                while time.monotonic() < end_t:
                    mdp._world.render()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            print("[FREEZE] released, evaluation will continue (or end)")
        if isinstance(result, torch.Tensor):
            done = result.to(dtype=torch.bool, device=mdp._device).flatten()
        else:
            done = torch.as_tensor(result, dtype=torch.bool, device=mdp._device)
            if done.ndim == 0:
                done = done.repeat(mdp._n_envs)
            else:
                done = done.flatten()
        horizon_done = episode_steps >= eval_horizon
        done = done | horizon_done
        if done.any():
            custom_consec[done] = 0
            episode_steps[done] = 0
        return result

    mdp.is_absorbing = freeze_on_success

    core = VectorCore(agent, mdp)
    if args.stochastic:
        core.evaluate(n_episodes=args.n_episodes, render=True)
    else:
        with deterministic_policy(agent):
            core.evaluate(n_episodes=args.n_episodes, render=True)

    if not state["frozen"]:
        print(f"\n[VIZ] env {args.viz_env_idx} 在 {args.n_episodes} episode "
              f"({state['step_idx']} 步) 内没达到 hold-{args.hold_steps}.")

    mdp.stop()


if __name__ == "__main__":
    main()
