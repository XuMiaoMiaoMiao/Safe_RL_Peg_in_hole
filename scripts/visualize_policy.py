"""跑训好的 policy, 在指定 env 第一次达到 hold-N 时冻结 IsaacSim 画面.

eval_sac.py 是数值验证, 这个脚本是肉眼验证. mushroom 的 VectorCore.evaluate
是全自动的, absorb 一触发 episode 立刻 reset, 你根本看不到"agent 稳稳停在
preinsert"那一帧. 这里通过 monkey-patch is_absorbing, 让 env 在指定 env
计数器达到 hold-N 时调 _world.render() 反复刷, 不调 _world.step(), 物理就冻住了.

stage 注意 (M1'/M2): 冻结由 env 的 success counter 触发, 而 success_mask =
(pos<pos_th) ∧ (axis_err<axis_th). M2a/M2b 评估时**必须**传 --rew_axis 和
--success_axis_threshold 与训练时一致, 否则 axis_th=inf 会让 freeze 在"位置
进阈但姿态没达标"时误触发, 看到的不是 M2 真正的成功状态.

用法:
    conda activate safe_rl
    # M1' 视觉验证
    python scripts/visualize_policy.py --preinsert_success_pos_threshold 0.10

    # M2a 视觉验证 (训练时 --rew_axis 2.0 --success_axis_threshold 0.5)
    python scripts/visualize_policy.py \\
        --preinsert_success_pos_threshold 0.10 \\
        --rew_axis 2.0 --success_axis_threshold 0.5

    python scripts/visualize_policy.py --freeze_seconds 30
    python scripts/visualize_policy.py --viz_env_idx 1
    python scripts/visualize_policy.py --hold_steps 30

注意:
- 必须 num_envs >= 2 (cloner bug).
- terminal_hold_bonus 强制设 0, 否则 env 自己会在 hold-N absorb 把 episode reset.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT / "results/best_agent.msh"))
    p.add_argument("--num_envs", type=int, default=2,
                   help="至少 2 (num_envs=1 触发 cloner bug)")
    p.add_argument("--viz_env_idx", type=int, default=0)
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=0.10,
                   help="跟 train 一致 (env/train/eval 默认 0.10m, M1'/M2 curriculum). "
                        "传更紧的值会让 freeze 条件更严, 可能看不到 hold-N 触发.")
    p.add_argument("--initial_joint_noise", type=float, default=0.1,
                   help="跟 env/train 默认 0.1 一致 (M1' 训练时的 reset 噪声).")
    p.add_argument("--action_scale", type=float, default=None,
                   help="覆盖 env 动作缩放. 与 train/eval 保持一致.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert offset (默认 0.05m)")
    p.add_argument("--rew_pos", type=float, default=None,
                   help="覆盖 env 的 pos_err reward 权重, 保持 CLI 与 train/eval 一致.")
    p.add_argument("--rew_axis", type=float, default=None,
                   help="覆盖 env 的 axis_err 权重. visualize 不算 reward, 但保留 "
                        "CLI 一致性 (env 内部根据这个值打印的 axis 项形式不同).")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="**必须与 train 一致**. M2a 用 0.5, M2b 用 0.2. 不传 = "
                        "默认 inf = M1' 行为, 会让 freeze 在 axis 还没对齐时误触发.")
    p.add_argument("--rew_clearance", type=float, default=None,
                   help="M2c clearance penalty 权重. 视觉脚本不看 reward, 但 env "
                        "要求 clearance_soft 有限时 rew_clearance > 0.")
    p.add_argument("--clearance_soft", type=float, default=None,
                   help="M2c clearance success gate (m). 传入后 freeze 条件也会要求 "
                        "min_clearance >= clearance_soft.")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="M2c hard threshold (m). 视觉验证一般不传, 避免提前 absorbing.")
    p.add_argument("--clearance_penalty_scale", type=float, default=None,
                   help="M2c penalty 归一化 scale (m), 保持 CLI 与 train/eval 一致.")
    p.add_argument("--clearance_success_threshold", type=float, default=None,
                   help="M2c success gate 阈值 (m). 与 train 一致 — 影响 freeze 触发.")
    p.add_argument("--log_clearance", action="store_true",
                   help="metric-only: 即使不启用 M2c gate 也计算并打印 clearance.")
    p.add_argument("--show_proxies", action="store_true",
                   help="spawn M2c sphere proxy marker (左红 / 右蓝半透明球), "
                        "跟随 policy 运动; 用来肉眼判断 clearance 球半径是否过大.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="M2c sphere proxy 的 arm/link 半径 (m). 默认 0.06. "
                        "用于 --show_proxies 对比球半径是否过大.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="M2c sphere proxy 的 EE/finger 半径 (m). 默认 0.03. "
                        "用于 --show_proxies 对比球半径是否过大.")
    p.add_argument("--use_m2d_obs", action="store_true",
                   help="把 env obs 切到 37 维 M2d 几何观测. 可视化 37 维 M2d "
                        "checkpoint 时必须传; 32 维 M2c checkpoint 不要传.")
    p.add_argument("--rew_axial_off", type=float, default=None,
                   help="M2d 轴向偏差 reward 权重. 视觉脚本不看 reward, 但 env "
                        "要求 finite axial gate 时该权重 > 0.")
    p.add_argument("--rew_radial", type=float, default=None,
                   help="M2d 横向偏差 reward 权重. 视觉脚本不看 reward, 但 env "
                        "要求 finite radial gate 时该权重 > 0.")
    p.add_argument("--axial_success_threshold", type=float, default=None,
                   help="M2d 轴向 success 阈值 (m). 传入后 freeze 按 M2d gate 触发.")
    p.add_argument("--radial_success_threshold", type=float, default=None,
                   help="M2d 横向 success 阈值 (m). 传入后 freeze 按 M2d gate 触发.")
    p.add_argument("--log_axial_radial", action="store_true",
                   help="metric-only: 即使不启用 M2d gate 也打印 axial/radial.")
    p.add_argument("--hold_steps", type=int, default=10,
                   help="连续 N 步在阈内即冻结. 默认 10 = 跟 train 的 hold_success_steps 一致")
    p.add_argument("--n_episodes", type=int, default=2,
                   help="VectorCore.evaluate 跑几集. 一般冻结发生在第一集就够看")
    p.add_argument("--freeze_seconds", type=float, default=15.0,
                   help="冻结多少秒, Ctrl-C 可提前退出")
    p.add_argument("--stochastic", action="store_true",
                   help="用 SAC 采样策略而不是 deterministic tanh(mu)")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) 必须落在 [0, {args.num_envs - 1}]"
        )

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=False,
        preinsert_success_pos_threshold=args.preinsert_success_pos_threshold,
        initial_joint_noise=args.initial_joint_noise,
        success_hold_steps=args.hold_steps,
        terminal_hold_bonus=0.0,
    )
    if args.action_scale is not None:
        env_kwargs["action_scale"] = args.action_scale
    if args.preinsert_offset is not None:
        env_kwargs["preinsert_offset"] = args.preinsert_offset
    if args.rew_pos is not None:
        env_kwargs["rew_pos"] = args.rew_pos
    if args.rew_axis is not None:
        env_kwargs["rew_axis"] = args.rew_axis
    if args.success_axis_threshold is not None:
        env_kwargs["success_axis_threshold"] = args.success_axis_threshold
    for key in ("rew_clearance", "clearance_soft", "clearance_hard",
                "clearance_penalty_scale", "clearance_success_threshold",
                "proxy_arm_radius", "proxy_ee_radius",
                "rew_axial_off", "rew_radial",
                "axial_success_threshold", "radial_success_threshold"):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value
    if args.log_clearance:
        env_kwargs["log_clearance"] = True
    if args.show_proxies:
        # proxy marker 每帧调用 _compute_min_clearance(); 打开 log_clearance 可确保
        # 非 M2c gate 场景下也初始化 clearance/proxy cache.
        env_kwargs["log_clearance"] = True
    if args.log_axial_radial:
        env_kwargs["log_axial_radial"] = True
    if args.use_m2d_obs:
        env_kwargs["use_m2d_obs"] = True
    mdp = DualArmPegHoleEnv(**env_kwargs)
    print(f"[VIZ STAGE] pos_th={mdp._preinsert_success_pos_threshold:.3f}m  "
          f"axis_th={mdp._success_axis_threshold:.3f}  "
          f"w_axis={mdp._w_axis:.3f}  "
          f"clearance_soft={mdp._clearance_soft:+.4f}m  "
          f"clearance_success={mdp._clearance_success_threshold:+.4f}m")
    if args.show_proxies:
        print(f"[VIZ PROXY] arm_radius={mdp._proxy_arm_radius:.3f}m  "
              f"ee_radius={mdp._proxy_ee_radius:.3f}m")
    if (math.isfinite(mdp._axial_success_threshold)
            or math.isfinite(mdp._radial_success_threshold)):
        print(f"[VIZ M2d] axial_th={mdp._axial_success_threshold:.4f}m  "
              f"radial_th={mdp._radial_success_threshold:.4f}m  "
              f"w_axial_off={mdp._w_axial_off:.3f}  "
              f"w_radial={mdp._w_radial:.3f}")

    from mushroom_rl.core import Agent, VectorCore

    agent = Agent.load(args.agent_path)
    print(f"[VIZ] loaded agent from {args.agent_path}")
    print(f"[VIZ] watching env {args.viz_env_idx} for {args.hold_steps} consecutive in_thresh steps")

    proxy_handles = None
    if args.show_proxies:
        from scripts.visualize_targets import _spawn_sphere_proxies, _update_sphere_proxies
        proxy_handles = _spawn_sphere_proxies(mdp)
        _update_sphere_proxies(mdp, args.viz_env_idx, proxy_handles)
    else:
        _update_sphere_proxies = None

    state = {"frozen": False, "step_idx": 0}
    original_is_absorbing = mdp.is_absorbing

    def _axial_radial_at(env_idx):
        """从 mdp._compute_preinsert_errors 拿 axial_dist / radial_err for one env.
        给 freeze / step 打印用 — 验证 M3 reward 公式前确认符号和量级.
        """
        try:
            info = mdp._compute_preinsert_errors()
            axial_dist = float(info["axial_dist"][env_idx])
            return (axial_dist,
                    abs(axial_dist - mdp._preinsert_offset),
                    float(info["radial_err"][env_idx]))
        except Exception:
            return (None, None, None)

    def freeze_on_success(obs):
        result = original_is_absorbing(obs)
        if proxy_handles is not None:
            _update_sphere_proxies(mdp, args.viz_env_idx, proxy_handles)
        state["step_idx"] += 1
        cnt = int(mdp._consecutive_inthresh[args.viz_env_idx].item())
        if cnt > 0 and state["step_idx"] % 5 == 0:
            pos_err = mdp._last_pos_err
            axis_err = mdp._last_axis_err
            clearance = mdp._last_min_clearance
            clearance_txt = ""
            if clearance is not None:
                clearance_txt = (
                    f"  clearance={float(clearance[args.viz_env_idx]):+.4f}m"
                )
            ax, ax_off, rad = _axial_radial_at(args.viz_env_idx)
            ar_txt = ""
            if ax is not None:
                ar_txt = (
                    f"  axial={ax:+.4f}m  axial_off={ax_off:.4f}m"
                    f"  radial={rad:.4f}m"
                )
            print(f"  step {state['step_idx']}  env {args.viz_env_idx}  "
                  f"hold count={cnt}  "
                  f"pos_err={float(pos_err[args.viz_env_idx]):.4f}m  "
                  f"axis_err={float(axis_err[args.viz_env_idx]):.4f}"
                  f"{clearance_txt}"
                  f"{ar_txt}")
        if not state["frozen"] and cnt >= args.hold_steps:
            pos_err = mdp._last_pos_err
            axis_err = mdp._last_axis_err
            clearance = mdp._last_min_clearance
            clearance_line = ""
            if clearance is not None:
                clearance_line = (
                    f"  clearance = {float(clearance[args.viz_env_idx]):+.4f}m  "
                    f"(success_threshold = {mdp._clearance_success_threshold:+.4f}m)\n"
                )
            ax, ax_off, rad = _axial_radial_at(args.viz_env_idx)
            ar_line = ""
            if ax is not None:
                ar_line = (
                    f"  axial_dist = {ax:+.4f}m  "
                    f"(>0 = peg_tip 在 hole 开口外侧, <0 = 已穿入)\n"
                    f"  axial_off  = {ax_off:.4f}m  "
                    f"(target preinsert_offset = {mdp._preinsert_offset:.4f}m)\n"
                    f"  radial_err = {rad:.4f}m  "
                    f"(M2d approach 起步阈值建议 0.04m, 后续收紧到 0.02m)\n"
                )
            print(
                f"\n[FREEZE] env {args.viz_env_idx} reached hold-{args.hold_steps} "
                f"at step {state['step_idx']}\n"
                f"  pos_err  = {float(pos_err[args.viz_env_idx]):.4f}m\n"
                f"  axis_err = {float(axis_err[args.viz_env_idx]):.4f}  "
                f"(0 = perfect; success_axis_threshold = {mdp._success_axis_threshold:.3f})\n"
                f"{clearance_line}"
                f"{ar_line}"
                f"  freezing for {args.freeze_seconds:.1f}s, Ctrl-C to exit early"
            )
            state["frozen"] = True
            end_t = time.monotonic() + args.freeze_seconds
            try:
                while time.monotonic() < end_t:
                    if proxy_handles is not None:
                        _update_sphere_proxies(mdp, args.viz_env_idx, proxy_handles)
                    mdp._world.render()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            print("[FREEZE] released, evaluation will continue (or end)")
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
