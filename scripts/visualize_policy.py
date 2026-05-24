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
import math
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
    p.add_argument("--default_pose_variant", type=str, default="easy",
                   choices=["easy", "harder"],
                   help="Reset pose: 'easy' = default HOME_JOINT_POS (arms swing "
                        "wide); 'harder' = crossed-forearm pose (distal forearm/EE "
                        "segments cross the midline). For benchmark.")
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
    # 2026-05-17: split collision termination + runtime arm collision APIs.
    # Must match training env for visualization to reflect actual policy behavior.
    p.add_argument("--sphere_collision_terminates",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Must match training. False = sphere collision is cost-only.")
    p.add_argument("--physx_collision_terminates",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Must match training. True = PhysX real contact terminates episode.")
    p.add_argument("--enable_physx_arm_collision",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Must match training. True = runtime apply capsule colliders to arm links.")
    p.add_argument("--enable_table_collision",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Must match training. True = geometry-plane arm/EE-vs-table safety.")
    p.add_argument("--table_collision_terminates",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Must match training. True = table clearance collision terminates episode.")
    p.add_argument("--table_z", type=float, default=None)
    p.add_argument("--table_clearance_hard", type=float, default=None)
    p.add_argument("--table_clearance_cost_margin", type=float, default=None)
    p.add_argument("--debug_show_physx_arm_collision", action="store_true",
                   help="Visualization only. Show the runtime PhysX arm collision "
                        "capsule proxies: left arm blue, right arm red.")
    p.add_argument("--debug_show_sphere_proxy", action="store_true",
                   help="Visualization only. Show the sphere-proxy clearance "
                        "boundary used by the env/cost: left arm orange, right arm cyan.")
    p.add_argument("--keep_collision_reward_penalty",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="CostEnv only. Must match training. Cliff at collision step.")
    p.add_argument("--drop_penetration_reward_for_cost",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="CostEnv only. Must match training. Drop -w·pen when cost=penetration.")
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
                   choices=["collision", "penetration", "clearance"])
    p.add_argument("--clearance_cost_margin", type=float, default=None)
    p.add_argument("--clearance_cost_clip_max", type=float, default=None,
                   help="Optional upper clip for clearance cost. Default None = "
                        "no upper clip, matching D-ATACOM-style positive violation.")
    p.add_argument("--cost_scale", type=float, default=None,
                   help="Multiplicative gain on info['cost']. Pass the same value used "
                        "at training time so visual diagnostics match wandb logs.")
    p.add_argument("--geom_progress_floor", type=float, default=None)
    p.add_argument("--rew_geom_advance", type=float, default=None)
    p.add_argument("--geom_d_gate_mode", type=str, default=None,
                   choices=["off", "alignment"])
    p.add_argument("--rew_geom_bad_entry", type=float, default=None)
    p.add_argument("--geom_bad_entry_radial_safe", type=float, default=None)
    p.add_argument("--geom_bad_entry_axis_safe", type=float, default=None)
    p.add_argument("--geom_bad_entry_pen_safe", type=float, default=None)
    return p.parse_args()


def _spawn_sphere_proxy_markers(mdp):
    """Spawn one env's sphere-proxy markers under /World/viz.

    The proxy geometry is the same one used by mdp._compute_min_clearance():
    8 joint-link spheres + 7 segment-midpoint spheres + 2 EE spheres per arm.
    """
    try:
        import omni.usd
        from pxr import UsdGeom, Sdf, Gf, Vt
    except ImportError:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    radii = mdp._proxy_radii_per_side.detach().cpu().tolist()

    def _make(path, color, opacity, radius):
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
        sphere.GetRadiusAttr().Set(float(radius))
        sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        sphere.GetDisplayOpacityAttr().Set([float(opacity)])
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(0.0, 0.0, 10.0))
        return t_op

    left = [
        _make(f"/World/viz/policy_sphere_left_{i:02d}", (1.0, 0.55, 0.20), 0.25, r)
        for i, r in enumerate(radii)
    ]
    right = [
        _make(f"/World/viz/policy_sphere_right_{i:02d}", (0.20, 0.85, 1.0), 0.25, r)
        for i, r in enumerate(radii)
    ]
    print(
        f"[VIZ SPHERES] spawned {len(left) + len(right)} sphere proxy markers "
        f"for one env (left=orange, right=cyan, opacity=0.25)"
    )
    return {"left": left, "right": right}


def _update_sphere_proxy_markers(mdp, sphere_handles, env_idx):
    """Update sphere marker translations to the selected env's current proxies."""
    if sphere_handles is None:
        return
    from pxr import Gf

    _, info = mdp._compute_min_clearance()
    left_p = info["left_proxies"][env_idx].detach().cpu()
    right_p = info["right_proxies"][env_idx].detach().cpu()
    for i, t_op in enumerate(sphere_handles["left"]):
        x, y, z = left_p[i].tolist()
        t_op.Set(Gf.Vec3d(x, y, z))
    for i, t_op in enumerate(sphere_handles["right"]):
        x, y, z = right_p[i].tolist()
        t_op.Set(Gf.Vec3d(x, y, z))


def main():
    args = parse_args()
    if args.num_envs < 2:
        raise ValueError("--num_envs must be >= 2 (num_envs=1 triggers cloner bugs)")
    # Mushroom VectorCore expects completed episodes to align with vectorized
    # env batches. For visualization, be permissive: users often ask for
    # `--n_episodes 1 --num_envs 4` just to look at env 0. Run one full vector
    # batch instead of failing at dataset.flatten() after the render.
    requested_episodes = int(args.n_episodes)
    min_episodes = args.num_envs
    if requested_episodes < min_episodes:
        args.n_episodes = min_episodes
        print(
            f"[VIZ] adjusted n_episodes {requested_episodes} -> {args.n_episodes} "
            f"to match num_envs={args.num_envs}"
        )
    elif requested_episodes % args.num_envs != 0:
        args.n_episodes = int(math.ceil(requested_episodes / args.num_envs) * args.num_envs)
        print(
            f"[VIZ] adjusted n_episodes {requested_episodes} -> {args.n_episodes} "
            f"so it is divisible by num_envs={args.num_envs}"
        )
    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) 必须落在 [0, {args.num_envs - 1}]"
        )
    if args.freeze_mode == "step":
        if args.freeze_after_step is None:
            raise ValueError("--freeze_mode step 需要同时传 --freeze_after_step N")
        if args.freeze_after_step <= 0:
            raise ValueError("--freeze_after_step 必须 > 0")

    from envs import DualArmPegHoleEnv, DualArmPegHoleCostEnv
    EnvClass = DualArmPegHoleCostEnv if args.cost_signal is not None else DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=False,
        preinsert_success_pos_threshold=args.preinsert_success_pos_threshold,
        initial_joint_noise=args.initial_joint_noise,
        default_pose_variant=args.default_pose_variant,
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
                "rew_geom_penetration", "geom_gate_penetration_sigma",
                "cost_signal", "clearance_cost_margin", "clearance_cost_clip_max",
                "cost_scale",
                "table_z", "table_clearance_hard", "table_clearance_cost_margin",
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
    env_kwargs["sphere_collision_terminates"] = bool(args.sphere_collision_terminates)
    env_kwargs["physx_collision_terminates"] = bool(args.physx_collision_terminates)
    env_kwargs["enable_physx_arm_collision"] = bool(args.enable_physx_arm_collision)
    env_kwargs["enable_table_collision"] = bool(args.enable_table_collision)
    env_kwargs["table_collision_terminates"] = bool(args.table_collision_terminates)
    env_kwargs["debug_show_physx_arm_collision"] = bool(args.debug_show_physx_arm_collision)
    if EnvClass is DualArmPegHoleCostEnv:
        env_kwargs["keep_collision_reward_penalty"] = bool(
            args.keep_collision_reward_penalty
        )
        env_kwargs["drop_penetration_reward_for_cost"] = bool(
            args.drop_penetration_reward_for_cost
        )
    mdp = EnvClass(**env_kwargs)
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
    sphere_handles = None
    if args.debug_show_sphere_proxy:
        sphere_handles = _spawn_sphere_proxy_markers(mdp)
        _update_sphere_proxy_markers(mdp, sphere_handles, args.viz_env_idx)

    def freeze_on_success(obs):
        result = original_is_absorbing(obs)
        _update_sphere_proxy_markers(mdp, sphere_handles, args.viz_env_idx)
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
