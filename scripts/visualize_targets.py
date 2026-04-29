"""可视化双臂场景 — preinsert 目标 + peg/hole frame, 不训练.

显示:
    - peg (红) / hole (绿) 已经从 USDA 挂在左/右末端
    - peg_axis 箭头 (浅红) 从 peg_tip 沿 peg 轴向外 5cm
    - hole_axis 箭头 (浅绿) 从 hole_entry 沿 hole 开口方向外 5cm
    - preinsert_target 球 (黄)     hole_entry 外 preinsert_offset 处, 预插入站位

肉眼验收:
    - 黄色 preinsert_target 应该在 hole 开口外 ~5cm, 沿 hole 轴方向.
    - peg/hole 的 axis 箭头分别从自身尖端 / 开口中心沿 self +Z 伸出.
    - reset 多次后 marker 稳定跟随; M0 后 frame 已改成解析式 (EE_pose ⊗ const offset),
      headless / render=False 也保证 fresh — 不依赖 XFormPrim 的 Fabric flush.

运行:
    conda activate safe_rl
    python scripts/visualize_targets.py
    python scripts/visualize_targets.py --preinsert_offset 0.08
    python scripts/visualize_targets.py --duration 30
    # 同时 spawn 19 球 sphere proxy (左红 / 右蓝, opacity=0.3), 看 M2c 几何包络:
    python scripts/visualize_targets.py --show_proxies --duration 30
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2,
                   help="至少 2, 避免 IsaacSim cloner 的 num_envs=1 bug")
    p.add_argument("--viz_env_idx", type=int, default=0,
                   help="用哪一个 env 的 peg/hole frame 驱动 marker")
    p.add_argument("--initial_joint_noise", type=float, default=0.0,
                   help="可视化时默认不加 reset 噪声, 更方便看默认姿态")
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=0.10,
                   help="success 触发的位置阈值 (env 默认 0.10m). 改这里只影响"
                        "[VIS PREINSERT STATS] 里 success_rate 的判定, 不改 marker.")
    p.add_argument("--success_axis_threshold", type=float, default=None,
                   help="success 触发的 axis_err 阈值. **不传 = 默认 inf = pos-only**, "
                        "此时输出的 success_rate 是 pos-only success 而非 M2 的 "
                        "pos∧axis success. M2 时传 0.5 / 0.2 与 train 一致.")
    p.add_argument("--preinsert_offset", type=float, default=None,
                   help="覆盖 env 的 preinsert_offset (默认 0.05m)")
    p.add_argument("--duration", type=float, default=0.0,
                   help=">0 时显示指定秒数后退出; <=0 时持续到 Ctrl-C")
    p.add_argument("--idle_dt", type=float, default=0.02,
                   help="主循环 sleep 间隔, 仅用于降低 CPU 占用")
    p.add_argument("--n_resets", type=int, default=1,
                   help="进入主可视化循环前采样多少次 reset, 用来打印 preinsert "
                        "误差在 reset 分布下的统计. n_resets=1 = 单次 reset.")
    p.add_argument("--show_proxies", action="store_true",
                   help="spawn 19 球 sphere proxy marker (M2c clearance 用), 半透明, "
                       "跟随 articulation. 帮你肉眼判断 arm/EE 半径是否合理. "
                       "默认关闭.")
    p.add_argument("--proxy_arm_radius", type=float, default=None,
                   help="M2c sphere proxy 的 arm/link 半径 (m). 默认 0.06.")
    p.add_argument("--proxy_ee_radius", type=float, default=None,
                   help="M2c sphere proxy 的 EE/finger 半径 (m). 默认 0.03.")
    return p.parse_args()


def _spawn_preinsert_markers(axis_length=0.10, axis_radius=0.004, sphere_radius=0.010):
    """在 /World/viz/ 下 spawn peg_axis / hole_axis 箭头 + preinsert_target 球."""
    try:
        import omni.usd
        from pxr import UsdGeom, Sdf, Gf, Vt
    except ImportError:
        return None
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    def _make_arrow(path, color, length, radius):
        head_length = 0.25 * length
        shaft_length = length - head_length
        xform_prim = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        xf = UsdGeom.Xformable(xform_prim.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp()
        r_op = xf.AddOrientOp()
        t_op.Set(Gf.Vec3d(0.0, 0.0, 10.0))
        r_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        shaft_path = path + "/Shaft"
        cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(shaft_path))
        cyl.GetRadiusAttr().Set(radius)
        cyl.GetHeightAttr().Set(shaft_length)
        cyl.GetAxisAttr().Set("Z")
        cyl.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        cxf = UsdGeom.Xformable(cyl.GetPrim())
        cxf.ClearXformOpOrder()
        cxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, shaft_length / 2.0))

        head_path = path + "/Head"
        cone = UsdGeom.Cone.Define(stage, Sdf.Path(head_path))
        cone.GetRadiusAttr().Set(2.2 * radius)
        cone.GetHeightAttr().Set(head_length)
        cone.GetAxisAttr().Set("Z")
        cone.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        hxf = UsdGeom.Xformable(cone.GetPrim())
        hxf.ClearXformOpOrder()
        hxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, shaft_length + head_length / 2.0))
        return t_op, r_op

    def _make_sphere(path, color, radius):
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
        sphere.GetRadiusAttr().Set(radius)
        sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(0.0, 0.0, 10.0))
        return t_op

    peg_t, peg_r = _make_arrow("/World/viz/peg_axis_arrow",
                                color=(1.0, 0.35, 0.35),
                                length=axis_length, radius=axis_radius)
    hole_t, hole_r = _make_arrow("/World/viz/hole_axis_arrow",
                                  color=(0.35, 1.0, 0.35),
                                  length=axis_length, radius=axis_radius)
    pre_t = _make_sphere("/World/viz/preinsert_target",
                         color=(1.0, 0.95, 0.15),
                         radius=sphere_radius)

    print(f"[VIS PREINSERT] spawned markers: /World/viz/{{peg_axis_arrow, "
          f"hole_axis_arrow, preinsert_target}}  (axis length={axis_length}m, "
          f"preinsert sphere r={sphere_radius}m)")
    return {"peg_t": peg_t, "peg_r": peg_r,
            "hole_t": hole_t, "hole_r": hole_r,
            "pre_t": pre_t}


def _spawn_sphere_proxies(mdp):
    """在 /World/viz/sphere_proxies/ 下 spawn 38 个半透明球 (左 19 + 右 19),
    跟随 mdp 的 sphere proxy 几何. 颜色: 左红 / 右蓝, alpha=0.3.

    球心走 world 坐标 (physics_view.get_link_transforms 已经是 world frame),
    不加 env_offset — 与 preinsert markers (env-local + env_offset) 不同.

    返回 list translate_op 用于每帧 update; mdp 未启用 _clearance_active 或
    IsaacSim 不可用时返回 None.
    """
    if not mdp._clearance_active:
        print("[VIS PROXIES] mdp._clearance_active=False (M2c 未启用且未设 log_clearance). "
              "sphere proxy 在 _create_observation 不算, 无法可视化. "
              "重跑加 --rew_clearance 0.5 --clearance_soft 0.0 之类启用 M2c, "
              "或 env 端开 log_clearance=True (CLI 暂未透出).")
        return None
    try:
        import omni.usd
        from pxr import UsdGeom, Sdf, Gf, Vt
    except ImportError:
        return None
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    n_per_side = mdp._n_proxies_per_side                 # 19
    radii = mdp._proxy_radii_per_side.detach().cpu().numpy()  # [19], configured radii
    handles = []

    def _make_sphere(path, color, radius):
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
        sphere.GetRadiusAttr().Set(float(radius))
        sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        sphere.GetDisplayOpacityAttr().Set(Vt.FloatArray([0.30]))
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.ClearXformOpOrder()
        t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(0.0, 0.0, 100.0))   # 初始放远处, 第一帧 update 后到位
        return t_op, sphere

    UsdGeom.Xform.Define(stage, Sdf.Path("/World/viz/sphere_proxies"))
    for side, color in [("left", (1.0, 0.4, 0.4)), ("right", (0.4, 0.4, 1.0))]:
        for i in range(n_per_side):
            path = f"/World/viz/sphere_proxies/{side}_{i:02d}"
            t_op, _ = _make_sphere(path, color, float(radii[i]))
            handles.append(t_op)

    print(f"[VIS PROXIES] spawned {2 * n_per_side} sphere proxies "
          f"(left=red / right=blue, opacity=0.3). 每侧 19 球: 8 关节 + 7 中点 + 4 EE.")
    return handles


def _update_sphere_proxies(mdp, viz_env_idx, proxy_handles):
    """每帧从 mdp._compute_min_clearance() 拿到 left_proxies / right_proxies,
    更新 marker 位置. 这两个张量已经是 world 坐标 (physics_view.get_link_transforms
    返回 world frame), 直接写入 /World/viz 下不需要再加 env_offset.
    """
    if proxy_handles is None:
        return
    from pxr import Gf
    _, info = mdp._compute_min_clearance()    # info["left_proxies"] [N_env, 19, 3] world coord
    left = info["left_proxies"][viz_env_idx].detach().cpu().numpy()
    right = info["right_proxies"][viz_env_idx].detach().cpu().numpy()
    n_per = mdp._n_proxies_per_side
    for i in range(n_per):
        x, y, z = left[i]
        proxy_handles[i].Set(Gf.Vec3d(float(x), float(y), float(z)))
    for i in range(n_per):
        x, y, z = right[i]
        proxy_handles[n_per + i].Set(Gf.Vec3d(float(x), float(y), float(z)))


def _update_preinsert_markers(frames, handles, env_idx, env_offset_world):
    """frames 里的 pos 是 env-local; marker 写到 /World 必须加 env_offset_world.

    env_offset_world 是 viz_env_idx 这个 env 的 base 在 world 里的位置 (mdp._task.env_pos).
    没有这个偏移, viz_env_idx>0 时 marker 会画在错的 env 上方.
    """
    if handles is None or frames is None:
        return
    from pxr import Gf
    ox, oy, oz = env_offset_world
    pt = frames["peg_tip_pos"][env_idx].cpu().tolist()
    pq = frames["peg_axis_quat"][env_idx].cpu().tolist()   # wxyz
    he = frames["hole_entry_pos"][env_idx].cpu().tolist()
    hq = frames["hole_axis_quat"][env_idx].cpu().tolist()
    pi = frames["preinsert_target_pos"][env_idx].cpu().tolist()

    handles["peg_t"].Set(Gf.Vec3d(pt[0] + ox, pt[1] + oy, pt[2] + oz))
    handles["peg_r"].Set(Gf.Quatf(pq[0], Gf.Vec3f(pq[1], pq[2], pq[3])))
    handles["hole_t"].Set(Gf.Vec3d(he[0] + ox, he[1] + oy, he[2] + oz))
    handles["hole_r"].Set(Gf.Quatf(hq[0], Gf.Vec3f(hq[1], hq[2], hq[3])))
    handles["pre_t"].Set(Gf.Vec3d(pi[0] + ox, pi[1] + oy, pi[2] + oz))


def _print_preinsert_frames(frames, errors, env_idx):
    if frames is None:
        print("[VIS PREINSERT] frames 不可用, 跳过 preinsert 可视化")
        return
    pt = frames["peg_tip_pos"][env_idx].tolist()
    pa = frames["peg_axis"][env_idx].tolist()
    he = frames["hole_entry_pos"][env_idx].tolist()
    ha = frames["hole_axis"][env_idx].tolist()
    pi = frames["preinsert_target_pos"][env_idx].tolist()
    align = float(torch.sum(frames["peg_axis"][env_idx] * frames["hole_axis"][env_idx]))
    def _fmt(v):
        return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"
    print(f"[VIS PREINSERT] env {env_idx} frames (env-local coords):")
    print(f"  peg_tip      pos={_fmt(pt)}  axis={_fmt(pa)}")
    print(f"  hole_entry   pos={_fmt(he)}  axis={_fmt(ha)}")
    print(f"  preinsert    pos={_fmt(pi)}")
    print(f"  dot(peg_axis, hole_axis) = {align:+.4f}  "
          "(理想插入对齐 = -1; 当前是默认姿态下两臂末端相对几何)")
    if errors is not None:
        print(f"[VIS PREINSERT] env {env_idx} errors:")
        print(f"  pos_err     = {float(errors['pos_err'][env_idx]):+.4f} m")
        print(f"  axis_dot    = {float(errors['axis_dot'][env_idx]):+.4f}  "
              "(-1 = 完美轴反平行)")
        print(f"  axis_err    = {float(errors['axis_err'][env_idx]):+.4f}  "
              "(0 = 完美对齐, 量级 ∈ [0, 2])")
        print(f"  axial_dist  = {float(errors['axial_dist'][env_idx]):+.4f} m  "
              "(>0: peg 在 hole 开口外侧)")
        print(f"  radial_err  = {float(errors['radial_err'][env_idx]):+.4f} m")
        print(f"  success     = {bool(errors['success_mask'][env_idx])}")


def _focus_camera_on_env(mdp, env_idx):
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        return
    world_pos, _ = mdp._task.robots.get_world_poses()
    base = world_pos[env_idx].detach().cpu().tolist()
    target = [base[0], base[1], base[2] + 0.45]
    eye = [target[0] + 2.0, target[1] - 1.6, target[2] + 1.0]
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    print(f"[VIS CAMERA] env {env_idx} eye={tuple(round(x, 3) for x in eye)} "
          f"target={tuple(round(x, 3) for x in target)}")


def main():
    args = parse_args()
    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(f"--viz_env_idx ({args.viz_env_idx}) 必须落在 [0, {args.num_envs - 1}]")

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=False,
        initial_joint_noise=args.initial_joint_noise,
        preinsert_success_pos_threshold=args.preinsert_success_pos_threshold,
    )
    if args.preinsert_offset is not None:
        env_kwargs["preinsert_offset"] = args.preinsert_offset
    if args.success_axis_threshold is not None:
        env_kwargs["success_axis_threshold"] = args.success_axis_threshold
    if args.proxy_arm_radius is not None:
        env_kwargs["proxy_arm_radius"] = args.proxy_arm_radius
    if args.proxy_ee_radius is not None:
        env_kwargs["proxy_ee_radius"] = args.proxy_ee_radius
    if args.show_proxies:
        # log_clearance 让 _create_observation 每步算 sphere proxy, 否则
        # _compute_min_clearance 用的 cache 是 None.
        env_kwargs["log_clearance"] = True

    from envs import DualArmPegHoleEnv

    if args.n_resets < 1:
        raise ValueError(f"--n_resets ({args.n_resets}) 必须 >= 1")

    mdp = DualArmPegHoleEnv(**env_kwargs)
    _focus_camera_on_env(mdp, args.viz_env_idx)
    mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)

    print(f"[VIS ENV] {env_kwargs}")

    obs = None
    frames = None
    errors = None
    batch_pos_err = []
    batch_axis_err = []
    batch_radial_err = []
    batch_axial_dist = []
    batch_success = []
    for r in range(args.n_resets):
        obs, _ = mdp.reset_all(mask)
        # M0+ 解析式 frame 不再依赖 XFormPrim; 但 reset_all 后 BODY_POS / BODY_ROT
        # view 仍可能 stale, 显式 step 一次同步.
        mdp._world.step(render=False)
        frames = mdp.get_preinsert_frames()
        errors = mdp._compute_preinsert_errors(frames)
        batch_pos_err.append(errors["pos_err"].detach())
        batch_axis_err.append(errors["axis_err"].detach())
        batch_radial_err.append(errors["radial_err"].detach())
        batch_axial_dist.append(errors["axial_dist"].detach())
        batch_success.append(errors["success_mask"].detach())

    pos = torch.cat(batch_pos_err)
    ax = torch.cat(batch_axis_err)
    rad = torch.cat(batch_radial_err)
    axd = torch.cat(batch_axial_dist)
    suc = torch.cat(batch_success)
    n_total = pos.numel()
    # success_rate 名字 + 阈值显式标出: axis_th=inf 时是 pos-only success, 不是
    # M2 的 pos∧axis success — 不标清楚容易误读成 M2 的乐观估计.
    is_pos_only = math.isinf(mdp._success_axis_threshold)
    success_label = "pos_success_rate" if is_pos_only else "success_rate (pos∧axis)"
    print(
        f"[VIS PREINSERT STATS] aggregated over n_resets={args.n_resets} × "
        f"num_envs={args.num_envs} = {n_total} samples  "
        f"(pos<{mdp._preinsert_success_pos_threshold:.3f}m, "
        f"axis<{mdp._success_axis_threshold:.3f})"
    )
    for name, t in [("pos_err", pos), ("axis_err", ax),
                    ("radial_err", rad), ("axial_dist", axd)]:
        print(
            f"  {name:11s} mean={float(t.mean()):+.4f}  "
            f"min={float(t.min()):+.4f}  max={float(t.max()):+.4f}  "
            f"std={float(t.std(unbiased=False)):+.4f}"
        )
    print(f"  {success_label} = {float(suc.float().mean()):.4f}")

    mdp._world.step(render=True)
    handles = _spawn_preinsert_markers()
    # env_pos: cloned env 的世界 base 偏移. frames 是 env-local, marker 在 /World/viz/...
    # 必须加这个偏移才能落在 viz_env_idx 指定的那个 env 上.
    env_offset_world = tuple(
        float(x) for x in mdp._task.env_pos[args.viz_env_idx].detach().cpu().tolist()
    )
    print(f"[VIS] env {args.viz_env_idx} world offset = {env_offset_world}")
    frames = mdp.get_preinsert_frames()
    errors = mdp._compute_preinsert_errors(frames)
    _print_preinsert_frames(frames, errors, args.viz_env_idx)
    _update_preinsert_markers(frames, handles, args.viz_env_idx, env_offset_world)
    proxy_handles = None
    if args.show_proxies:
        proxy_handles = _spawn_sphere_proxies(mdp)
        _update_sphere_proxies(mdp, args.viz_env_idx, proxy_handles)

    if args.duration <= 0:
        print("[VIS] 窗口已打开. 观察完后按 Ctrl-C 退出.")
    else:
        print(f"[VIS] 窗口将保持 {args.duration:.1f}s.")

    start_t = time.monotonic()
    try:
        while True:
            mdp._world.step(render=True)
            frames = mdp.get_preinsert_frames()
            _update_preinsert_markers(frames, handles, args.viz_env_idx, env_offset_world)
            _update_sphere_proxies(mdp, args.viz_env_idx, proxy_handles)
            if args.duration > 0 and time.monotonic() - start_t >= args.duration:
                break
            time.sleep(args.idle_dt)
    except KeyboardInterrupt:
        pass
    finally:
        mdp.stop()


if __name__ == "__main__":
    main()
