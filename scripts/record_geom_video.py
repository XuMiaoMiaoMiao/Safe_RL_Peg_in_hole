"""Record geom-stage SAC checkpoints to mp4 (standalone).

Records one checkpoint with the 41D geom env arguments needed by current
insert-stage policies. The cluster YAML uses Replicator's headless offscreen
camera backend; local runs may use the passive viewport capture backend.

Self-contained: the three Replicator helpers (_setup_offscreen_camera,
_get_frame, _make_writer) used to be imported from scripts.record_video and
are now inlined below.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import compute_geom_metrics, deterministic_policy
from scripts._eval_utils import parse_home_weights


# ── Replicator offscreen render helpers (inlined from former record_video.py) ──
# headless 模式里 viewport 的 RTX Hydra Engine 被关闭, viewport-based render
# product 的 get_data() 会因 overscan=None 而崩 ("HydraEngine rtx failed
# creating scene renderer"). 用 rep.create.camera() 建独立 prim + render
# product, 有自己的渲染 context, 不依赖 viewport, headless 下正常工作.


def _setup_offscreen_camera(width: int, height: int, position, look_at):
    """Build a standalone offscreen camera + render product.

    Returns (annot, rp): annotator and render product, fed to _get_frame.
    """
    import omni.replicator.core as rep

    camera = rep.create.camera(
        position=tuple(position),
        look_at=tuple(look_at),
    )
    rp = rep.create.render_product(camera, (width, height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    return annot, rp


def _get_frame(annot, width: int, height: int) -> np.ndarray:
    """Trigger one render pass on the standalone camera, return BGR uint8 frame.

    rep.orchestrator.step() drives the replicator render — fully decoupled from
    viewport / _world.render().
    """
    import omni.replicator.core as rep

    rep.orchestrator.step(rt_subframes=1)
    raw = annot.get_data()

    if hasattr(raw, 'numpy'):
        arr = raw.numpy()
    else:
        arr = np.asarray(raw)

    if arr.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    rgb = arr[..., :3]
    bgr = rgb[..., ::-1].copy()   # RGB → BGR (cv2), copy removes negative stride
    return bgr.astype(np.uint8)


def _make_writer(out_path: Path, fps: int, width: int, height: int):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter cannot open: {out_path}\n"
            "Verify opencv-python-headless is installed (pip show opencv-python-headless)"
        )
    return writer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--viz_env_idx", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "videos"))
    p.add_argument("--tag", type=str, default="geom")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--capture_backend", type=str, default="viewport",
                   choices=("viewport", "replicator"),
                   help="viewport=默认, 用 Kit viewport schedule_capture 抓帧, 不调用 "
                        "rep.orchestrator.step(), 轨迹应与 visualize_policy 一致. "
                        "replicator=旧 headless 离屏相机后端, 可能干扰 geom insert rollout.")
    p.add_argument("--final_hold_seconds", type=float, default=0.0,
                   help="episode 结束后额外重复写入最后一帧多久. 默认 0, 避免视频后半段静止.")
    p.add_argument("--stop_on_absorbing", action="store_true",
                   help="默认目标 env 只按 horizon 结束录制; 传此项才在 absorbing 时提前结束.")
    p.add_argument("--stochastic", action="store_true")

    p.add_argument("--initial_joint_noise", type=float, default=None)
    p.add_argument("--default_pose_variant", type=str, default=None,
                   choices=("easy", "harder"))
    p.add_argument("--preinsert_offset", type=float, default=None)
    p.add_argument("--rew_home", type=float, default=None)
    p.add_argument("--home_weights", type=parse_home_weights, default=None)
    p.add_argument("--hold_success_steps", type=int, default=10)
    p.add_argument("--exclude_ee_from_physx_self_collision", action="store_true")
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="Sphere-proxy hard absorbing threshold. Use -inf to disable "
                        "the sphere-proxy hard termination during recording.")
    p.add_argument("--proxy_arm_radius", type=float, default=None)
    p.add_argument("--proxy_ee_radius", type=float, default=None)
    p.add_argument("--horizon", type=int, default=200)

    p.add_argument("--geom_stage", type=str, default="insert",
                   choices=("prepos", "preaxis", "insert"))
    p.add_argument("--geom_eval_epoch", type=int, default=None)
    p.add_argument("--geom_d_target_neg", type=float, default=-0.08)
    p.add_argument("--geom_d_target_pos", type=float, default=0.03)
    p.add_argument("--geom_d_target_ramp_start", type=int, default=0)
    p.add_argument("--geom_d_target_ramp_end", type=int, default=20)
    p.add_argument("--geom_progress_floor", type=float, default=0.0)

    p.add_argument("--geom_d_sat", type=float, default=0.30)
    p.add_argument("--geom_radial_sat", type=float, default=1.0)
    p.add_argument("--geom_d_th", type=float, default=0.020)
    p.add_argument("--geom_r_tip_th", type=float, default=0.015)
    p.add_argument("--geom_r_max_th", type=float, default=0.025)
    p.add_argument("--geom_axis_th", type=float, default=0.300)
    p.add_argument("--geom_insert_d_ins", type=float, default=0.025)
    p.add_argument("--geom_insert_r_max_th", type=float, default=0.025)
    p.add_argument("--geom_pen_th", type=float, default=0.001)

    p.add_argument("--rew_geom_d", type=float, default=1.0)
    p.add_argument("--rew_geom_radial_tip", type=float, default=0.0)
    p.add_argument("--rew_geom_radial_max", type=float, default=5.0)
    p.add_argument("--rew_geom_axis", type=float, default=1.0)
    p.add_argument("--rew_geom_progress", type=float, default=0.0)
    p.add_argument("--rew_geom_advance", type=float, default=25.0)
    p.add_argument("--geom_gate_radial_sigma", type=float, default=0.025)
    p.add_argument("--geom_gate_axis_sigma", type=float, default=0.30)
    p.add_argument("--rew_geom_penetration", type=float, default=20.0)
    p.add_argument("--geom_gate_penetration_sigma", type=float, default=0.005)
    p.add_argument("--rew_geom_soft_success", type=float, default=1.25)
    p.add_argument("--geom_soft_d_sigma", type=float, default=0.018)
    p.add_argument("--geom_soft_radial_sigma", type=float, default=0.009)
    p.add_argument("--geom_soft_axis_sigma", type=float, default=0.15)
    p.add_argument("--geom_soft_penetration_sigma", type=float, default=0.0025)
    p.add_argument("--geom_d_gate_mode", type=str, default="off",
                   choices=("off", "alignment"))
    p.add_argument("--rew_geom_bad_entry", type=float, default=0.30)
    p.add_argument("--geom_bad_entry_radial_safe", type=float, default=0.014)
    p.add_argument("--geom_bad_entry_axis_safe", type=float, default=0.10)
    p.add_argument("--geom_bad_entry_pen_safe", type=float, default=0.00075)
    p.add_argument("--cost_signal", type=str, default="collision",
                   choices=("collision", "penetration"))
    return p.parse_args()


class _ReplicatorFrameGrabber:
    def __init__(self, annot, width: int, height: int):
        self._annot = annot
        self._width = width
        self._height = height

    def get_frame(self):
        return _get_frame(self._annot, self._width, self._height)


class _ViewportFrameGrabber:
    """Passive viewport capture.

    This backend deliberately avoids omni.replicator.core.orchestrator.step().
    It schedules a capture from the current viewport and calls world.render()
    to flush rendering. world.render() is the same non-physics render refresh
    used by visualize_policy.py during freeze.
    """

    def __init__(self, mdp, width: int, height: int, eye, target):
        self._mdp = mdp
        self._width = int(width)
        self._height = int(height)
        self._loop = asyncio.get_event_loop()

        import omni.kit.viewport.utility as viewport_utils
        from isaacsim.core.utils.viewports import set_camera_view

        window = viewport_utils.create_viewport_window(
            "Geom Recorder",
            width=self._width,
            height=self._height,
            camera_path="/OmniverseKit_Persp",
        )
        self._viewport = window.viewport_api if window else viewport_utils.get_active_viewport()
        if self._viewport is None:
            raise RuntimeError("Could not create or find an active viewport for recording")

        set_camera_view(
            eye=np.asarray(eye, dtype=np.float64),
            target=np.asarray(target, dtype=np.float64),
            camera_prim_path="/OmniverseKit_Persp",
        )

        # Warm up the viewport render path without stepping physics.
        for _ in range(5):
            self._mdp._world.render()

    @staticmethod
    def _buffer_to_bgr(buffer, buffer_size: int, width: int, height: int, byte_format):
        if isinstance(buffer, (bytes, bytearray, memoryview)):
            data = np.frombuffer(buffer, dtype=np.uint8, count=buffer_size)
        else:
            # Official Kit renderer capture tests convert this opaque buffer
            # through omni.kit.renderer_capture, not ctypes.string_at().
            import omni.kit.renderer_capture

            raw = omni.kit.renderer_capture.convert_raw_bytes_to_list(
                buffer, buffer_size, width, height, byte_format
            )
            data = np.asarray(raw, dtype=np.uint8)

        if data.ndim == 1:
            channels = max(1, data.size // max(1, width * height))
            arr = data.reshape((height, width, channels))
        elif data.ndim == 2:
            arr = data.reshape((height, width, data.shape[-1]))
        elif data.ndim == 3:
            arr = data
        else:
            raise RuntimeError(f"Unexpected viewport buffer shape: {data.shape}")

        channels = arr.shape[-1]
        if channels >= 4:
            rgb = arr[:, :, :3]
        elif channels == 3:
            rgb = arr
        else:
            rgb = np.repeat(arr[:, :, :1], 3, axis=2)
        return rgb[:, :, ::-1].copy()

    def get_frame(self):
        import omni.kit.viewport.utility as viewport_utils

        captured = {"frame": None}

        def _on_capture(buffer, buffer_size, width, height, byte_format):
            captured["frame"] = self._buffer_to_bgr(
                buffer, buffer_size, width, height, byte_format
            )

        cap = viewport_utils.capture_viewport_to_buffer(self._viewport, _on_capture)
        # schedule_capture needs a render update. This refreshes graphics only,
        # not physics; visualize_policy.py uses the same call while frozen.
        self._mdp._world.render()

        async def _wait():
            return await asyncio.wait_for(cap.wait_for_result(completion_frames=0), timeout=5.0)

        self._loop.run_until_complete(_wait())
        if captured["frame"] is None:
            raise RuntimeError("Viewport capture completed but no frame was delivered")

        frame = captured["frame"]
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            # Keep cv2 writer dimensions stable if the viewport rounded/scaled.
            import cv2
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame


def _record_one_geom_agent(mdp, agent, args, frame_grabber, out_dir: Path, prefix: str) -> int:
    # Use VectorCore.evaluate for the actual rollout. This keeps policy execution,
    # env reset semantics, and horizon handling identical to eval_sac/visualize.
    # We only hook is_absorbing to capture a rendered frame after each env step.
    from mushroom_rl.core import VectorCore

    episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=mdp._device)
    final_hold_frames = max(0, int(round(args.final_hold_seconds * args.fps)))
    state = {
        "ep_idx": 0,
        "writer": None,
        "ep_frames": 0,
        "last_frame": None,
        "done_recording": False,
    }
    original_is_absorbing = mdp.is_absorbing

    def _as_bool_tensor(value):
        if isinstance(value, torch.Tensor):
            out = value.to(dtype=torch.bool, device=mdp._device).flatten()
        else:
            out = torch.as_tensor(value, dtype=torch.bool, device=mdp._device)
            if out.ndim == 0:
                out = out.repeat(args.num_envs)
            else:
                out = out.flatten()
        return out

    def _close_episode(reason, absorbing):
        writer = state["writer"]
        if writer is None:
            return
        idx = args.viz_env_idx
        if mdp._cached_d is not None:
            print(
                f"[REC GEOM] episode {state['ep_idx']:03d} final "
                f"reason={reason} step={int(episode_steps[idx].item())} "
                f"d={float(mdp._cached_d[idx]):+.4f}m "
                f"radial_max={float(mdp._cached_radial_max[idx]):.4f}m "
                f"axis_err={float(mdp._cached_axis_err[idx]):.4f} "
                f"penetration={float(mdp._cached_penetration_max[idx]) * 1000:.2f}mm "
                f"absorbing={bool(absorbing[idx].item())}",
                flush=True,
            )
        if state["last_frame"] is not None:
            for _ in range(final_hold_frames):
                writer.write(state["last_frame"])
                state["ep_frames"] += 1
        writer.release()
        state["writer"] = None
        print(
            f"[REC GEOM] episode {state['ep_idx']:03d} done: "
            f"{state['ep_frames']} frames "
            f"({state['ep_frames'] / max(1, args.fps):.2f}s)",
            flush=True,
        )
        state["ep_idx"] += 1
        if state["ep_idx"] >= args.n_episodes:
            state["done_recording"] = True

    def hooked_is_absorbing(obs):
        result = original_is_absorbing(obs)
        absorbing = _as_bool_tensor(result)
        episode_steps.add_(1)

        if not state["done_recording"]:
            if state["writer"] is None:
                out_path = out_dir / f"{prefix}episode_{state['ep_idx']:03d}.mp4"
                state["writer"] = _make_writer(out_path, args.fps, args.width, args.height)
                state["ep_frames"] = 0
                state["last_frame"] = None
                print(f"[REC GEOM] episode {state['ep_idx']:03d} -> {out_path}", flush=True)

            frame = frame_grabber.get_frame()
            state["writer"].write(frame)
            state["last_frame"] = frame
            state["ep_frames"] += 1

            target_horizon = episode_steps[args.viz_env_idx] >= mdp.info.horizon
            target_absorb = absorbing[args.viz_env_idx]
            if target_horizon.item():
                _close_episode("horizon", absorbing)
            elif args.stop_on_absorbing and target_absorb.item():
                _close_episode("absorbing", absorbing)

        done = absorbing | (episode_steps >= mdp.info.horizon)
        if done.any():
            episode_steps[done] = 0
        return result

    mdp.is_absorbing = hooked_is_absorbing
    # VectorCore's n_episodes is total episodes across all vector envs, not
    # "episodes for viz_env_idx".  With num_envs=2 and n_episodes=1, both envs
    # finish at horizon and dataset flatten sees 2 env streams but a 1-episode
    # mask.  Run enough total vector episodes while only writing viz_env_idx.
    eval_episodes = max(args.num_envs, args.n_episodes * args.num_envs)
    print(
        f"[REC GEOM] recording via VectorCore.evaluate: "
        f"target_recorded_episodes={args.n_episodes}, "
        f"vector_eval_episodes={eval_episodes}, "
        f"horizon={mdp.info.horizon}, fps={args.fps}, "
        f"final_hold_frames={final_hold_frames}",
        flush=True,
    )
    core = VectorCore(agent, mdp)
    try:
        if args.stochastic:
            dataset = core.evaluate(n_episodes=eval_episodes, render=False, quiet=True)
        else:
            with deterministic_policy(agent):
                dataset = core.evaluate(n_episodes=eval_episodes, render=False, quiet=True)
    finally:
        if state["writer"] is not None:
            state["writer"].release()
            state["writer"] = None
        mdp.is_absorbing = original_is_absorbing

    if mdp._geom_stage is not None:
        try:
            mg = compute_geom_metrics(dataset, mdp, args.hold_success_steps)
            print(
                f"[REC GEOM] dataset check: geom_hold_rate={mg['geom_hold_rate']:.3f} "
                f"max_run={mg['geom_max_run_mean']:.1f} "
                f"final_success={mg['geom_final_success_rate']:.3f} "
                f"final_d={mg['geom_final_d_mean']:+.4f}m "
                f"final_pen={mg['geom_final_penetration_mean'] * 1000:.2f}mm",
                flush=True,
            )
        except Exception as exc:
            print(f"[REC GEOM] dataset check skipped: {exc}", flush=True)
    return state["ep_idx"]


def main():
    args = parse_args()
    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) must be in [0, {args.num_envs - 1}]"
        )

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=(args.capture_backend == "replicator"),
        horizon=args.horizon,
        success_hold_steps=args.hold_success_steps,
        terminal_hold_bonus=0.0,
        geom_stage=args.geom_stage,
        geom_d_target_neg=args.geom_d_target_neg,
        geom_d_target_pos=args.geom_d_target_pos,
        geom_d_target_ramp_start=args.geom_d_target_ramp_start,
        geom_d_target_ramp_end=args.geom_d_target_ramp_end,
        geom_progress_floor=args.geom_progress_floor,
        geom_d_sat=args.geom_d_sat,
        geom_radial_sat=args.geom_radial_sat,
        geom_d_th=args.geom_d_th,
        geom_r_tip_th=args.geom_r_tip_th,
        geom_r_max_th=args.geom_r_max_th,
        geom_axis_th=args.geom_axis_th,
        geom_insert_d_ins=args.geom_insert_d_ins,
        geom_insert_r_max_th=args.geom_insert_r_max_th,
        geom_pen_th=args.geom_pen_th,
        rew_geom_d=args.rew_geom_d,
        rew_geom_radial_tip=args.rew_geom_radial_tip,
        rew_geom_radial_max=args.rew_geom_radial_max,
        rew_geom_axis=args.rew_geom_axis,
        rew_geom_progress=args.rew_geom_progress,
        rew_geom_advance=args.rew_geom_advance,
        geom_gate_radial_sigma=args.geom_gate_radial_sigma,
        geom_gate_axis_sigma=args.geom_gate_axis_sigma,
        rew_geom_penetration=args.rew_geom_penetration,
        geom_gate_penetration_sigma=args.geom_gate_penetration_sigma,
        rew_geom_soft_success=args.rew_geom_soft_success,
        geom_soft_d_sigma=args.geom_soft_d_sigma,
        geom_soft_radial_sigma=args.geom_soft_radial_sigma,
        geom_soft_axis_sigma=args.geom_soft_axis_sigma,
        geom_soft_penetration_sigma=args.geom_soft_penetration_sigma,
        geom_d_gate_mode=args.geom_d_gate_mode,
        rew_geom_bad_entry=args.rew_geom_bad_entry,
        geom_bad_entry_radial_safe=args.geom_bad_entry_radial_safe,
        geom_bad_entry_axis_safe=args.geom_bad_entry_axis_safe,
        geom_bad_entry_pen_safe=args.geom_bad_entry_pen_safe,
        cost_signal=args.cost_signal,
        preinsert_offset=(
            args.preinsert_offset
            if args.preinsert_offset is not None
            else abs(float(args.geom_d_target_neg))
        ),
    )
    if args.initial_joint_noise is not None:
        env_kwargs["initial_joint_noise"] = args.initial_joint_noise
    if args.default_pose_variant is not None:
        env_kwargs["default_pose_variant"] = args.default_pose_variant
    if args.rew_home is not None:
        env_kwargs["rew_home"] = args.rew_home
    if args.home_weights is not None:
        env_kwargs["home_weights"] = args.home_weights
    if args.clearance_hard is not None:
        env_kwargs["clearance_hard"] = args.clearance_hard
    if args.proxy_arm_radius is not None:
        env_kwargs["proxy_arm_radius"] = args.proxy_arm_radius
    if args.proxy_ee_radius is not None:
        env_kwargs["proxy_ee_radius"] = args.proxy_ee_radius
    if args.exclude_ee_from_physx_self_collision:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[REC GEOM] creating DualArmPegHoleEnv ...", flush=True)
    mdp = DualArmPegHoleEnv(**env_kwargs)
    if args.geom_eval_epoch is not None:
        geom_eval_epoch = args.geom_eval_epoch
    elif mdp._geom_stage == "insert":
        geom_eval_epoch = mdp._geom_d_target_ramp_end
    else:
        geom_eval_epoch = 0
    mdp.set_geom_epoch(geom_eval_epoch)
    print(
        f"[REC GEOM] stage={mdp._geom_stage} set_geom_epoch({geom_eval_epoch}) "
        f"d_target_eff={mdp._geom_d_target_eff:+.4f}m "
        f"pen_th={mdp._geom_pen_th * 1000:.1f}mm",
        flush=True,
    )

    print(f"[REC GEOM] setting up {args.capture_backend} camera ...", flush=True)
    world_pos, _ = mdp._task.robots.get_world_poses()
    base = world_pos[args.viz_env_idx].detach().cpu().tolist()
    cam_target = [base[0], base[1], base[2] + 0.45]
    cam_eye = [cam_target[0] + 2.0, cam_target[1] - 1.6, cam_target[2] + 1.0]
    print(
        f"[REC GEOM] camera eye={tuple(round(x, 3) for x in cam_eye)} "
        f"target={tuple(round(x, 3) for x in cam_target)}",
        flush=True,
    )
    if args.capture_backend == "replicator":
        rec_annot, _ = _setup_offscreen_camera(args.width, args.height, cam_eye, cam_target)
        frame_grabber = _ReplicatorFrameGrabber(rec_annot, args.width, args.height)

        import omni.replicator.core as rep

        print("[REC GEOM] warming up Replicator render pipeline ...", flush=True)
        for _ in range(10):
            rep.orchestrator.step(rt_subframes=1)
    else:
        frame_grabber = _ViewportFrameGrabber(mdp, args.width, args.height, cam_eye, cam_target)

    from mushroom_rl.core import Agent

    agent_path = Path(args.agent_path)
    print(f"[REC GEOM] loading agent: {agent_path}", flush=True)
    agent = Agent.load(str(agent_path))
    prefix = f"{args.tag}_{agent_path.stem}_" if args.tag else f"{agent_path.stem}_"
    done = _record_one_geom_agent(mdp, agent, args, frame_grabber, out_dir, prefix)
    mdp.stop()
    print(f"[REC GEOM] done. {done} video(s) saved in: {out_dir}")


if __name__ == "__main__":
    main()
