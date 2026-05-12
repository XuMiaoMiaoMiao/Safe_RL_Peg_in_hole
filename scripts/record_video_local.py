"""Record a local visualization video for one .msh checkpoint.

This is a lightweight local entry point. It does not use Hydra, Apptainer, or
Slurm. Put checkpoints under results/video_local/checkpoint if you want a
stable project-local place for manual video recording.

Example:
    python scripts/record_video_local.py \
        --checkpoint_path results/video_local/checkpoint/best_agent.msh \
        --n_episodes 1 \
        --use_axis_resid_obs \
        --preinsert_success_pos_threshold 0.10 \
        --rew_home 0.0005 \
        --clearance_hard=-inf
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import deterministic_policy, parse_home_weights
from scripts.record_video import (
    _get_frame,
    _make_writer,
    _setup_offscreen_camera,
)

VIDEO_LOCAL_ROOT = PROJECT_ROOT / "results" / "video_local"
DEFAULT_CHECKPOINT_DIR = VIDEO_LOCAL_ROOT / "checkpoint"
DEFAULT_OUTPUT_DIR = VIDEO_LOCAL_ROOT / "video"


def _project_path(value: str | Path) -> Path:
    """Resolve relative paths from the project root for repeatable CLI use."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record local mp4 video for one mushroom-rl .msh checkpoint."
    )

    p.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=(
            "Path to the .msh checkpoint. Relative paths are resolved from the "
            f"project root. Suggested location: {DEFAULT_CHECKPOINT_DIR}"
        ),
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Video output directory. Defaults to results/video_local/video.",
    )
    p.add_argument("--tag", type=str, default="", help="Optional filename prefix.")

    # Local rendering defaults.
    p.add_argument("--headless", action="store_true", help="Run without Isaac Sim UI.")
    p.add_argument(
        "--num_envs",
        type=int,
        default=2,
        help="Must be at least 2 because of the cloner issue; only viz_env_idx is recorded.",
    )
    p.add_argument("--viz_env_idx", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument(
        "--viewport_capture_wait_frames",
        type=int,
        default=8,
        help="How many render ticks to wait for each viewport capture file.",
    )
    p.add_argument(
        "--frame_source",
        choices=("viewport", "replicator"),
        default="viewport",
        help=(
            "viewport records the same render path as the Isaac Sim UI. "
            "replicator uses the old offscreen render product path."
        ),
    )
    p.add_argument(
        "--no_world_render_sync",
        action="store_true",
        help=(
            "Do not call mdp._world.render() after each physics step. The default "
            "keeps local UI/Fabric transforms synchronized before grabbing frames."
        ),
    )
    p.add_argument(
        "--no_force_render_step",
        action="store_true",
        help=(
            "Do not force internal mdp._world.step(render=False) calls to render=True. "
            "The default is needed when local recordings show stale USD/default poses."
        ),
    )
    p.add_argument(
        "--debug_motion_every",
        type=int,
        default=25,
        help="Print action and joint motion diagnostics every N steps. Use 0 to disable.",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Use SAC sampling policy. Default is deterministic tanh(mu).",
    )

    # Env parameters. Keep these aligned with the checkpoint's training command.
    p.add_argument("--initial_joint_noise", type=float, default=None)
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None)
    p.add_argument("--preinsert_offset", type=float, default=None)
    p.add_argument("--rew_action", type=float, default=None)
    p.add_argument("--rew_axis", type=float, default=None)
    p.add_argument("--rew_success", type=float, default=None)
    p.add_argument("--rew_pos_success", type=float, default=None)
    p.add_argument("--rew_home", type=float, default=None)
    p.add_argument("--home_weights", type=parse_home_weights, default=None)
    p.add_argument("--axis_gate_radius", type=float, default=None)
    p.add_argument("--success_axis_threshold", type=float, default=None)
    p.add_argument("--hold_success_steps", type=int, default=10)
    p.add_argument("--clearance_hard", type=float, default=None)
    p.add_argument("--proxy_arm_radius", type=float, default=None)
    p.add_argument("--proxy_ee_radius", type=float, default=None)
    p.add_argument(
        "--exclude_ee_from_physx_self_collision",
        action="store_true",
        help=(
            "Stage 3 recording: exclude EE links from PhysX self-collision groups "
            "so peg-hole contact does not trigger hard absorbing."
        ),
    )
    p.add_argument(
        "--use_axis_resid_obs",
        action="store_true",
        help="Use the 34-D axis_resid observation. Must match training.",
    )

    return p.parse_args()


def _build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs = dict(
        num_envs=args.num_envs,
        headless=args.headless,
        success_hold_steps=args.hold_success_steps,
        # Keep recording episodes visually complete instead of cutting at hold-N.
        terminal_hold_bonus=0.0,
    )

    for key in (
        "initial_joint_noise",
        "preinsert_success_pos_threshold",
        "preinsert_offset",
        "rew_action",
        "rew_axis",
        "rew_success",
        "rew_pos_success",
        "rew_home",
        "home_weights",
        "axis_gate_radius",
        "success_axis_threshold",
        "clearance_hard",
        "proxy_arm_radius",
        "proxy_ee_radius",
    ):
        value = getattr(args, key)
        if value is not None:
            env_kwargs[key] = value

    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    if args.exclude_ee_from_physx_self_collision:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True

    return env_kwargs


@contextmanager
def _force_world_step_render(mdp, enabled: bool):
    """Temporarily force Isaac world.step(..., render=True) for local recording.

    The training/eval environment intentionally steps with render=False. That keeps
    tensor observations fresh, but local RTX/Replicator output can keep showing stale
    USD articulation poses. For recording, every internal physics step must also
    render so the viewport/render product sees the same pose as the policy.
    """
    if not enabled:
        yield
        return

    original_step = mdp._world.step

    def step_with_render(*args, **kwargs):
        if "render" in kwargs:
            kwargs["render"] = True
        elif args and isinstance(args[0], bool):
            args = (True, *args[1:])
        else:
            kwargs["render"] = True
        return original_step(*args, **kwargs)

    mdp._world.step = step_with_render
    try:
        yield
    finally:
        mdp._world.step = original_step


def _setup_viewport_camera(position, look_at):
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import get_active_viewport

    set_camera_view(
        eye=position,
        target=look_at,
        camera_prim_path="/OmniverseKit_Persp",
    )
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active Isaac Sim viewport found for local recording.")
    return viewport


def _read_viewport_frame(mdp, viewport, frame_path: Path, width: int, height: int,
                         wait_frames: int):
    import cv2
    from omni.kit.viewport.utility import capture_viewport_to_file

    try:
        frame_path.unlink()
    except FileNotFoundError:
        pass

    capture_viewport_to_file(viewport, file_path=str(frame_path))

    frame = None
    for _ in range(wait_frames):
        mdp._world.render()
        if frame_path.exists() and frame_path.stat().st_size > 0:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is not None:
                break
        time.sleep(0.001)

    if frame is None:
        raise RuntimeError(
            f"Viewport capture did not produce a readable frame after "
            f"{wait_frames} render ticks: {frame_path}"
        )
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def _record_one_agent_local(mdp, agent, args, get_frame, out_dir: Path, prefix: str) -> int:
    """Record one agent, forcing render synchronization for local visualization."""
    env_mask = torch.ones(args.num_envs, dtype=torch.bool, device=mdp._device)

    obs, _ = mdp.reset_all(env_mask)
    if not args.no_world_render_sync:
        mdp._world.render()

    episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=mdp._device)
    prev_joint_pos = obs[:, :14].clone()

    ep_idx = 0
    step_count = 0
    writer = None
    ep_frames = 0

    print(f"[LOCAL REC] agent '{prefix.rstrip('_')}': recording {args.n_episodes} episode(s)")

    try:
        while ep_idx < args.n_episodes:
            if args.stochastic:
                action, _ = agent.policy.draw_action(obs)
            else:
                with deterministic_policy(agent):
                    action, _ = agent.policy.draw_action(obs)

            next_obs, reward, absorbing, _info = mdp.step_all(env_mask, action)
            step_count += 1
            episode_steps[env_mask] += 1
            last = absorbing | (episode_steps >= mdp.info.horizon)

            if not args.no_world_render_sync:
                mdp._world.render()

            if args.debug_motion_every and step_count % args.debug_motion_every == 0:
                joint_delta = (next_obs[:, :14] - prev_joint_pos).abs().mean().item()
                action_abs = torch.as_tensor(action).abs().mean().item()
                reward_mean = torch.as_tensor(reward).float().mean().item()
                print(
                    f"[LOCAL REC] step={step_count:04d} "
                    f"mean|action|={action_abs:.4f} "
                    f"mean|dq|={joint_delta:.6f} "
                    f"reward={reward_mean:.3f}"
                )
                prev_joint_pos = next_obs[:, :14].clone()

            other_last = last.clone()
            other_last[args.viz_env_idx] = False
            if other_last.any():
                reset_obs, _ = mdp.reset_all(other_last)
                if not args.no_world_render_sync:
                    mdp._world.render()
                next_obs = next_obs.clone()
                next_obs[other_last] = reset_obs[other_last]
                episode_steps[other_last] = 0

            frame = get_frame()

            if writer is None:
                out_path = out_dir / f"{prefix}episode_{ep_idx:03d}.mp4"
                writer = _make_writer(out_path, args.fps, args.width, args.height)
                ep_frames = 0
                print(f"[LOCAL REC]   episode {ep_idx:03d} -> {out_path}")

            writer.write(frame)
            ep_frames += 1

            if last[args.viz_env_idx].item():
                writer.release()
                writer = None
                print(
                    f"[LOCAL REC]   episode {ep_idx:03d} done: {ep_frames} frames "
                    f"(total steps: {step_count})"
                )
                ep_idx += 1

                if ep_idx < args.n_episodes:
                    obs, _ = mdp.reset_all(env_mask)
                    if not args.no_world_render_sync:
                        mdp._world.render()
                    episode_steps.zero_()
                    prev_joint_pos = obs[:, :14].clone()
                    continue

            obs = next_obs

            if step_count > args.n_episodes * mdp.info.horizon * 3:
                print("[LOCAL REC] warning: exceeded expected step limit; stopping")
                break

    finally:
        if writer is not None:
            writer.release()
            print(f"[LOCAL REC] interrupted; released writer for episode {ep_idx:03d}")

    return ep_idx


def main() -> None:
    args = parse_args()

    if not (0 <= args.viz_env_idx < args.num_envs):
        raise ValueError(
            f"--viz_env_idx ({args.viz_env_idx}) must be in [0, {args.num_envs - 1}]"
        )

    checkpoint_path = _project_path(args.checkpoint_path)
    if checkpoint_path.suffix != ".msh":
        raise ValueError(f"--checkpoint_path must point to a .msh file: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    out_dir = _project_path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from envs import DualArmPegHoleEnv

    env_kwargs = _build_env_kwargs(args)
    print(f"[LOCAL REC] checkpoint: {checkpoint_path}")
    print(f"[LOCAL REC] output_dir: {out_dir}")
    print(f"[LOCAL REC] env_kwargs: {env_kwargs}")

    mdp = DualArmPegHoleEnv(**env_kwargs)
    try:
        world_pos, _ = mdp._task.robots.get_world_poses()
        base = world_pos[args.viz_env_idx].detach().cpu().tolist()
        cam_target = [base[0], base[1], base[2] + 0.45]
        cam_eye = [cam_target[0] + 2.0, cam_target[1] - 1.6, cam_target[2] + 1.0]
        print(
            "[LOCAL REC] camera "
            f"eye={tuple(round(x, 3) for x in cam_eye)} "
            f"target={tuple(round(x, 3) for x in cam_target)}"
        )
        from mushroom_rl.core import Agent

        agent = Agent.load(str(checkpoint_path))
        prefix = f"{args.tag}_{checkpoint_path.stem}_" if args.tag else f"{checkpoint_path.stem}_"
        with tempfile.TemporaryDirectory(prefix="record_video_local_") as tmp:
            if args.frame_source == "viewport":
                if args.headless:
                    raise ValueError("--frame_source viewport requires non-headless Isaac Sim.")
                viewport = _setup_viewport_camera(cam_eye, cam_target)
                frame_path = Path(tmp) / "viewport_frame.png"

                print("[LOCAL REC] warming up viewport render pipeline ...", flush=True)
                for _ in range(args.warmup_steps):
                    mdp._world.step(render=True)

                def get_frame():
                    return _read_viewport_frame(
                        mdp,
                        viewport,
                        frame_path,
                        args.width,
                        args.height,
                        args.viewport_capture_wait_frames,
                    )
            else:
                rec_annot, _rec_rp = _setup_offscreen_camera(
                    args.width, args.height, cam_eye, cam_target
                )

                import omni.replicator.core as rep

                print("[LOCAL REC] warming up replicator render pipeline ...", flush=True)
                for _ in range(args.warmup_steps):
                    rep.orchestrator.step(rt_subframes=1)

                def get_frame():
                    return _get_frame(rec_annot, args.width, args.height)

            with _force_world_step_render(mdp, enabled=not args.no_force_render_step):
                done = _record_one_agent_local(mdp, agent, args, get_frame, out_dir, prefix)
    finally:
        mdp.stop()

    print(f"[LOCAL REC] done. Recorded {done} video(s) under: {out_dir}")


if __name__ == "__main__":
    main()
