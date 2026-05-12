"""Record a visualize_policy.py run by capturing the X11 screen.

This script is intentionally outside the IsaacSim render loop.  It starts
ffmpeg x11grab, runs the command after ``--`` (typically visualize_policy.py),
then stops ffmpeg when that command exits.

Why this exists:
  scripts.record_video / record_geom_video use Replicator offscreen rendering.
  Replicator requires rep.orchestrator.step() to fetch frames, and that extra
  step perturbs geom-stage policy rollouts.  Screen capture is passive: it
  records exactly the GUI window you see, without touching the simulation.

Example:
  python scripts/record_visualize_screen.py \\
      --output results/videos/v8_visual.mp4 \\
      --geometry 1920x1080+0,0 \\
      -- \\
      python scripts/visualize_policy.py --agent_path ... --geom_stage insert
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def _parse_geometry(text: str) -> tuple[int, int, int, int]:
    """Parse WxH+X,Y or WxH+X+Y into width, height, x, y."""
    m = re.fullmatch(r"(\d+)x(\d+)(?:\+(\d+)(?:[,+](\d+))?)?", text.strip())
    if not m:
        raise ValueError(
            "--geometry must be like 1920x1080+0,0 or 1920x1080+0+0"
        )
    width = int(m.group(1))
    height = int(m.group(2))
    x = int(m.group(3) or 0)
    y = int(m.group(4) or 0)
    return width, height, x, y


def _display_size(display: str) -> tuple[int, int]:
    out = subprocess.check_output(
        ["xdpyinfo", "-display", display],
        text=True,
        stderr=subprocess.STDOUT,
    )
    m = re.search(r"dimensions:\s+(\d+)x(\d+)\s+pixels", out)
    if not m:
        raise RuntimeError("Could not parse screen dimensions from xdpyinfo")
    return int(m.group(1)), int(m.group(2))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record visualize_policy.py by passive X11 screen capture."
    )
    p.add_argument("--output", required=True, help="Output .mp4 path.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--geometry",
        default=None,
        help="Capture region: WxH+X,Y. Default: full DISPLAY from xdpyinfo.",
    )
    p.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="veryfast")
    p.add_argument(
        "--stop_delay",
        type=float,
        default=0.5,
        help="Seconds to keep recording after the command exits.",
    )
    p.add_argument(
        "--draw_mouse",
        action="store_true",
        help="Include mouse cursor in the recording.",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after --, e.g. python scripts/visualize_policy.py ...",
    )
    args = p.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("Missing command after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.geometry is None:
        width, height = _display_size(args.display)
        x = y = 0
    else:
        width, height, x, y = _parse_geometry(args.geometry)

    # yuv420p / libx264 need even dimensions.
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        raise SystemExit(f"Invalid capture size: {width}x{height}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-f",
        "x11grab",
        "-draw_mouse",
        "1" if args.draw_mouse else "0",
        "-framerate",
        str(args.fps),
        "-video_size",
        f"{width}x{height}",
        "-i",
        f"{args.display}+{x},{y}",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]

    print("[SCREEN REC] starting ffmpeg:")
    print("  " + " ".join(ffmpeg_cmd), flush=True)
    ffmpeg = subprocess.Popen(ffmpeg_cmd)

    print("[SCREEN REC] running command:")
    print("  " + " ".join(args.command), flush=True)
    cmd_rc = 1
    try:
        cmd = subprocess.Popen(args.command)
        cmd_rc = cmd.wait()
        if args.stop_delay > 0:
            time.sleep(args.stop_delay)
    finally:
        print("[SCREEN REC] stopping ffmpeg...", flush=True)
        if ffmpeg.poll() is None:
            ffmpeg.send_signal(signal.SIGINT)
            try:
                ffmpeg.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                ffmpeg.terminate()
                try:
                    ffmpeg.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    ffmpeg.kill()
                    ffmpeg.wait()

    if ffmpeg.returncode not in (0, 255):
        print(f"[SCREEN REC] ffmpeg exited with {ffmpeg.returncode}", flush=True)
    print(f"[SCREEN REC] wrote {out_path}", flush=True)
    return cmd_rc


if __name__ == "__main__":
    raise SystemExit(main())
