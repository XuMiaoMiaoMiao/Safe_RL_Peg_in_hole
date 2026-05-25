#!/usr/bin/env python3
"""Plot SAC vs LagSAC benchmark — 1×3 paper-style figure (matplotlib).

Three panels per pose:

    Discounted Return  |  Maximum Violation per Episode  |  Episodic Sum of Cost

Aggregation:
    --band ci    mean  + 95 % CI       (t-distribution; default)
    --band sem   mean  ± σ / √n
    --band std   mean  ± σ
    --band none  mean line only

Strict mode (default) refuses to plot runs whose logs lack
`rollout_ep_max_violation` / `rollout_ep_cost` / `cost_scale=1.0`. Pass
``--allow_legacy_proxy`` to backfill from a per-step proxy with a clearly
warned subtitle (explicit ``cost_scale != 1.0`` still hard-fails — that case
lies about the panel and cannot be opted out of).

Typical use::

    python scripts/plot_paper_benchmark.py \\
        --logs /tmp/final_bench_logs \\
        --out_dir results/plots/final_$(date +%Y%m%d) \\
        --settings final

Each pose produces a PNG + sibling SVG. The CSV ``stage1_paper_curves.csv``
exports every (setting, algo, metric, epoch) cell as
``center, band_lo, band_hi, n, band``.

Borrows MushroomRL's matplotlib pattern: one ``ax.plot`` + ``ax.fill_between``
per algorithm (see ``mushroom_rl/utils/plot.py:plot_mean_conf``).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


# ── constants ────────────────────────────────────────────────────────────────

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COLORS = {"LagSAC": "#2563eb", "SAC": "#d97706"}
ORDER = ("LagSAC", "SAC")

PANELS = (
    ("J",                  "Discounted Return",                "J (higher is better)"),
    ("maximum_violation",  "Maximum Violation per Episode",    "mean per-episode max cost (rollout)"),
    ("rollout_ep_cost",    "Episodic Sum of Cost",             "mean per-episode cost sum (rollout)"),
)

BAND_LABEL = {
    "ci":   "mean line, shaded = 95 % CI (t-distribution)",
    "sem":  "mean line, shaded = ± σ / √n",
    "std":  "mean line, shaded = ± σ",
    "none": "mean line only",
}

# CSV exports panel metrics plus a few summary-table metrics.
CSV_METRICS = (
    "J", "maximum_violation", "success",
    "eval_step_cost", "eval_ep_cost", "rollout_ep_cost",
    "hard_collision_events", "epoch_collision_total", "epoch_collision_table",
)


# ── log parsing ──────────────────────────────────────────────────────────────

def run_meta(path: Path) -> tuple[str, str, int] | None:
    """Parse algo/setting/seed from a log file stem like ``lag_final_s7``."""
    m = re.match(r"(lag|sac)_(default|harder|final)_s(\d+)$", path.stem)
    if not m:
        return None
    algo = "LagSAC" if m.group(1) == "lag" else "SAC"
    return m.group(2), algo, int(m.group(3))


def grab_float(line: str, key: str) -> float | None:
    """Return ``key=NUM`` or ``key: NUM`` from a line, or ``None``."""
    for pat in (rf"{re.escape(key)}=({NUM})", rf"{re.escape(key)}:\s*({NUM})"):
        m = re.search(pat, line)
        if m:
            return float(m.group(1))
    return None


def grab_int(line: str, key: str) -> int | None:
    val = grab_float(line, key)
    return None if val is None else int(val)


# Keys grabbed off a "cost/collision stats" line (per epoch).
_COST_KEYS_FLOAT = (
    "eval_step_cost", "eval_ep_cost", "rollout_ep_cost",
    "rollout_ep_max_violation", "eval_ep_max_violation",
)
_COST_KEYS_INT = (
    "epoch_collision_total", "epoch_collision_sphere",
    "epoch_collision_physx", "epoch_collision_table",
    "epoch_absorb_total", "epoch_absorb_table",
)


def parse_run(path: Path) -> list[dict]:
    """Parse one training-log file into a list of per-epoch row dicts.

    Also captures the run-level ``cost_scale`` (printed once at env setup) and
    stamps it on every row so ``main()`` can assert it equals 1.0.
    """
    rows: list[dict] = []
    current: dict | None = None
    run_cost_scale: float | None = None

    for line in path.read_text(errors="replace").splitlines():
        # One-shot capture of cost_scale (printed once before epoch 1).
        if run_cost_scale is None and "cost_scale=" in line:
            run_cost_scale = grab_float(line, "cost_scale")

        # "Epoch N | J: ... R: ... best_J: ... best_geom: ..." starts a row.
        m = re.search(
            rf"Epoch\s+(\d+)\s+\|\s+J:\s*({NUM})\s+R:\s*({NUM})"
            rf".*?best_J:\s*({NUM}).*?best_geom:\s*({NUM})",
            line,
        )
        if m:
            current = {
                "epoch":     int(m.group(1)),
                "J":         float(m.group(2)),
                "R":         float(m.group(3)),
                "best_J":    float(m.group(4)),
                "best_geom": float(m.group(5)),
            }
            rows.append(current)
            continue

        if current is None:
            continue

        if "geom eval" in line:
            for key in ("geom_step_rate", "geom_hold_rate", "geom_max_run_mean", "final_success_rate"):
                val = grab_float(line, key)
                if val is not None:
                    current[key] = val
            current["success"] = current.get("geom_hold_rate", current.get("best_geom", 0.0))
            continue

        if "eval_ep_cost=" in line:
            for key in _COST_KEYS_FLOAT:
                val = grab_float(line, key)
                if val is not None:
                    current[key] = val
            for key in _COST_KEYS_INT:
                val = grab_int(line, key)
                if val is not None:
                    current[key] = val
            current["hard_collision_events"] = (
                current.get("epoch_collision_physx", 0)
                + current.get("epoch_collision_table", 0)
            )
            # Legacy per-step proxy stored under a separate key; main() opts
            # into it via --allow_legacy_proxy for replays of old logs.
            current["maximum_violation_proxy"] = max(
                current.get("eval_step_cost", 0.0),
                current.get("rollout_ep_cost", 0.0) / 100.0,
            )
            if "rollout_ep_max_violation" in current:
                current["maximum_violation"] = current["rollout_ep_max_violation"]
            # else: leave unset; main() either fails (strict) or backfills.

    # Per-row defaults for fields with sensible zeros.
    for row in rows:
        row.setdefault("success", row.get("best_geom", 0.0))
        for k in ("eval_step_cost", "eval_ep_cost"):
            row.setdefault(k, 0.0)
        for k in ("epoch_collision_total", "epoch_collision_table", "hard_collision_events"):
            row.setdefault(k, 0)
        # Deliberately NOT defaulting: rollout_ep_cost, rollout_ep_max_violation,
        # maximum_violation. Their absence is the strict-mode signal that this
        # run predates the rollout cost tracker.
        if run_cost_scale is not None:
            row["cost_scale"] = run_cost_scale
    return rows


def collect(log_dir: Path,
            exclude_seeds: set[int] | None = None,
            settings: set[str] | None = None,
            ) -> dict[tuple[str, str, int], list[dict]]:
    """Discover all training-log files under ``log_dir`` and parse them."""
    runs: dict[tuple[str, str, int], list[dict]] = {}
    exclude_seeds = exclude_seeds or set()
    for path in sorted(log_dir.glob("*.log")):
        meta = run_meta(path)
        if meta is None:
            continue
        setting, _algo, seed = meta
        if settings is not None and setting not in settings:
            continue
        if seed in exclude_seeds:
            continue
        rows = parse_run(path)
        if rows:
            runs[meta] = rows
    return runs


# ── aggregation ──────────────────────────────────────────────────────────────

def summarize(values, band: str) -> tuple[float, float, float]:
    """Return ``(center, band_lo, band_hi)`` for one (epoch, metric) cell.

    ``ci`` uses the t-distribution (proper for finite n; matches MushroomRL's
    ``utils.plot.get_mean_and_confidence``).
    """
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n == 0:
        return math.nan, math.nan, math.nan

    center = float(vals.mean())
    if n < 2:
        return center, center, center

    sd = float(vals.std(ddof=1))
    se = sd / math.sqrt(n)
    if band == "ci":
        # st.t.interval returns (lo, hi) around 0; take the half-width.
        # Guard se==0 (all seeds identical → degenerate t-dist, scipy returns NaN).
        if se > 0:
            _, half = st.t.interval(0.95, n - 1, scale=se)
            err = float(half)
        else:
            err = 0.0
    elif band == "sem":
        err = se
    elif band == "std":
        err = sd
    elif band == "none":
        err = 0.0
    else:
        raise ValueError(f"Unknown band: {band!r}")
    return center, center - err, center + err


def aggregate(runs, setting: str, metric: str, band: str,
              ) -> dict[str, list[dict]]:
    """Compute per-algo time series for one (setting, metric)."""
    out: dict[str, list[dict]] = {}
    for algo in ORDER:
        by_epoch: dict[int, list[float]] = {}
        for (s, a, _seed), rows in runs.items():
            if s != setting or a != algo:
                continue
            for row in rows:
                if metric in row:
                    by_epoch.setdefault(row["epoch"], []).append(float(row[metric]))
        series = []
        for epoch in sorted(by_epoch):
            c, lo, hi = summarize(by_epoch[epoch], band)
            series.append({"epoch": epoch, "center": c, "band_lo": lo,
                           "band_hi": hi, "n": len(by_epoch[epoch])})
        out[algo] = series
    return out


def write_csv(runs, out_path: Path, band: str, settings: list[str]) -> None:
    """Dump every aggregated cell to CSV for downstream tables / summaries."""
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "algo", "metric", "epoch",
                    "center", "band_lo", "band_hi", "n", "band"])
        for setting in settings:
            for metric in CSV_METRICS:
                grouped = aggregate(runs, setting, metric, band)
                for algo in ORDER:
                    for row in grouped[algo]:
                        w.writerow([
                            setting, algo, metric, row["epoch"],
                            f"{row['center']:.8g}",
                            f"{row['band_lo']:.8g}",
                            f"{row['band_hi']:.8g}",
                            row["n"], band,
                        ])


# ── plotting ─────────────────────────────────────────────────────────────────

def _plot_series(ax, epochs, center, lo, hi, color, label, zorder: int = 1) -> None:
    """Draw one mean line + shaded band on ``ax``, with sparse open-circle
    markers. End-of-line labels are placed by ``_place_end_labels`` after all
    curves on the axes are drawn, so overlapping labels can be offset.

    Pattern adapted from MushroomRL's ``utils.plot.plot_mean_conf``.
    """
    # ~12 marker samples per curve (avoid clutter for 60-epoch runs).
    every = max(1, len(epochs) // 12)
    ax.plot(epochs, center, color=color, linewidth=2.4, label=label,
            marker="o", markersize=4.0, markevery=every,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4,
            zorder=zorder + 1)
    if np.any(hi > lo):
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.18, linewidth=0,
                        zorder=zorder)


def _place_end_labels(ax, end_points: list, min_separation_px: float = 11.0,
                      offset_px: float = 7.0) -> None:
    """Annotate each curve's end with its label, splitting overlaps vertically.

    ``end_points``: list of ``(label, x_end, y_end, color, zorder)`` tuples.
    If two adjacent labels (sorted by y) are closer than ``min_separation_px``
    in screen space, push the upper one up by ``offset_px`` and the lower one
    down by ``offset_px``. Labels far apart get no offset (no visual nudge).
    """
    if not end_points:
        return
    fig = ax.figure
    fig.canvas.draw()  # realize transforms before querying pixel coords

    # Sort by y descending (top first).
    pts = sorted(end_points, key=lambda p: -p[2])
    px_ys = [ax.transData.transform((x, y))[1] for _, x, y, _, _ in pts]

    offsets = [0.0] * len(pts)
    for i in range(len(pts) - 1):
        if px_ys[i] - px_ys[i + 1] < min_separation_px:
            offsets[i]     = max(offsets[i],     +offset_px)
            offsets[i + 1] = min(offsets[i + 1], -offset_px)

    for (label, x, y, color, z), dy in zip(pts, offsets):
        ax.annotate(label, xy=(x, y), xytext=(5, dy),
                    textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold",
                    va="center", zorder=z + 2)


def plot_setting(runs, setting: str, stem: Path, band: str,
                 subtitle_note: str = "") -> tuple[Path, Path]:
    """Render the 1×3 figure for one pose. Returns the (png, svg) paths.

    Header layout (top to bottom):
      ~y=0.96   bold suptitle
      ~y=0.915  italic gray caption (panel-source description + notes)
      ~y=0.87   centered legend row (markers + algo names) | band note (right)
                ← thin gap →
      panels    1×3 axes (top of axes ≈ 0.78)

    Manual ``subplots_adjust`` rather than ``constrained_layout`` so the three
    header lines (title / caption / legend) sit exactly where we want.
    """
    from matplotlib.lines import Line2D

    pose = "harder" if setting == "final" else setting
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    fig.subplots_adjust(top=0.78, bottom=0.11, left=0.045, right=0.985, wspace=0.30)

    # ── header line 1: title ──────────────────────────────────────────────
    fig.text(0.5, 0.955, f"SAC vs LagSAC, Stage 1 {pose} pose",
             ha="center", va="center", fontsize=15, fontweight="bold")

    # ── header line 2: caption (panel sources + optional notes) ──────────
    caption_parts = [
        "Benchmark 1×3 panels. Return=J (eval); safety panels=rollout "
        "(training-time). Success rate reported in summary table (not shown)."
    ]
    if subtitle_note:
        caption_parts.append(subtitle_note)
    fig.text(0.5, 0.915, " ".join(caption_parts),
             ha="center", va="center", fontsize=9, color="#6b7280", style="italic")

    # ── header line 3: legend row (center) + band note (right) ───────────
    handles = [
        Line2D([0], [0], color=COLORS[a], marker="o", markersize=6,
               markerfacecolor="white", markeredgecolor=COLORS[a],
               markeredgewidth=1.5, linewidth=2.4, label=a)
        for a in ORDER
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.87),
               ncol=len(ORDER), frameon=False, fontsize=10,
               columnspacing=2.5, handletextpad=0.5)
    fig.text(0.985, 0.87, BAND_LABEL.get(band, f"band = {band}"),
             ha="right", va="center", fontsize=9, color="#6b7280")

    # ── panels ───────────────────────────────────────────────────────────
    # zorder map: draw SAC first, LagSAC on top so its line is never masked.
    zorder_map = {"SAC": 1, "LagSAC": 3}

    for ax, (metric, panel_title, ylabel) in zip(axes, PANELS):
        series = aggregate(runs, setting, metric, band)
        end_points: list = []
        for algo in ORDER:
            rows = series.get(algo, [])
            if not rows:
                continue
            epochs = np.array([r["epoch"] for r in rows])
            center = np.array([r["center"] for r in rows])
            lo     = np.array([r["band_lo"] for r in rows])
            hi     = np.array([r["band_hi"] for r in rows])
            _plot_series(ax, epochs, center, lo, hi,
                         COLORS[algo], algo, zorder=zorder_map[algo])
            end_points.append(
                (algo, float(epochs[-1]), float(center[-1]),
                 COLORS[algo], zorder_map[algo])
            )

        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        # Right-margin headroom so the inline curve label fits inside the axes.
        xlo, xhi = ax.get_xlim()
        ax.set_xlim(xlo, xhi + (xhi - xlo) * 0.04)
        # Place inline labels last so we can split any vertical overlap.
        _place_end_labels(ax, end_points)

    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


# ── strict-mode validation ──────────────────────────────────────────────────

def _validate_runs(runs, allow_legacy_proxy: bool) -> bool:
    """Apply strict-mode checks; mutate ``runs`` to backfill if legacy.

    Returns ``True`` iff any backfill happened (so caller can warn in subtitle).
    """
    # cost_scale != 1.0 is a hard fail regardless of legacy flag — silently
    # plotting a scaled cost on a panel labelled "Maximum Violation" would lie.
    scaled = [(k, rs[0]["cost_scale"]) for k, rs in runs.items()
              if rs and "cost_scale" in rs[0] and rs[0]["cost_scale"] != 1.0]
    if scaled:
        raise SystemExit(
            "cost_scale != 1.0 detected — the Maximum Violation panel assumes "
            f"info['cost'] is the raw violation. Offending runs: {scaled}"
        )

    required = ("rollout_ep_cost", "maximum_violation", "cost_scale")
    missing: dict = {}
    for k, rs in runs.items():
        for r in rs:
            for f in required:
                if f not in r:
                    missing.setdefault(k, set()).add(f)

    if missing and not allow_legacy_proxy:
        lines = ["Strict mode: required fields missing from logs:"]
        for k, fs in list(missing.items())[:10]:
            lines.append(f"  {k!r}: missing {sorted(fs)}")
        if len(missing) > 10:
            lines.append(f"  ... and {len(missing) - 10} more")
        lines.append("Pass --allow_legacy_proxy to backfill from a per-step proxy "
                     "(missing cost_scale tolerated; explicit != 1.0 still fails).")
        raise SystemExit("\n".join(lines))

    if missing:
        for rs in runs.values():
            for r in rs:
                r.setdefault("rollout_ep_cost", r.get("eval_ep_cost", 0.0))
                r.setdefault("maximum_violation", r.get("maximum_violation_proxy", 0.0))
        return True
    return False


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--logs", type=Path, default=Path("/tmp/overnight_logs"),
                   help="Directory with *_{default,harder,final}_s<seed>.log files.")
    p.add_argument("--out_dir", type=Path, default=Path("results/plots/overnight_paper"),
                   help="Where to write PNG + SVG + CSV.")
    p.add_argument("--band", choices=("ci", "sem", "std", "none"), default="ci",
                   help="Shaded-band aggregation: 'ci' = t-distribution 95%% CI "
                        "(default), 'sem' = ±σ/√n, 'std' = ±σ, 'none' = mean only.")
    p.add_argument("--settings", nargs="+", default=["default", "harder"],
                   help="Pose settings to plot: default, harder, final.")
    p.add_argument("--exclude_seeds", nargs="*", type=int, default=[],
                   help="Paired seeds to exclude from all algorithms.")
    p.add_argument("--allow_legacy_proxy", action="store_true",
                   help="Backfill missing rollout fields with a per-step proxy "
                        "for old logs (subtitle will warn). cost_scale != 1.0 "
                        "still hard-fails. Default is strict.")
    args = p.parse_args()

    settings = list(dict.fromkeys(args.settings))
    runs = collect(args.logs, set(args.exclude_seeds), set(settings))
    if not runs:
        raise SystemExit(f"No Stage 1 SAC/LagSAC logs found in {args.logs}")

    backfilled = _validate_runs(runs, args.allow_legacy_proxy)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "stage1_paper_curves.csv"
    write_csv(runs, csv_path, args.band, settings)
    outputs = [csv_path]

    notes = []
    if args.exclude_seeds:
        excluded = ", ".join(str(s) for s in sorted(args.exclude_seeds))
        notes.append(f"Excluded paired seed(s): {excluded}.")
    if backfilled:
        notes.append("LEGACY MODE: max-violation / episodic-cost backfilled from per-step proxies.")
    subtitle_note = "  ".join(notes)

    suffix = "_filtered" if args.exclude_seeds else ""
    for setting in settings:
        stem = args.out_dir / f"stage1_{setting}_paper{suffix}"
        png, svg = plot_setting(runs, setting, stem, args.band, subtitle_note)
        outputs.extend([png, svg])

    print("Wrote benchmark plots:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
