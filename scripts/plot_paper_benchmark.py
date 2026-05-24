#!/usr/bin/env python3
"""Draw SAC vs LagSAC benchmark comparison plots from training logs.

This script intentionally has no matplotlib dependency. It writes clean SVG
figures directly, so it works in the current safe_rl environment.

The panel names track the benchmark metrics used by this project. When logs
do not contain the exact statistic, the plotted proxy is stated in the
subtitle and exported CSV.

Default:
  python scripts/plot_paper_benchmark.py

Outputs:
  results/plots/overnight_paper/stage1_default_paper.svg
  results/plots/overnight_paper/stage1_harder_paper.svg
  results/plots/overnight_paper/stage1_paper_curves.csv

Final benchmark example:
  python scripts/plot_paper_benchmark.py \
    --logs /tmp/final_bench_logs \
    --out_dir results/plots/final_benchmark_filtered \
    --settings final \
    --exclude_seeds 3
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import statistics
from pathlib import Path


NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COLORS = {
    "LagSAC": "#2563eb",
    "SAC": "#d97706",
}
ORDER = ["LagSAC", "SAC"]


def run_meta(path: Path) -> tuple[str, str, int] | None:
    name = path.stem
    m = re.match(r"(lag|sac)_(default|harder|final)_s(\d+)$", name)
    if not m:
        return None
    algo = "LagSAC" if m.group(1) == "lag" else "SAC"
    setting = m.group(2)
    seed = int(m.group(3))
    return setting, algo, seed


def grab_float(line: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=({NUM})", line)
    if match:
        return float(match.group(1))
    match = re.search(rf"{re.escape(key)}:\s*({NUM})", line)
    if match:
        return float(match.group(1))
    return None


def grab_int(line: str, key: str) -> int | None:
    val = grab_float(line, key)
    return None if val is None else int(val)


def parse_run(path: Path) -> list[dict]:
    rows: list[dict] = []
    current: dict | None = None
    run_cost_scale: float | None = None
    for line in path.read_text(errors="replace").splitlines():
        # One-shot capture from the env setup log (printed before epoch 1).
        # The Maximum Violation panel requires cost_scale == 1.0 (so that
        # info["cost"] == raw violation); main() asserts this after collect().
        if run_cost_scale is None and "cost_scale=" in line:
            val = grab_float(line, "cost_scale")
            if val is not None:
                run_cost_scale = val
        epoch_match = re.search(
            rf"Epoch\s+(\d+)\s+\|\s+J:\s*({NUM})\s+R:\s*({NUM})"
            rf".*?best_J:\s*({NUM}).*?best_geom:\s*({NUM})",
            line,
        )
        if epoch_match:
            current = {
                "epoch": int(epoch_match.group(1)),
                "J": float(epoch_match.group(2)),
                "R": float(epoch_match.group(3)),
                "best_J": float(epoch_match.group(4)),
                "best_geom": float(epoch_match.group(5)),
            }
            rows.append(current)
            continue

        if current is None:
            continue

        if "geom eval" in line:
            for key in (
                "geom_step_rate",
                "geom_hold_rate",
                "geom_max_run_mean",
                "final_success_rate",
            ):
                val = grab_float(line, key)
                if val is not None:
                    current[key] = val
            current["success"] = current.get("geom_hold_rate", current.get("best_geom", 0.0))
            continue

        if "eval_ep_cost=" in line:
            for key in (
                "eval_step_cost",
                "eval_ep_cost",
                "rollout_ep_cost",
                "rollout_ep_max_violation",
                "eval_ep_max_violation",
                "epoch_collision_total",
                "epoch_collision_sphere",
                "epoch_collision_physx",
                "epoch_collision_table",
                "epoch_absorb_total",
                "epoch_absorb_table",
            ):
                if key.startswith("epoch_"):
                    val = grab_int(line, key)
                else:
                    val = grab_float(line, key)
                if val is not None:
                    current[key] = val
            current["table_contact"] = current.get("epoch_collision_table", 0)
            current["hard_collision_events"] = (
                current.get("epoch_collision_physx", 0)
                + current.get("epoch_collision_table", 0)
            )
            # Paper "Maximum Violation per Episode" = real per-episode max cost
            # logged by the training scripts (rollout = training-time safety).
            # The legacy per-step proxy is stored under a SEPARATE key so main()
            # can opt into it via --allow_legacy_proxy when replotting old logs.
            # In strict (default) mode, missing rollout_ep_max_violation is a
            # hard error — silent fallback would lie about the panel's meaning.
            current["maximum_violation_proxy"] = max(
                current.get("eval_step_cost", 0.0),
                current.get("rollout_ep_cost", 0.0) / 100.0,
            )
            if "rollout_ep_max_violation" in current:
                current["maximum_violation"] = current["rollout_ep_max_violation"]
            # else: leave maximum_violation unset; main() decides strict-fail vs
            # legacy-backfill.

    for row in rows:
        row.setdefault("success", row.get("best_geom", 0.0))
        row.setdefault("eval_step_cost", 0.0)
        row.setdefault("eval_ep_cost", 0.0)
        row.setdefault("epoch_collision_total", 0)
        row.setdefault("epoch_collision_table", 0)
        row.setdefault("hard_collision_events", 0)
        # Deliberately NOT setdefault: rollout_ep_cost, rollout_ep_max_violation,
        # maximum_violation. Their absence is the strict-mode signal that this
        # run predates the rollout tracker. main() either fails (default) or
        # backfills from proxy / eval (--allow_legacy_proxy).
        if run_cost_scale is not None:
            row["cost_scale"] = run_cost_scale
    return rows


def collect(
    log_dir: Path,
    exclude_seeds: set[int] | None = None,
    settings: set[str] | None = None,
) -> dict[tuple[str, str, int], list[dict]]:
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


def _summary(values: list[float], band: str) -> tuple[float, float, float]:
    """Return (center, band_lo, band_hi) for one (epoch, metric) cell.

    band="ci"   — center = mean,   band = mean ± 1.96 · σ/√n  (normal-approx
                  95% confidence interval; default benchmark plot).
                  Uses normal approx (1.96) not t-dist; for n<10 the t-dist
                  correction would widen the band slightly (n=15: ×1.09).
    band="sem"  — center = mean,   band = mean ± σ/√n.
    band="std"  — center = mean,   band = mean ± σ.
    band="none" — center = mean,   band = [center, center] (no shade).
    """
    if not values:
        return math.nan, math.nan, math.nan
    center = statistics.mean(values)
    if len(values) <= 1:
        return center, center, center
    std = statistics.stdev(values)
    if band == "ci":
        err = 1.96 * std / math.sqrt(len(values))
    elif band == "sem":
        err = std / math.sqrt(len(values))
    elif band == "none":
        err = 0.0
    else:  # "std"
        err = std
    return center, center - err, center + err


def aggregate(
    runs: dict[tuple[str, str, int], list[dict]],
    setting: str,
    metric: str,
    band: str,
) -> dict[str, list[dict]]:
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
            vals = by_epoch[epoch]
            center, lo, hi = _summary(vals, band)
            series.append({"epoch": epoch, "center": center,
                           "band_lo": lo, "band_hi": hi, "n": len(vals)})
        out[algo] = series
    return out


def write_csv(
    runs: dict[tuple[str, str, int], list[dict]],
    out_path: Path,
    band: str,
    settings: list[str],
) -> None:
    metrics = [
        "J",
        "maximum_violation",
        "success",
        "eval_step_cost",
        "eval_ep_cost",
        "rollout_ep_cost",
        "hard_collision_events",
        "epoch_collision_total",
        "epoch_collision_table",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "algo", "metric", "epoch",
                         "center", "band_lo", "band_hi", "n", "band"])
        for setting in settings:
            for metric in metrics:
                grouped = aggregate(runs, setting, metric, band)
                for algo in ORDER:
                    for row in grouped[algo]:
                        writer.writerow(
                            [
                                setting,
                                algo,
                                metric,
                                row["epoch"],
                                f"{row['center']:.8g}",
                                f"{row['band_lo']:.8g}",
                                f"{row['band_hi']:.8g}",
                                row["n"],
                                band,
                            ]
                        )


def nice_bounds(values: list[float], metric: str) -> tuple[float, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return 0.0, 1.0
    if metric == "success":
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if lo == hi:
        pad = max(abs(lo) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    if metric in {
        "eval_ep_cost",
        "eval_step_cost",
        "maximum_violation",
        "hard_collision_events",
        "epoch_collision_total",
        "epoch_collision_table",
    }:
        return 0.0, hi + pad
    return lo - pad, hi + pad


def ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda x: abs(raw - x * mag)) * mag
    start = math.ceil(lo / step) * step
    vals = []
    v = start
    while v <= hi + 1e-9:
        vals.append(v)
        v += step
    return vals or [lo, hi]


def fmt_tick(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def panel_svg(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    ylabel: str,
    metric: str,
    series: dict[str, list[dict]],
) -> str:
    margin_l, margin_r, margin_t, margin_b = 58, 18, 34, 44
    px0, py0 = x + margin_l, y + margin_t
    pw, ph = w - margin_l - margin_r, h - margin_t - margin_b

    epochs = [p["epoch"] for rows in series.values() for p in rows]
    if not epochs:
        return ""
    xlo, xhi = min(epochs), max(epochs)
    y_values = []
    for rows in series.values():
        for p in rows:
            y_values.extend([p["band_lo"], p["band_hi"]])
    ylo, yhi = nice_bounds(y_values, metric)

    def sx(epoch: float) -> float:
        if xhi == xlo:
            return px0 + pw / 2
        return px0 + (epoch - xlo) / (xhi - xlo) * pw

    def sy(val: float) -> float:
        if yhi == ylo:
            return py0 + ph / 2
        return py0 + ph - (val - ylo) / (yhi - ylo) * ph

    parts = []
    parts.append(f'<g class="panel">')
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white"/>')
    parts.append(f'<text x="{x + w / 2:.1f}" y="{y + 18}" text-anchor="middle" class="title">{html.escape(title)}</text>')

    for tv in ticks(ylo, yhi):
        yy = sy(tv)
        parts.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px0 + pw}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{px0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{fmt_tick(tv)}</text>')
    for tv in ticks(xlo, xhi, 6):
        xx = sx(tv)
        parts.append(f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py0 + ph}" class="grid"/>')
        parts.append(f'<text x="{xx:.1f}" y="{py0 + ph + 18}" text-anchor="middle" class="tick">{fmt_tick(tv)}</text>')

    parts.append(f'<line x1="{px0}" y1="{py0 + ph}" x2="{px0 + pw}" y2="{py0 + ph}" class="axis"/>')
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + ph}" class="axis"/>')
    parts.append(f'<text x="{px0 + pw / 2:.1f}" y="{y + h - 8}" text-anchor="middle" class="label">Epoch</text>')
    parts.append(
        f'<text x="{x + 14}" y="{py0 + ph / 2:.1f}" transform="rotate(-90 {x + 14} {py0 + ph / 2:.1f})" '
        f'text-anchor="middle" class="label">{html.escape(ylabel)}</text>'
    )

    for algo in ORDER:
        rows = series.get(algo, [])
        if not rows:
            continue
        color = COLORS[algo]
        upper = [(sx(p["epoch"]), sy(p["band_hi"])) for p in rows]
        lower = [(sx(p["epoch"]), sy(p["band_lo"])) for p in reversed(rows)]
        if any(p["band_hi"] > p["band_lo"] for p in rows):
            polygon = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in upper + lower)
            parts.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.16"/>')
        path = " ".join(
            ("M" if i == 0 else "L") + f" {sx(p['epoch']):.1f} {sy(p['center']):.1f}"
            for i, p in enumerate(rows)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        for p in rows[:: max(1, len(rows) // 10)]:
            parts.append(
                f'<circle cx="{sx(p["epoch"]):.1f}" cy="{sy(p["center"]):.1f}" r="2.4" '
                f'fill="white" stroke="{color}" stroke-width="1.5"/>'
            )
        last = rows[-1]
        label_x = min(sx(last["epoch"]) + 8, px0 + pw - 46)
        label_y = sy(last["center"]) + (-8 if algo == "LagSAC" else 14)
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" fill="{color}" '
            f'class="curve-label">{algo}</text>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def write_svg(
    runs: dict[tuple[str, str, int], list[dict]],
    setting: str,
    out_path: Path,
    band: str,
    subtitle_note: str = "",
) -> None:
    panels = [
        ("J", "Discounted Return", "J (higher is better)"),
        ("maximum_violation", "Maximum Violation per Episode",
         "mean per-episode max cost (rollout)"),
        ("rollout_ep_cost", "Episodic Sum of Cost",
         "mean per-episode cost sum (rollout)"),
    ]
    width, height = 1440, 500
    panel_w, panel_h = 430, 330
    positions = [(40, 110), (510, 110), (980, 110)]
    setting_title = "harder pose" if setting == "final" else f"{setting} pose"
    title = f"SAC vs LagSAC, Stage 1 {setting_title}"
    subtitle = (
        "Benchmark 1×3 panels. Return=J (eval); "
        "safety panels=rollout (training-time). "
        "Success rate reported in summary table (not shown)."
    )
    if subtitle_note:
        subtitle = f"{subtitle} {subtitle_note}"
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".suptitle{font-size:22px;font-weight:700}",
        ".subtitle{font-size:13px;fill:#4b5563}",
        ".title{font-size:15px;font-weight:700}",
        ".label{font-size:12px;fill:#374151}",
        ".curve-label{font-size:12px;font-weight:700}",
        ".tick{font-size:11px;fill:#4b5563}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".axis{stroke:#111827;stroke-width:1.2}",
        "</style>",
        f'<text x="{width / 2}" y="30" text-anchor="middle" class="suptitle">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="52" text-anchor="middle" class="subtitle">{html.escape(subtitle)}</text>',
    ]
    legend_y = 80
    legend_x = (width // 2) - 200
    band_label = {
        "ci":  "mean line, shaded = 95% CI (mean ± 1.96·σ/√n)",
        "sem": "mean line, shaded = ± sem (σ/√n)",
        "std": "mean line, shaded = ± std",
        "none": "mean line, no band",
    }.get(band, f"band = {band}")
    for i, algo in enumerate(ORDER):
        color = COLORS[algo]
        x = legend_x + i * 165
        body.append(f'<rect x="{x}" y="{legend_y - 10}" width="18" height="10" fill="{color}" opacity="0.16"/>')
        body.append(f'<line x1="{x}" y1="{legend_y - 5}" x2="{x + 36}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{x + 18}" cy="{legend_y - 5}" r="3" fill="white" stroke="{color}" stroke-width="1.5"/>')
        body.append(f'<text x="{x + 44}" y="{legend_y - 1}" class="label">{algo}</text>')
    body.append(f'<text x="{legend_x + 340}" y="{legend_y - 1}" class="subtitle">{band_label}</text>')
    for (metric, panel_title, ylabel), (x, y) in zip(panels, positions):
        series = aggregate(runs, setting, metric, band)
        body.append(panel_svg(x, y, panel_w, panel_h, panel_title, ylabel, metric, series))
    body.append("</svg>")
    out_path.write_text("\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("/tmp/overnight_logs"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/plots/overnight_paper"))
    parser.add_argument("--band", choices=["ci", "sem", "std", "none"], default="ci",
                        help="Shaded-band convention. 'ci' = 95%% confidence interval "
                             "(default), 'sem' = ±σ/√n, 'std' = ±σ.")
    parser.add_argument(
        "--allow_legacy_proxy",
        action="store_true",
        help="Replay old logs lacking rollout_ep_max_violation / rollout_ep_cost "
             "/ cost_scale: backfill Maximum Violation from the per-step proxy, "
             "Episodic Sum of Cost from eval_ep_cost, and skip the "
             "MISSING-cost_scale requirement (explicit cost_scale != 1.0 still "
             "hard-fails — that case lies about the panel and cannot be opted "
             "out of). Default (strict) refuses to plot such runs.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["default", "harder"],
        help="Settings to plot: default, harder, final.",
    )
    parser.add_argument(
        "--exclude_seeds",
        nargs="*",
        type=int,
        default=[],
        help="Paired seeds to exclude from all algorithms.",
    )
    args = parser.parse_args()

    settings = list(dict.fromkeys(args.settings))
    runs = collect(args.logs, set(args.exclude_seeds), set(settings))
    if not runs:
        raise SystemExit(f"No Stage 1 SAC/LagSAC logs found in {args.logs}")

    # Hard fail: cost_scale explicitly != 1.0. The Maximum Violation panel
    # interprets info['cost'] as raw violation; a scaled cost would silently
    # lie. This check fires regardless of --allow_legacy_proxy.
    scaled = [
        (k, rs[0]["cost_scale"]) for k, rs in runs.items()
        if rs and "cost_scale" in rs[0] and rs[0]["cost_scale"] != 1.0
    ]
    if scaled:
        raise SystemExit(
            "cost_scale != 1.0 detected — the Maximum Violation panel assumes "
            "info['cost'] is the raw violation. Offending runs (key, cost_scale): "
            f"{scaled}"
        )

    # Strict benchmark mode (default): require rollout_ep_cost, rollout_ep_max_violation
    # (→ row["maximum_violation"]) and cost_scale present in every row. Missing
    # any of these means the run predates the new tracker — silently plotting it
    # with a proxy would mislabel the panel. Opt out with --allow_legacy_proxy.
    required_per_row = ("rollout_ep_cost", "maximum_violation", "cost_scale")
    missing = {}
    for k, rs in runs.items():
        for r in rs:
            for f in required_per_row:
                if f not in r:
                    missing.setdefault(k, set()).add(f)
    if missing and not args.allow_legacy_proxy:
        lines = ["Strict benchmark mode: required fields missing from logs:"]
        for k, fs in list(missing.items())[:10]:
            lines.append(f"  {k!r}: missing {sorted(fs)}")
        if len(missing) > 10:
            lines.append(f"  ... and {len(missing) - 10} more")
        lines.append(
            "These runs predate the rollout cost tracker (or were logged before "
            "the cost_scale startup line was added)."
        )
        lines.append(
            "Use --allow_legacy_proxy to plot anyway (Maximum Violation falls "
            "back to the per-step proxy; Episodic Sum of Cost falls back to "
            "eval_ep_cost; missing cost_scale is tolerated — but explicit "
            "cost_scale != 1.0 still hard-fails)."
        )
        raise SystemExit("\n".join(lines))

    if args.allow_legacy_proxy and missing:
        # Backfill missing fields so plotting can proceed. Subtitle will warn.
        for rs in runs.values():
            for r in rs:
                r.setdefault("rollout_ep_cost", r.get("eval_ep_cost", 0.0))
                r.setdefault("maximum_violation",
                             r.get("maximum_violation_proxy", 0.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "stage1_paper_curves.csv"
    write_csv(runs, csv_path, args.band, settings)
    outputs = [csv_path]
    subtitle_note = ""
    if args.exclude_seeds:
        excluded = ", ".join(str(s) for s in sorted(args.exclude_seeds))
        subtitle_note = f"Excluded paired seed(s): {excluded}."
    if args.allow_legacy_proxy and missing:
        legacy = ("LEGACY MODE: max-violation = step-cost proxy / "
                  "episodic-cost = eval_ep_cost where rollout fields missing.")
        subtitle_note = (f"{subtitle_note} {legacy}".strip())
    for setting in settings:
        suffix = ""
        if args.exclude_seeds:
            suffix = "_filtered"
        out_path = args.out_dir / f"stage1_{setting}_paper{suffix}.svg"
        write_svg(runs, setting, out_path, args.band, subtitle_note)
        outputs.append(out_path)

    print("Wrote benchmark plots:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
