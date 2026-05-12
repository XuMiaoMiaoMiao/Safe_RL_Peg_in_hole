"""Plot geom reward curves without Isaac or matplotlib.

This is a pure-Python diagnostic for the geom reward:

    r = -w_d * clip(|d - target|, d_sat)
        -w_radial_tip * clip(radial_tip, radial_sat)
        -w_radial_max * clip(radial_max, radial_sat)
        -w_axis * axis_err
        + optional soft well

It writes a self-contained HTML file with inline SVG plots.
"""

from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=Path("results/reward_plots/geom_reward_landscape.html"))
    p.add_argument("--log", type=Path, default=None,
                   help="Optional train_sac output.log to overlay d_err_mean.")
    p.add_argument("--d-neg", type=float, default=-0.08)
    p.add_argument("--d-pos", type=float, default=-0.02)
    p.add_argument("--ramp-end", type=int, default=40)
    p.add_argument("--w-d", type=float, default=5.0)
    p.add_argument("--w-radial-tip", type=float, default=0.0)
    p.add_argument("--w-radial-max", type=float, default=5.0)
    p.add_argument("--w-axis", type=float, default=1.0)
    p.add_argument("--d-sat", type=float, default=0.30)
    p.add_argument("--radial-sat", type=float, default=1.0)
    p.add_argument("--soft-w", type=float, default=0.0)
    p.add_argument("--soft-d-sigma", type=float, default=0.02)
    p.add_argument("--soft-radial-sigma", type=float, default=0.015)
    p.add_argument("--soft-axis-sigma", type=float, default=0.30)
    return p.parse_args()


def reward(d, target, radial_tip, radial_max, axis_err, args):
    d_err = abs(d - target)
    r = (
        -args.w_d * min(d_err, args.d_sat)
        -args.w_radial_tip * min(radial_tip, args.radial_sat)
        -args.w_radial_max * min(radial_max, args.radial_sat)
        -args.w_axis * axis_err
    )
    if args.soft_w > 0.0:
        soft = math.exp(
            - (d_err / args.soft_d_sigma) ** 2
            - (radial_max / args.soft_radial_sigma) ** 2
            - (axis_err / args.soft_axis_sigma) ** 2
        )
        r += args.soft_w * soft
    return r


def target_at(epoch, args):
    t = min(max(epoch / max(args.ramp_end, 1), 0.0), 1.0)
    return args.d_neg + t * (args.d_pos - args.d_neg)


def parse_log(path: Path):
    if path is None or not path.is_file():
        return []
    schedule_re = re.compile(
        r"raw_epoch=(?P<raw>\d+) actor_epoch=(?P<actor>\d+): "
        r"d_target_eff=(?P<target>[+-]?[0-9.]+)m"
    )
    eval_re = re.compile(
        r"d_err_mean=(?P<derr>[0-9.]+)m .*"
        r"radial_max_min=(?P<rmax>[0-9.]+)m .*"
        r"axis_err_min=(?P<axis>[0-9.]+)"
    )
    rows = []
    pending = None
    for line in path.read_text(errors="replace").splitlines():
        m = schedule_re.search(line)
        if m:
            pending = {
                "raw": int(m.group("raw")),
                "actor": int(m.group("actor")),
                "target": float(m.group("target")),
            }
            continue
        m = eval_re.search(line)
        if m and pending is not None:
            row = dict(pending)
            row.update({
                "d_err": float(m.group("derr")),
                "radial_max_min": float(m.group("rmax")),
                "axis_err_min": float(m.group("axis")),
            })
            rows.append(row)
            pending = None
    return rows


def svg_line_plot(series, width=760, height=280, title="", y_label="reward"):
    pad_l, pad_r, pad_t, pad_b = 55, 20, 35, 45
    all_points = [p for _, pts, _ in series for p in pts]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x0 == x1:
        x1 = x0 + 1.0
    if y0 == y1:
        y0 -= 1.0
        y1 += 1.0
    y_pad = 0.08 * (y1 - y0)
    y0 -= y_pad
    y1 += y_pad

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * (width - pad_l - pad_r)

    def sy(y):
        return pad_t + (y1 - y) / (y1 - y0) * (height - pad_t - pad_b)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{width/2:.1f}" y="20" text-anchor="middle" '
        f'font-size="14" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#333"/>',
        f'<text x="{width/2:.1f}" y="{height-8}" text-anchor="middle" font-size="11">x</text>',
        f'<text x="14" y="{height/2:.1f}" text-anchor="middle" font-size="11" '
        f'transform="rotate(-90 14 {height/2:.1f})">{html.escape(y_label)}</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + frac * (y1 - y0)
        yy = sy(y)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" text-anchor="end" font-size="10">{y:.3f}</text>')
    legend_y = pad_t + 10
    for idx, (name, pts, dash) in enumerate(series):
        color = colors[idx % len(colors)]
        d_attr = ' stroke-dasharray="5 4"' if dash else ""
        path = " ".join(
            ("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}"
            for i, (x, y) in enumerate(pts)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"{d_attr}/>')
        lx = width - 245
        ly = legend_y + idx * 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+22}" y2="{ly}" stroke="{color}" stroke-width="2"{d_attr}/>')
        parts.append(f'<text x="{lx+28}" y="{ly+4}" font-size="11">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_heatmap(args, target, axis_err=0.05, width=760, height=360):
    pad_l, pad_r, pad_t, pad_b = 55, 85, 35, 45
    nx, ny = 90, 45
    d_min, d_max = -0.12, 0.06
    r_min, r_max = 0.0, 0.08
    vals = []
    for j in range(ny):
        row = []
        radial = r_min + (j + 0.5) / ny * (r_max - r_min)
        for i in range(nx):
            d = d_min + (i + 0.5) / nx * (d_max - d_min)
            row.append(reward(d, target, radial, radial, axis_err, args))
        vals.extend(row)
    v0, v1 = min(vals), max(vals)

    def color(v):
        t = (v - v0) / (v1 - v0 + 1e-9)
        # blue -> yellow -> red
        if t < 0.5:
            q = t / 0.5
            r = int(40 + q * 210)
            g = int(90 + q * 140)
            b = int(170 - q * 120)
        else:
            q = (t - 0.5) / 0.5
            r = int(250)
            g = int(230 - q * 130)
            b = int(50 - q * 20)
        return f"#{r:02x}{g:02x}{b:02x}"

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    cell_w = plot_w / nx
    cell_h = plot_h / ny
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{width/2:.1f}" y="20" text-anchor="middle" font-size="14" font-weight="600">'
        f'Reward heatmap, target d={target:+.3f}, axis={axis_err:.2f}</text>',
    ]
    for j in range(ny):
        radial = r_min + (j + 0.5) / ny * (r_max - r_min)
        for i in range(nx):
            d = d_min + (i + 0.5) / nx * (d_max - d_min)
            v = reward(d, target, radial, radial, axis_err, args)
            x = pad_l + i * cell_w
            y = pad_t + (ny - 1 - j) * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w+0.2:.1f}" height="{cell_h+0.2:.1f}" fill="{color(v)}"/>')
    # target line
    tx = pad_l + (target - d_min) / (d_max - d_min) * plot_w
    parts.append(f'<line x1="{tx:.1f}" y1="{pad_t}" x2="{tx:.1f}" y2="{pad_t+plot_h}" stroke="#111" stroke-width="2"/>')
    parts.append(f'<text x="{tx+4:.1f}" y="{pad_t+14}" font-size="11">target</text>')
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-size="11">d (m)</text>')
    parts.append(f'<text x="15" y="{height/2:.1f}" text-anchor="middle" font-size="11" transform="rotate(-90 15 {height/2:.1f})">radial_max (m)</text>')
    # color legend
    lx = width - pad_r + 25
    ly = pad_t
    for k in range(80):
        t = k / 79
        v = v0 + t * (v1 - v0)
        parts.append(f'<rect x="{lx}" y="{ly + (79-k)*2.7:.1f}" width="18" height="3" fill="{color(v)}"/>')
    parts.append(f'<text x="{lx+24}" y="{ly+5}" font-size="10">{v1:.2f}</text>')
    parts.append(f'<text x="{lx+24}" y="{ly+218}" font-size="10">{v0:.2f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_rows = parse_log(args.log)

    d_values = [(-0.12 + i * (0.18 / 240)) for i in range(241)]
    target_curves = []
    for target in [args.d_neg, (args.d_neg + args.d_pos) / 2.0, args.d_pos, 0.03]:
        pts = [
            (d, reward(d, target, radial_tip=0.02, radial_max=0.02, axis_err=0.05, args=args))
            for d in d_values
        ]
        target_curves.append((f"target {target:+.3f}", pts, False))

    ramp_series = []
    epochs = list(range(0, args.ramp_end + 1))
    ramp_series.append((
        "ideal tracking: d=target, rmax=.015, axis=.05",
        [(e, reward(target_at(e, args), target_at(e, args), 0.015, 0.015, 0.05, args)) for e in epochs],
        False,
    ))
    ramp_series.append((
        "stuck at d_neg, rmax=.020, axis=.02",
        [(e, reward(args.d_neg, target_at(e, args), 0.020, 0.020, 0.02, args)) for e in epochs],
        False,
    ))
    ramp_series.append((
        "tracks d but loose geometry: rmax=.030, axis=.30",
        [(e, reward(target_at(e, args), target_at(e, args), 0.030, 0.030, 0.30, args)) for e in epochs],
        False,
    ))

    log_d_err_svg = ""
    log_axial_svg = ""
    if log_rows:
        log_d_err_svg = svg_line_plot(
            [("log d_err_mean", [(r["actor"], r["d_err"]) for r in log_rows], False),
             ("3cm target", [(r["actor"], 0.03) for r in log_rows], True)],
            title=f"Observed d_err_mean from {args.log.name}",
            y_label="d_err (m)",
        )
        log_axial_svg = svg_line_plot(
            [("observed axial reward = -w_d*d_err_mean",
              [(r["actor"], -args.w_d * min(r["d_err"], args.d_sat)) for r in log_rows], False)],
            title="Observed axial reward component from log",
            y_label="r_geom_d",
        )

    axial_gain_to_pos = args.w_d * abs(args.d_pos - args.d_neg)
    axial_gain_to_insert = args.w_d * abs(0.03 - args.d_neg)
    radial_pen_2cm = args.w_radial_max * 0.02
    radial_pen_3cm = args.w_radial_max * 0.03
    axis_pen_030 = args.w_axis * 0.30

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Geom Reward Landscape</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; }}
    .plot {{ margin: 22px 0; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
    table {{ border-collapse: collapse; margin: 12px 0; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 9px; }}
  </style>
</head>
<body>
  <h1>Geom reward diagnostic</h1>
  <p>Formula: <code>-w_d|d-target| - w_rtip*radial_tip - w_rmax*radial_max - w_axis*axis</code>, with clipping on d/radial.</p>
  <table>
    <tr><th>parameter</th><th>value</th></tr>
    <tr><td>w_d</td><td>{args.w_d}</td></tr>
    <tr><td>w_radial_tip</td><td>{args.w_radial_tip}</td></tr>
    <tr><td>w_radial_max</td><td>{args.w_radial_max}</td></tr>
    <tr><td>w_axis</td><td>{args.w_axis}</td></tr>
    <tr><td>d target ramp</td><td>{args.d_neg:+.3f} to {args.d_pos:+.3f}, {args.ramp_end} actor epochs</td></tr>
  </table>
  <h2>Useful scale checks</h2>
  <ul>
    <li>Axial reward gain from staying at {args.d_neg:+.3f} to tracking {args.d_pos:+.3f}: <b>{axial_gain_to_pos:.3f}/step</b>.</li>
    <li>Axial reward gain from staying at {args.d_neg:+.3f} to true insert target +0.030: <b>{axial_gain_to_insert:.3f}/step</b>.</li>
    <li>Radial_max penalty at 2cm: <b>-{radial_pen_2cm:.3f}/step</b>; at 3cm: <b>-{radial_pen_3cm:.3f}/step</b>.</li>
    <li>Axis penalty at axis_err=0.30: <b>-{axis_pen_030:.3f}/step</b>.</li>
  </ul>
  <div class="plot">{svg_line_plot(target_curves, title="Reward vs d at fixed radial_max=.02, axis=.05", y_label="reward / step")}</div>
  <div class="plot">{svg_line_plot(ramp_series, title="Reward along d-target ramp for stylized policies", y_label="reward / step")}</div>
  <div class="plot">{svg_heatmap(args, target=args.d_pos, axis_err=0.05)}</div>
  <div class="plot">{log_d_err_svg}</div>
  <div class="plot">{log_axial_svg}</div>
</body>
</html>
"""
    args.output.write_text(html_doc)
    print(f"Wrote {args.output}")
    print(f"Axial gain d_neg->{args.d_pos:+.3f}: {axial_gain_to_pos:.3f}/step")
    print(f"Radial penalty 2cm: -{radial_pen_2cm:.3f}/step")
    print(f"Axis penalty axis_err=0.30: -{axis_pen_030:.3f}/step")
    if log_rows:
        print(f"Parsed {len(log_rows)} log eval rows from {args.log}")


if __name__ == "__main__":
    main()
