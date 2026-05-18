#!/usr/bin/env python3
"""Paper-style analysis for /tmp/overnight_logs.

This complements scripts/analyze_overnight.py with stricter parsing for SAC
completion lines and with D-ATACOM-style safety metrics:
return, success, episodic cost, and maximum violation. Table contact is counted
as the same safety class as arm-arm collision.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path("/tmp/overnight_logs")
REPORT = Path("/tmp/overnight_report.md")
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def fnum(v: str | None, default: float | None = None) -> float | None:
    if v is None or v.lower() == "n/a":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{digits}f}"
    return str(v)


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    xs = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def mean_std_str(vals: list[float], digits: int = 2) -> str:
    m, s = mean_std(vals)
    if m is None:
        return "-"
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def condition(name: str) -> str:
    if name.startswith("lag_default_s"):
        return "stage1_default_lagsac"
    if name.startswith("sac_default_s"):
        return "stage1_default_sac"
    if name.startswith("lag_harder_s"):
        return "stage1_harder_lagsac"
    if name.startswith("sac_harder_s"):
        return "stage1_harder_sac"
    if name == "lag_stage3_b_route_calib":
        return "stage3_lagsac_cost100"
    if name == "lag_stage3_b_route_calib_strict":
        return "stage3_lagsac_cost50"
    if name == "sac_stage3_b_route_calib":
        return "stage3_sac_overnight_failed"
    return name


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    name = path.stem
    out = {"run": name, "condition": condition(name), "status": "incomplete"}

    if "训练完成." in text:
        out["status"] = "completed"
    if "Traceback" in text or "RuntimeError" in text or "error: unrecognized arguments" in text:
        out["status"] = "crashed"

    lines = re.findall(r"训练完成\.[^\n]+", text)
    if lines:
        line = lines[-1]
        m = re.search(rf"best J\s*=\s*({NUM})", line)
        if m:
            out["best_J"] = float(m.group(1))
        # LagSAC: best_geom_rate. SAC: best geom_hold_rate.
        m = re.search(rf"best[_ ]geom(?:_hold)?_rate\s*=\s*({NUM}|n/a)", line)
        if m:
            out["best_geom_rate"] = fnum(m.group(1), 0.0)
            out["best_geom_was_na"] = m.group(1).lower() == "n/a"
        elif "no geom_hold_rate success" in line:
            out["best_geom_rate"] = 0.0
            out["best_geom_was_na"] = True
        m = re.search(rf"(?:max_hold_mean|geom_max_run_mean)=({NUM})", line)
        if m:
            out["max_run_mean"] = float(m.group(1))
        m = re.search(rf"final λ\s*=\s*({NUM})", line)
        if m:
            out["final_lambda"] = float(m.group(1))

    # Fallbacks.
    if "best_J" not in out:
        vals = [float(v) for v in re.findall(rf"best_J:\s*({NUM})", text)]
        if vals:
            out["best_J"] = max(vals)
    if "best_geom_rate" not in out:
        vals = [float(v) for v in re.findall(rf"best_geom:\s*({NUM})", text)]
        if vals:
            out["best_geom_rate"] = max(vals)
    if "max_run_mean" not in out:
        vals = [float(v) for v in re.findall(rf"best_geom_max_run_mean\s+({NUM})", text)]
        if vals:
            out["max_run_mean"] = vals[-1]

    epochs = [int(v) for v in re.findall(r"Epoch\s+(\d+)\s+\|", text)]
    if epochs:
        out["epochs_run"] = max(epochs)

    eval_cost = [float(v) for v in re.findall(rf"eval_ep_cost=({NUM})", text)]
    rollout_cost = [float(v) for v in re.findall(rf"rollout_ep_cost=({NUM})", text)]
    if eval_cost:
        out["final_eval_ep_cost"] = eval_cost[-1]
        out["mean_eval_ep_cost"] = statistics.mean(eval_cost)
        out["max_eval_ep_cost"] = max(eval_cost)
    if rollout_cost:
        out["final_rollout_ep_cost"] = rollout_cost[-1]
        out["mean_rollout_ep_cost"] = statistics.mean(rollout_cost)
        out["max_rollout_ep_cost"] = max(rollout_cost)

    for prefix in ("collision", "absorb"):
        for src in ("total", "sphere", "physx", "table"):
            vals = [int(v) for v in re.findall(rf"epoch_{prefix}_{src}=(\d+)", text)]
            if vals:
                out[f"cumulative_{prefix}_{src}"] = sum(vals)
                out[f"max_epoch_{prefix}_{src}"] = max(vals)
                out[f"final_epoch_{prefix}_{src}"] = vals[-1]

    lambdas = [float(v) for v in re.findall(rf"(?:λ|lam):?\s*=?\s*({NUM})", text)]
    if lambdas:
        out.setdefault("final_lambda", lambdas[-1])
        out["max_lambda"] = max(lambdas)

    out["physx_arm_applied"] = (
        "verified PhysX arm collision patch fired" in text
        or "applied=14" in text
        or "--enable_physx_arm_collision" in text
    )
    if out["status"] == "crashed":
        out["error_tail"] = "\n".join(text.splitlines()[-30:])
    return out


def group_stats(runs: list[dict]) -> dict[str, dict[str, list[float]]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    keys = [
        "best_J",
        "best_geom_rate",
        "max_run_mean",
        "cumulative_collision_total",
        "cumulative_collision_sphere",
        "cumulative_collision_physx",
        "cumulative_collision_table",
        "max_epoch_collision_total",
        "max_epoch_collision_table",
        "mean_eval_ep_cost",
        "max_eval_ep_cost",
        "mean_rollout_ep_cost",
        "max_rollout_ep_cost",
        "final_lambda",
        "max_lambda",
    ]
    for run in runs:
        g = groups[run["condition"]]
        g["N"].append(1)
        for key in keys:
            if isinstance(run.get(key), (int, float)):
                g[key].append(run[key])
    return groups


def mean(groups: dict[str, dict[str, list[float]]], cond: str, key: str) -> float | None:
    return mean_std(groups.get(cond, {}).get(key, []))[0]


def reduction(baseline: float | None, method: float | None) -> float | None:
    if baseline is None or method is None or baseline == 0:
        return None
    return 100.0 * (baseline - method) / baseline


def write_report(runs: list[dict]) -> None:
    groups = group_stats(runs)
    completed = [r for r in runs if r["status"] == "completed"]
    crashed = [r for r in runs if r["status"] == "crashed"]

    md: list[str] = []
    md += [
        "# Overnight Benchmark Report",
        "",
        f"Source logs: `{LOG_DIR}`",
        f"Runs: {len(runs)} total, {len(completed)} completed, {len(crashed)} crashed.",
        "",
        "## Metric Mapping",
        "",
        "Following the D-ATACOM / long-term safety paper style, this report separates:",
        "- **Task return**: `best_J`.",
        "- **Task quality**: `best_geom_rate` and `max_run_mean` / `max_hold_mean`.",
        "- **Episodic sum of cost**: `eval_ep_cost` and `rollout_ep_cost` summaries.",
        "- **Maximum violation per episode**: `max_epoch_collision_total` and `max_eval_ep_cost`.",
        "- **Safety breakdown**: `sphere`, `physx`, and `table`; table contact is counted as the same safety violation class as arm collision.",
        "",
        "## Aggregate Results",
        "",
        "| Condition | N | best_J | best_geom | max_run | cum_collision_total | cum_table | max_epoch_collision | max_eval_ep_cost | final_lambda |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "stage1_default_lagsac",
        "stage1_default_sac",
        "stage1_harder_lagsac",
        "stage1_harder_sac",
        "stage3_lagsac_cost100",
        "stage3_lagsac_cost50",
        "stage3_sac_overnight_failed",
    ]
    for cond in order:
        if cond not in groups:
            continue
        g = groups[cond]
        md.append(
            f"| `{cond}` | {len(g['N'])} | "
            f"{mean_std_str(g['best_J'], 2)} | "
            f"{mean_std_str(g['best_geom_rate'], 3)} | "
            f"{mean_std_str(g['max_run_mean'], 1)} | "
            f"{mean_std_str(g['cumulative_collision_total'], 1)} | "
            f"{mean_std_str(g['cumulative_collision_table'], 1)} | "
            f"{mean_std_str(g['max_epoch_collision_total'], 1)} | "
            f"{mean_std_str(g['max_eval_ep_cost'], 2)} | "
            f"{mean_std_str(g['final_lambda'], 4)} |"
        )

    md += [
        "",
        "## SAC vs LagSAC Deltas",
        "",
        "| Setting | Delta best_J (LagSAC - SAC) | Total collision reduction | Table collision reduction | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for setting, lag, sac in [
        ("stage1_default", "stage1_default_lagsac", "stage1_default_sac"),
        ("stage1_harder", "stage1_harder_lagsac", "stage1_harder_sac"),
    ]:
        dj = None
        lj, sj = mean(groups, lag, "best_J"), mean(groups, sac, "best_J")
        if lj is not None and sj is not None:
            dj = lj - sj
        total_red = reduction(
            mean(groups, sac, "cumulative_collision_total"),
            mean(groups, lag, "cumulative_collision_total"),
        )
        table_red = reduction(
            mean(groups, sac, "cumulative_collision_table"),
            mean(groups, lag, "cumulative_collision_table"),
        )
        interp = "LagSAC safer and higher return" if (dj or 0) > 0 and (total_red or 0) > 0 else "mixed"
        md.append(f"| `{setting}` | {fmt(dj, 2)} | {fmt(total_red, 1)}% | {fmt(table_red, 1)}% | {interp} |")

    md += [
        "",
        "## Per-Run Detail",
        "",
        "| Run | Status | Ep | best_J | best_geom | max_run | cum_total | cum_sphere | cum_physx | cum_table | max_epoch_total | max_eval_cost | final_eval_cost | final_lambda |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        md.append(
            f"| `{r['run']}` | {r['status']} | "
            f"{r.get('epochs_run', '-')} | "
            f"{fmt(r.get('best_J'), 2)} | "
            f"{fmt(r.get('best_geom_rate'), 3)} | "
            f"{fmt(r.get('max_run_mean'), 1)} | "
            f"{fmt(r.get('cumulative_collision_total'), 0)} | "
            f"{fmt(r.get('cumulative_collision_sphere'), 0)} | "
            f"{fmt(r.get('cumulative_collision_physx'), 0)} | "
            f"{fmt(r.get('cumulative_collision_table'), 0)} | "
            f"{fmt(r.get('max_epoch_collision_total'), 0)} | "
            f"{fmt(r.get('max_eval_ep_cost'), 2)} | "
            f"{fmt(r.get('final_eval_ep_cost'), 2)} | "
            f"{fmt(r.get('final_lambda'), 4)} |"
        )

    if crashed:
        md += ["", "## Crashed Runs", ""]
        for r in crashed:
            md += [f"### `{r['run']}`", "```text", r.get("error_tail", "unknown error"), "```"]

    physx_ok = sum(1 for r in runs if r.get("physx_arm_applied"))
    md += [
        "",
        "## Infrastructure",
        "",
        f"- PhysX arm collision / B-route flag detected: {physx_ok} / {len(runs)} logs.",
        "- Timeout recovery worked after restart: the 02:13 run completed overnight instead of hanging.",
        "",
        "## Interpretation",
        "",
        "- Stage 1 default pose: LagSAC has the stronger result: higher mean return and substantially fewer safety events.",
        "- Stage 1 harder pose: task success saturates for both algorithms, so reward is not discriminative. LagSAC lowers total collisions, but table contacts are not improved; table contact should be treated as a real safety failure.",
        "- Stage 3 LagSAC calibration: both cost budgets produced zero arm/table safety events, but also zero insertion success. This is safe but not useful yet; the currently running SAC Stage 3 retry is required for the missing baseline.",
        "- Lambda collapsed to near zero in most B-route runs because late-epoch costs were below the selected budgets. If table safety should be binding, the next sweep should lower or weight the table cost separately.",
        "",
    ]
    REPORT.write_text("\n".join(md))


def main() -> None:
    if not LOG_DIR.is_dir():
        raise SystemExit(f"No log dir found: {LOG_DIR}")
    runs = [parse_log(p) for p in sorted(LOG_DIR.glob("*.log"))]
    write_report(runs)
    completed = sum(1 for r in runs if r["status"] == "completed")
    crashed = sum(1 for r in runs if r["status"] == "crashed")
    print(f"Report written to {REPORT}")
    print(f"Runs: {len(runs)} total, {completed} completed, {crashed} crashed")


if __name__ == "__main__":
    main()
