#!/usr/bin/env python3
"""Parse /tmp/overnight_logs/*.log and produce a markdown summary.

Run anytime — script is read-only on logs, writes report to:
  /tmp/overnight_report.md

Usage:
  python scripts/analyze_overnight.py
"""

import re
import json
import statistics
from pathlib import Path

LOG_DIR = Path("/tmp/overnight_logs")
REPORT = Path("/tmp/overnight_report.md")
SUMMARY = Path("/tmp/overnight_summary.txt")


def parse_log(log_path: Path) -> dict:
    """Extract key metrics from a single training log."""
    if not log_path.is_file():
        return {"status": "log_missing"}
    try:
        text = log_path.read_text(errors="replace")
    except Exception as e:
        return {"status": f"read_error: {e}"}

    out = {"status": "incomplete", "log": str(log_path)}

    # Final completion line
    m = re.search(r"训练完成\.\s+best J\s+=\s+(-?[\d.]+)", text)
    if m:
        out["best_J"] = float(m.group(1))
        out["status"] = "completed"

    # best geom rate
    m = re.search(r"best[_ ]geom[_ ]?rate\s*=\s*([\d.]+)", text)
    if m:
        out["best_geom_rate"] = float(m.group(1))
    # max_hold_mean / max_run_mean
    m = re.search(r"max_(?:hold|run)_mean=([\d.]+)", text)
    if m:
        out["max_run_mean"] = float(m.group(1))

    # final lambda (LagSAC only)
    m = re.search(r"final λ\s*=\s*([\d.]+)", text)
    if m:
        out["final_lambda"] = float(m.group(1))

    # Last-epoch and cumulative safety counters. Treat table contact as the
    # same safety class as arm-arm contact, but keep source split for diagnosis.
    # Prefer explicit *_total fields (OR mask, no double counting when sources
    # fire together); fall back to source sums for older logs.
    coll_total_lines = re.findall(r"epoch_collision_total=(\d+)", text)
    coll_lines = re.findall(
        r"epoch_collision_sphere=(\d+)[^\n]*epoch_collision_physx=(\d+)(?:[^\n]*epoch_collision_table=(\d+))?",
        text
    )
    if coll_lines:
        cs, cp, ct = coll_lines[-1]
        out["final_epoch_collision_sphere"] = int(cs)
        out["final_epoch_collision_physx"] = int(cp)
        out["final_epoch_collision_table"] = int(ct or 0)
        all_cs = [int(x[0]) for x in coll_lines]
        all_cp = [int(x[1]) for x in coll_lines]
        all_ct = [int(x[2] or 0) for x in coll_lines]
        out["cumulative_collision_sphere"] = sum(all_cs)
        out["cumulative_collision_physx"] = sum(all_cp)
        out["cumulative_collision_table"] = sum(all_ct)
        out["cumulative_collision_total"] = (
            sum(int(x) for x in coll_total_lines)
            if coll_total_lines
            else sum(all_cs) + sum(all_cp) + sum(all_ct)
        )

    absorb_total_lines = re.findall(r"epoch_absorb_total=(\d+)", text)
    absorb_lines = re.findall(
        r"epoch_absorb_sphere=(\d+)[^\n]*epoch_absorb_physx=(\d+)(?:[^\n]*epoch_absorb_table=(\d+))?",
        text
    )
    if absorb_lines:
        as_, ap, at = absorb_lines[-1]
        out["final_epoch_absorb_sphere"] = int(as_)
        out["final_epoch_absorb_physx"] = int(ap)
        out["final_epoch_absorb_table"] = int(at or 0)
        out["final_epoch_absorb_total"] = (
            int(absorb_total_lines[-1])
            if absorb_total_lines
            else int(as_) + int(ap) + int(at or 0)
        )

    # Last eval_ep_cost
    eval_costs = re.findall(r"eval_ep_cost=([\d.]+)", text)
    if eval_costs:
        out["final_eval_ep_cost"] = float(eval_costs[-1])

    # epoch count actually run
    eps = re.findall(r"Epoch (\d+) \|", text)
    if eps:
        out["epochs_run"] = int(eps[-1])

    # PhysX arm collision verification
    if "verified PhysX arm collision patch fired" in text:
        out["physx_arm_applied"] = True
    elif "applied=14" in text:
        out["physx_arm_applied"] = True

    # Detect failures
    if "Traceback" in text or "RuntimeError" in text:
        out["status"] = "crashed"
        # Last few lines of error
        tb_match = re.search(r"(Traceback[\s\S]+)", text)
        if tb_match:
            out["error_tail"] = tb_match.group(1)[-500:]

    return out


def fmt_num(v, fmt=".3f"):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:{fmt}}"
    return str(v)


def aggregate(runs: list[dict], key: str):
    """Return mean ± std of a key across runs that have it."""
    vals = [r[key] for r in runs if key in r and r[key] is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main():
    if not LOG_DIR.is_dir():
        print(f"No logs found at {LOG_DIR}")
        return

    runs = {}
    for log in sorted(LOG_DIR.glob("*.log")):
        runs[log.stem] = parse_log(log)

    if not runs:
        print(f"No log files in {LOG_DIR}")
        return

    # Group runs by prefix: lag_default_*, sac_default_*, lag_harder_*, sac_harder_*, etc.
    groups = {}
    for name, data in runs.items():
        parts = name.split("_s")
        if len(parts) >= 2 and parts[-1].isdigit():
            key = "_s".join(parts[:-1])  # everything except last _sN
        else:
            key = name
        groups.setdefault(key, []).append((name, data))

    md = []
    md.append("# Overnight Benchmark Report")
    md.append("")
    md.append(f"Total runs analyzed: {len(runs)}")
    md.append("")

    md.append("## Aggregate by Condition\n")
    md.append("| Condition | N | best_J (mean±std) | best_geom (mean) | max_run_mean (mean±std) | cum_collision_total | cum_collision_sphere | cum_collision_table | final_λ (last seed) |")
    md.append("|---|---:|---|---:|---|---:|---:|---:|---:|")
    for key in sorted(groups):
        run_list = [d for (n, d) in groups[key] if d["status"] == "completed"]
        n = len(run_list)
        if n == 0:
            md.append(f"| `{key}` | 0 | — | — | — | — | (no completed runs) |")
            continue
        jm, js = aggregate(run_list, "best_J")
        gm, _ = aggregate(run_list, "best_geom_rate")
        mm, ms = aggregate(run_list, "max_run_mean")
        total_cm, _ = aggregate(run_list, "cumulative_collision_total")
        cm, _ = aggregate(run_list, "cumulative_collision_sphere")
        table_cm, _ = aggregate(run_list, "cumulative_collision_table")
        finalL = run_list[-1].get("final_lambda")
        md.append(
            f"| `{key}` | {n} | "
            f"{fmt_num(jm, '.2f')} ± {fmt_num(js, '.2f')} | "
            f"{fmt_num(gm, '.3f')} | "
            f"{fmt_num(mm, '.1f')} ± {fmt_num(ms, '.1f')} | "
            f"{int(total_cm) if total_cm else '—'} | "
            f"{int(cm) if cm else '—'} | "
            f"{int(table_cm) if table_cm else '—'} | "
            f"{fmt_num(finalL, '.4f')} |"
        )
    md.append("")

    md.append("## Per-Run Detail\n")
    md.append("| Run | Status | epochs | best_J | best_geom | max_run | cum_coll_total | cum_coll_S | cum_coll_P | cum_coll_T | final_absorb_T | final_λ |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(runs):
        d = runs[name]
        md.append(
            f"| `{name}` | {d['status']} | "
            f"{d.get('epochs_run', '—')} | "
            f"{fmt_num(d.get('best_J'), '.2f')} | "
            f"{fmt_num(d.get('best_geom_rate'), '.3f')} | "
            f"{fmt_num(d.get('max_run_mean'), '.1f')} | "
            f"{d.get('cumulative_collision_total', '—')} | "
            f"{d.get('cumulative_collision_sphere', '—')} | "
            f"{d.get('cumulative_collision_physx', '—')} | "
            f"{d.get('cumulative_collision_table', '—')} | "
            f"{d.get('final_epoch_absorb_table', '—')} | "
            f"{fmt_num(d.get('final_lambda'), '.4f')} |"
        )
    md.append("")

    crashed = [n for n, d in runs.items() if d["status"] == "crashed"]
    if crashed:
        md.append("## Crashed Runs\n")
        for n in crashed:
            md.append(f"### {n}\n```\n{runs[n].get('error_tail', '?')}\n```\n")

    # PhysX verification
    physx_ok = sum(1 for d in runs.values() if d.get("physx_arm_applied"))
    md.append(f"\n## Infrastructure Verification\n")
    md.append(f"- PhysX arm collision applied: {physx_ok} / {len(runs)} runs")
    md.append("")

    REPORT.write_text("\n".join(md))
    print(f"Report written to {REPORT}")
    print(f"  Total runs: {len(runs)}")
    print(f"  Completed: {sum(1 for d in runs.values() if d['status']=='completed')}")
    print(f"  Crashed:   {len(crashed)}")
    print(f"  Conditions: {len(groups)}")


if __name__ == "__main__":
    main()
