#!/usr/bin/env python3
"""Backfill W&B config.plot_group and summary.plot_group from run.group.

Why this exists:
  wandb.init(group=...) stores the value as run metadata. In W&B Reports, the
  "Group by" selector is much more reliable with normal config keys. New runs
  now log config["plot_group"] directly (and mirror it into summary); this script fixes already-finished runs.

Example:
  python scripts/backfill_wandb_plot_group.py ENTITY/PROJECT
      --groups sac_final_benchmark_9seed final_bench_20260520
      --apply

Default is dry-run. Pass --apply to update W&B.
"""

from __future__ import annotations

import argparse
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "project_path",
        help="W&B project path, e.g. ENTITY/PROJECT.",
    )
    p.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Only update runs whose run.group is in this list. Default: all runs with a group.",
    )
    p.add_argument(
        "--key",
        default="plot_group",
        help="Config key to write. Default: plot_group.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually update W&B. Without this flag the script only prints planned changes.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing config key if it already exists.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of runs to inspect, useful for testing.",
    )
    return p.parse_args()


def iter_limited(runs: Iterable, limit: int | None):
    for i, run in enumerate(runs):
        if limit is not None and i >= limit:
            break
        yield run


def main() -> None:
    args = parse_args()

    import wandb

    target_groups = set(args.groups) if args.groups else None
    api = wandb.Api()

    inspected = matched = planned = updated = skipped_existing = skipped_no_group = 0
    for run in iter_limited(api.runs(args.project_path), args.limit):
        inspected += 1
        group = getattr(run, "group", None)
        if not group:
            skipped_no_group += 1
            continue
        if target_groups is not None and group not in target_groups:
            continue
        matched += 1

        # api.runs() yields lightweight run objects. Force-load full data so
        # existing config values are not missed by the list-query cache.
        run.load(force=True)
        old_config = run.config.get(args.key)
        old_summary = run.summary.get(args.key)
        config_done = old_config == group
        summary_done = old_summary == group

        conflicting = (
            (old_config is not None and old_config != group)
            or (old_summary is not None and old_summary != group)
        )
        if conflicting and not args.overwrite:
            skipped_existing += 1
            print(
                f"skip existing: {run.path} config.{args.key}={old_config!r} "
                f"summary.{args.key}={old_summary!r} run.group={group!r} "
                "(use --overwrite to replace)"
            )
            continue
        if config_done and summary_done:
            skipped_existing += 1
            print(f"ok existing: {run.path} config/summary {args.key}={group!r}")
            continue

        planned += 1
        action = "update" if args.apply else "dry-run"
        print(
            f"{action}: {run.path} "
            f"config.{args.key}: {old_config!r} -> {group!r}; "
            f"summary.{args.key}: {old_summary!r} -> {group!r}"
        )
        if args.apply:
            if not config_done:
                run.config[args.key] = group
                run.update()
            if not summary_done:
                run.summary[args.key] = group
                run.summary.update()
            updated += 1

    print(
        "summary: "
        f"inspected={inspected} matched={matched} planned={planned} "
        f"updated={updated} skipped_existing={skipped_existing} "
        f"skipped_no_group={skipped_no_group}"
    )
    if not args.apply and planned:
        print("dry-run only; rerun with --apply to write changes.")


if __name__ == "__main__":
    main()
