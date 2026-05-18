#!/usr/bin/env python3
"""Run the Lagrangian SAC geom chain locally from experiment YAMLs.

This is the local-machine counterpart of `scripts/train_hydra.py`: it reuses
the same `conf/experiment/lag_stage*.yaml` parameter source, but does not wrap
the command in Apptainer/Slurm.  Run it from the same conda/Isaac environment
used for direct training on this machine.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESERVED_KEYS = {"train_script", "extra_args"}
NEGATABLE_BOOL_KEYS = {
    "keep_collision_reward_penalty",
    "drop_penetration_reward_for_cost",
    "sphere_collision_terminates",
    "physx_collision_terminates",
    "enable_physx_arm_collision",
    "enable_table_collision",
    "table_collision_terminates",
}
DEFAULT_STAGES = (
    ("stage1g_prepos", "conf/experiment/lag_stage1_prepos.yaml"),
    ("stage2g_preaxis", "conf/experiment/lag_stage2_preaxis.yaml"),
    ("stage3g_insert", "conf/experiment/lag_stage3_insert.yaml"),
)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return [str(item) for item in value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _append_scalar_arg(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    value_s = str(value)
    if value_s.startswith("-"):
        cmd.append(f"{flag}={value_s}")
    else:
        cmd.extend([flag, value_s])


def _append_arg(cmd: list[str], key: str, value) -> None:
    if value is None:
        return
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        elif key in NEGATABLE_BOOL_KEYS:
            cmd.append(f"--no-{key}")
        return
    if isinstance(value, DictConfig):
        raise TypeError(f"Nested config is not supported for train.{key}")
    if isinstance(value, (ListConfig, list, tuple)):
        items = _as_list(value)
        if items:
            cmd.append(flag)
            cmd.extend(items)
        return
    _append_scalar_arg(cmd, flag, value)


def _build_command(
    cfg: DictConfig,
    python_bin: str,
    tag: str,
    keep_wandb_names: bool,
    wandb_entity: str | None,
    wandb_project: str | None,
    no_wandb: bool,
) -> list[str]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "train_script" not in cfg:
        raise ValueError("YAML must define train_script")
    script = PROJECT_ROOT / str(cfg.train_script)
    if not script.is_file():
        raise FileNotFoundError(script)

    if not keep_wandb_names:
        base_name = str(cfg.get("wandb_run_name", script.stem))
        cfg.wandb_run_name = f"{base_name}_{tag}"
        cfg.wandb_group = tag
    if wandb_entity is not None:
        cfg.wandb_entity = wandb_entity
    if wandb_project is not None:
        cfg.wandb_project = wandb_project
    if no_wandb:
        cfg.no_wandb = True

    cmd = [python_bin, str(script)]
    for key, value in cfg.items():
        if key in RESERVED_KEYS:
            continue
        _append_arg(cmd, key, value)
    cmd.extend(_as_list(cfg.get("extra_args", [])))
    return cmd


def _copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst_dir / src.name)


def _snapshot_outputs(
    stage_name: str,
    tag: str,
    cfg: DictConfig,
    cfg_path: Path,
    command: list[str],
    stage_start_time: float,
) -> Path:
    train_script = str(cfg.get("train_script", ""))
    if train_script.endswith("train_sac.py"):
        prefix = "SAC"
        best_hold_name = "best_hold.msh"
        best_agent_name = "best_agent.msh"
        final_agent_name = "final_agent.msh"
    else:
        prefix = "LagSAC"
        best_hold_name = "best_hold_lag.msh"
        best_agent_name = "best_agent_lag.msh"
        final_agent_name = "final_agent_lag.msh"

    best_hold = PROJECT_ROOT / "results" / best_hold_name
    if not best_hold.is_file():
        raise FileNotFoundError(
            f"{stage_name} finished without {best_hold}. "
            "Check the training log before continuing to the next stage."
        )
    if best_hold.stat().st_mtime < stage_start_time:
        raise RuntimeError(
            f"{stage_name} did not produce a fresh {best_hold}. "
            "The existing file is stale from an earlier stage/run, so this stage "
            "did not reach the best-hold criterion. Stop the chain and inspect "
            "the training log instead of warm-starting the next stage from it."
        )
    out_dir = PROJECT_ROOT / "results" / "checkpoints" / "saved" / f"{prefix}_{stage_name}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(PROJECT_ROOT / "results" / best_hold_name, out_dir)
    _copy_if_exists(PROJECT_ROOT / "results" / best_agent_name, out_dir)
    _copy_if_exists(PROJECT_ROOT / "results" / final_agent_name, out_dir)
    shutil.copy2(cfg_path, out_dir / cfg_path.name)
    (out_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    return out_dir


def _select_stages(start_stage: int, stop_stage: int) -> tuple[tuple[str, str], ...]:
    if not (1 <= start_stage <= 3 and 1 <= stop_stage <= 3 and start_stage <= stop_stage):
        raise ValueError("--start_stage/--stop_stage must satisfy 1 <= start <= stop <= 3")
    return DEFAULT_STAGES[start_stage - 1:stop_stage]


def _stage_table_from_args(args) -> tuple[tuple[str, str], ...]:
    return (
        ("stage1g_prepos", args.stage1_cfg),
        ("stage2g_preaxis", args.stage2_cfg),
        ("stage3g_insert", args.stage3_cfg),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run train_sac_lagrangian.py (default: current interpreter).",
    )
    parser.add_argument(
        "--tag",
        default=f"lag_local_compare_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Shared W&B group and saved checkpoint suffix.",
    )
    parser.add_argument("--start_stage", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--stop_stage", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument(
        "--stage1_cfg",
        default="conf/experiment/lag_stage1_prepos.yaml",
        help="YAML config for stage 1.",
    )
    parser.add_argument(
        "--stage2_cfg",
        default="conf/experiment/lag_stage2_preaxis.yaml",
        help="YAML config for stage 2.",
    )
    parser.add_argument(
        "--stage3_cfg",
        default="conf/experiment/lag_stage3_insert.yaml",
        help="YAML config for stage 3.",
    )
    parser.add_argument(
        "--keep_wandb_names",
        action="store_true",
        help="Do not override wandb_group/wandb_run_name from YAML.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="Override YAML wandb_entity for local runs.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="Override YAML wandb_project for local runs.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B by passing --no_wandb to train_sac_lagrangian.py.",
    )
    parser.add_argument(
        "--skip_snapshot",
        action="store_true",
        help=(
            "Run the selected YAML without requiring/copying best_hold*.msh at "
            "the end. Useful for short calibration runs where W&B/logs are the "
            "main output."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    global DEFAULT_STAGES
    DEFAULT_STAGES = _stage_table_from_args(args)
    stages = _select_stages(args.start_stage, args.stop_stage)
    print(f"[local-lag-chain] project_root={PROJECT_ROOT}")
    print(f"[local-lag-chain] tag={args.tag}")
    print(f"[local-lag-chain] python={args.python}")
    print(f"[local-lag-chain] stages={[name for name, _ in stages]}")

    for stage_name, rel_cfg in stages:
        cfg_path = PROJECT_ROOT / rel_cfg
        cfg = OmegaConf.load(cfg_path)
        cmd = _build_command(
            cfg,
            args.python,
            args.tag,
            args.keep_wandb_names,
            args.wandb_entity,
            args.wandb_project,
            args.no_wandb,
        )
        print("\n" + "=" * 100)
        print(f"[local-lag-chain] {stage_name}: {cfg_path}")
        print(shlex.join(cmd))
        if args.dry_run:
            continue
        stage_start_time = time.time()
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        if args.skip_snapshot:
            print("[local-lag-chain] skip_snapshot=true; not copying checkpoint files")
            continue
        out_dir = _snapshot_outputs(stage_name, args.tag, cfg, cfg_path, cmd, stage_start_time)
        print(f"[local-lag-chain] saved checkpoint snapshot: {out_dir}")


if __name__ == "__main__":
    main()
