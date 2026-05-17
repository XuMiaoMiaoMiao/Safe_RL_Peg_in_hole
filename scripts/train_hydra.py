"""
Hydra + Apptainer 训练入口 (train_hydra.py)

使用示例：
  python scripts/train_hydra.py --multirun experiment@train=phase1
  python scripts/train_hydra.py --multirun experiment@train=phase2

该脚本是项目的统一训练启动器。任务差异由 `conf/experiment/*.yaml`
决定，包括要调用哪个训练脚本、要传哪些训练参数。

该脚本负责：
1) 解析 Hydra 配置 (conf/config.yaml)
2) 将 `cfg.train` 泛化映射为训练脚本 CLI 参数
3) 在 Apptainer 容器中执行训练
4) 支持本地运行 或 通过 Hydra Submitit 提交到 Slurm (HPC)

说明：
- --multirun: 启用 Hydra sweep（可扩展为 grid/random sweep）
- `experiment@train=phase1|phase2`: 选择任务配置与训练脚本
- 每个 override 组合会生成一个独立 Slurm job
- 日志和输出保存在 Hydra 自动创建的目录中
"""


from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESERVED_TRAIN_KEYS = {"train_script", "extra_args"}

# argparse args declared with action=argparse.BooleanOptionalAction in the
# downstream train scripts. For those args, yaml `key: false` must emit
# `--no-{key}` instead of silently dropping the flag (which leaves the
# default=True in effect — the silent bug fixed 2026-05-17).
#
# All other bool args use action="store_true" (default=False), where `false`
# is a no-op and `--no-{key}` would crash argparse. Keep this whitelist tight.
#
# Source of truth: grep "BooleanOptionalAction" in scripts/train_*.py.
BOOLEAN_OPTIONAL_ARGS: set[str] = {
    "keep_collision_reward_penalty",
    "drop_penetration_reward_for_cost",
}


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
    s = str(value)
    # Negative numbers (e.g. -inf, -0.5) start with '-' and would be mistaken
    # for a new option flag by argparse when passed as a separate token.
    if s.startswith("-"):
        cmd.append(f"{flag}={s}")
    else:
        cmd.extend([flag, s])


def _append_vector_arg(cmd: list[str], flag: str, value) -> None:
    items = _as_list(value)
    if not items:
        return
    cmd.append(flag)
    cmd.extend(items)


def _append_bool_arg(cmd: list[str], flag: str, enabled: bool, key: str) -> None:
    if enabled:
        cmd.append(flag)
    elif key in BOOLEAN_OPTIONAL_ARGS:
        # BooleanOptionalAction default is typically True; explicit `false`
        # in yaml must translate to `--no-{key}` to actually disable the flag.
        cmd.append(f"--no-{key}")
    # else: store_true arg with `false` value is a no-op (default=False).


def _append_train_arg(cmd: list[str], key: str, value) -> None:
    flag = f"--{key}"

    if value is None:
        return
    if isinstance(value, bool):
        _append_bool_arg(cmd, flag, value, key)
        return
    if isinstance(value, DictConfig):
        raise TypeError(
            f"`train.{key}` is a nested mapping, which the generic launcher cannot "
            "translate into CLI arguments"
        )
    if isinstance(value, (ListConfig, list, tuple)):
        _append_vector_arg(cmd, flag, value)
        return

    _append_scalar_arg(cmd, flag, value)


def _validate_cfg(cfg: DictConfig) -> None:
    train_script = cfg.train.get("train_script")
    if not train_script:
        raise ValueError("`train.train_script` must be set by the selected experiment config")

    host_train_script = PROJECT_ROOT / str(train_script)
    if not host_train_script.is_file():
        raise FileNotFoundError(f"train script not found: {host_train_script}")

    extra_args = cfg.train.get("extra_args", [])
    if not isinstance(extra_args, (ListConfig, list, tuple)):
        raise TypeError("`train.extra_args` must be a list of additional CLI tokens")


def _build_train_command(cfg: DictConfig) -> list[str]:
    train_cfg = cfg.train
    cmd = [str(cfg.project.isaac_python), str(train_cfg.train_script)]

    for key, value in train_cfg.items():
        if key in RESERVED_TRAIN_KEYS:
            continue
        _append_train_arg(cmd, key, value)

    cmd.extend(_as_list(train_cfg.get("extra_args", [])))

    return cmd


def _build_apptainer_command(cfg: DictConfig) -> list[str]:
    train_cmd = _build_train_command(cfg)
    train_cmd_str = shlex.join(train_cmd)
    container_root = str(cfg.project.container_project_root)
    inner_cmd = f"cd {shlex.quote(container_root)} && {train_cmd_str}"

    cmd = [str(cfg.apptainer.executable), "exec"]
    if cfg.apptainer.cleanenv:
        cmd.append("--cleanenv")
    if cfg.apptainer.nv:
        cmd.append("--nv")
    for bind in _as_list(cfg.apptainer.binds):
        cmd.extend(["--bind", bind])
    cmd.append(str(cfg.apptainer.image))
    cmd.append(str(cfg.apptainer.shell))
    cmd.extend(_as_list(cfg.apptainer.shell_args))
    cmd.append(inner_cmd)
    return cmd


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    _validate_cfg(cfg)

    job = HydraConfig.get().job
    print(f"[hydra] job_name={job.name}")
    print(f"[hydra] overrides={list(job.override_dirname.split(',')) if job.override_dirname else []}")
    print(f"[hydra] cwd={Path.cwd()}")
    print(f"[hydra] original_cwd={get_original_cwd()}")
    print("[hydra] resolved_config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    cmd = _build_apptainer_command(cfg)
    print("[launch] command:")
    print(shlex.join(cmd))

    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
