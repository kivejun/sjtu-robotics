from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from summer_camp_rl.registry import TaskSpec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _expand_path(path: str | os.PathLike[str]) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return _repo_root() / value


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section must be a mapping: {name}")
    return value


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _base_command(cfg: Mapping[str, Any], command: str, checkpoint: str | None) -> list[str]:
    task_cfg = _section(cfg, "task")
    external_cfg = _section(cfg, "external")
    training_cfg = _section(cfg, "training")
    evaluation_cfg = _section(cfg, "evaluation")

    isaaclab_path = _expand_path(str(external_cfg.get("isaaclab_path", "external/IsaacLab")))
    isaaclab_sh = isaaclab_path / "isaaclab.sh"
    _require_file(isaaclab_sh, "IsaacLab launcher")

    isaac_task_name = str(task_cfg.get("isaac_task_name", "Isaac-Reach-Franka-v0"))
    headless = bool(training_cfg.get("headless", True))

    if command == "train":
        script = "scripts/reinforcement_learning/rsl_rl/train.py"
        num_envs = int(training_cfg.get("num_envs", 16))
        cmd = [
            str(isaaclab_sh),
            "-p",
            script,
            "--task",
            isaac_task_name,
            "--num_envs",
            str(num_envs),
            "--max_iterations",
            str(int(training_cfg.get("max_iterations", 20))),
        ]
    elif command in {"eval", "play", "record"}:
        script = "scripts/reinforcement_learning/rsl_rl/play.py"
        num_envs = int(evaluation_cfg.get("num_envs", training_cfg.get("num_envs", 16)))
        cmd = [
            str(isaaclab_sh),
            "-p",
            script,
            "--task",
            isaac_task_name,
            "--num_envs",
            str(num_envs),
        ]
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        if command == "record":
            cmd.extend(["--video", "--video_length", str(int(evaluation_cfg.get("video_length", 200)))])
    else:
        raise ValueError(f"Unsupported command for Franka Reach runner: {command}")

    if headless:
        cmd.append("--headless")
    return cmd


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    env.setdefault("TERM", "xterm")
    # Avoid Ubuntu 20.04 XCB crashes when running Isaac Sim in headless mode.
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("XAUTHORITY", None)
    return env


def run(
    command: str,
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    task: TaskSpec,
) -> int:
    task_cfg = _section(cfg, "task")
    config_task_name = task_cfg.get("name")
    if config_task_name != task.name:
        raise ValueError(f"Config task.name={config_task_name!r} does not match requested task {task.name!r}")

    cmd = _base_command(cfg, command, args.checkpoint)
    isaaclab_path = _expand_path(str(_section(cfg, "external").get("isaaclab_path", "external/IsaacLab")))

    print("IsaacLab command:")
    print("  " + " ".join(cmd))
    print("Working directory:")
    print(f"  {isaaclab_path}")

    if args.dry_run:
        print("Dry-run only; simulator/training was not launched.")
        return 0

    completed = subprocess.run(cmd, cwd=isaaclab_path, env=_env(), check=False)
    return completed.returncode
