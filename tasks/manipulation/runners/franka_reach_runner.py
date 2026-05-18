from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
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
    recording_cfg = _section(cfg, "recording")

    isaaclab_path = _expand_path(str(external_cfg.get("isaaclab_path", "external/IsaacLab")))
    isaaclab_sh = isaaclab_path / "isaaclab.sh"
    _require_file(isaaclab_sh, "IsaacLab launcher")

    isaac_task_name = str(task_cfg.get("isaac_task_name", "Isaac-Reach-Franka-v0"))
    isaac_play_task_name = str(task_cfg.get("isaac_play_task_name", isaac_task_name))
    if command == "train":
        headless = bool(training_cfg.get("headless", True))
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
        if checkpoint:
            checkpoint_path = _expand_path(checkpoint)
            cmd.extend(
                [
                    "--resume",
                    "--load_run",
                    checkpoint_path.parent.name,
                    "--checkpoint",
                    checkpoint_path.name,
                ]
            )
    elif command in {"eval", "play", "record"}:
        script = "scripts/reinforcement_learning/rsl_rl/play.py"
        playback_task_name = isaac_play_task_name
        if command == "record":
            num_envs = int(recording_cfg.get("num_envs", evaluation_cfg.get("num_envs", 1)))
            headless = bool(recording_cfg.get("headless", True))
        elif command == "eval":
            num_envs = int(evaluation_cfg.get("num_envs", training_cfg.get("num_envs", 16)))
            headless = bool(evaluation_cfg.get("headless", True))
        else:
            num_envs = int(evaluation_cfg.get("num_envs", training_cfg.get("num_envs", 16)))
            headless = bool(evaluation_cfg.get("headless", False))
        cmd = [
            str(isaaclab_sh),
            "-p",
            script,
            "--task",
            playback_task_name,
            "--num_envs",
            str(num_envs),
        ]
        if checkpoint:
            cmd.extend(["--checkpoint", str(_expand_path(checkpoint))])
        if command == "record":
            video_length = int(recording_cfg.get("video_length", evaluation_cfg.get("video_length", 200)))
            cmd.extend(["--video", "--video_length", str(video_length)])
            rendering_mode = recording_cfg.get("rendering_mode")
            if rendering_mode:
                cmd.extend(["--rendering_mode", str(rendering_mode)])
    else:
        raise ValueError(f"Unsupported command for Franka Reach runner: {command}")

    if headless:
        cmd.append("--headless")
    return cmd


def _preserve_display(cfg: Mapping[str, Any], command: str) -> bool:
    if command != "play":
        return False
    evaluation_cfg = _section(cfg, "evaluation")
    return not bool(evaluation_cfg.get("headless", False))


def _env(*, preserve_display: bool) -> dict[str, str]:
    env = os.environ.copy()
    python_path = Path(sys.executable).resolve()
    if python_path.parent.name == "bin":
        conda_prefix = python_path.parents[1]
        env.setdefault("CONDA_PREFIX", str(conda_prefix))
        env.setdefault("CONDA_DEFAULT_ENV", conda_prefix.name)
        env["PATH"] = f"{python_path.parent}:{env.get('PATH', '')}"
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    if env.get("TERM") in {None, "", "dumb"}:
        env["TERM"] = "xterm"
    if not preserve_display:
        # Avoid Ubuntu 20.04 XCB crashes when running Isaac Sim headless jobs.
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        env.pop("XAUTHORITY", None)
    return env


def _with_isaacsim_env(env: dict[str, str], isaaclab_path: Path) -> dict[str, str]:
    setup_script = isaaclab_path / "_isaac_sim" / "setup_conda_env.sh"
    if not setup_script.exists():
        return env

    clean_env = env.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("LD_LIBRARY_PATH", None)
    command = f"source {shlex.quote(str(setup_script))} >/dev/null && env -0"
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=isaaclab_path,
        env=clean_env,
        check=True,
        capture_output=True,
    )
    loaded_env: dict[str, str] = {}
    for entry in completed.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, _, value = entry.partition(b"=")
        loaded_env[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
    return loaded_env


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

    env = _env(preserve_display=_preserve_display(cfg, command))
    env = _with_isaacsim_env(env, isaaclab_path)
    completed = subprocess.run(cmd, cwd=isaaclab_path, env=env, check=False)
    return completed.returncode
