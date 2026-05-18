from __future__ import annotations

import argparse
import os
import re
import shutil
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


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _checkpoint_to_load_args(checkpoint: str | None) -> tuple[str | None, str | None]:
    if not checkpoint:
        return None, None
    checkpoint_path = _expand_path(checkpoint)
    _require_file(checkpoint_path, "SEA-Nav checkpoint")
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)
    if not match:
        raise ValueError(f"Expected checkpoint filename like model_20.pt: {checkpoint_path}")
    return checkpoint_path.parent.name, match.group(1)


def _base_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    external_cfg = _section(cfg, "external_repo")
    sea_nav_path = _expand_path(str(external_cfg.get("path", "external/SEA-Nav-Code")))
    train_script = sea_nav_path / "training" / "legged_gym" / "legged_gym" / "scripts" / "train.py"
    play_script = sea_nav_path / "training" / "legged_gym" / "legged_gym" / "scripts" / "play.py"
    _require_file(train_script, "SEA-Nav train script")
    _require_file(play_script, "SEA-Nav play script")
    return sea_nav_path, train_script, play_script


def _python_command(cfg: Mapping[str, Any], script: Path) -> list[str]:
    runtime_cfg = _section(cfg, "runtime")
    conda_env = str(runtime_cfg.get("conda_env", "sea_nav"))
    return ["conda", "run", "-n", conda_env, "python", str(script)]


def _command(cfg: Mapping[str, Any], command: str, checkpoint: str | None) -> tuple[list[str], Path, dict[str, str]]:
    sea_nav_path, train_script, play_script = _base_paths(cfg)
    training_cfg = _section(cfg, "training")
    evaluation_cfg = _section(cfg, "evaluation")
    recording_cfg = _section(cfg, "recording")

    task_name = str(training_cfg.get("task_name", "go2_pos_rough"))
    experiment_name = str(training_cfg.get("experiment_name", "Go2_pos_rough"))
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_DISABLED", "true")
    env.pop("SEA_NAV_ENABLE_WANDB", None)

    if command == "train":
        cmd = _python_command(cfg, train_script)
        cmd.extend(
            [
                "--task",
                task_name,
                "--experiment_name",
                experiment_name,
                "--num_envs",
                str(int(training_cfg.get("num_envs", 16))),
                "--max_iterations",
                str(int(training_cfg.get("max_iterations", 20))),
                "--seed",
                str(int(training_cfg.get("seed", 42))),
            ]
        )
        if _bool(training_cfg.get("headless"), True):
            cmd.append("--headless")
        load_run, checkpoint_number = _checkpoint_to_load_args(checkpoint)
        if load_run and checkpoint_number:
            cmd.extend(["--resume", "--load_run", load_run, "--checkpoint", checkpoint_number])
    elif command in {"eval", "play", "record"}:
        cfg_for_mode = recording_cfg if command == "record" else evaluation_cfg
        cmd = _python_command(cfg, play_script)
        cmd.extend(
            [
                "--task",
                task_name,
                "--experiment_name",
                experiment_name,
                "--num_envs",
                str(int(cfg_for_mode.get("num_envs", 1))),
            ]
        )
        load_run, checkpoint_number = _checkpoint_to_load_args(checkpoint)
        if load_run and checkpoint_number:
            cmd.extend(["--load_run", load_run, "--checkpoint", checkpoint_number])
        if command == "record":
            env["SEA_NAV_RECORD_VIDEO"] = "1"
            env["SEA_NAV_PLAY_HEADLESS"] = "1" if _bool(recording_cfg.get("headless"), True) else "0"
            env["SEA_NAV_VIDEO_LENGTH"] = str(int(recording_cfg.get("video_length", 2000)))
            env["SEA_NAV_TOTAL_EPISODES"] = str(int(recording_cfg.get("total_episodes", 3)))
        elif command == "eval":
            env["SEA_NAV_PLAY_HEADLESS"] = "1"
            env["SEA_NAV_TOTAL_EPISODES"] = str(int(evaluation_cfg.get("total_episodes", 10)))
        else:
            env["SEA_NAV_PLAY_HEADLESS"] = "1" if _bool(evaluation_cfg.get("headless"), False) else "0"
            env["SEA_NAV_TOTAL_EPISODES"] = str(int(evaluation_cfg.get("total_episodes", 10)))
    else:
        raise ValueError(f"Unsupported command for SEA-Nav runner: {command}")

    return cmd, sea_nav_path, env


def _copy_recorded_video(cfg: Mapping[str, Any], checkpoint: str | None) -> Path | None:
    if not checkpoint:
        return None
    sea_nav_path, _, _ = _base_paths(cfg)
    training_cfg = _section(cfg, "training")
    outputs_cfg = _section(cfg, "outputs")
    recording_cfg = _section(cfg, "recording")
    experiment_name = str(training_cfg.get("experiment_name", "Go2_pos_rough"))
    load_run, checkpoint_number = _checkpoint_to_load_args(checkpoint)
    if not load_run or not checkpoint_number:
        return None
    source = (
        sea_nav_path
        / "training"
        / "legged_gym"
        / "logs"
        / experiment_name
        / "exported"
        / f"{load_run}_{checkpoint_number}.mp4"
    )
    if not source.exists():
        return None
    video_dir = _expand_path(str(outputs_cfg.get("video_dir", "outputs/videos/navigation/sea_nav_baseline")))
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"sea_nav_demo_model_{checkpoint_number}_{int(recording_cfg.get('video_length', 2000))}steps.mp4"
    shutil.copy2(source, dest)
    return dest


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

    cmd, cwd, env = _command(cfg, command, args.checkpoint)

    print("SEA-Nav command:")
    print("  " + " ".join(cmd))
    print("Working directory:")
    print(f"  {cwd}")

    if args.dry_run:
        print("Dry-run only; simulator/training was not launched.")
        return 0

    completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    if command == "record" and completed.returncode == 0:
        copied = _copy_recorded_video(cfg, args.checkpoint)
        if copied is not None:
            print(f"Copied demo video to: {copied}")
    return completed.returncode
