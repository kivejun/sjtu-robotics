from __future__ import annotations

import argparse
import os
import shlex
import shutil
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


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _base_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    external_cfg = _section(cfg, "external")
    robot_lab_path = _expand_path(str(external_cfg.get("robot_lab_path", "external/robot_lab")))
    isaaclab_path = _expand_path(str(external_cfg.get("isaaclab_path", "external/IsaacLab")))
    isaaclab_launcher = isaaclab_path / "isaaclab.sh"
    train_script = robot_lab_path / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    play_script = robot_lab_path / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    _require_file(isaaclab_launcher, "IsaacLab launcher")
    _require_file(train_script, "robot_lab RSL-RL train script")
    _require_file(play_script, "robot_lab RSL-RL play script")
    return robot_lab_path, isaaclab_launcher, train_script, play_script


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    if not path.exists():
        return
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(path) if not current else f"{path}:{current}"


def _preserve_current_python_env(env: dict[str, str]) -> None:
    python_path = Path(sys.executable).resolve()
    if python_path.parent.name != "bin":
        return
    conda_prefix = python_path.parents[1]
    env.setdefault("CONDA_PREFIX", str(conda_prefix))
    env.setdefault("CONDA_DEFAULT_ENV", conda_prefix.name)
    env["PATH"] = f"{python_path.parent}:{env.get('PATH', '')}"


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


def _isaaclab_command(isaaclab_launcher: Path, script: Path) -> list[str]:
    return [str(isaaclab_launcher), "-p", str(script)]


def _checkpoint_path(checkpoint: str | None) -> Path | None:
    if not checkpoint:
        return None
    checkpoint_path = _expand_path(checkpoint)
    _require_file(checkpoint_path, "robot_lab checkpoint")
    return checkpoint_path


def _append_common_args(cmd: list[str], task_name: str, experiment_name: str, num_envs: int) -> None:
    cmd.extend(
        [
            "--task",
            task_name,
            "--experiment_name",
            experiment_name,
            "--num_envs",
            str(num_envs),
        ]
    )


def _command(cfg: Mapping[str, Any], command: str, checkpoint: str | None) -> tuple[list[str], Path, dict[str, str]]:
    robot_lab_path, isaaclab_launcher, train_script, play_script = _base_paths(cfg)
    robot_lab_cfg = _section(cfg, "robot_lab")
    training_cfg = _section(cfg, "training")
    evaluation_cfg = _section(cfg, "evaluation")
    recording_cfg = _section(cfg, "recording")

    task_name = str(robot_lab_cfg["task_name"])
    experiment_name = str(robot_lab_cfg.get("experiment_name", task_name.replace("RobotLab-Isaac-", "").replace("-v0", "")))
    env = os.environ.copy()
    _preserve_current_python_env(env)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    if env.get("TERM") in {None, "", "dumb"}:
        env["TERM"] = "xterm-256color"
    env = _with_isaacsim_env(env, isaaclab_launcher.parent)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    if env.get("TERM") in {None, "", "dumb"}:
        env["TERM"] = "xterm-256color"

    # Isaac Sim 4.5 expects its bundled warp module. A newer conda-installed
    # warp-lang can shadow it and break RobotLab rendering/video recording.
    _prepend_pythonpath(env, isaaclab_launcher.parent / "_isaac_sim" / "extscache" / "omni.warp.core-1.5.0+lx64")

    if command == "train":
        cmd = _isaaclab_command(isaaclab_launcher, train_script)
        _append_common_args(cmd, task_name, experiment_name, int(training_cfg.get("num_envs", 64)))
        cmd.extend(
            [
                "--max_iterations",
                str(int(training_cfg.get("max_iterations", 20))),
                "--seed",
                str(int(training_cfg.get("seed", 42))),
            ]
        )
        logger = training_cfg.get("logger")
        if logger:
            cmd.extend(["--logger", str(logger)])
        if _bool(training_cfg.get("headless"), True):
            cmd.append("--headless")
            env.pop("DISPLAY", None)
            env.pop("WAYLAND_DISPLAY", None)
            env.pop("XAUTHORITY", None)
        checkpoint_path = _checkpoint_path(checkpoint)
        if checkpoint_path is not None:
            cmd.extend(["--resume", "--load_run", checkpoint_path.parent.name, "--checkpoint", checkpoint_path.name])
    elif command in {"eval", "play", "record"}:
        mode_cfg = recording_cfg if command == "record" else evaluation_cfg
        cmd = _isaaclab_command(isaaclab_launcher, play_script)
        _append_common_args(cmd, task_name, experiment_name, int(mode_cfg.get("num_envs", 1)))
        checkpoint_path = _checkpoint_path(checkpoint)
        if checkpoint_path is not None:
            cmd.extend(["--checkpoint", str(checkpoint_path)])
        if command == "record":
            cmd.extend(["--video", "--video_length", str(int(recording_cfg.get("video_length", 1200)))])
        if _bool(mode_cfg.get("headless"), command != "play"):
            cmd.append("--headless")
            env.pop("DISPLAY", None)
            env.pop("WAYLAND_DISPLAY", None)
            env.pop("XAUTHORITY", None)
    else:
        raise ValueError(f"Unsupported command for robot_lab locomotion runner: {command}")

    return cmd, robot_lab_path, env


def _copy_recorded_video(cfg: Mapping[str, Any], checkpoint: str | None) -> Path | None:
    checkpoint_path = _checkpoint_path(checkpoint)
    if checkpoint_path is None:
        return None
    outputs_cfg = _section(cfg, "outputs")
    recording_cfg = _section(cfg, "recording")
    video_candidates = sorted((checkpoint_path.parent / "videos" / "play").glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not video_candidates:
        return None
    video_dir = _expand_path(str(outputs_cfg.get("video_dir", "outputs/videos/locomotion")))
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"{_section(cfg, 'task')['name'].split('/')[-1]}_demo_{checkpoint_path.stem}_{int(recording_cfg.get('video_length', 1200))}steps.mp4"
    shutil.copy2(video_candidates[-1], dest)
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

    print("robot_lab command:")
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
