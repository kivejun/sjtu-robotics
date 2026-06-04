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


def _apply_dynamic_obstacle_env(cfg: Mapping[str, Any], env: dict[str, str]) -> None:
    environment_cfg = _section(cfg, "environment")
    mapping = {
        "enable_dynamic_obstacles": "SEA_NAV_DYNAMIC_ENABLE",
        "use_legacy_dynamic_reward": "SEA_NAV_DYNAMIC_LEGACY_REWARD",
        "num_dynamic_obstacles": "SEA_NAV_DYNAMIC_NUM_OBSTACLES",
        "obstacle_radius": "SEA_NAV_DYNAMIC_OBSTACLE_RADIUS",
        "obstacle_speed": "SEA_NAV_DYNAMIC_OBSTACLE_SPEED",
        "obstacle_collision_distance": "SEA_NAV_DYNAMIC_COLLISION_DISTANCE",
        "obstacle_amplitude_scale": "SEA_NAV_DYNAMIC_AMPLITUDE_SCALE",
        "obstacle_amplitude_min": "SEA_NAV_DYNAMIC_AMPLITUDE_MIN",
        "obstacle_amplitude_max": "SEA_NAV_DYNAMIC_AMPLITUDE_MAX",
        "dynamic_obstacle_state_k": "SEA_NAV_DYNAMIC_OBS_K",
        "ttc_horizon": "SEA_NAV_DYNAMIC_TTC_HORIZON",
        "path_block_distance": "SEA_NAV_DYNAMIC_PATH_BLOCK_DISTANCE",
        "path_block_width": "SEA_NAV_DYNAMIC_PATH_BLOCK_WIDTH",
        "path_block_ttc": "SEA_NAV_DYNAMIC_PATH_BLOCK_TTC",
        "path_block_rise": "SEA_NAV_DYNAMIC_PATH_BLOCK_RISE",
        "path_block_fall": "SEA_NAV_DYNAMIC_PATH_BLOCK_FALL",
        "path_block_threshold": "SEA_NAV_DYNAMIC_PATH_BLOCK_THRESHOLD",
        "preferred_speed": "SEA_NAV_DYNAMIC_PREFERRED_SPEED",
        "wait_speed": "SEA_NAV_DYNAMIC_WAIT_SPEED",
        "ttc_safe_distance": "SEA_NAV_DYNAMIC_TTC_SAFE_DISTANCE",
        "blocked_timeout": "SEA_NAV_DYNAMIC_BLOCKED_TIMEOUT",
    }
    for cfg_key, env_key in mapping.items():
        if cfg_key in environment_cfg:
            env[env_key] = str(environment_cfg[cfg_key])

    if "include_dynamic_obstacle_state" in environment_cfg:
        env["SEA_NAV_DYNAMIC_OBS_STATE"] = "1" if _bool(environment_cfg["include_dynamic_obstacle_state"]) else "0"

    motion_modes = environment_cfg.get("obstacle_motion_modes")
    if motion_modes is not None:
        if not isinstance(motion_modes, list):
            raise TypeError("environment.obstacle_motion_modes must be a list")
        env["SEA_NAV_DYNAMIC_MOTION_MODES"] = ",".join(str(mode) for mode in motion_modes)

    reward_cfg = _section(cfg, "reward")
    reward_mapping = {
        "progress": "SEA_NAV_REWARD_PROGRESS",
        "preferred_velocity": "SEA_NAV_REWARD_PREFERRED_VELOCITY",
        "dynamic_ttc": "SEA_NAV_REWARD_DYNAMIC_TTC",
        "dynamic_clearance": "SEA_NAV_REWARD_DYNAMIC_CLEARANCE",
        "wait": "SEA_NAV_REWARD_WAIT",
        "blocked_fast_penalty": "SEA_NAV_REWARD_BLOCKED_FAST_PENALTY",
        "detour": "SEA_NAV_REWARD_DETOUR",
        "near_goal_stop": "SEA_NAV_REWARD_NEAR_GOAL_STOP",
        "dynamic_collision": "SEA_NAV_REWARD_DYNAMIC_COLLISION",
        "progress_blocked_scale": "SEA_NAV_REWARD_PROGRESS_BLOCKED_SCALE",
        "progress_near_goal_scale": "SEA_NAV_REWARD_PROGRESS_NEAR_GOAL_SCALE",
        "preferred_sigma": "SEA_NAV_REWARD_PREFERRED_SIGMA",
        "ttc_min_closing_speed": "SEA_NAV_REWARD_TTC_MIN_CLOSING_SPEED",
        "clearance_safe_distance": "SEA_NAV_REWARD_CLEARANCE_SAFE_DISTANCE",
        "clearance_front_margin": "SEA_NAV_REWARD_CLEARANCE_FRONT_MARGIN",
        "wait_max_speed": "SEA_NAV_REWARD_WAIT_MAX_SPEED",
        "wait_heading_weight": "SEA_NAV_REWARD_WAIT_HEADING_WEIGHT",
        "wait_timeout_decay": "SEA_NAV_REWARD_WAIT_TIMEOUT_DECAY",
        "blocked_fast_speed_threshold": "SEA_NAV_REWARD_BLOCKED_FAST_SPEED_THRESHOLD",
        "blocked_fast_ttc_threshold": "SEA_NAV_REWARD_BLOCKED_FAST_TTC_THRESHOLD",
        "blocked_fast_closing_speed_threshold": "SEA_NAV_REWARD_BLOCKED_FAST_CLOSING_SPEED_THRESHOLD",
        "detour_lateral_weight": "SEA_NAV_REWARD_DETOUR_LATERAL_WEIGHT",
        "detour_yaw_weight": "SEA_NAV_REWARD_DETOUR_YAW_WEIGHT",
        "detour_blocked_scale": "SEA_NAV_REWARD_DETOUR_BLOCKED_SCALE",
        "detour_timeout_scale": "SEA_NAV_REWARD_DETOUR_TIMEOUT_SCALE",
        "near_goal_distance": "SEA_NAV_REWARD_NEAR_GOAL_DISTANCE",
        "near_goal_speed": "SEA_NAV_REWARD_NEAR_GOAL_SPEED",
        "near_goal_max_speed": "SEA_NAV_REWARD_NEAR_GOAL_MAX_SPEED",
        "near_goal_max_yaw_rate": "SEA_NAV_REWARD_NEAR_GOAL_MAX_YAW_RATE",
    }
    for cfg_key, env_key in reward_mapping.items():
        if cfg_key in reward_cfg:
            env[env_key] = str(reward_cfg[cfg_key])

    pipeline_cfg = _section(cfg, "pipeline")
    pipeline_mapping = {
        "controller": "SEA_NAV_PLAY_CONTROLLER",
        "preferred_speed": "SEA_NAV_DWA_PREFERRED_SPEED",
        "wait_speed": "SEA_NAV_DWA_WAIT_SPEED",
        "max_lateral_speed": "SEA_NAV_DWA_MAX_LATERAL_SPEED",
        "max_yaw_rate": "SEA_NAV_DWA_MAX_YAW_RATE",
        "ttc_horizon": "SEA_NAV_DWA_TTC_HORIZON",
        "safe_distance": "SEA_NAV_DWA_SAFE_DISTANCE",
        "static_clearance": "SEA_NAV_DWA_STATIC_CLEARANCE",
        "static_inflation": "SEA_NAV_DWA_STATIC_INFLATION",
        "obstacle_height": "SEA_NAV_DWA_OBSTACLE_HEIGHT",
        "waypoint_lookahead": "SEA_NAV_DWA_WAYPOINT_LOOKAHEAD",
        "astar_replan_period": "SEA_NAV_DWA_ASTAR_REPLAN_PERIOD",
        "control_period": "SEA_NAV_DWA_CONTROL_PERIOD",
        "goal_weight": "SEA_NAV_DWA_GOAL_WEIGHT",
        "velocity_weight": "SEA_NAV_DWA_VELOCITY_WEIGHT",
        "ttc_weight": "SEA_NAV_DWA_TTC_WEIGHT",
        "clearance_weight": "SEA_NAV_DWA_CLEARANCE_WEIGHT",
        "yaw_weight": "SEA_NAV_DWA_YAW_WEIGHT",
        "smoothness_weight": "SEA_NAV_DWA_SMOOTHNESS_WEIGHT",
        "wait_bonus": "SEA_NAV_DWA_WAIT_BONUS",
        "hybrid_safe_distance": "SEA_NAV_HYBRID_SAFE_DISTANCE",
        "hybrid_critical_distance": "SEA_NAV_HYBRID_CRITICAL_DISTANCE",
        "hybrid_ttc_horizon": "SEA_NAV_HYBRID_TTC_HORIZON",
        "hybrid_stop_ttc": "SEA_NAV_HYBRID_STOP_TTC",
        "hybrid_slow_ttc": "SEA_NAV_HYBRID_SLOW_TTC",
        "hybrid_front_ray_clearance": "SEA_NAV_HYBRID_FRONT_RAY_CLEARANCE",
        "hybrid_max_lateral_speed": "SEA_NAV_HYBRID_MAX_LATERAL_SPEED",
        "hybrid_escape_yaw_rate": "SEA_NAV_HYBRID_ESCAPE_YAW_RATE",
        "hybrid_policy_weight": "SEA_NAV_HYBRID_POLICY_WEIGHT",
        "hybrid_progress_weight": "SEA_NAV_HYBRID_PROGRESS_WEIGHT",
        "hybrid_ttc_weight": "SEA_NAV_HYBRID_TTC_WEIGHT",
        "hybrid_clearance_weight": "SEA_NAV_HYBRID_CLEARANCE_WEIGHT",
        "hybrid_static_weight": "SEA_NAV_HYBRID_STATIC_WEIGHT",
        "hybrid_wait_bonus": "SEA_NAV_HYBRID_WAIT_BONUS",
        "hybrid_smoothness_weight": "SEA_NAV_HYBRID_SMOOTHNESS_WEIGHT",
    }
    for cfg_key, env_key in pipeline_mapping.items():
        if cfg_key in pipeline_cfg:
            env[env_key] = str(pipeline_cfg[cfg_key])


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
    return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", str(script)]


def _command(cfg: Mapping[str, Any], command: str, checkpoint: str | None) -> tuple[list[str], Path, dict[str, str]]:
    sea_nav_path, train_script, play_script = _base_paths(cfg)
    training_cfg = _section(cfg, "training")
    evaluation_cfg = _section(cfg, "evaluation")
    recording_cfg = _section(cfg, "recording")

    task_name = str(training_cfg.get("task_name", "go2_pos_rough"))
    experiment_name = str(training_cfg.get("experiment_name", "Go2_pos_rough"))
    env = os.environ.copy()
    wandb_enabled = env.get("SEA_NAV_ENABLE_WANDB") == "1"
    if wandb_enabled:
        env.pop("WANDB_DISABLED", None)
        if env.get("WANDB_MODE", "").lower() == "offline":
            env.pop("WANDB_MODE", None)
    else:
        env.setdefault("WANDB_MODE", "offline")
        env.setdefault("WANDB_DISABLED", "true")

    if command == "train":
        _apply_dynamic_obstacle_env(cfg, env)
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
        if _bool(training_cfg.get("reset_optimizer"), False):
            env["SEA_NAV_RESET_OPTIMIZER"] = "1"
        load_run, checkpoint_number = _checkpoint_to_load_args(checkpoint)
        if load_run and checkpoint_number:
            cmd.extend(["--resume", "--load_run", load_run, "--checkpoint", checkpoint_number])
    elif command in {"eval", "play", "record"}:
        _apply_dynamic_obstacle_env(cfg, env)
        cfg_for_mode = recording_cfg if command == "record" else evaluation_cfg
        mode_experiment_name = str(cfg_for_mode.get("experiment_name", experiment_name))
        cmd = _python_command(cfg, play_script)
        cmd.extend(
            [
                "--task",
                task_name,
                "--experiment_name",
                mode_experiment_name,
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
    sea_nav_path, _, _ = _base_paths(cfg)
    training_cfg = _section(cfg, "training")
    outputs_cfg = _section(cfg, "outputs")
    recording_cfg = _section(cfg, "recording")
    experiment_name = str(recording_cfg.get("experiment_name", training_cfg.get("experiment_name", "Go2_pos_rough")))
    if checkpoint:
        load_run, checkpoint_number = _checkpoint_to_load_args(checkpoint)
        if not load_run or not checkpoint_number:
            return None
        video_stem = f"{load_run}_{checkpoint_number}"
        dest_name = f"sea_nav_demo_model_{checkpoint_number}_{int(recording_cfg.get('video_length', 2000))}steps.mp4"
    else:
        pipeline_cfg = _section(cfg, "pipeline")
        if not pipeline_cfg:
            return None
        video_stem = "dwa_pipeline_n1"
        dest_name = f"sea_nav_demo_pipeline_{int(recording_cfg.get('video_length', 2000))}steps.mp4"
    source = (
        sea_nav_path
        / "training"
        / "legged_gym"
        / "logs"
        / experiment_name
        / "exported"
        / f"{video_stem}.mp4"
    )
    if not source.exists():
        return None
    video_dir = _expand_path(str(outputs_cfg.get("video_dir", "outputs/videos/navigation/sea_nav_baseline")))
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / dest_name
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
