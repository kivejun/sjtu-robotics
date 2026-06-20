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
    training_cfg = _section(cfg, "training")
    if "policy_class_name" in training_cfg:
        env["SEA_NAV_POLICY_CLASS_NAME"] = str(training_cfg["policy_class_name"])

    policy_cfg = _section(cfg, "policy")
    policy_mapping = {
        "transformer_dim": "SEA_NAV_POLICY_TRANSFORMER_DIM",
        "transformer_heads": "SEA_NAV_POLICY_TRANSFORMER_HEADS",
        "transformer_layers": "SEA_NAV_POLICY_TRANSFORMER_LAYERS",
        "transformer_dropout": "SEA_NAV_POLICY_TRANSFORMER_DROPOUT",
        "dynamic_token_dim": "SEA_NAV_POLICY_DYNAMIC_TOKEN_DIM",
        "dynamic_global_dim": "SEA_NAV_POLICY_DYNAMIC_GLOBAL_DIM",
    }
    for cfg_key, env_key in policy_mapping.items():
        if cfg_key in policy_cfg:
            env[env_key] = str(policy_cfg[cfg_key])

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
        "phase_randomization": "SEA_NAV_DYNAMIC_PHASE_RANDOMIZATION",
        "focus_near_robot": "SEA_NAV_DYNAMIC_FOCUS_NEAR_ROBOT",
        "focus_all_obstacles": "SEA_NAV_DYNAMIC_FOCUS_ALL_OBSTACLES",
        "focus_distance_min": "SEA_NAV_DYNAMIC_FOCUS_DISTANCE_MIN",
        "focus_distance_max": "SEA_NAV_DYNAMIC_FOCUS_DISTANCE_MAX",
        "focus_lateral_min": "SEA_NAV_DYNAMIC_FOCUS_LATERAL_MIN",
        "focus_lateral_max": "SEA_NAV_DYNAMIC_FOCUS_LATERAL_MAX",
        "focus_phase_min": "SEA_NAV_DYNAMIC_FOCUS_PHASE_MIN",
        "focus_phase_max": "SEA_NAV_DYNAMIC_FOCUS_PHASE_MAX",
        "delayed_static_activation_min": "SEA_NAV_DYNAMIC_DELAYED_STATIC_ACTIVATION_MIN",
        "delayed_static_activation_max": "SEA_NAV_DYNAMIC_DELAYED_STATIC_ACTIVATION_MAX",
        "delayed_static_distance_min": "SEA_NAV_DYNAMIC_DELAYED_STATIC_DISTANCE_MIN",
        "delayed_static_distance_max": "SEA_NAV_DYNAMIC_DELAYED_STATIC_DISTANCE_MAX",
        "delayed_static_lateral_min": "SEA_NAV_DYNAMIC_DELAYED_STATIC_LATERAL_MIN",
        "delayed_static_lateral_max": "SEA_NAV_DYNAMIC_DELAYED_STATIC_LATERAL_MAX",
        "training_interaction_scene": "SEA_NAV_DYNAMIC_TRAINING_INTERACTION_SCENE",
        "training_interaction_robot_jitter": "SEA_NAV_DYNAMIC_TRAINING_ROBOT_JITTER",
        "training_interaction_goal_jitter": "SEA_NAV_DYNAMIC_TRAINING_GOAL_JITTER",
        "training_interaction_obstacle_frac_jitter": "SEA_NAV_DYNAMIC_TRAINING_OBSTACLE_FRAC_JITTER",
        "training_interaction_obstacle_lateral_jitter": "SEA_NAV_DYNAMIC_TRAINING_OBSTACLE_LATERAL_JITTER",
        "demo_interaction_scene": "SEA_NAV_DYNAMIC_DEMO_INTERACTION_SCENE",
        "demo_interaction_face_goal": "SEA_NAV_DYNAMIC_DEMO_FACE_GOAL",
        "demo_terrain_level": "SEA_NAV_DYNAMIC_DEMO_TERRAIN_LEVEL",
        "demo_terrain_type": "SEA_NAV_DYNAMIC_DEMO_TERRAIN_TYPE",
        "demo_goal_distance_min": "SEA_NAV_DYNAMIC_DEMO_GOAL_DISTANCE_MIN",
        "demo_robot_frac_x": "SEA_NAV_DYNAMIC_DEMO_ROBOT_FRAC_X",
        "demo_robot_frac_y": "SEA_NAV_DYNAMIC_DEMO_ROBOT_FRAC_Y",
        "demo_goal_frac_x": "SEA_NAV_DYNAMIC_DEMO_GOAL_FRAC_X",
        "demo_goal_frac_y": "SEA_NAV_DYNAMIC_DEMO_GOAL_FRAC_Y",
        "demo_force_path_obstacles": "SEA_NAV_DYNAMIC_DEMO_FORCE_PATH_OBSTACLES",
        "demo_path_lateral_spacing": "SEA_NAV_DYNAMIC_DEMO_PATH_LATERAL_SPACING",
        "demo_path_obstacle_frac_start": "SEA_NAV_DYNAMIC_DEMO_PATH_OBSTACLE_FRAC_START",
        "demo_path_obstacle_frac_step": "SEA_NAV_DYNAMIC_DEMO_PATH_OBSTACLE_FRAC_STEP",
        "demo_path_obstacle_fracs": "SEA_NAV_DYNAMIC_DEMO_PATH_OBSTACLE_FRACS",
        "demo_path_lateral_offsets": "SEA_NAV_DYNAMIC_DEMO_PATH_LATERAL_OFFSETS",
        "demo_motion_range_scale": "SEA_NAV_DYNAMIC_DEMO_MOTION_RANGE_SCALE",
        "demo_pedestrian_range_scale": "SEA_NAV_DYNAMIC_DEMO_PEDESTRIAN_RANGE_SCALE",
        "demo_backtrack_range_scale": "SEA_NAV_DYNAMIC_DEMO_BACKTRACK_RANGE_SCALE",
        "demo_random_range_scale": "SEA_NAV_DYNAMIC_DEMO_RANDOM_RANGE_SCALE",
        "demo_motion_range_max": "SEA_NAV_DYNAMIC_DEMO_MOTION_RANGE_MAX",
        "dynamic_obstacle_state_k": "SEA_NAV_DYNAMIC_OBS_K",
        "dynamic_obstacle_global_dim": "SEA_NAV_DYNAMIC_OBS_GLOBAL_DIM",
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
        "episode_length_s": "SEA_NAV_ENV_EPISODE_LENGTH_S",
    }
    for cfg_key, env_key in mapping.items():
        if cfg_key in environment_cfg:
            value = environment_cfg[cfg_key]
            if isinstance(value, list):
                env[env_key] = ",".join(str(item) for item in value)
            else:
                env[env_key] = str(value)

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
        "termination": "SEA_NAV_REWARD_TERMINATION",
        "collision": "SEA_NAV_REWARD_COLLISION",
        "close_obst_vel": "SEA_NAV_REWARD_CLOSE_OBST_VEL",
        "stuck": "SEA_NAV_REWARD_STUCK",
        "velo_dir": "SEA_NAV_REWARD_VELO_DIR",
        "reach_pos_target_tight": "SEA_NAV_REWARD_REACH_POS_TARGET_TIGHT",
        "ang_vel_xy": "SEA_NAV_REWARD_ANG_VEL_XY",
        "proximity": "SEA_NAV_REWARD_PROXIMITY",
        "dynamic_ttc": "SEA_NAV_REWARD_DYNAMIC_TTC",
        "dynamic_clearance": "SEA_NAV_REWARD_DYNAMIC_CLEARANCE",
        "wait": "SEA_NAV_REWARD_WAIT",
        "blocked_fast_penalty": "SEA_NAV_REWARD_BLOCKED_FAST_PENALTY",
        "detour": "SEA_NAV_REWARD_DETOUR",
        "near_goal_stop": "SEA_NAV_REWARD_NEAR_GOAL_STOP",
        "dynamic_collision": "SEA_NAV_REWARD_DYNAMIC_COLLISION",
        "successful_avoidance": "SEA_NAV_REWARD_SUCCESSFUL_AVOIDANCE",
        "avoidance_clearance": "SEA_NAV_REWARD_AVOIDANCE_CLEARANCE",
        "risk_reduction": "SEA_NAV_REWARD_RISK_REDUCTION",
        "free_space_action": "SEA_NAV_REWARD_FREE_SPACE_ACTION",
        "unsafe_ttc": "SEA_NAV_REWARD_UNSAFE_TTC",
        "nav_action_smoothness": "SEA_NAV_REWARD_NAV_ACTION_SMOOTHNESS",
        "escape_direction": "SEA_NAV_REWARD_ESCAPE_DIRECTION",
        "threat_direction_penalty": "SEA_NAV_REWARD_THREAT_DIRECTION_PENALTY",
        "stable_velocity": "SEA_NAV_REWARD_STABLE_VELOCITY",
        "resume_ready": "SEA_NAV_REWARD_RESUME_READY",
        "rebot_distance": "SEA_NAV_REWARD_REBOT_DISTANCE",
        "rebot_collision": "SEA_NAV_REWARD_REBOT_COLLISION",
        "rebot_walk": "SEA_NAV_REWARD_REBOT_WALK",
        "rebot_energy": "SEA_NAV_REWARD_REBOT_ENERGY",
        "rebot_contact": "SEA_NAV_REWARD_REBOT_CONTACT",
        "rebot_diversity": "SEA_NAV_REWARD_REBOT_DIVERSITY",
        "rebot_threat": "SEA_NAV_REWARD_REBOT_THREAT",
        "rebot_direction": "SEA_NAV_REWARD_REBOT_DIRECTION",
        "static_collision": "SEA_NAV_REWARD_STATIC_COLLISION",
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
        "avoid_high_risk_distance": "SEA_NAV_REWARD_AVOID_HIGH_RISK_DISTANCE",
        "avoid_low_risk_distance": "SEA_NAV_REWARD_AVOID_LOW_RISK_DISTANCE",
        "avoid_high_risk_ttc": "SEA_NAV_REWARD_AVOID_HIGH_RISK_TTC",
        "avoid_low_risk_ttc": "SEA_NAV_REWARD_AVOID_LOW_RISK_TTC",
        "avoid_min_static_clearance": "SEA_NAV_REWARD_AVOID_MIN_STATIC_CLEARANCE",
        "avoidance_clearance_safe_distance": "SEA_NAV_REWARD_AVOIDANCE_CLEARANCE_SAFE_DISTANCE",
        "avoidance_clearance_collision_distance": "SEA_NAV_REWARD_AVOIDANCE_CLEARANCE_COLLISION_DISTANCE",
        "avoidance_clearance_min_static_clearance": "SEA_NAV_REWARD_AVOIDANCE_CLEARANCE_MIN_STATIC_CLEARANCE",
        "rebot_distance_sigma": "SEA_NAV_REWARD_REBOT_DISTANCE_SIGMA",
        "rebot_distance_collision_distance": "SEA_NAV_REWARD_REBOT_DISTANCE_COLLISION_DISTANCE",
        "rebot_walk_contact_threshold": "SEA_NAV_REWARD_REBOT_WALK_CONTACT_THRESHOLD",
        "rebot_energy_scale": "SEA_NAV_REWARD_REBOT_ENERGY_SCALE",
        "rebot_contact_force_scale": "SEA_NAV_REWARD_REBOT_CONTACT_FORCE_SCALE",
        "rebot_threat_lambda": "SEA_NAV_REWARD_REBOT_THREAT_LAMBDA",
        "rebot_threat_eta": "SEA_NAV_REWARD_REBOT_THREAT_ETA",
        "rebot_threat_sigma": "SEA_NAV_REWARD_REBOT_THREAT_SIGMA",
        "risk_reduction_distance_weight": "SEA_NAV_REWARD_RISK_REDUCTION_DISTANCE_WEIGHT",
        "risk_reduction_ttc_weight": "SEA_NAV_REWARD_RISK_REDUCTION_TTC_WEIGHT",
        "free_space_safe_distance": "SEA_NAV_REWARD_FREE_SPACE_SAFE_DISTANCE",
        "free_space_speed_threshold": "SEA_NAV_REWARD_FREE_SPACE_SPEED_THRESHOLD",
        "unsafe_ttc_threshold": "SEA_NAV_REWARD_UNSAFE_TTC_THRESHOLD",
        "nav_action_smoothness_sigma": "SEA_NAV_REWARD_NAV_ACTION_SMOOTHNESS_SIGMA",
        "nav_action_smoothness_min_phase_weight": "SEA_NAV_REWARD_NAV_ACTION_SMOOTHNESS_MIN_PHASE_WEIGHT",
        "escape_speed_sigma": "SEA_NAV_REWARD_ESCAPE_SPEED_SIGMA",
        "threat_speed_sigma": "SEA_NAV_REWARD_THREAT_SPEED_SIGMA",
        "stable_max_speed": "SEA_NAV_REWARD_STABLE_MAX_SPEED",
        "stable_max_yaw_rate": "SEA_NAV_REWARD_STABLE_MAX_YAW_RATE",
        "resume_min_dynamic_distance": "SEA_NAV_REWARD_RESUME_MIN_DYNAMIC_DISTANCE",
        "resume_min_ttc": "SEA_NAV_REWARD_RESUME_MIN_TTC",
        "resume_min_static_clearance": "SEA_NAV_REWARD_RESUME_MIN_STATIC_CLEARANCE",
        "resume_max_speed": "SEA_NAV_REWARD_RESUME_MAX_SPEED",
        "resume_max_yaw_rate": "SEA_NAV_REWARD_RESUME_MAX_YAW_RATE",
        "resume_heading_weight": "SEA_NAV_REWARD_RESUME_HEADING_WEIGHT",
    }
    for cfg_key, env_key in reward_mapping.items():
        if cfg_key in reward_cfg:
            env[env_key] = str(reward_cfg[cfg_key])

    replay_cfg = _section(cfg, "replay")
    replay_mapping = {
        "enable_collision_replay": "SEA_NAV_REPLAY_ENABLE_COLLISION_REPLAY",
        "replay_prob": "SEA_NAV_REPLAY_PROB",
        "early_reset_prob_range": "SEA_NAV_REPLAY_EARLY_RESET_PROB_RANGE",
        "undo_steps_range": "SEA_NAV_REPLAY_UNDO_STEPS_RANGE",
        "max_collision_points": "SEA_NAV_REPLAY_MAX_COLLISION_POINTS",
    }
    for cfg_key, env_key in replay_mapping.items():
        if cfg_key in replay_cfg:
            value = replay_cfg[cfg_key]
            if isinstance(value, list):
                env[env_key] = ",".join(str(item) for item in value)
            else:
                env[env_key] = str(value)

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
        "hybrid_dynamic_rollout_horizon": "SEA_NAV_HYBRID_DYNAMIC_ROLLOUT_HORIZON",
        "hybrid_crossing_window_horizon": "SEA_NAV_HYBRID_CROSSING_WINDOW_HORIZON",
        "hybrid_static_rollout_horizon": "SEA_NAV_HYBRID_STATIC_ROLLOUT_HORIZON",
        "hybrid_rollout_dt": "SEA_NAV_HYBRID_ROLLOUT_DT",
        "hybrid_simple_speed_filter": "SEA_NAV_HYBRID_SIMPLE_SPEED_FILTER",
        "hybrid_soft_scale": "SEA_NAV_HYBRID_SOFT_SCALE",
        "hybrid_crossing_commit_time": "SEA_NAV_HYBRID_CROSSING_COMMIT_TIME",
        "hybrid_max_wait_time": "SEA_NAV_HYBRID_MAX_WAIT_TIME",
        "hybrid_crossing_speed": "SEA_NAV_HYBRID_CROSSING_SPEED",
        "hybrid_robot_radius": "SEA_NAV_HYBRID_ROBOT_RADIUS",
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
        "emergency_policy_checkpoint": "SEA_NAV_EMERGENCY_POLICY_CHECKPOINT",
        "emergency_policy_blend": "SEA_NAV_EMERGENCY_POLICY_BLEND",
        "emergency_policy_trigger_distance": "SEA_NAV_EMERGENCY_POLICY_TRIGGER_DISTANCE",
        "emergency_policy_trigger_ttc": "SEA_NAV_EMERGENCY_POLICY_TRIGGER_TTC",
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
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
    return [conda_exe, "run", "--no-capture-output", "-n", conda_env, "python", str(script)]


def _command(cfg: Mapping[str, Any], command: str, checkpoint: str | None) -> tuple[list[str], Path, dict[str, str]]:
    sea_nav_path, train_script, play_script = _base_paths(cfg)
    training_cfg = _section(cfg, "training")
    evaluation_cfg = _section(cfg, "evaluation")
    recording_cfg = _section(cfg, "recording")

    task_name = str(training_cfg.get("task_name", "go2_pos_rough"))
    experiment_name = str(training_cfg.get("experiment_name", "Go2_pos_rough"))
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
        if "num_steps_per_env" in training_cfg:
            env["SEA_NAV_RUNNER_NUM_STEPS_PER_ENV"] = str(training_cfg["num_steps_per_env"])
        if "log_interval" in training_cfg:
            env["SEA_NAV_RUNNER_LOG_INTERVAL"] = str(training_cfg["log_interval"])
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
            if _bool(recording_cfg.get("segment_by_episode"), False):
                env["SEA_NAV_SEGMENT_VIDEO_BY_EPISODE"] = "1"
            if "camera_mode" in recording_cfg:
                env["SEA_NAV_RECORD_CAMERA_MODE"] = str(recording_cfg["camera_mode"])
            if "topdown_height" in recording_cfg:
                env["SEA_NAV_RECORD_TOPDOWN_HEIGHT"] = str(recording_cfg["topdown_height"])
            if "camera_fov" in recording_cfg:
                env["SEA_NAV_RECORD_FOV"] = str(recording_cfg["camera_fov"])
            if "width" in recording_cfg:
                env["SEA_NAV_RECORD_WIDTH"] = str(int(recording_cfg["width"]))
            if "height" in recording_cfg:
                env["SEA_NAV_RECORD_HEIGHT"] = str(int(recording_cfg["height"]))
            if _bool(recording_cfg.get("gui_style"), False):
                env["SEA_NAV_RECORD_GUI_STYLE"] = "1"
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
    video_dir = _expand_path(str(outputs_cfg.get("video_dir", "outputs/videos/navigation/sea_nav_baseline")))
    video_dir.mkdir(parents=True, exist_ok=True)
    if _bool(recording_cfg.get("segment_by_episode"), False):
        episode_source_dir = (
            sea_nav_path
            / "training"
            / "legged_gym"
            / "logs"
            / experiment_name
            / "exported"
            / "episodes"
        )
        episode_sources = sorted(episode_source_dir.glob(f"{video_stem}_episode_*.mp4"))
        if episode_sources:
            copied_dir = video_dir / f"sea_nav_demo_model_{checkpoint_number}_episodes"
            copied_dir.mkdir(parents=True, exist_ok=True)
            for episode_source in episode_sources:
                suffix = episode_source.stem.replace(f"{video_stem}_", "")
                shutil.copy2(episode_source, copied_dir / f"sea_nav_demo_model_{checkpoint_number}_{suffix}.mp4")
            return copied_dir
    if not source.exists():
        return None
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
