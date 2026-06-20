# SPDX-License-Identifier: BSD-3-Clause

import os

from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg, Go2PosRoughCfgPPO


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_str(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_list(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_float_list(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _env_int_list(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


class Go2DynamicObstaclesCfg(Go2PosRoughCfg):
    class env(Go2PosRoughCfg.env):
        debug_viz = True
        episode_length_s = _env_float("SEA_NAV_ENV_EPISODE_LENGTH_S", Go2PosRoughCfg.env.episode_length_s)
        include_dynamic_obstacle_state = _env_bool("SEA_NAV_DYNAMIC_OBS_STATE", False)
        dynamic_obstacle_state_k = _env_int("SEA_NAV_DYNAMIC_OBS_K", 3)
        dynamic_obstacle_global_dim = _env_int("SEA_NAV_DYNAMIC_OBS_GLOBAL_DIM", 2)
        num_dynamic_obstacle_obs = (
            dynamic_obstacle_state_k * 6 + dynamic_obstacle_global_dim if include_dynamic_obstacle_state else 0
        )
        num_obs_one_step = (
            Go2PosRoughCfg.env.num_props
            + Go2PosRoughCfg.env.num_rays
            + Go2PosRoughCfg.env.num_goal_obs
            + num_dynamic_obstacle_obs
        )
        num_observations = num_obs_one_step * Go2PosRoughCfg.env.his_len

    class terrain(Go2PosRoughCfg.terrain):
        terrain_types = ["hard_room"]
        terrain_proportions = [1.0]

    class replay(Go2PosRoughCfg.replay):
        enable_collision_replay = _env_bool(
            "SEA_NAV_REPLAY_ENABLE_COLLISION_REPLAY",
            Go2PosRoughCfg.replay.enable_collision_replay,
        )
        replay_prob = _env_float("SEA_NAV_REPLAY_PROB", Go2PosRoughCfg.replay.replay_prob)
        early_reset_prob_range = _env_float_list(
            "SEA_NAV_REPLAY_EARLY_RESET_PROB_RANGE",
            Go2PosRoughCfg.replay.early_reset_prob_range,
        )
        undo_steps_range = _env_int_list(
            "SEA_NAV_REPLAY_UNDO_STEPS_RANGE",
            Go2PosRoughCfg.replay.undo_steps_range,
        )
        max_collision_points = _env_int(
            "SEA_NAV_REPLAY_MAX_COLLISION_POINTS",
            Go2PosRoughCfg.replay.max_collision_points,
        )

    class dynamic_obstacles:
        enable = _env_bool("SEA_NAV_DYNAMIC_ENABLE", True)
        use_legacy_reward = _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False)
        num_obstacles = _env_int("SEA_NAV_DYNAMIC_NUM_OBSTACLES", 3)
        radius = _env_float("SEA_NAV_DYNAMIC_OBSTACLE_RADIUS", 0.35)
        speed = _env_float("SEA_NAV_DYNAMIC_OBSTACLE_SPEED", 0.45)
        min_dist = 0.1
        max_dist = 3.0
        collision_distance = _env_float("SEA_NAV_DYNAMIC_COLLISION_DISTANCE", 0.55)
        motion_modes = _env_list(
            "SEA_NAV_DYNAMIC_MOTION_MODES",
            ["pedestrian_like", "back_and_forth", "random_rigid_body"],
        )
        amplitude_scale = _env_float("SEA_NAV_DYNAMIC_AMPLITUDE_SCALE", 0.55)
        amplitude_min = _env_float("SEA_NAV_DYNAMIC_AMPLITUDE_MIN", 0.35)
        amplitude_max = _env_float("SEA_NAV_DYNAMIC_AMPLITUDE_MAX", 1.35)
        phase_randomization = _env_bool("SEA_NAV_DYNAMIC_PHASE_RANDOMIZATION", True)
        focus_near_robot = _env_bool("SEA_NAV_DYNAMIC_FOCUS_NEAR_ROBOT", False)
        focus_all_obstacles = _env_bool("SEA_NAV_DYNAMIC_FOCUS_ALL_OBSTACLES", False)
        focus_distance_min = _env_float("SEA_NAV_DYNAMIC_FOCUS_DISTANCE_MIN", 0.90)
        focus_distance_max = _env_float("SEA_NAV_DYNAMIC_FOCUS_DISTANCE_MAX", 1.40)
        focus_lateral_min = _env_float("SEA_NAV_DYNAMIC_FOCUS_LATERAL_MIN", 0.25)
        focus_lateral_max = _env_float("SEA_NAV_DYNAMIC_FOCUS_LATERAL_MAX", 0.75)
        focus_phase_min = _env_float("SEA_NAV_DYNAMIC_FOCUS_PHASE_MIN", 0.35)
        focus_phase_max = _env_float("SEA_NAV_DYNAMIC_FOCUS_PHASE_MAX", 0.85)
        delayed_static_activation_min = _env_float("SEA_NAV_DYNAMIC_DELAYED_STATIC_ACTIVATION_MIN", 0.45)
        delayed_static_activation_max = _env_float("SEA_NAV_DYNAMIC_DELAYED_STATIC_ACTIVATION_MAX", 1.10)
        delayed_static_distance_min = _env_float("SEA_NAV_DYNAMIC_DELAYED_STATIC_DISTANCE_MIN", focus_distance_min)
        delayed_static_distance_max = _env_float("SEA_NAV_DYNAMIC_DELAYED_STATIC_DISTANCE_MAX", focus_distance_max)
        delayed_static_lateral_min = _env_float("SEA_NAV_DYNAMIC_DELAYED_STATIC_LATERAL_MIN", focus_lateral_min)
        delayed_static_lateral_max = _env_float("SEA_NAV_DYNAMIC_DELAYED_STATIC_LATERAL_MAX", focus_lateral_max)
        training_interaction_scene = _env_bool("SEA_NAV_DYNAMIC_TRAINING_INTERACTION_SCENE", False)
        training_interaction_robot_jitter = _env_float("SEA_NAV_DYNAMIC_TRAINING_ROBOT_JITTER", 0.0)
        training_interaction_goal_jitter = _env_float("SEA_NAV_DYNAMIC_TRAINING_GOAL_JITTER", 0.0)
        training_interaction_obstacle_frac_jitter = _env_float("SEA_NAV_DYNAMIC_TRAINING_OBSTACLE_FRAC_JITTER", 0.0)
        training_interaction_obstacle_lateral_jitter = _env_float("SEA_NAV_DYNAMIC_TRAINING_OBSTACLE_LATERAL_JITTER", 0.0)
        demo_interaction_scene = _env_bool("SEA_NAV_DYNAMIC_DEMO_INTERACTION_SCENE", False)
        demo_interaction_face_goal = _env_bool("SEA_NAV_DYNAMIC_DEMO_FACE_GOAL", True)
        demo_terrain_level = _env_int("SEA_NAV_DYNAMIC_DEMO_TERRAIN_LEVEL", 1)
        demo_terrain_type = _env_int("SEA_NAV_DYNAMIC_DEMO_TERRAIN_TYPE", 0)
        demo_goal_distance_min = _env_float("SEA_NAV_DYNAMIC_DEMO_GOAL_DISTANCE_MIN", 3.5)
        demo_robot_frac_x = _env_float("SEA_NAV_DYNAMIC_DEMO_ROBOT_FRAC_X", 0.52)
        demo_robot_frac_y = _env_float("SEA_NAV_DYNAMIC_DEMO_ROBOT_FRAC_Y", 0.42)
        demo_goal_frac_x = _env_float("SEA_NAV_DYNAMIC_DEMO_GOAL_FRAC_X", 0.52)
        demo_goal_frac_y = _env_float("SEA_NAV_DYNAMIC_DEMO_GOAL_FRAC_Y", 0.78)
        demo_force_path_obstacles = _env_bool("SEA_NAV_DYNAMIC_DEMO_FORCE_PATH_OBSTACLES", False)
        demo_path_lateral_spacing = _env_float("SEA_NAV_DYNAMIC_DEMO_PATH_LATERAL_SPACING", 0.12)
        demo_path_obstacle_frac_start = _env_float("SEA_NAV_DYNAMIC_DEMO_PATH_OBSTACLE_FRAC_START", 0.30)
        demo_path_obstacle_frac_step = _env_float("SEA_NAV_DYNAMIC_DEMO_PATH_OBSTACLE_FRAC_STEP", 0.18)
        demo_path_obstacle_fracs = _env_float_list("SEA_NAV_DYNAMIC_DEMO_PATH_OBSTACLE_FRACS", [])
        demo_path_lateral_offsets = _env_float_list("SEA_NAV_DYNAMIC_DEMO_PATH_LATERAL_OFFSETS", [])
        demo_motion_range_scale = _env_float("SEA_NAV_DYNAMIC_DEMO_MOTION_RANGE_SCALE", 1.0)
        demo_pedestrian_range_scale = _env_float("SEA_NAV_DYNAMIC_DEMO_PEDESTRIAN_RANGE_SCALE", demo_motion_range_scale)
        demo_backtrack_range_scale = _env_float("SEA_NAV_DYNAMIC_DEMO_BACKTRACK_RANGE_SCALE", demo_motion_range_scale)
        demo_random_range_scale = _env_float("SEA_NAV_DYNAMIC_DEMO_RANDOM_RANGE_SCALE", demo_motion_range_scale)
        demo_motion_range_max = _env_float("SEA_NAV_DYNAMIC_DEMO_MOTION_RANGE_MAX", amplitude_max)
        ttc_horizon = _env_float("SEA_NAV_DYNAMIC_TTC_HORIZON", 3.0)
        path_block_distance = _env_float("SEA_NAV_DYNAMIC_PATH_BLOCK_DISTANCE", 2.5)
        path_block_width = _env_float("SEA_NAV_DYNAMIC_PATH_BLOCK_WIDTH", 0.75)
        path_block_ttc = _env_float("SEA_NAV_DYNAMIC_PATH_BLOCK_TTC", 2.5)
        path_block_rise = _env_float("SEA_NAV_DYNAMIC_PATH_BLOCK_RISE", 0.20)
        path_block_fall = _env_float("SEA_NAV_DYNAMIC_PATH_BLOCK_FALL", 0.08)
        path_block_threshold = _env_float("SEA_NAV_DYNAMIC_PATH_BLOCK_THRESHOLD", 0.45)
        preferred_speed = _env_float("SEA_NAV_DYNAMIC_PREFERRED_SPEED", 0.5)
        wait_speed = _env_float("SEA_NAV_DYNAMIC_WAIT_SPEED", 0.08)
        ttc_safe_distance = _env_float("SEA_NAV_DYNAMIC_TTC_SAFE_DISTANCE", collision_distance)
        blocked_timeout = _env_float("SEA_NAV_DYNAMIC_BLOCKED_TIMEOUT", 3.0)
        mode_amplitude_scale = {
            "pedestrian_like": 1.10,
            "back_and_forth": 1.15,
            "random_rigid_body": 1.60,
        }

    class rewards(Go2PosRoughCfg.rewards):
        class scales(Go2PosRoughCfg.rewards.scales):
            termination = _env_float("SEA_NAV_REWARD_TERMINATION", Go2PosRoughCfg.rewards.scales.termination)
            collision = _env_float("SEA_NAV_REWARD_COLLISION", Go2PosRoughCfg.rewards.scales.collision)
            close_obst_vel = _env_float("SEA_NAV_REWARD_CLOSE_OBST_VEL", Go2PosRoughCfg.rewards.scales.close_obst_vel)
            stuck = _env_float("SEA_NAV_REWARD_STUCK", Go2PosRoughCfg.rewards.scales.stuck)
            velo_dir = _env_float("SEA_NAV_REWARD_VELO_DIR", Go2PosRoughCfg.rewards.scales.velo_dir)
            reach_pos_target_tight = _env_float(
                "SEA_NAV_REWARD_REACH_POS_TARGET_TIGHT",
                Go2PosRoughCfg.rewards.scales.reach_pos_target_tight,
            )
            ang_vel_xy = _env_float("SEA_NAV_REWARD_ANG_VEL_XY", Go2PosRoughCfg.rewards.scales.ang_vel_xy)
            proximity = _env_float("SEA_NAV_REWARD_PROXIMITY", Go2PosRoughCfg.rewards.scales.proximity)
            progress = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_PROGRESS", 5.0)
            preferred_velocity = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_PREFERRED_VELOCITY", 2.0)
            )
            dynamic_ttc = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_DYNAMIC_TTC", -4.0)
            dynamic_clearance = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_DYNAMIC_CLEARANCE", -2.0)
            )
            wait = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_WAIT", 2.0)
            blocked_fast_penalty = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_BLOCKED_FAST_PENALTY", -1.5)
            )
            detour = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_DETOUR", -0.8)
            near_goal_stop = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_NEAR_GOAL_STOP", 2.0)
            )
            dynamic_collision = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_DYNAMIC_COLLISION", -60.0)
            )
            successful_avoidance = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_SUCCESSFUL_AVOIDANCE", 0.0)
            )
            avoidance_clearance = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_AVOIDANCE_CLEARANCE", 0.0)
            )
            risk_reduction = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_RISK_REDUCTION", 0.0)
            )
            free_space_action = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_FREE_SPACE_ACTION", 0.0)
            )
            unsafe_ttc = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_UNSAFE_TTC", 0.0)
            nav_action_smoothness = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_NAV_ACTION_SMOOTHNESS", 0.0)
            )
            static_collision = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_STATIC_COLLISION", 0.0)
            )
            escape_direction = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_ESCAPE_DIRECTION", 0.0)
            )
            threat_direction_penalty = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_THREAT_DIRECTION_PENALTY", 0.0)
            )
            stable_velocity = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_STABLE_VELOCITY", 0.0)
            )
            resume_ready = (
                0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_RESUME_READY", 0.0)
            )
            rebot_distance = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_DISTANCE", 0.0)
            rebot_collision = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_COLLISION", 0.0)
            rebot_walk = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_WALK", 0.0)
            rebot_energy = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_ENERGY", 0.0)
            rebot_contact = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_CONTACT", 0.0)
            rebot_diversity = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_DIVERSITY", 0.0)
            rebot_threat = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_THREAT", 0.0)
            rebot_direction = 0.0 if _env_bool("SEA_NAV_DYNAMIC_LEGACY_REWARD", False) else _env_float("SEA_NAV_REWARD_REBOT_DIRECTION", 0.0)

        class progress_config:
            clip = 0.25
            blocked_scale = _env_float("SEA_NAV_REWARD_PROGRESS_BLOCKED_SCALE", 0.10)
            near_goal_scale = _env_float("SEA_NAV_REWARD_PROGRESS_NEAR_GOAL_SCALE", 0.20)
            near_goal_distance = _env_float("SEA_NAV_REWARD_NEAR_GOAL_DISTANCE", 0.60)

        class preferred_velocity_config:
            target_speed = _env_float("SEA_NAV_DYNAMIC_PREFERRED_SPEED", 0.5)
            wait_speed = _env_float("SEA_NAV_DYNAMIC_WAIT_SPEED", 0.08)
            near_goal_speed = _env_float("SEA_NAV_REWARD_NEAR_GOAL_SPEED", 0.03)
            near_goal_distance = _env_float("SEA_NAV_REWARD_NEAR_GOAL_DISTANCE", 0.60)
            sigma = _env_float("SEA_NAV_REWARD_PREFERRED_SIGMA", 0.45)

        class dynamic_ttc_config:
            horizon = _env_float("SEA_NAV_DYNAMIC_TTC_HORIZON", 3.0)
            min_closing_speed = _env_float("SEA_NAV_REWARD_TTC_MIN_CLOSING_SPEED", 0.03)

        class dynamic_clearance_config:
            safe_distance = _env_float("SEA_NAV_REWARD_CLEARANCE_SAFE_DISTANCE", 1.05)
            front_margin = _env_float("SEA_NAV_REWARD_CLEARANCE_FRONT_MARGIN", -0.15)

        class wait_config:
            max_speed = _env_float("SEA_NAV_REWARD_WAIT_MAX_SPEED", 0.12)
            heading_weight = _env_float("SEA_NAV_REWARD_WAIT_HEADING_WEIGHT", 1.0)
            timeout = _env_float("SEA_NAV_DYNAMIC_BLOCKED_TIMEOUT", 3.0)
            timeout_decay = _env_float("SEA_NAV_REWARD_WAIT_TIMEOUT_DECAY", 0.35)

        class blocked_fast_penalty_config:
            speed_threshold = _env_float("SEA_NAV_REWARD_BLOCKED_FAST_SPEED_THRESHOLD", 0.18)
            ttc_threshold = _env_float("SEA_NAV_REWARD_BLOCKED_FAST_TTC_THRESHOLD", 2.0)
            closing_speed_threshold = _env_float("SEA_NAV_REWARD_BLOCKED_FAST_CLOSING_SPEED_THRESHOLD", 0.05)

        class detour_config:
            lateral_weight = _env_float("SEA_NAV_REWARD_DETOUR_LATERAL_WEIGHT", 1.0)
            yaw_weight = _env_float("SEA_NAV_REWARD_DETOUR_YAW_WEIGHT", 0.5)
            blocked_scale = _env_float("SEA_NAV_REWARD_DETOUR_BLOCKED_SCALE", 0.05)
            timeout_scale = _env_float("SEA_NAV_REWARD_DETOUR_TIMEOUT_SCALE", 0.35)

        class near_goal_stop_config:
            distance = _env_float("SEA_NAV_REWARD_NEAR_GOAL_DISTANCE", 0.60)
            max_speed = _env_float("SEA_NAV_REWARD_NEAR_GOAL_MAX_SPEED", 0.10)
            max_yaw_rate = _env_float("SEA_NAV_REWARD_NEAR_GOAL_MAX_YAW_RATE", 0.35)

        class avoidance_stage1_config:
            high_risk_distance = _env_float("SEA_NAV_REWARD_AVOID_HIGH_RISK_DISTANCE", 1.05)
            low_risk_distance = _env_float("SEA_NAV_REWARD_AVOID_LOW_RISK_DISTANCE", 1.35)
            high_risk_ttc = _env_float("SEA_NAV_REWARD_AVOID_HIGH_RISK_TTC", 1.30)
            low_risk_ttc = _env_float("SEA_NAV_REWARD_AVOID_LOW_RISK_TTC", 2.50)
            min_static_clearance = _env_float("SEA_NAV_REWARD_AVOID_MIN_STATIC_CLEARANCE", 0.45)

        class avoidance_clearance_config:
            safe_distance = _env_float("SEA_NAV_REWARD_AVOIDANCE_CLEARANCE_SAFE_DISTANCE", 1.10)
            collision_distance = _env_float("SEA_NAV_REWARD_AVOIDANCE_CLEARANCE_COLLISION_DISTANCE", 0.55)
            min_static_clearance = _env_float("SEA_NAV_REWARD_AVOIDANCE_CLEARANCE_MIN_STATIC_CLEARANCE", 0.35)

        class rebot_distance_config:
            sigma = _env_float("SEA_NAV_REWARD_REBOT_DISTANCE_SIGMA", 0.35)
            collision_distance = _env_float("SEA_NAV_REWARD_REBOT_DISTANCE_COLLISION_DISTANCE", 0.55)

        class rebot_walk_config:
            contact_threshold = _env_float("SEA_NAV_REWARD_REBOT_WALK_CONTACT_THRESHOLD", 1.0)

        class rebot_energy_config:
            scale = _env_float("SEA_NAV_REWARD_REBOT_ENERGY_SCALE", 50.0)

        class rebot_contact_config:
            force_scale = _env_float("SEA_NAV_REWARD_REBOT_CONTACT_FORCE_SCALE", 100.0)

        class rebot_threat_config:
            lambda_speed = _env_float("SEA_NAV_REWARD_REBOT_THREAT_LAMBDA", 0.45)
            eta = _env_float("SEA_NAV_REWARD_REBOT_THREAT_ETA", 1.1)
            sigma = _env_float("SEA_NAV_REWARD_REBOT_THREAT_SIGMA", 0.45)

        class risk_reduction_config:
            distance_weight = _env_float("SEA_NAV_REWARD_RISK_REDUCTION_DISTANCE_WEIGHT", 0.55)
            ttc_weight = _env_float("SEA_NAV_REWARD_RISK_REDUCTION_TTC_WEIGHT", 0.45)

        class free_space_action_config:
            safe_distance = _env_float("SEA_NAV_REWARD_FREE_SPACE_SAFE_DISTANCE", 0.65)
            speed_threshold = _env_float("SEA_NAV_REWARD_FREE_SPACE_SPEED_THRESHOLD", 0.08)

        class unsafe_ttc_config:
            threshold = _env_float("SEA_NAV_REWARD_UNSAFE_TTC_THRESHOLD", 1.0)

        class nav_action_smoothness_config:
            sigma = _env_float("SEA_NAV_REWARD_NAV_ACTION_SMOOTHNESS_SIGMA", 0.50)
            min_phase_weight = _env_float("SEA_NAV_REWARD_NAV_ACTION_SMOOTHNESS_MIN_PHASE_WEIGHT", 0.20)

        class escape_direction_config:
            high_risk_only = _env_bool("SEA_NAV_REWARD_ESCAPE_HIGH_RISK_ONLY", True)
            speed_sigma = _env_float("SEA_NAV_REWARD_ESCAPE_SPEED_SIGMA", 0.45)

        class threat_direction_penalty_config:
            high_risk_only = _env_bool("SEA_NAV_REWARD_THREAT_HIGH_RISK_ONLY", True)
            speed_sigma = _env_float("SEA_NAV_REWARD_THREAT_SPEED_SIGMA", 0.45)

        class stable_velocity_config:
            max_speed = _env_float("SEA_NAV_REWARD_STABLE_MAX_SPEED", 0.18)
            max_yaw_rate = _env_float("SEA_NAV_REWARD_STABLE_MAX_YAW_RATE", 0.35)
            low_risk_only = _env_bool("SEA_NAV_REWARD_STABLE_LOW_RISK_ONLY", True)

        class resume_ready_config:
            min_dynamic_distance = _env_float("SEA_NAV_REWARD_RESUME_MIN_DYNAMIC_DISTANCE", 1.35)
            min_ttc = _env_float("SEA_NAV_REWARD_RESUME_MIN_TTC", 2.50)
            min_static_clearance = _env_float("SEA_NAV_REWARD_RESUME_MIN_STATIC_CLEARANCE", 0.55)
            max_speed = _env_float("SEA_NAV_REWARD_RESUME_MAX_SPEED", 0.22)
            max_yaw_rate = _env_float("SEA_NAV_REWARD_RESUME_MAX_YAW_RATE", 0.40)
            heading_weight = _env_float("SEA_NAV_REWARD_RESUME_HEADING_WEIGHT", 0.35)

    class visualization(Go2PosRoughCfg.visualization):
        draw_dynamic_obstacles = True
        points = dict(Go2PosRoughCfg.visualization.points)
        points.update(
            {
                "dynamic_obstacle_marker": [0.18, 12, (1.0, 0.45, 0.0)],
                "dynamic_collision_marker": [0.12, 8, (1.0, 0.0, 0.0)],
            }
        )


class Go2DynamicObstaclesCfgPPO(Go2PosRoughCfgPPO):
    class policy(Go2PosRoughCfgPPO.policy):
        transformer_dim = _env_int("SEA_NAV_POLICY_TRANSFORMER_DIM", 64)
        transformer_heads = _env_int("SEA_NAV_POLICY_TRANSFORMER_HEADS", 4)
        transformer_layers = _env_int("SEA_NAV_POLICY_TRANSFORMER_LAYERS", 2)
        transformer_dropout = _env_float("SEA_NAV_POLICY_TRANSFORMER_DROPOUT", 0.0)
        dynamic_token_dim = _env_int("SEA_NAV_POLICY_DYNAMIC_TOKEN_DIM", 6)
        dynamic_global_dim = _env_int("SEA_NAV_POLICY_DYNAMIC_GLOBAL_DIM", 2)

    class runner(Go2PosRoughCfgPPO.runner):
        experiment_name = "Go2_dynamic_obstacles"
        max_iterations = 20
        num_steps_per_env = _env_int("SEA_NAV_RUNNER_NUM_STEPS_PER_ENV", Go2PosRoughCfgPPO.runner.num_steps_per_env)
        log_interval = _env_int("SEA_NAV_RUNNER_LOG_INTERVAL", 10)
        policy_class_name = _env_str("SEA_NAV_POLICY_CLASS_NAME", Go2PosRoughCfgPPO.runner.policy_class_name)
