# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse
from legged_gym.utils.custom_terrain import is_far_from_obstacles
from scipy.signal import convolve2d

from legged_gym.envs.base.legged_robot_pos import LeggedRobotPos
from legged_gym.utils.torch_math import yaw_quat


class LeggedRobotDynamicPos(LeggedRobotPos):
    """SEA-Nav position task with analytical moving obstacles.

    The obstacles are not physical actors during training. They are maintained as
    tensor state, fused into the existing ray observation, and drawn for GUI/demo.
    """

    BACKTRACK_LATERAL_SCALE = 0.30
    RAND_AXIS_SCALE = 1.05
    RAND_PERP_SCALE = 0.72

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.dynamic_obstacles
        self.num_dynamic_obstacles = int(cfg.num_obstacles)
        self.dynamic_obstacle_radius = float(cfg.radius)
        self.dynamic_collision_distance = float(cfg.collision_distance)
        self.dynamic_obstacle_pos = torch.zeros(self.num_envs, self.num_dynamic_obstacles, 2, device=self.device)
        self.dynamic_obstacle_vel = torch.zeros_like(self.dynamic_obstacle_pos)
        self.dynamic_obstacle_base = torch.zeros_like(self.dynamic_obstacle_pos)
        self.dynamic_obstacle_axis = torch.zeros_like(self.dynamic_obstacle_pos)
        self.dynamic_obstacle_perp = torch.zeros_like(self.dynamic_obstacle_pos)
        self.dynamic_obstacle_amp = torch.ones(self.num_envs, self.num_dynamic_obstacles, device=self.device)
        self.dynamic_obstacle_phase = torch.zeros(self.num_envs, self.num_dynamic_obstacles, device=self.device)
        self.dynamic_obstacle_omega = torch.ones(self.num_envs, self.num_dynamic_obstacles, device=self.device)
        self.dynamic_collision_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dynamic_collision_step = torch.zeros(self.num_envs, device=self.device)
        self.min_dynamic_obstacle_distance = torch.ones(self.num_envs, device=self.device) * 100.0
        self.dynamic_collision_pos_hist = torch.zeros(self.num_envs, self.num_dynamic_obstacles, 3, device=self.device)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)
        self.distance_progress = torch.zeros(self.num_envs, device=self.device)
        self.dynamic_nearest_distance = torch.ones(self.num_envs, device=self.device) * 100.0
        self.dynamic_nearest_ttc = torch.ones(self.num_envs, device=self.device) * 100.0
        self.dynamic_nearest_closing_speed = torch.zeros(self.num_envs, device=self.device)
        self.dynamic_min_ttc = torch.ones(self.num_envs, device=self.device) * 100.0
        self.dynamic_min_ttc_closing_speed = torch.zeros(self.num_envs, device=self.device)
        self.prev_dynamic_nearest_distance = torch.ones(self.num_envs, device=self.device) * 100.0
        self.prev_dynamic_min_ttc = torch.ones(self.num_envs, device=self.device) * 100.0
        self.avoidance_had_high_risk = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.avoidance_success_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_nav_actions_after_clip = torch.zeros(self.num_envs, 3, device=self.device)
        self.dynamic_path_blocked = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dynamic_path_block_score = torch.zeros(self.num_envs, device=self.device)
        self.dynamic_path_block_time = torch.zeros(self.num_envs, device=self.device)
        self.dynamic_nearest_rel_pos_local = torch.zeros(self.num_envs, 2, device=self.device)
        self.dynamic_nearest_rel_vel_local = torch.zeros(self.num_envs, 2, device=self.device)
        if hasattr(self, "rays"):
            self.static_rays = torch.ones_like(self.rays) * float(getattr(cfg, "max_dist", 3.0))
        self._dynamic_free_cell_cache = {}
        self._reset_dynamic_obstacles(torch.arange(self.num_envs, device=self.device))
        self.prev_distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        self._update_dynamic_obstacle_features()

    def _dynamic_obstacles_enabled(self):
        return bool(getattr(self.cfg.dynamic_obstacles, "enable", True))

    def _set_dynamic_obstacles_inactive(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        far_xy = torch.ones(len(env_ids), self.num_dynamic_obstacles, 2, device=self.device) * 1.0e6
        self.dynamic_obstacle_pos[env_ids] = far_xy
        self.dynamic_obstacle_vel[env_ids] = 0.0
        self.dynamic_obstacle_base[env_ids] = far_xy
        self.dynamic_collision_step[env_ids] = 0.0
        self.min_dynamic_obstacle_distance[env_ids] = 100.0
        self.dynamic_nearest_distance[env_ids] = 100.0
        self.dynamic_nearest_ttc[env_ids] = 100.0
        self.dynamic_nearest_closing_speed[env_ids] = 0.0
        self.dynamic_min_ttc[env_ids] = 100.0
        self.dynamic_min_ttc_closing_speed[env_ids] = 0.0
        self.prev_dynamic_nearest_distance[env_ids] = 100.0
        self.prev_dynamic_min_ttc[env_ids] = 100.0
        self.avoidance_had_high_risk[env_ids] = False
        self.avoidance_success_latched[env_ids] = False
        self.dynamic_path_blocked[env_ids] = False
        self.dynamic_path_block_score[env_ids] = 0.0
        self.dynamic_path_block_time[env_ids] = 0.0
        self.dynamic_nearest_rel_pos_local[env_ids] = 0.0
        self.dynamic_nearest_rel_vel_local[env_ids] = 0.0

    def _demo_interaction_enabled(self) -> bool:
        return bool(getattr(self.cfg.dynamic_obstacles, "demo_interaction_scene", False))

    def _training_interaction_enabled(self) -> bool:
        return bool(getattr(self.cfg.dynamic_obstacles, "training_interaction_scene", False))

    def _get_clear_free_cells_uncached(self, room: np.ndarray, min_clearance: int) -> np.ndarray:
        free = (room <= 0.1).astype(np.uint8)
        kernel_size = 2 * min_clearance + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        clear_count = convolve2d(free, kernel, mode="same", boundary="fill", fillvalue=0)
        return np.argwhere(clear_count == kernel.size).astype(np.int32)

    def _nearest_clear_cell(self, room: np.ndarray, desired_xy: np.ndarray, min_clearance: int) -> Optional[np.ndarray]:
        free_cells = self._get_clear_free_cells_uncached(room, min_clearance)
        if free_cells.size == 0:
            return None
        score = np.sum((free_cells.astype(np.float32) - desired_xy[None, :]) ** 2, axis=1)
        return free_cells[int(np.argmin(score))].astype(np.int32)

    def _path_is_clear_for_demo(self, room: np.ndarray, start_xy: np.ndarray, goal_xy: np.ndarray, min_clearance: int) -> bool:
        delta = goal_xy.astype(np.float32) - start_xy.astype(np.float32)
        distance = np.linalg.norm(delta)
        if distance < 1e-6:
            return False
        num_samples = max(8, int(distance / 4.0))
        for alpha in np.linspace(0.0, 0.55, num_samples):
            probe = np.rint(start_xy + alpha * delta).astype(np.int32)
            if probe[0] < 1 or probe[0] >= room.shape[0] - 1 or probe[1] < 1 or probe[1] >= room.shape[1] - 1:
                return False
            if room[probe[0], probe[1]] > 0.1:
                return False
            if not is_far_from_obstacles(room, probe.tolist(), min_clearance):
                return False
        return True

    def _sample_demo_interaction_pair(self, room: np.ndarray, grid_shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        min_clearance = 4
        free_cells = self._get_clear_free_cells_uncached(room, min_clearance)
        if free_cells.size == 0:
            return None

        cfg = self.cfg.dynamic_obstacles
        goal_distance_min_m = float(getattr(self.cfg.dynamic_obstacles, "demo_goal_distance_min", 3.5))
        min_goal_distance = max(20.0, goal_distance_min_m / self.terrain.cfg.horizontal_scale)
        margin = np.array([0.18 * grid_shape[0], 0.18 * grid_shape[1]], dtype=np.float32)
        center = np.array([0.50 * grid_shape[0], 0.50 * grid_shape[1]], dtype=np.float32)
        desired_robot = np.array(
            [
                float(getattr(cfg, "demo_robot_frac_x", 0.52)) * grid_shape[0],
                float(getattr(cfg, "demo_robot_frac_y", 0.42)) * grid_shape[1],
            ],
            dtype=np.float32,
        )
        desired_goal = np.array(
            [
                float(getattr(cfg, "demo_goal_frac_x", 0.52)) * grid_shape[0],
                float(getattr(cfg, "demo_goal_frac_y", 0.78)) * grid_shape[1],
            ],
            dtype=np.float32,
        )
        inside = (
            (free_cells[:, 0] > margin[0])
            & (free_cells[:, 0] < grid_shape[0] - margin[0])
            & (free_cells[:, 1] > margin[1])
            & (free_cells[:, 1] < grid_shape[1] - margin[1])
        )
        central_cells = free_cells[inside]
        if central_cells.size > 0:
            free_cells = central_cells

        robot_order = np.argsort(np.sum((free_cells - desired_robot[None, :]) ** 2, axis=1))
        goal_order = np.argsort(np.sum((free_cells - desired_goal[None, :]) ** 2, axis=1))
        robot_candidates = free_cells[robot_order[: min(240, len(robot_order))]]
        goal_candidates = free_cells[goal_order[: min(400, len(goal_order))]]

        best_pair = None
        best_score = float("inf")
        for robot_xy in robot_candidates:
            delta = goal_candidates.astype(np.float32) - robot_xy.astype(np.float32)[None, :]
            dist = np.linalg.norm(delta, axis=1)
            desired_delta = desired_goal - desired_robot
            desired_norm = np.linalg.norm(desired_delta)
            if desired_norm < 1e-6:
                desired_dir = np.array([1.0, 0.0], dtype=np.float32)
            else:
                desired_dir = desired_delta / desired_norm
            forward_mask = np.sum(delta * desired_dir[None, :], axis=1) > 0.55 * min_goal_distance
            valid_goals = goal_candidates[(dist >= min_goal_distance) & forward_mask]
            if valid_goals.size == 0:
                continue
            for goal_xy in valid_goals[:80]:
                if not self._path_is_clear_for_demo(room, robot_xy, goal_xy, min_clearance):
                    continue
                score = (
                    np.sum((robot_xy.astype(np.float32) - desired_robot) ** 2)
                    + 0.75 * np.sum((goal_xy.astype(np.float32) - desired_goal) ** 2)
                    + 0.2 * np.sum((robot_xy.astype(np.float32) - center) ** 2)
                )
                if score < best_score:
                    best_score = float(score)
                    best_pair = (robot_xy.astype(np.int32), goal_xy.astype(np.int32))

        return best_pair

    def _apply_demo_interaction_origins(self, env_ids):
        if not self._demo_interaction_enabled() or len(env_ids) == 0:
            return

        cfg = self.cfg.dynamic_obstacles
        fixed_row = int(getattr(cfg, "demo_terrain_level", 1))
        fixed_col = int(getattr(cfg, "demo_terrain_type", 0))
        fixed_row = int(np.clip(fixed_row, 0, self.cfg.terrain.num_rows - 1))
        fixed_col = int(np.clip(fixed_col, 0, self.cfg.terrain.num_cols - 1))

        for env_idx_tensor in env_ids:
            env_idx = int(env_idx_tensor.item())
            self.terrain_levels[env_idx] = fixed_row
            self.terrain_types[env_idx] = fixed_col
            room = self.terrain.select_room(fixed_row, fixed_col)
            grid_shape = room.shape
            pair = self._sample_demo_interaction_pair(room, grid_shape)
            if pair is None:
                continue
            robot_xy, goal_xy = pair
            robot_world = self._room_idx_to_world_xy(env_idx, robot_xy, grid_shape)
            goal_world = self._room_idx_to_world_xy(env_idx, goal_xy, grid_shape)
            self.env_origins[env_idx, 0] = float(robot_world[0])
            self.env_origins[env_idx, 1] = float(robot_world[1])
            self.position_targets[env_idx, 0] = float(goal_world[0])
            self.position_targets[env_idx, 1] = float(goal_world[1])

    def _apply_training_interaction_origins(self, env_ids):
        if not self._training_interaction_enabled() or len(env_ids) == 0:
            return

        cfg = self.cfg.dynamic_obstacles
        fixed_row = int(getattr(cfg, "demo_terrain_level", 1))
        fixed_col = int(getattr(cfg, "demo_terrain_type", 0))
        fixed_row = int(np.clip(fixed_row, 0, self.cfg.terrain.num_rows - 1))
        fixed_col = int(np.clip(fixed_col, 0, self.cfg.terrain.num_cols - 1))

        for env_idx_tensor in env_ids:
            env_idx = int(env_idx_tensor.item())
            self.terrain_levels[env_idx] = fixed_row
            self.terrain_types[env_idx] = fixed_col
            room = self.terrain.select_room(fixed_row, fixed_col)
            grid_shape = room.shape
            robot_jitter = float(getattr(cfg, "training_interaction_robot_jitter", 0.0))
            goal_jitter = float(getattr(cfg, "training_interaction_goal_jitter", 0.0))
            robot_frac_jitter = np.random.uniform(-robot_jitter, robot_jitter, size=2) if robot_jitter > 0.0 else np.zeros(2)
            goal_frac_jitter = np.random.uniform(-goal_jitter, goal_jitter, size=2) if goal_jitter > 0.0 else np.zeros(2)

            robot_desired = np.array(
                [
                    np.clip(float(getattr(cfg, "demo_robot_frac_x", 0.52)) + robot_frac_jitter[0], 0.18, 0.82)
                    * grid_shape[0],
                    np.clip(float(getattr(cfg, "demo_robot_frac_y", 0.34)) + robot_frac_jitter[1], 0.18, 0.82)
                    * grid_shape[1],
                ],
                dtype=np.float32,
            )
            goal_desired = np.array(
                [
                    np.clip(float(getattr(cfg, "demo_goal_frac_x", 0.52)) + goal_frac_jitter[0], 0.18, 0.82)
                    * grid_shape[0],
                    np.clip(float(getattr(cfg, "demo_goal_frac_y", 0.78)) + goal_frac_jitter[1], 0.18, 0.82)
                    * grid_shape[1],
                ],
                dtype=np.float32,
            )

            robot_xy = self._nearest_clear_cell(room, robot_desired, min_clearance=4)
            goal_xy = self._nearest_clear_cell(room, goal_desired, min_clearance=4)
            if robot_xy is None or goal_xy is None:
                continue

            robot_world = self._room_idx_to_world_xy(env_idx, robot_xy, grid_shape)
            goal_world = self._room_idx_to_world_xy(env_idx, goal_xy, grid_shape)
            self.env_origins[env_idx, 0] = float(robot_world[0])
            self.env_origins[env_idx, 1] = float(robot_world[1])
            self.position_targets[env_idx, 0] = float(goal_world[0])
            self.position_targets[env_idx, 1] = float(goal_world[1])

    def _get_env_origins(self):
        super()._get_env_origins()
        env_ids = torch.arange(self.num_envs, device=self.device)
        if self._training_interaction_enabled():
            self._apply_training_interaction_origins(env_ids)
        else:
            self._apply_demo_interaction_origins(env_ids)

    def _update_terrain_curriculum(self, env_ids):
        super()._update_terrain_curriculum(env_ids)
        if self._training_interaction_enabled():
            self._apply_training_interaction_origins(env_ids)
        else:
            self._apply_demo_interaction_origins(env_ids)

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        interaction_enabled = self._demo_interaction_enabled() or self._training_interaction_enabled()
        if not interaction_enabled or not bool(getattr(self.cfg.dynamic_obstacles, "demo_interaction_face_goal", True)):
            return
        goal_vec = self.position_targets[env_ids, :2] - self.root_states[env_ids, :2]
        yaw = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
        zeros = torch.zeros_like(yaw)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _world_xy_to_room_idx(self, env_id: int, xy_world: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
        row = int(self.terrain_levels[env_id].item())
        col = int(self.terrain_types[env_id].item())
        grid_x, grid_y = grid_shape
        rel_x = np.clip(xy_world[0] / self.terrain.env_length - row, 0.0, 0.9999)
        rel_y = np.clip(xy_world[1] / self.terrain.env_width - col, 0.0, 0.9999)
        return np.array(
            [
                int(rel_x * grid_x),
                int(rel_y * grid_y),
            ],
            dtype=np.int32,
        )

    def _room_idx_to_world_xy(self, env_id: int, idx_xy: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
        row = int(self.terrain_levels[env_id].item())
        col = int(self.terrain_types[env_id].item())
        grid_x, grid_y = grid_shape
        return np.array(
            [
                (row + (float(idx_xy[0]) + 0.5) / grid_x) * self.terrain.env_length,
                (col + (float(idx_xy[1]) + 0.5) / grid_y) * self.terrain.env_width,
            ],
            dtype=np.float32,
        )

    def _sample_base_candidate(
        self,
        free_cells: np.ndarray,
        desired_xy: np.ndarray,
        robot_xy: np.ndarray,
        goal_xy: np.ndarray,
        used_xy: List[np.ndarray],
    ) -> np.ndarray:
        if free_cells.size == 0:
            return desired_xy.astype(np.int32)

        valid_arr = free_cells
        mask = np.linalg.norm(valid_arr - robot_xy[None, :], axis=1) >= 10
        mask &= np.linalg.norm(valid_arr - goal_xy[None, :], axis=1) >= 10
        for prev_xy in used_xy:
            mask &= np.linalg.norm(valid_arr - prev_xy[None, :], axis=1) >= 8

        valid_arr = valid_arr[mask]
        if valid_arr.size == 0:
            return desired_xy.astype(np.int32)

        score = np.sum((valid_arr - desired_xy[None, :]) ** 2, axis=1)
        best_idx = int(np.argmin(score))
        return valid_arr[best_idx].astype(np.int32)

    def _get_clear_free_cells(self, row: int, col: int, room: np.ndarray, min_clearance: int) -> np.ndarray:
        cache_key = (row, col, min_clearance)
        cached = self._dynamic_free_cell_cache.get(cache_key)
        if cached is not None:
            return cached

        free = (room <= 0.1).astype(np.uint8)
        kernel_size = 2 * min_clearance + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        clear_count = convolve2d(free, kernel, mode="same", boundary="fill", fillvalue=0)
        valid = np.argwhere(clear_count == kernel.size).astype(np.int32)
        self._dynamic_free_cell_cache[cache_key] = valid
        return valid

    def _max_clear_distance(
        self,
        room: np.ndarray,
        base_xy: np.ndarray,
        direction_xy: np.ndarray,
        min_clearance: int,
        max_steps: int = 18,
    ) -> float:
        step_dir = np.array(direction_xy, dtype=np.float32)
        norm = np.linalg.norm(step_dir)
        if norm < 1e-6:
            return 0.0
        step_dir = step_dir / norm

        free_steps = 0
        for step in range(1, max_steps + 1):
            probe = np.rint(base_xy + step * step_dir).astype(np.int32)
            if probe[0] < 1 or probe[0] >= room.shape[0] - 1 or probe[1] < 1 or probe[1] >= room.shape[1] - 1:
                break
            if room[probe[0], probe[1]] > 0.1:
                break
            if not is_far_from_obstacles(room, probe.tolist(), min_clearance):
                break
            free_steps = step
        return float(free_steps)

    def _reset_dynamic_obstacles(self, env_ids):
        if len(env_ids) == 0:
            return
        if not self._dynamic_obstacles_enabled():
            self._set_dynamic_obstacles_inactive(env_ids)
            self.dynamic_collision_count[env_ids] = 0
            self.dynamic_collision_pos_hist[env_ids] = 0.0
            return

        cfg = self.cfg.dynamic_obstacles
        speed = float(cfg.speed)
        min_clearance = max(3, int(math.ceil((self.dynamic_obstacle_radius + 0.15) / self.terrain.cfg.horizontal_scale)))
        amplitude_scale = float(getattr(cfg, "amplitude_scale", 0.45))
        amplitude_min = float(getattr(cfg, "amplitude_min", 0.25))
        amplitude_max = float(getattr(cfg, "amplitude_max", 1.20))
        mode_amplitude_scale = dict(getattr(cfg, "mode_amplitude_scale", {}))

        for env_idx_tensor in env_ids:
            env_idx = int(env_idx_tensor.item())
            row = int(self.terrain_levels[env_idx].item())
            col = int(self.terrain_types[env_idx].item())
            room = self.terrain.select_room(row, col)
            grid_shape = room.shape

            start = self.env_origins[env_idx, :2].detach().cpu().numpy()
            goal = self.position_targets[env_idx, :2].detach().cpu().numpy()
            direction = goal - start
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
                norm = 1.0
            axis = direction / norm
            perp = np.array([-axis[1], axis[0]], dtype=np.float32)

            robot_xy = self._world_xy_to_room_idx(env_idx, start, grid_shape)
            goal_xy = self._world_xy_to_room_idx(env_idx, goal, grid_shape)
            free_cells = self._get_clear_free_cells(row, col, room, min_clearance)
            used_xy: List[np.ndarray] = []
            force_path_obstacles = bool(getattr(cfg, "demo_force_path_obstacles", False))
            path_frac_start = float(getattr(cfg, "demo_path_obstacle_frac_start", 0.30))
            path_frac_step = float(getattr(cfg, "demo_path_obstacle_frac_step", 0.18))
            path_lateral_spacing = float(getattr(cfg, "demo_path_lateral_spacing", 0.12))
            path_fracs = list(getattr(cfg, "demo_path_obstacle_fracs", []))
            path_lateral_offsets = list(getattr(cfg, "demo_path_lateral_offsets", []))
            path_frac_jitter = float(getattr(cfg, "training_interaction_obstacle_frac_jitter", 0.0))
            path_lateral_jitter = float(getattr(cfg, "training_interaction_obstacle_lateral_jitter", 0.0))

            for obs_idx, mode in enumerate(self.cfg.dynamic_obstacles.motion_modes[: self.num_dynamic_obstacles]):
                focus_near_robot = bool(getattr(cfg, "focus_near_robot", False)) and obs_idx == 0
                if force_path_obstacles:
                    if obs_idx < len(path_fracs):
                        frac = float(path_fracs[obs_idx])
                    else:
                        frac = path_frac_start + path_frac_step * obs_idx
                    if path_frac_jitter > 0.0:
                        frac += np.random.uniform(-path_frac_jitter, path_frac_jitter)
                    frac = float(np.clip(frac, 0.18, 0.82))
                    if obs_idx < len(path_lateral_offsets):
                        lateral_offset = float(path_lateral_offsets[obs_idx])
                    else:
                        lateral_index = obs_idx - (self.num_dynamic_obstacles - 1) * 0.5
                        lateral_offset = lateral_index * path_lateral_spacing
                    if path_lateral_jitter > 0.0:
                        lateral_offset += np.random.uniform(-path_lateral_jitter, path_lateral_jitter)
                    desired_world = start + frac * direction + lateral_offset * perp
                    if mode == "pedestrian_like":
                        motion_world = perp
                    elif mode == "back_and_forth":
                        motion_world = axis
                    else:
                        motion_world = 0.65 * axis + 0.35 * perp
                elif focus_near_robot:
                    front_min = float(getattr(cfg, "focus_distance_min", 0.90))
                    front_max = float(getattr(cfg, "focus_distance_max", 1.40))
                    lateral_min = float(getattr(cfg, "focus_lateral_min", 0.25))
                    lateral_max = float(getattr(cfg, "focus_lateral_max", 0.75))
                    front_dist = np.random.uniform(front_min, front_max)
                    lateral = np.random.choice([-1.0, 1.0]) * np.random.uniform(lateral_min, lateral_max)
                    desired_world = start + front_dist * axis + lateral * perp
                    motion_world = perp
                else:
                    frac = 0.35 + 0.18 * obs_idx
                    offset = (obs_idx - (self.num_dynamic_obstacles - 1) * 0.5) * 9.0
                    desired_world = start + frac * direction
                    if mode == "pedestrian_like":
                        desired_world = desired_world + offset * self.terrain.cfg.horizontal_scale * perp
                        motion_world = perp
                    elif mode == "back_and_forth":
                        desired_world = desired_world + offset * 0.35 * self.terrain.cfg.horizontal_scale * perp
                        motion_world = axis
                    else:
                        desired_world = desired_world - offset * self.terrain.cfg.horizontal_scale * perp
                        motion_world = 0.65 * axis + 0.35 * perp

                desired_xy = self._world_xy_to_room_idx(env_idx, desired_world, grid_shape)
                base_xy = self._sample_base_candidate(free_cells, desired_xy, robot_xy, goal_xy, used_xy)
                used_xy.append(base_xy)

                motion_room = np.array(
                    [
                        motion_world[0] * grid_shape[0] / self.terrain.env_length,
                        motion_world[1] * grid_shape[1] / self.terrain.env_width,
                    ],
                    dtype=np.float32,
                )
                axis_room = np.array(
                    [
                        axis[0] * grid_shape[0] / self.terrain.env_length,
                        axis[1] * grid_shape[1] / self.terrain.env_width,
                    ],
                    dtype=np.float32,
                )
                perp_room = np.array(
                    [
                        perp[0] * grid_shape[0] / self.terrain.env_length,
                        perp[1] * grid_shape[1] / self.terrain.env_width,
                    ],
                    dtype=np.float32,
                )
                fwd_steps = self._max_clear_distance(room, base_xy, motion_room, min_clearance)
                back_steps = self._max_clear_distance(room, base_xy, -motion_room, min_clearance)
                motion_clear = min(fwd_steps, back_steps)

                axis_clear = min(
                    self._max_clear_distance(room, base_xy, axis_room, min_clearance),
                    self._max_clear_distance(room, base_xy, -axis_room, min_clearance),
                )
                perp_clear = min(
                    self._max_clear_distance(room, base_xy, perp_room, min_clearance),
                    self._max_clear_distance(room, base_xy, -perp_room, min_clearance),
                )

                if mode == "back_and_forth":
                    motion_clear = min(motion_clear, perp_clear / max(self.BACKTRACK_LATERAL_SCALE, 1e-3))
                elif mode == "random_rigid_body":
                    motion_clear = min(
                        motion_clear,
                        axis_clear / max(self.RAND_AXIS_SCALE, 1e-3),
                        perp_clear / max(self.RAND_PERP_SCALE, 1e-3),
                    )

                amp_steps = max(2.5, amplitude_scale * motion_clear)
                mode_scale = float(mode_amplitude_scale.get(mode, 1.0))
                amp_world = float(
                    min(
                        amplitude_max,
                        max(amplitude_min, amp_steps * self.terrain.cfg.horizontal_scale * mode_scale),
                    )
                )
                if force_path_obstacles:
                    demo_range_scale = float(getattr(cfg, "demo_motion_range_scale", 1.0))
                    if mode == "pedestrian_like":
                        demo_range_scale = float(getattr(cfg, "demo_pedestrian_range_scale", demo_range_scale))
                    elif mode == "back_and_forth":
                        demo_range_scale = float(getattr(cfg, "demo_backtrack_range_scale", demo_range_scale))
                    else:
                        demo_range_scale = float(getattr(cfg, "demo_random_range_scale", demo_range_scale))
                    demo_range_max = float(getattr(cfg, "demo_motion_range_max", amplitude_max))
                    clearance_cap = max(
                        amplitude_min,
                        0.90 * motion_clear * self.terrain.cfg.horizontal_scale,
                    )
                    amp_world = float(
                        min(
                            demo_range_max,
                            clearance_cap,
                            max(amplitude_min, amp_world * demo_range_scale),
                        )
                    )

                base_world = self._room_idx_to_world_xy(env_idx, base_xy, grid_shape)
                self.dynamic_obstacle_base[env_idx, obs_idx] = torch.tensor(base_world, device=self.device)
                self.dynamic_obstacle_axis[env_idx, obs_idx] = torch.tensor(axis, device=self.device)
                self.dynamic_obstacle_perp[env_idx, obs_idx] = torch.tensor(perp, device=self.device)
                self.dynamic_obstacle_amp[env_idx, obs_idx] = amp_world
                if force_path_obstacles:
                    if mode == "pedestrian_like":
                        # Start on one side of the start-goal corridor and
                        # immediately cross it, making the interaction visible
                        # in GUI validation instead of relying on random hits.
                        phase = -0.75 if obs_idx % 2 == 0 else math.pi - 0.75
                    elif mode == "back_and_forth":
                        phase = 0.0
                    else:
                        phase = 0.5 * math.pi
                elif focus_near_robot:
                    phase_min = float(getattr(cfg, "focus_phase_min", 0.35))
                    phase_max = float(getattr(cfg, "focus_phase_max", 0.85))
                    alpha = np.random.uniform(phase_min, phase_max)
                    # Start on either side of the path, with velocity pointing
                    # toward the path center. This produces a near-term crossing
                    # event instead of a mostly harmless far obstacle.
                    phase = -alpha if np.random.rand() < 0.5 else math.pi - alpha
                elif bool(getattr(cfg, "phase_randomization", True)):
                    phase = torch.rand(1, device=self.device).item() * 2.0 * math.pi
                else:
                    phase = obs_idx * 2.0 * math.pi / max(1, self.num_dynamic_obstacles)
                self.dynamic_obstacle_phase[env_idx, obs_idx] = phase
                self.dynamic_obstacle_omega[env_idx, obs_idx] = speed / max(amp_world, 0.3)

        self._update_dynamic_obstacles(env_ids=env_ids, force=True)
        self.dynamic_collision_count[env_ids] = 0
        self.min_dynamic_obstacle_distance[env_ids] = 100.0
        self.dynamic_collision_pos_hist[env_ids] = 0.0

    def _update_dynamic_obstacles(self, env_ids=None, force=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        if not self._dynamic_obstacles_enabled():
            self._set_dynamic_obstacles_inactive(env_ids)
            return

        t = self.episode_length_buf[env_ids].float() * self.dt
        phase = self.dynamic_obstacle_phase[env_ids] + t[:, None] * self.dynamic_obstacle_omega[env_ids]

        for obs_idx, mode in enumerate(self.cfg.dynamic_obstacles.motion_modes[: self.num_dynamic_obstacles]):
            if mode == "pedestrian_like":
                move_axis = self.dynamic_obstacle_perp[env_ids, obs_idx]
                phase_i = phase[:, obs_idx].unsqueeze(1)
                amp = self.dynamic_obstacle_amp[env_ids, obs_idx].unsqueeze(1)
                omega = self.dynamic_obstacle_omega[env_ids, obs_idx].unsqueeze(1)
                pos = self.dynamic_obstacle_base[env_ids, obs_idx] + amp * torch.sin(phase_i) * move_axis
                vel = amp * omega * torch.cos(phase_i) * move_axis
            elif mode == "back_and_forth":
                axis = self.dynamic_obstacle_axis[env_ids, obs_idx]
                perp = self.dynamic_obstacle_perp[env_ids, obs_idx]
                phase_i = phase[:, obs_idx].unsqueeze(1)
                amp = self.dynamic_obstacle_amp[env_ids, obs_idx].unsqueeze(1)
                omega = self.dynamic_obstacle_omega[env_ids, obs_idx].unsqueeze(1)
                pos = (
                    self.dynamic_obstacle_base[env_ids, obs_idx]
                    + amp * torch.sin(phase_i) * axis
                    + (self.BACKTRACK_LATERAL_SCALE * amp) * torch.sin(2.0 * phase_i) * perp
                )
                vel = (
                    amp * omega * torch.cos(phase_i) * axis
                    + (2.0 * self.BACKTRACK_LATERAL_SCALE * amp) * omega * torch.cos(2.0 * phase_i) * perp
                )
            else:
                axis = self.dynamic_obstacle_axis[env_ids, obs_idx]
                perp = self.dynamic_obstacle_perp[env_ids, obs_idx]
                amp = self.dynamic_obstacle_amp[env_ids, obs_idx].unsqueeze(1)
                omega = self.dynamic_obstacle_omega[env_ids, obs_idx].unsqueeze(1)
                phase_i = phase[:, obs_idx].unsqueeze(1)
                orbit_axis = self.RAND_AXIS_SCALE * amp * torch.cos(phase_i) * axis
                orbit_perp = self.RAND_PERP_SCALE * amp * torch.sin(phase_i) * perp
                pos = self.dynamic_obstacle_base[env_ids, obs_idx] + orbit_axis + orbit_perp
                vel = (
                    -self.RAND_AXIS_SCALE * amp * omega * torch.sin(phase_i) * axis
                    + self.RAND_PERP_SCALE * amp * omega * torch.cos(phase_i) * perp
                )
            self.dynamic_obstacle_pos[env_ids, obs_idx] = pos
            self.dynamic_obstacle_vel[env_ids, obs_idx] = vel

    def update_percetion(self):
        self.prev_distance = self.distance.clone()
        self._update_dynamic_obstacles()
        super().update_percetion()
        progress_clip = float(getattr(self.cfg.rewards.progress_config, "clip", 0.25))
        self.distance_progress = torch.clamp(self.prev_distance - self.distance, min=-progress_clip, max=progress_clip)
        self.distance_progress = torch.where(
            self.episode_length_buf <= 1,
            torch.zeros_like(self.distance_progress),
            self.distance_progress,
        )
        self._update_dynamic_obstacle_features()

    def _update_dynamic_obstacle_features(self):
        if not self._dynamic_obstacles_enabled():
            self._set_dynamic_obstacles_inactive()
            return

        self.prev_dynamic_nearest_distance[:] = self.dynamic_nearest_distance
        self.prev_dynamic_min_ttc[:] = self.dynamic_min_ttc

        cfg = self.cfg.dynamic_obstacles
        horizon = float(getattr(cfg, "ttc_horizon", 3.0))
        safe_distance = float(getattr(cfg, "ttc_safe_distance", self.dynamic_collision_distance))
        path_block_distance = float(getattr(cfg, "path_block_distance", 2.5))
        path_block_width = float(getattr(cfg, "path_block_width", 0.75))
        path_block_ttc = float(getattr(cfg, "path_block_ttc", horizon))

        diff = self.dynamic_obstacle_pos - self.root_states[:, None, :2]
        dist = torch.norm(diff, dim=-1).clamp(min=1e-4)
        robot_vel_world = self.root_states[:, 7:9]
        rel_vel_robot = robot_vel_world[:, None, :] - self.dynamic_obstacle_vel
        closing_speed = torch.sum(diff * rel_vel_robot, dim=-1) / dist
        valid_closing = closing_speed > 0.05
        distance_to_unsafe = (dist - safe_distance).clamp(min=0.0)
        ttc = torch.where(
            valid_closing,
            distance_to_unsafe / closing_speed.clamp(min=1e-4),
            torch.ones_like(dist) * (horizon + 1.0),
        )

        nearest_idx = torch.argmin(dist, dim=1)
        gather_idx = nearest_idx[:, None, None].expand(-1, 1, 2)
        nearest_diff = torch.gather(diff, 1, gather_idx).squeeze(1)
        nearest_rel_vel = torch.gather(self.dynamic_obstacle_vel - robot_vel_world[:, None, :], 1, gather_idx).squeeze(1)
        self.dynamic_nearest_distance = torch.gather(dist, 1, nearest_idx[:, None]).squeeze(1)
        self.dynamic_nearest_ttc = torch.gather(ttc, 1, nearest_idx[:, None]).squeeze(1)
        self.dynamic_nearest_closing_speed = torch.gather(closing_speed, 1, nearest_idx[:, None]).squeeze(1)
        valid_ttc = torch.where(valid_closing, ttc, torch.ones_like(ttc) * (horizon + 1.0))
        self.dynamic_min_ttc, min_ttc_idx = torch.min(valid_ttc, dim=1)
        self.dynamic_min_ttc_closing_speed = torch.gather(closing_speed, 1, min_ttc_idx[:, None]).squeeze(1)

        nearest_diff_xyz = torch.cat((nearest_diff, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        nearest_rel_vel_xyz = torch.cat((nearest_rel_vel, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        yaw = yaw_quat(self.base_quat)
        self.dynamic_nearest_rel_pos_local = quat_rotate_inverse(yaw, nearest_diff_xyz)[:, :2]
        self.dynamic_nearest_rel_vel_local = quat_rotate_inverse(yaw, nearest_rel_vel_xyz)[:, :2]

        goal_vec = self.position_targets[:, :2] - self.root_states[:, :2]
        goal_dist = torch.norm(goal_vec, dim=-1).clamp(min=1e-4)
        goal_dir = goal_vec / goal_dist[:, None]
        along_goal = torch.sum(diff * goal_dir[:, None, :], dim=-1)
        lateral_vec = diff - along_goal[:, :, None] * goal_dir[:, None, :]
        lateral_dist = torch.norm(lateral_vec, dim=-1)
        raw_blocks = (
            (along_goal > 0.0)
            & (along_goal < torch.minimum(goal_dist[:, None], torch.ones_like(along_goal) * path_block_distance))
            & (lateral_dist < path_block_width)
            & (ttc < path_block_ttc)
        )
        raw_path_blocked = torch.any(raw_blocks, dim=1)
        rise = float(getattr(cfg, "path_block_rise", 0.20))
        fall = float(getattr(cfg, "path_block_fall", 0.08))
        threshold = float(getattr(cfg, "path_block_threshold", 0.45))
        self.dynamic_path_block_score = torch.where(
            raw_path_blocked,
            (self.dynamic_path_block_score + rise).clamp(max=1.0),
            (self.dynamic_path_block_score - fall).clamp(min=0.0),
        )
        self.dynamic_path_blocked = self.dynamic_path_block_score > threshold
        self.dynamic_path_block_time = torch.where(
            self.dynamic_path_blocked,
            self.dynamic_path_block_time + self.dt,
            torch.zeros_like(self.dynamic_path_block_time),
        )

    def _get_dynamic_obstacle_state_obs(self):
        k = int(getattr(self.cfg.env, "dynamic_obstacle_state_k", 3))
        if k <= 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        cfg = self.cfg.dynamic_obstacles
        max_dist = float(getattr(cfg, "max_dist", 3.0))
        horizon = float(getattr(cfg, "ttc_horizon", 3.0))
        safe_distance = float(getattr(cfg, "ttc_safe_distance", self.dynamic_collision_distance))
        vel_scale = max(float(getattr(cfg, "preferred_speed", 0.5)), 1e-3)

        diff = self.dynamic_obstacle_pos - self.root_states[:, None, :2]
        dist = torch.norm(diff, dim=-1).clamp(min=1e-4)
        robot_vel_world = self.root_states[:, 7:9]
        rel_vel = self.dynamic_obstacle_vel - robot_vel_world[:, None, :]
        rel_vel_robot = robot_vel_world[:, None, :] - self.dynamic_obstacle_vel
        closing_speed = torch.sum(diff * rel_vel_robot, dim=-1) / dist
        distance_to_unsafe = (dist - safe_distance).clamp(min=0.0)
        ttc = torch.where(
            closing_speed > 0.05,
            distance_to_unsafe / closing_speed.clamp(min=1e-4),
            torch.ones_like(dist) * (horizon + 1.0),
        )

        order = torch.argsort(dist, dim=1)
        take = min(k, self.num_dynamic_obstacles)
        order = order[:, :take]
        gather_xy = order[:, :, None].expand(-1, -1, 2)
        sorted_diff = torch.gather(diff, 1, gather_xy)
        sorted_vel = torch.gather(rel_vel, 1, gather_xy)
        sorted_dist = torch.gather(dist, 1, order)
        sorted_ttc = torch.gather(ttc, 1, order)

        quat = yaw_quat(self.base_quat).unsqueeze(1).repeat(1, take, 1)
        diff_xyz = torch.cat((sorted_diff, torch.zeros(self.num_envs, take, 1, device=self.device)), dim=-1)
        vel_xyz = torch.cat((sorted_vel, torch.zeros(self.num_envs, take, 1, device=self.device)), dim=-1)
        local_diff = quat_rotate_inverse(quat.reshape(-1, 4), diff_xyz.reshape(-1, 3)).reshape(self.num_envs, take, 3)[:, :, :2]
        local_vel = quat_rotate_inverse(quat.reshape(-1, 4), vel_xyz.reshape(-1, 3)).reshape(self.num_envs, take, 3)[:, :, :2]

        per_obstacle = torch.cat(
            (
                (local_diff / max_dist).clamp(min=-1.0, max=1.0),
                (local_vel / vel_scale).clamp(min=-2.0, max=2.0),
                (sorted_dist / max_dist).clamp(min=0.0, max=1.0).unsqueeze(-1),
                (sorted_ttc / horizon).clamp(min=0.0, max=1.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        if take < k:
            pad = torch.zeros(self.num_envs, k - take, 6, device=self.device)
            per_obstacle = torch.cat((per_obstacle, pad), dim=1)

        global_flags = torch.stack(
            (
                self.dynamic_path_blocked.float(),
                (self.dynamic_nearest_distance / max_dist).clamp(min=0.0, max=1.0),
            ),
            dim=-1,
        )
        return torch.cat((per_obstacle.reshape(self.num_envs, -1), global_flags), dim=-1)

    def compute_observations(self):
        if not getattr(self.cfg.env, "include_dynamic_obstacle_state", False):
            return super().compute_observations()

        self._update_replay_buffer()
        self.prop_buf = torch.cat(
            (
                self.projected_gravity,
                self.slr_commands[:, :3] * self.commands_scale[:3],
                self.base_lin_vel * 1.0,
                self.base_ang_vel * 1.0,
            ),
            dim=-1,
        )

        noise_scales = self.cfg.noise.noise_scales
        noise_vec = torch.cat(
            (
                torch.ones(3) * noise_scales.gravity,
                torch.zeros(3),
                torch.ones(3) * noise_scales.lin_vel * 1.0,
                torch.ones(3) * noise_scales.ang_vel * 1.0,
            ),
            dim=0,
        )

        if self.cfg.noise.add_noise:
            self.prop_buf += (2 * torch.rand_like(self.prop_buf) - 1) * noise_vec.to(self.device)

        self._get_perception()

        env_ids = (self.episode_length_buf % int(self.cfg.commands.delay_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        if len(env_ids) != 0:
            resample_time_idx = -torch.randint(2, 4, (len(env_ids),), device=self.device) - 1
            self.delay_rays[env_ids] = self.rays_hist[env_ids, resample_time_idx, :]
            self.delay_goal[env_ids] = self.goal_hist[env_ids, resample_time_idx, :]

        env_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        self.pos_hist[env_ids] = torch.where(
            (self.episode_length_buf[env_ids] <= 1)[:, None, None],
            torch.stack([self.root_states[env_ids, :2]] * self.cfg.env.his_len, dim=1),
            torch.cat([self.pos_hist[env_ids, 1:], self.root_states[env_ids, :2].unsqueeze(1)], dim=1),
        )

        obs_buf = torch.cat(
            (
                self.prop_buf,
                torch.log2(self.delay_rays.clip(min=0.1, max=5.0)),
                self._get_dynamic_obstacle_state_obs(),
                self.delay_goal,
            ),
            dim=-1,
        )

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.his_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
        )
        self.obs_buf = self.obs_history_buf.view(self.num_envs, -1)

    def _get_dynamic_obstacle_rays(self):
        cfg = self.cfg.dynamic_obstacles
        max_dist = float(cfg.max_dist)
        min_dist = float(cfg.min_dist)
        if not self._dynamic_obstacles_enabled():
            return torch.ones_like(self.rays) * max_dist

        diff_xy = self.dynamic_obstacle_pos - self.root_states[:, None, :2]
        diff_xyz = torch.cat((diff_xy, torch.zeros(self.num_envs, self.num_dynamic_obstacles, 1, device=self.device)), dim=-1)
        quat = yaw_quat(self.base_quat).unsqueeze(1).repeat(1, self.num_dynamic_obstacles, 1)
        local = quat_rotate_inverse(quat.reshape(-1, 4), diff_xyz.reshape(-1, 3)).reshape(
            self.num_envs, self.num_dynamic_obstacles, 3
        )[:, :, :2]

        ray_dirs = torch.stack((torch.cos(self.ray_angles), torch.sin(self.ray_angles)), dim=-1)
        centers = local[:, :, None, :]
        dirs = ray_dirs[None, None, :, :]
        proj = torch.sum(centers * dirs, dim=-1)
        center_norm_sq = torch.sum(centers * centers, dim=-1)
        radius_sq = self.dynamic_obstacle_radius**2
        discriminant = proj * proj - (center_norm_sq - radius_sq)
        valid = (discriminant > 0.0) & (proj > 0.0)
        distance = proj - torch.sqrt(torch.clamp(discriminant, min=0.0))
        distance = torch.where(valid & (distance > min_dist), distance, torch.ones_like(distance) * max_dist)
        return torch.min(distance, dim=1).values.clamp(min=min_dist, max=max_dist)

    def _get_rays(self, env_ids=None):
        super()._get_rays(env_ids=env_ids)
        self.static_rays = self.rays.clone()
        if env_ids is not None:
            # The base task currently refreshes all env rays. Keep the analytical
            # obstacle fusion all-env as well to avoid partial-state edge cases.
            pass
        dynamic_rays = self._get_dynamic_obstacle_rays()
        self.rays = torch.minimum(self.rays, dynamic_rays)

    def check_termination(self):
        super().check_termination()
        if not self._dynamic_obstacles_enabled():
            self.extras.setdefault("episode", {})
            self.extras["episode"]["dynamic_collision_count"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["min_dynamic_obstacle_distance"] = torch.tensor(100.0, device=self.device)
            self.extras["episode"]["dynamic_path_blocked"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["dynamic_path_block_time"] = torch.tensor(0.0, device=self.device)
            self.extras["episode"]["dynamic_nearest_ttc"] = torch.tensor(10.0, device=self.device)
            return

        dist = torch.norm(self.dynamic_obstacle_pos - self.root_states[:, None, :2], dim=-1)
        min_dist = torch.min(dist, dim=1).values
        self.min_dynamic_obstacle_distance = torch.minimum(self.min_dynamic_obstacle_distance, min_dist)
        collision = min_dist < self.dynamic_collision_distance
        self.dynamic_collision_step = collision.float()
        self.dynamic_collision_count += collision.long()
        if collision.any():
            self.dynamic_collision_pos_hist[collision, 0, :2] = self.root_states[collision, :2]
            self.dynamic_collision_pos_hist[collision, 0, 2] = self.root_states[collision, 2]
        self.reset_buf |= collision
        self.terminate_buf |= collision
        self.last_episode_dynamic_collision = collision.clone()
        self.last_episode_dynamic_min_distance = self.min_dynamic_obstacle_distance.clone()
        self.last_episode_dynamic_path_block_time = self.dynamic_path_block_time.clone()
        self.extras.setdefault("episode", {})
        self.extras["episode"]["dynamic_collision_count"] = torch.mean(self.dynamic_collision_count.float())
        self.extras["episode"]["min_dynamic_obstacle_distance"] = torch.mean(self.min_dynamic_obstacle_distance)
        self.extras["episode"]["dynamic_path_blocked"] = torch.mean(self.dynamic_path_blocked.float())
        self.extras["episode"]["dynamic_path_block_time"] = torch.mean(self.dynamic_path_block_time)
        self.extras["episode"]["dynamic_nearest_ttc"] = torch.mean(self.dynamic_nearest_ttc.clamp(max=10.0))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_dynamic_obstacles(env_ids)
        self.prev_distance[env_ids] = torch.norm(self.position_targets[env_ids, :2] - self.root_states[env_ids, :2], dim=1)
        self.distance_progress[env_ids] = 0.0
        self.dynamic_collision_step[env_ids] = 0.0
        self.dynamic_nearest_distance[env_ids] = 100.0
        self.dynamic_nearest_ttc[env_ids] = 100.0
        self.dynamic_nearest_closing_speed[env_ids] = 0.0
        self.dynamic_min_ttc[env_ids] = 100.0
        self.dynamic_min_ttc_closing_speed[env_ids] = 0.0
        self.prev_dynamic_nearest_distance[env_ids] = 100.0
        self.prev_dynamic_min_ttc[env_ids] = 100.0
        self.avoidance_had_high_risk[env_ids] = False
        self.avoidance_success_latched[env_ids] = False
        self.prev_nav_actions_after_clip[env_ids] = 0.0
        self.dynamic_path_block_score[env_ids] = 0.0
        self.dynamic_path_block_time[env_ids] = 0.0
        self._update_dynamic_obstacle_features()

    def _reward_progress(self):
        cfg = self.cfg.rewards.progress_config
        blocked_scale = float(getattr(cfg, "blocked_scale", 0.10))
        near_goal_scale = float(getattr(cfg, "near_goal_scale", 0.20))
        near_goal_distance = float(getattr(cfg, "near_goal_distance", 0.60))
        scale = torch.ones_like(self.distance)
        scale = torch.where(self.dynamic_path_blocked, torch.ones_like(scale) * blocked_scale, scale)
        scale = torch.where(self.distance < near_goal_distance, torch.ones_like(scale) * near_goal_scale, scale)
        return self.distance_progress * self.far_goal * scale

    def _reward_preferred_velocity(self):
        cfg = self.cfg.rewards.preferred_velocity_config
        target_speed = float(getattr(cfg, "target_speed", 0.5))
        wait_speed = float(getattr(cfg, "wait_speed", 0.08))
        near_goal_speed = float(getattr(cfg, "near_goal_speed", 0.03))
        near_goal_distance = float(getattr(cfg, "near_goal_distance", 0.60))
        sigma = max(float(getattr(cfg, "sigma", 0.5)), 1e-3)

        goal_dir = self.goal_local_pos / (self.distance.unsqueeze(1) + 1e-4)
        speed = torch.where(
            self.dynamic_path_blocked,
            torch.ones_like(self.distance) * wait_speed,
            torch.ones_like(self.distance) * target_speed,
        )
        speed = torch.where(
            self.distance < near_goal_distance,
            torch.ones_like(self.distance) * near_goal_speed,
            speed,
        )
        target_vel = goal_dir * speed[:, None]
        vel_error = torch.norm(self.base_lin_vel[:, :2] - target_vel, dim=-1)
        return torch.exp(-torch.square(vel_error / sigma))

    def _reward_dynamic_ttc(self):
        cfg = self.cfg.rewards.dynamic_ttc_config
        horizon = max(float(getattr(cfg, "horizon", 3.0)), 1e-3)
        min_closing_speed = float(getattr(cfg, "min_closing_speed", 0.05))
        approaching = self.dynamic_min_ttc_closing_speed > min_closing_speed
        in_horizon = self.dynamic_min_ttc < horizon
        return torch.exp(-self.dynamic_min_ttc.clamp(max=horizon) / horizon) * approaching * in_horizon

    def _reward_dynamic_clearance(self):
        cfg = self.cfg.rewards.dynamic_clearance_config
        safe_distance = max(float(getattr(cfg, "safe_distance", 0.9)), 1e-3)
        front_margin = float(getattr(cfg, "front_margin", -0.15))
        in_front = self.dynamic_nearest_rel_pos_local[:, 0] > front_margin
        return ((safe_distance - self.dynamic_nearest_distance).clip(min=0.0) / safe_distance) * in_front.float() * self.far_goal

    def _reward_wait(self):
        cfg = self.cfg.rewards.wait_config
        max_speed = max(float(getattr(cfg, "max_speed", 0.12)), 1e-3)
        heading_weight = float(getattr(cfg, "heading_weight", 1.0))
        timeout = max(float(getattr(cfg, "timeout", 3.0)), 1e-3)
        timeout_decay = float(getattr(cfg, "timeout_decay", 0.35))
        planar_speed = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        goal_alignment = (self.goal_local_pos[:, 0] / (self.distance + 1e-4)).clip(min=0.0)
        wait_score = torch.exp(-torch.square(planar_speed / max_speed))
        timeout_factor = torch.where(
            self.dynamic_path_block_time <= timeout,
            torch.ones_like(self.dynamic_path_block_time),
            torch.ones_like(self.dynamic_path_block_time) * timeout_decay,
        )
        return self.dynamic_path_blocked.float() * timeout_factor * wait_score * (1.0 + heading_weight * goal_alignment)

    def _reward_blocked_fast_penalty(self):
        cfg = self.cfg.rewards.blocked_fast_penalty_config
        speed_threshold = float(getattr(cfg, "speed_threshold", 0.18))
        ttc_threshold = float(getattr(cfg, "ttc_threshold", 2.0))
        closing_speed_threshold = float(getattr(cfg, "closing_speed_threshold", 0.05))
        planar_speed = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        fast_when_blocked = (planar_speed - speed_threshold).clip(min=0.0)
        dangerous = (
            self.dynamic_path_blocked
            & (self.dynamic_min_ttc < ttc_threshold)
            & (self.dynamic_min_ttc_closing_speed > closing_speed_threshold)
        )
        return dangerous.float() * fast_when_blocked

    def _reward_detour(self):
        cfg = self.cfg.rewards.detour_config
        lateral_weight = float(getattr(cfg, "lateral_weight", 1.0))
        yaw_weight = float(getattr(cfg, "yaw_weight", 0.5))
        blocked_scale = float(getattr(cfg, "blocked_scale", 0.2))
        timeout_scale = float(getattr(cfg, "timeout_scale", blocked_scale))
        penalty = lateral_weight * torch.square(self.base_lin_vel[:, 1]) + yaw_weight * torch.square(self.base_ang_vel[:, 2])
        blocked_timed_out = self.dynamic_path_block_time > float(getattr(self.cfg.dynamic_obstacles, "blocked_timeout", 3.0))
        scale = torch.where(
            self.dynamic_path_blocked & (~blocked_timed_out),
            torch.ones_like(penalty) * blocked_scale,
            torch.ones_like(penalty),
        )
        scale = torch.where(
            blocked_timed_out,
            torch.ones_like(penalty) * timeout_scale,
            scale,
        )
        return penalty * scale * self.far_goal

    def _reward_stuck(self):
        if getattr(self.cfg.dynamic_obstacles, "use_legacy_reward", False):
            return super()._reward_stuck()
        return super()._reward_stuck() * (~self.dynamic_path_blocked).float()

    def _reward_near_goal_stop(self):
        cfg = self.cfg.rewards.near_goal_stop_config
        distance = float(getattr(cfg, "distance", 0.60))
        max_speed = max(float(getattr(cfg, "max_speed", 0.10)), 1e-3)
        max_yaw_rate = max(float(getattr(cfg, "max_yaw_rate", 0.35)), 1e-3)
        near_goal = self.distance < distance
        planar_speed = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        yaw_rate = torch.abs(self.base_ang_vel[:, 2])
        stop_score = torch.exp(-torch.square(planar_speed / max_speed)) * torch.exp(-torch.square(yaw_rate / max_yaw_rate))
        return near_goal.float() * stop_score

    def _reward_dynamic_collision(self):
        return self.dynamic_collision_step

    def _stage1_high_risk(self):
        cfg = self.cfg.rewards.avoidance_stage1_config
        high_dist = float(getattr(cfg, "high_risk_distance", 1.05))
        high_ttc = float(getattr(cfg, "high_risk_ttc", 1.30))
        approaching = self.dynamic_min_ttc_closing_speed > 0.05
        return (self.dynamic_nearest_distance < high_dist) | (approaching & (self.dynamic_min_ttc < high_ttc))

    def _stage1_low_risk(self):
        cfg = self.cfg.rewards.avoidance_stage1_config
        low_dist = float(getattr(cfg, "low_risk_distance", 1.35))
        low_ttc = float(getattr(cfg, "low_risk_ttc", 2.50))
        approaching = self.dynamic_min_ttc_closing_speed > 0.05
        return (self.dynamic_nearest_distance > low_dist) & ((~approaching) | (self.dynamic_min_ttc > low_ttc))

    def _stage1_static_clearance(self):
        cfg = self.cfg.rewards.avoidance_stage1_config
        min_clearance = float(getattr(cfg, "min_static_clearance", 0.45))
        rays = getattr(self, "static_rays", self.rays)
        return torch.min(rays, dim=1).values > min_clearance

    def _reward_successful_avoidance(self):
        high_risk = self._stage1_high_risk()
        self.avoidance_had_high_risk |= high_risk
        no_dynamic_collision = self.dynamic_collision_step < 0.5
        no_static_collision = ~getattr(self, "last_collision_active", torch.zeros_like(high_risk))
        success = (
            self.avoidance_had_high_risk
            & self._stage1_low_risk()
            & self._stage1_static_clearance()
            & no_dynamic_collision
            & no_static_collision
            & (~self.avoidance_success_latched)
        )
        self.avoidance_success_latched |= success
        return success.float()

    def _reward_risk_reduction(self):
        cfg = self.cfg.rewards.risk_reduction_config
        dist_weight = float(getattr(cfg, "distance_weight", 0.55))
        ttc_weight = float(getattr(cfg, "ttc_weight", 0.45))
        avoid_cfg = self.cfg.rewards.avoidance_stage1_config
        high_dist = max(float(getattr(avoid_cfg, "high_risk_distance", 1.05)), 1e-3)
        low_ttc = max(float(getattr(avoid_cfg, "low_risk_ttc", 2.50)), 1e-3)

        was_relevant = self.avoidance_had_high_risk | self._stage1_high_risk()
        dist_gain = ((self.dynamic_nearest_distance - self.prev_dynamic_nearest_distance) / high_dist).clip(min=0.0, max=1.0)
        prev_ttc = self.prev_dynamic_min_ttc.clamp(max=low_ttc)
        curr_ttc = self.dynamic_min_ttc.clamp(max=low_ttc)
        ttc_gain = ((curr_ttc - prev_ttc) / low_ttc).clip(min=0.0, max=1.0)
        return was_relevant.float() * (dist_weight * dist_gain + ttc_weight * ttc_gain)

    def _reward_free_space_action(self):
        cfg = self.cfg.rewards.free_space_action_config
        safe_distance = float(getattr(cfg, "safe_distance", 0.65))
        speed_threshold = float(getattr(cfg, "speed_threshold", 0.08))
        rays = getattr(self, "static_rays", self.rays)
        static_clear = torch.min(rays, dim=1).values > safe_distance
        planar_speed = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        moved = planar_speed > speed_threshold
        # This reward deliberately ignores goal direction: Stage 1 first learns
        # to escape dynamic risk into a static-safe free space.
        return static_clear.float() * moved.float() * (self._stage1_high_risk() | self.avoidance_had_high_risk).float()

    def _reward_unsafe_ttc(self):
        cfg = self.cfg.rewards.unsafe_ttc_config
        threshold = max(float(getattr(cfg, "threshold", 1.0)), 1e-3)
        approaching = self.dynamic_min_ttc_closing_speed > 0.05
        unsafe = (threshold - self.dynamic_min_ttc).clip(min=0.0) / threshold
        return unsafe * approaching.float()

    def _reward_nav_action_smoothness(self):
        cfg = self.cfg.rewards.nav_action_smoothness_config
        sigma = max(float(getattr(cfg, "sigma", 0.50)), 1e-3)
        action_delta = torch.norm(self.nav_actions_after_clip - self.prev_nav_actions_after_clip, dim=-1)
        self.prev_nav_actions_after_clip[:] = self.nav_actions_after_clip
        return torch.square(action_delta / sigma)

    def _reward_static_collision(self):
        return super()._reward_collision()

    def _draw_dynamic_obstacles_vis(self):
        env_idx = 0
        base_z = float(self.root_states[env_idx, 2].detach().cpu().item() + 0.42)
        points = torch.zeros(self.num_dynamic_obstacles, 3, device=self.device)
        points[:, :2] = self.dynamic_obstacle_pos[env_idx]
        points[:, 2] = base_z

        mode_colors = {
            "pedestrian_like": (0.0, 0.95, 0.95),
            "back_and_forth": (0.2, 0.35, 1.0),
            "random_rigid_body": (1.0, 0.45, 0.0),
        }

        for obs_idx, mode in enumerate(self.cfg.dynamic_obstacles.motion_modes[: self.num_dynamic_obstacles]):
            pos_xy = self.dynamic_obstacle_pos[env_idx, obs_idx].detach().cpu().numpy()
            base_xy = self.dynamic_obstacle_base[env_idx, obs_idx].detach().cpu().numpy()
            axis_xy = self.dynamic_obstacle_axis[env_idx, obs_idx].detach().cpu().numpy()
            perp_xy = self.dynamic_obstacle_perp[env_idx, obs_idx].detach().cpu().numpy()
            amp = float(self.dynamic_obstacle_amp[env_idx, obs_idx].detach().cpu().item())
            color = mode_colors.get(mode, (1.0, 0.7, 0.0))
            color_vec = gymapi.Vec3(float(color[0]), float(color[1]), float(color[2]))

            if mode == "pedestrian_like":
                start_xy = base_xy - amp * perp_xy
                end_xy = base_xy + amp * perp_xy
                gymutil.draw_line(
                    gymapi.Vec3(float(start_xy[0]), float(start_xy[1]), base_z),
                    gymapi.Vec3(float(end_xy[0]), float(end_xy[1]), base_z),
                    color_vec,
                    self.gym,
                    self.viewer,
                    self.envs[env_idx],
                )
                torso_pose = gymapi.Transform(gymapi.Vec3(float(pos_xy[0]), float(pos_xy[1]), base_z + 0.35), r=None)
                torso_geom = gymutil.WireframeBoxGeometry(0.14, 0.14, 0.52, None, color=color)
                gymutil.draw_lines(torso_geom, self.gym, self.viewer, self.envs[env_idx], torso_pose)
                head_pose = gymapi.Transform(gymapi.Vec3(float(pos_xy[0]), float(pos_xy[1]), base_z + 0.70), r=None)
                head_geom = gymutil.WireframeSphereGeometry(0.08, 10, 10, None, color=color)
                gymutil.draw_lines(head_geom, self.gym, self.viewer, self.envs[env_idx], head_pose)
            elif mode == "back_and_forth":
                pts = []
                for theta in np.linspace(0.0, 2.0 * math.pi, 33, endpoint=False):
                    curve_xy = (
                        base_xy
                        + amp * math.sin(theta) * axis_xy
                        + (self.BACKTRACK_LATERAL_SCALE * amp) * math.sin(2.0 * theta) * perp_xy
                    )
                    pts.append(curve_xy)
                pts.append(pts[0])
                for start_xy, end_xy in zip(pts[:-1], pts[1:]):
                    gymutil.draw_line(
                        gymapi.Vec3(float(start_xy[0]), float(start_xy[1]), base_z),
                        gymapi.Vec3(float(end_xy[0]), float(end_xy[1]), base_z),
                        color_vec,
                        self.gym,
                        self.viewer,
                        self.envs[env_idx],
                    )
                loop_markers = [
                    base_xy + 0.55 * amp * axis_xy,
                    base_xy - 0.55 * amp * axis_xy,
                ]
                for marker_xy in loop_markers:
                    marker_pose = gymapi.Transform(gymapi.Vec3(float(marker_xy[0]), float(marker_xy[1]), base_z), r=None)
                    marker_geom = gymutil.WireframeSphereGeometry(0.10, 8, 8, None, color=color)
                    gymutil.draw_lines(marker_geom, self.gym, self.viewer, self.envs[env_idx], marker_pose)
                box_pose = gymapi.Transform(gymapi.Vec3(float(pos_xy[0]), float(pos_xy[1]), base_z + 0.18), r=None)
                box_geom = gymutil.WireframeBoxGeometry(0.36, 0.24, 0.30, None, color=color)
                gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[env_idx], box_pose)
                accent_geom = gymutil.WireframeBoxGeometry(0.14, 0.40, 0.10, None, color=(0.75, 0.85, 1.0))
                gymutil.draw_lines(accent_geom, self.gym, self.viewer, self.envs[env_idx], box_pose)
            else:
                pts = []
                for theta in np.linspace(0.0, 2.0 * math.pi, 17, endpoint=False):
                    orbit_xy = (
                        base_xy
                        + self.RAND_AXIS_SCALE * amp * math.cos(theta) * axis_xy
                        + self.RAND_PERP_SCALE * amp * math.sin(theta) * perp_xy
                    )
                    pts.append(orbit_xy)
                pts.append(pts[0])
                for start_xy, end_xy in zip(pts[:-1], pts[1:]):
                    gymutil.draw_line(
                        gymapi.Vec3(float(start_xy[0]), float(start_xy[1]), base_z),
                        gymapi.Vec3(float(end_xy[0]), float(end_xy[1]), base_z),
                        color_vec,
                        self.gym,
                        self.viewer,
                        self.envs[env_idx],
                    )
                body_pose = gymapi.Transform(gymapi.Vec3(float(pos_xy[0]), float(pos_xy[1]), base_z + 0.20), r=None)
                body_geom = gymutil.WireframeBoxGeometry(0.40, 0.24, 0.28, None, color=color)
                gymutil.draw_lines(body_geom, self.gym, self.viewer, self.envs[env_idx], body_pose)
                accent_geom = gymutil.WireframeBoxGeometry(0.18, 0.44, 0.12, None, color=(1.0, 0.8, 0.2))
                gymutil.draw_lines(accent_geom, self.gym, self.viewer, self.envs[env_idx], body_pose)

            current_pose = gymapi.Transform(gymapi.Vec3(float(pos_xy[0]), float(pos_xy[1]), base_z), r=None)
            current_geom = gymutil.WireframeSphereGeometry(0.10, 12, 12, None, color=color)
            gymutil.draw_lines(current_geom, self.gym, self.viewer, self.envs[env_idx], current_pose)

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        if getattr(self.cfg.visualization, "draw_dynamic_obstacles", True):
            self._draw_dynamic_obstacles_vis()
