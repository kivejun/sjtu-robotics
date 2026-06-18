# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import sys
import heapq
import math


from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import time
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules import ActorCritic, DifferentiableSafeActorCritic, TransformerActorCritic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import numpy as np
import torch
import time
import cv2
from isaacgym import gymapi


POLICY_CLASSES = {
    "ActorCritic": ActorCritic,
    "DifferentiableSafeActorCritic": DifferentiableSafeActorCritic,
    "TransformerActorCritic": TransformerActorCritic,
}


def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def resolve_policy_checkpoint(train_cfg, load_run, checkpoint):
    if load_run is None or checkpoint is None:
        return None
    checkpoint_str = str(checkpoint)
    if checkpoint_str.endswith(".pt") and os.path.isabs(checkpoint_str):
        return checkpoint_str
    if checkpoint_str.endswith(".pt"):
        return os.path.abspath(checkpoint_str)
    return os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        str(load_run),
        f"model_{checkpoint}.pt",
    )


def load_inference_policy_from_checkpoint(env, train_cfg, checkpoint_path, num_dynamic_obstacle_obs):
    train_cfg_dict = class_to_dict(train_cfg)
    policy_name = train_cfg_dict["runner"]["policy_class_name"]
    policy_class = POLICY_CLASSES.get(policy_name)
    if policy_class is None:
        raise ValueError(f"Unsupported policy class for manual loading: {policy_name}")
    actor_critic = policy_class(
        num_actions=env.num_nav_actions,
        num_props=env.num_props,
        his_len=env.cfg.env.his_len,
        num_rays=env.rays.shape[1],
        num_dynamic_obstacle_obs=num_dynamic_obstacle_obs,
        **train_cfg_dict["policy"],
    ).to(env.device)
    loaded = torch.load(checkpoint_path, map_location=env.device)
    actor_critic.load_state_dict(loaded["model_state_dict"])
    actor_critic.eval()
    print(f"Loaded manual policy from: {checkpoint_path} (dynamic_obs_dim={num_dynamic_obstacle_obs})")
    return actor_critic.act_inference


def strip_dynamic_obstacle_obs(env, observations):
    dyn_dim = int(getattr(env.cfg.env, "num_dynamic_obstacle_obs", 0))
    if dyn_dim <= 0:
        return observations
    his_len = int(env.cfg.env.his_len)
    full_step = int(env.cfg.env.num_obs_one_step)
    base_step = full_step - dyn_dim
    prop_ray_dim = int(env.num_props + env.rays.shape[1])
    obs_hist = observations.reshape(observations.shape[0], his_len, full_step)
    stripped = torch.cat(
        (
            obs_hist[:, :, :prop_ray_dim],
            obs_hist[:, :, -2:],
        ),
        dim=-1,
    )
    return stripped.reshape(observations.shape[0], his_len * base_step)


def yaw_from_xyzw(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def rotate_local_to_world(vec_local, yaw):
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack(
        (
            c * vec_local[:, 0] - s * vec_local[:, 1],
            s * vec_local[:, 0] + c * vec_local[:, 1],
        ),
        dim=-1,
    )


def shortest_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def inflate_occupancy(occ, radius_cells):
    if radius_cells <= 0:
        return occ
    inflated = occ.copy()
    obstacle_idx = np.argwhere(occ)
    rows, cols = occ.shape
    for r, c in obstacle_idx:
        r0, r1 = max(0, r - radius_cells), min(rows, r + radius_cells + 1)
        c0, c1 = max(0, c - radius_cells), min(cols, c + radius_cells + 1)
        inflated[r0:r1, c0:c1] = True
    return inflated


def world_xy_to_room_idx(env, env_id, xy):
    row = int(env.terrain_levels[env_id].item())
    col = int(env.terrain_types[env_id].item())
    room = env.terrain.select_room(row, col)
    gx = int(round(((float(xy[0]) / env.terrain.env_length) - row) * room.shape[0]))
    gy = int(round(((float(xy[1]) / env.terrain.env_width) - col) * room.shape[1]))
    gx = int(np.clip(gx, 0, room.shape[0] - 1))
    gy = int(np.clip(gy, 0, room.shape[1] - 1))
    return gx, gy


def room_idx_to_world_xy(env, env_id, idx):
    row = int(env.terrain_levels[env_id].item())
    col = int(env.terrain_types[env_id].item())
    room = env.terrain.select_room(row, col)
    x = (row + (float(idx[0]) + 0.5) / room.shape[0]) * env.terrain.env_length
    y = (col + (float(idx[1]) + 0.5) / room.shape[1]) * env.terrain.env_width
    return np.array([x, y], dtype=np.float32)


def nearest_free_cell(occ, start):
    rows, cols = occ.shape
    sr = int(np.clip(start[0], 0, rows - 1))
    sc = int(np.clip(start[1], 0, cols - 1))
    if not occ[sr, sc]:
        return sr, sc
    queue = [(sr, sc)]
    seen = {(sr, sc)}
    for r, c in queue:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in seen:
                continue
            if not occ[nr, nc]:
                return nr, nc
            seen.add((nr, nc))
            queue.append((nr, nc))
    return sr, sc


def astar_path(occ, start, goal):
    start = nearest_free_cell(occ, start)
    goal = nearest_free_cell(occ, goal)
    if start == goal:
        return [start]
    rows, cols = occ.shape
    neighbors = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, 1.414),
        (1, -1, 1.414),
        (-1, 1, 1.414),
        (-1, -1, 1.414),
    )
    open_set = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)
        for dr, dc, cost in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or occ[nr, nc]:
                continue
            nxt = (nr, nc)
            tentative = g_score[current] + cost
            if tentative >= g_score.get(nxt, 1.0e9):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative
            heuristic = np.hypot(goal[0] - nr, goal[1] - nc)
            heapq.heappush(open_set, (tentative + heuristic, nxt))
    return [start, goal]


def get_static_planner_state(env, env_id=0):
    cache = getattr(env, "_pipeline_global_planner_cache", {})
    row = int(env.terrain_levels[env_id].item())
    col = int(env.terrain_types[env_id].item())
    room = env.terrain.select_room(row, col)
    goal_xy = env.position_targets[env_id, :2].detach().cpu().numpy()
    robot_xy = env.root_states[env_id, :2].detach().cpu().numpy()
    goal_idx = world_xy_to_room_idx(env, env_id, goal_xy)
    robot_idx = world_xy_to_room_idx(env, env_id, robot_xy)
    current_step = int(env.episode_length_buf[env_id].item())
    replan_period_s = env_float("SEA_NAV_DWA_ASTAR_REPLAN_PERIOD", 1.0)
    replan_steps = max(1, int(round(replan_period_s / max(float(env.dt), 1e-6))))
    last_replan_step = int(cache.get("last_replan_step", -10**9))
    replan_due = current_step <= 1 or current_step - last_replan_step >= replan_steps
    key = (row, col, goal_idx)
    if cache.get("key") != key or replan_due:
        obstacle_height = env_float("SEA_NAV_DWA_OBSTACLE_HEIGHT", 0.12)
        inflation_m = env_float("SEA_NAV_DWA_STATIC_INFLATION", 0.35)
        res = float(env.terrain.cfg.horizontal_scale)
        base_occ = room > obstacle_height
        start = nearest_free_cell(base_occ, robot_idx)
        goal = nearest_free_cell(base_occ, goal_idx)
        path_idx = []
        occ = base_occ
        for inflation_scale in (1.0, 0.5, 0.25, 0.0):
            radius_cells = int(np.ceil(inflation_m * inflation_scale / max(res, 1e-6)))
            occ_try = inflate_occupancy(base_occ, radius_cells)
            start_try = nearest_free_cell(occ_try, start)
            goal_try = nearest_free_cell(occ_try, goal)
            path_try = astar_path(occ_try, start_try, goal_try)
            if len(path_try) > 2 or start_try == goal_try:
                occ = occ_try
                path_idx = path_try
                break
        if not path_idx:
            path_idx = astar_path(base_occ, start, goal)
            occ = base_occ
        replan_count = int(cache.get("replan_count", 0)) + 1
        cache = {
            "key": key,
            "robot_idx": robot_idx,
            "last_replan_step": current_step,
            "replan_count": replan_count,
            "occ": occ,
            "path_idx": path_idx,
            "path_world": [room_idx_to_world_xy(env, env_id, idx) for idx in path_idx],
        }
        env._pipeline_global_planner_cache = cache
        if replan_count == 1 or replan_count % 5 == 0:
            print(f"[N1 pipeline] A* path cells: {len(path_idx)}")
    return env._pipeline_global_planner_cache


def get_local_waypoint_world(env, env_id=0):
    state = get_static_planner_state(env, env_id)
    robot_xy = env.root_states[env_id, :2].detach().cpu().numpy()
    path = state["path_world"]
    if len(path) == 0:
        return env.position_targets[env_id, :2].detach().cpu().numpy()
    lookahead = env_float("SEA_NAV_DWA_WAYPOINT_LOOKAHEAD", 1.0)
    distances = np.array([np.linalg.norm(p - robot_xy) for p in path], dtype=np.float32)
    nearest = int(np.argmin(distances))
    waypoint = path[-1]
    for p in path[nearest:]:
        if np.linalg.norm(p - robot_xy) >= lookahead:
            waypoint = p
            break
    return waypoint


def is_static_rollout_safe(env, env_id, action, horizon=1.6, dt=0.2):
    state = get_static_planner_state(env, env_id)
    occ = state["occ"]
    pos = env.root_states[env_id, :2].detach().cpu().numpy().astype(np.float32)
    yaw = float(yaw_from_xyzw(env.base_quat[env_id : env_id + 1]).item())
    act = action.detach().cpu().numpy().astype(np.float32)
    steps = max(1, int(horizon / dt))
    for _ in range(steps):
        c, s = np.cos(yaw), np.sin(yaw)
        vel_world = np.array([c * act[0] - s * act[1], s * act[0] + c * act[1]], dtype=np.float32)
        pos = pos + vel_world * dt
        yaw = yaw + float(act[2]) * dt
        idx = world_xy_to_room_idx(env, env_id, pos)
        if occ[idx]:
            return False
    return True


def is_dynamic_rollout_safe(env, env_id, action, horizon=2.0, dt=0.2):
    if not hasattr(env, "dynamic_obstacle_pos"):
        return True
    safe_distance = env_float("SEA_NAV_DWA_SAFE_DISTANCE", 0.75)
    pos = env.root_states[env_id, :2].detach().cpu().numpy().astype(np.float32)
    yaw = float(yaw_from_xyzw(env.base_quat[env_id : env_id + 1]).item())
    act = action.detach().cpu().numpy().astype(np.float32)
    obs_pos = env.dynamic_obstacle_pos[env_id].detach().cpu().numpy().astype(np.float32)
    obs_vel = env.dynamic_obstacle_vel[env_id].detach().cpu().numpy().astype(np.float32)
    steps = max(1, int(horizon / dt))
    for step in range(steps):
        c, s = np.cos(yaw), np.sin(yaw)
        vel_world = np.array([c * act[0] - s * act[1], s * act[0] + c * act[1]], dtype=np.float32)
        pos = pos + vel_world * dt
        yaw = yaw + float(act[2]) * dt
        predicted_obs = obs_pos + obs_vel * (step + 1) * dt
        if np.min(np.linalg.norm(predicted_obs - pos[None, :], axis=1)) < safe_distance:
            return False
    return True


def predict_dynamic_obstacle_trajectory(env, env_id, horizon=4.0, dt=0.15):
    """Predict analytical dynamic obstacle positions using the same motion model as the env."""
    if not hasattr(env, "dynamic_obstacle_pos"):
        return None

    steps = max(1, int(np.ceil(horizon / dt)))
    times = torch.arange(1, steps + 1, device=env.device, dtype=torch.float32) * dt
    current_t = env.episode_length_buf[env_id].float() * env.dt

    if not all(
        hasattr(env, name)
        for name in (
            "dynamic_obstacle_base",
            "dynamic_obstacle_axis",
            "dynamic_obstacle_perp",
            "dynamic_obstacle_amp",
            "dynamic_obstacle_phase",
            "dynamic_obstacle_omega",
        )
    ):
        pos = env.dynamic_obstacle_pos[env_id][:, None, :]
        vel = env.dynamic_obstacle_vel[env_id][:, None, :]
        return pos + vel * times[None, :, None]

    modes = list(getattr(env.cfg.dynamic_obstacles, "motion_modes", []))
    base = env.dynamic_obstacle_base[env_id]
    axis = env.dynamic_obstacle_axis[env_id]
    perp = env.dynamic_obstacle_perp[env_id]
    amp = env.dynamic_obstacle_amp[env_id]
    phase0 = env.dynamic_obstacle_phase[env_id]
    omega = env.dynamic_obstacle_omega[env_id]
    phase = phase0[:, None] + (current_t + times)[None, :] * omega[:, None]

    traj = torch.zeros(env.num_dynamic_obstacles, steps, 2, device=env.device)
    backtrack_lateral_scale = float(getattr(env, "BACKTRACK_LATERAL_SCALE", 0.35))
    rand_axis_scale = float(getattr(env, "RAND_AXIS_SCALE", 1.0))
    rand_perp_scale = float(getattr(env, "RAND_PERP_SCALE", 1.0))
    for obs_idx in range(env.num_dynamic_obstacles):
        mode = modes[obs_idx % len(modes)] if modes else "pedestrian_like"
        phase_i = phase[obs_idx]
        if mode == "pedestrian_like":
            traj[obs_idx] = base[obs_idx][None, :] + amp[obs_idx] * torch.sin(phase_i)[:, None] * perp[obs_idx][None, :]
        elif mode == "back_and_forth":
            traj[obs_idx] = (
                base[obs_idx][None, :]
                + amp[obs_idx] * torch.sin(phase_i)[:, None] * axis[obs_idx][None, :]
                + (backtrack_lateral_scale * amp[obs_idx]) * torch.sin(2.0 * phase_i)[:, None] * perp[obs_idx][None, :]
            )
        else:
            traj[obs_idx] = (
                base[obs_idx][None, :]
                + rand_axis_scale * amp[obs_idx] * torch.cos(phase_i)[:, None] * axis[obs_idx][None, :]
                + rand_perp_scale * amp[obs_idx] * torch.sin(phase_i)[:, None] * perp[obs_idx][None, :]
            )
    return traj


def rollout_robot_xy(env, env_id, action, horizon=2.0, dt=0.15):
    steps = max(1, int(np.ceil(horizon / dt)))
    pos = env.root_states[env_id, :2].detach().clone()
    yaw = yaw_from_xyzw(env.base_quat[env_id : env_id + 1]).detach().clone().squeeze(0)
    act = action.detach()
    positions = []
    for _ in range(steps):
        c, s = torch.cos(yaw), torch.sin(yaw)
        vel_world = torch.stack((c * act[0] - s * act[1], s * act[0] + c * act[1]))
        pos = pos + vel_world * dt
        yaw = yaw + act[2] * dt
        positions.append(pos.clone())
    return torch.stack(positions, dim=0)


def is_dynamic_trajectory_rollout_safe(env, env_id, action, horizon=4.0, dt=0.15, safe_distance=None):
    if not hasattr(env, "dynamic_obstacle_pos"):
        return True
    if safe_distance is None:
        safe_distance = env_float("SEA_NAV_HYBRID_SAFE_DISTANCE", 0.78)
    robot_traj = rollout_robot_xy(env, env_id, action, horizon=horizon, dt=dt)
    obs_traj = predict_dynamic_obstacle_trajectory(env, env_id, horizon=horizon, dt=dt)
    if obs_traj is None:
        return True
    steps = min(robot_traj.shape[0], obs_traj.shape[1])
    diff = obs_traj[:, :steps, :] - robot_traj[None, :steps, :]
    min_dist = torch.min(torch.norm(diff, dim=-1)).item()
    return min_dist >= safe_distance


def is_static_footprint_rollout_safe(env, env_id, action, horizon=1.8, dt=0.15):
    robot_radius = env_float("SEA_NAV_HYBRID_ROBOT_RADIUS", 0.45)
    state = get_static_planner_state(env, env_id)
    occ = state["occ"]
    positions = rollout_robot_xy(env, env_id, action, horizon=horizon, dt=dt).detach().cpu().numpy()
    sample_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    sample_offsets = [np.array([0.0, 0.0], dtype=np.float32)]
    sample_offsets.extend(np.stack((np.cos(sample_angles), np.sin(sample_angles)), axis=1).astype(np.float32) * robot_radius)
    for pos in positions:
        for offset in sample_offsets:
            idx = world_xy_to_room_idx(env, env_id, pos + offset)
            if occ[idx]:
                return False
    return True


def has_safe_crossing_window(env, env_id, goal_dir, raw_action, horizon=4.0, dt=0.15):
    crossing_speed = env_float("SEA_NAV_HYBRID_CROSSING_SPEED", 0.35)
    crossing_window_horizon = env_float("SEA_NAV_HYBRID_CROSSING_WINDOW_HORIZON", 1.6)
    crossing_action = torch.zeros(3, device=env.device)
    crossing_action[:2] = goal_dir[env_id] * crossing_speed
    crossing_action[2] = torch.clamp(raw_action[env_id, 2], min=-0.35, max=0.35)
    # We only need a full safe gap long enough to clear the conflict region.
    # Predicting much farther than that makes the filter overreact to periodic
    # obstacles that will come back after the robot has already crossed.
    dynamic_safe = is_dynamic_trajectory_rollout_safe(
        env,
        env_id,
        crossing_action,
        horizon=min(horizon, crossing_window_horizon),
        dt=dt,
    )
    static_safe = is_static_footprint_rollout_safe(
        env,
        env_id,
        crossing_action,
        horizon=min(horizon, env_float("SEA_NAV_HYBRID_STATIC_ROLLOUT_HORIZON", 1.8)),
        dt=dt,
    )
    return dynamic_safe and static_safe


def dynamic_ttc_for_velocity(env, vel_world):
    if not hasattr(env, "dynamic_obstacle_pos"):
        inf = torch.ones(env.num_envs, device=env.device) * 100.0
        return inf, inf

    horizon = env_float("SEA_NAV_HYBRID_TTC_HORIZON", env_float("SEA_NAV_DWA_TTC_HORIZON", 3.0))
    safe_distance = env_float(
        "SEA_NAV_HYBRID_SAFE_DISTANCE",
        env_float(
            "SEA_NAV_DWA_SAFE_DISTANCE",
            float(getattr(getattr(env.cfg, "dynamic_obstacles", object()), "collision_distance", 0.60)),
        ),
    )
    diff = env.dynamic_obstacle_pos - env.root_states[:, None, :2]
    dist = torch.norm(diff, dim=-1).clamp(min=1e-4)
    rel_vel_robot = vel_world[:, None, :] - env.dynamic_obstacle_vel
    closing_speed = torch.sum(diff * rel_vel_robot, dim=-1) / dist
    distance_to_unsafe = (dist - safe_distance).clamp(min=0.0)
    ttc = torch.where(
        closing_speed > 0.03,
        distance_to_unsafe / closing_speed.clamp(min=1e-4),
        torch.ones_like(dist) * (horizon + 1.0),
    )
    min_ttc = torch.min(ttc, dim=1).values
    min_dist = torch.min(dist, dim=1).values
    return min_ttc, min_dist


def static_clearance_for_velocity(env, candidate):
    if not hasattr(env, "rays") or not hasattr(env, "ray_angles"):
        return torch.ones(env.num_envs, device=env.device) * 5.0
    speed = torch.norm(candidate[:, :2], dim=-1)
    angle = torch.atan2(candidate[:, 1], candidate[:, 0].clamp(min=1e-4))
    ray_angles = env.ray_angles.to(env.device)
    idx = torch.argmin(torch.abs(shortest_angle(ray_angles[None, :] - angle[:, None])), dim=1)
    ray_dist = env.rays.gather(1, idx[:, None]).squeeze(1)
    return torch.where(speed > 0.05, ray_dist, torch.ones_like(ray_dist) * 5.0)


def forward_static_clearance(env):
    if not hasattr(env, "rays") or not hasattr(env, "ray_angles"):
        return torch.ones(env.num_envs, device=env.device) * 5.0
    ray_angles = env.ray_angles.to(env.device)
    front_mask = torch.abs(ray_angles) < np.deg2rad(18.0)
    if not torch.any(front_mask):
        idx = torch.argmin(torch.abs(ray_angles))
        return env.rays[:, idx]
    return torch.min(env.rays[:, front_mask], dim=1).values


def hybrid_safety_filter_actions(env, policy_actions, emergency_policy_actions=None):
    """Keep SEA-Nav navigation behavior, but filter unsafe velocity commands around moving obstacles.

    The policy supplies the nominal local velocity command. This safety layer searches
    a small velocity set around that command and picks the closest command that stays
    outside TTC/VO danger zones and does not drive into nearby static rays.
    """
    raw = torch.clip(policy_actions, -3.0, 3.0).to(env.device)
    nav_min = getattr(env, "nav_clip_min", torch.tensor([-1.0, -1.0, -1.0], device=env.device))
    nav_max = getattr(env, "nav_clip_max", torch.tensor([1.0, 1.0, 1.0], device=env.device))
    raw = torch.clip(raw, nav_min, nav_max)

    if not hasattr(env, "_hybrid_last_action"):
        env._hybrid_last_action = torch.zeros(env.num_envs, 3, device=env.device)
        env._hybrid_state = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._hybrid_state_time = torch.zeros(env.num_envs, device=env.device)

    safe_distance = env_float("SEA_NAV_HYBRID_SAFE_DISTANCE", 0.78)
    critical_distance = env_float("SEA_NAV_HYBRID_CRITICAL_DISTANCE", 0.52)
    ttc_horizon = env_float("SEA_NAV_HYBRID_TTC_HORIZON", 3.0)
    stop_ttc = env_float("SEA_NAV_HYBRID_STOP_TTC", 1.05)
    slow_ttc = env_float("SEA_NAV_HYBRID_SLOW_TTC", 2.2)
    dynamic_rollout_horizon = env_float("SEA_NAV_HYBRID_DYNAMIC_ROLLOUT_HORIZON", 4.0)
    static_rollout_horizon = env_float("SEA_NAV_HYBRID_STATIC_ROLLOUT_HORIZON", 1.8)
    rollout_dt = env_float("SEA_NAV_HYBRID_ROLLOUT_DT", 0.15)
    crossing_commit_time = env_float("SEA_NAV_HYBRID_CROSSING_COMMIT_TIME", 1.8)
    max_wait_time = env_float("SEA_NAV_HYBRID_MAX_WAIT_TIME", 0.8)
    crossing_speed = env_float("SEA_NAV_HYBRID_CROSSING_SPEED", 0.35)
    front_ray_clearance = env_float("SEA_NAV_HYBRID_FRONT_RAY_CLEARANCE", 0.48)
    max_lateral_speed = env_float("SEA_NAV_HYBRID_MAX_LATERAL_SPEED", 0.45)
    escape_yaw_rate = env_float("SEA_NAV_HYBRID_ESCAPE_YAW_RATE", 0.65)

    policy_weight = env_float("SEA_NAV_HYBRID_POLICY_WEIGHT", 3.0)
    progress_weight = env_float("SEA_NAV_HYBRID_PROGRESS_WEIGHT", 1.4)
    ttc_weight = env_float("SEA_NAV_HYBRID_TTC_WEIGHT", 7.0)
    clearance_weight = env_float("SEA_NAV_HYBRID_CLEARANCE_WEIGHT", 4.5)
    static_weight = env_float("SEA_NAV_HYBRID_STATIC_WEIGHT", 2.0)
    wait_bonus = env_float("SEA_NAV_HYBRID_WAIT_BONUS", 2.0)
    smoothness_weight = env_float("SEA_NAV_HYBRID_SMOOTHNESS_WEIGHT", 0.5)

    yaw = yaw_from_xyzw(env.base_quat)
    goal_world = env.position_targets[:, :2] - env.root_states[:, :2]
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    goal_local = torch.stack(
        (
            c * goal_world[:, 0] - s * goal_world[:, 1],
            s * goal_world[:, 0] + c * goal_world[:, 1],
        ),
        dim=-1,
    )
    goal_dist = torch.norm(goal_local, dim=-1).clamp(min=1e-4)
    goal_dir = goal_local / goal_dist[:, None]
    near_goal = goal_dist < 0.45

    blocked_hint = getattr(env, "dynamic_path_blocked", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    raw_vel_world = rotate_local_to_world(raw[:, :2], yaw)
    raw_min_ttc, raw_min_dyn_dist = dynamic_ttc_for_velocity(env, raw_vel_world)
    front_dynamic = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "dynamic_nearest_rel_pos_local"):
        nearest_local = env.dynamic_nearest_rel_pos_local
        along_goal = torch.sum(nearest_local * goal_dir, dim=-1)
        lateral_to_goal = torch.abs(goal_dir[:, 0] * nearest_local[:, 1] - goal_dir[:, 1] * nearest_local[:, 0])
        front_dynamic = (along_goal > 0.0) & (along_goal < 3.2) & (lateral_to_goal < 0.75)
    # The policy is already good at static navigation. Keep the filter as a
    # dynamic-obstacle safety override instead of letting the path-blocked hint
    # suppress nominal navigation whenever an obstacle is merely nearby.
    dynamic_risk = front_dynamic & (raw_min_dyn_dist < 1.20) & ((raw_min_ttc < slow_ttc) | (raw_min_dyn_dist < safe_distance))
    critical_dynamic_risk = front_dynamic & (
        (raw_min_dyn_dist < critical_distance)
        | ((raw_min_dyn_dist < 0.85) & (raw_min_ttc < stop_ttc))
    )
    stop_required = critical_dynamic_risk
    front_raw_clearance = forward_static_clearance(env)
    front_static_risk = (front_raw_clearance < min(front_ray_clearance, 0.22)) & (raw[:, 0] > 0.08)
    monitor_dynamic = dynamic_risk | blocked_hint | (env._hybrid_state != 0)
    if not torch.any(monitor_dynamic):
        env._hybrid_state = torch.zeros_like(env._hybrid_state)
        env._hybrid_state_time = torch.zeros_like(env._hybrid_state_time)
        env._hybrid_last_action = raw
        return raw

    # If the nominal SEA-Nav command is predicted to keep a safe future
    # clearance, do not optimize it away. This preserves the trained static
    # navigation behavior and makes the filter a true safety layer.
    raw_rollout_safe = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(env.num_envs):
        if dynamic_risk[env_id]:
            raw_rollout_safe[env_id] = is_dynamic_trajectory_rollout_safe(
                env,
                env_id,
                raw[env_id],
                horizon=dynamic_rollout_horizon,
                dt=rollout_dt,
                safe_distance=safe_distance,
            )
    raw_rollout_safe = raw_rollout_safe & (~critical_dynamic_risk)
    dynamic_risk = dynamic_risk & (~raw_rollout_safe)
    monitor_dynamic = dynamic_risk | blocked_hint | (env._hybrid_state != 0)
    if not torch.any(monitor_dynamic):
        env._hybrid_state = torch.zeros_like(env._hybrid_state)
        env._hybrid_state_time = torch.zeros_like(env._hybrid_state_time)
        env._hybrid_last_action = raw
        return raw

    if os.environ.get("SEA_NAV_HYBRID_SIMPLE_SPEED_FILTER", "1") == "1":
        action = raw.clone()
        soft_risk = dynamic_risk
        hard_risk = front_dynamic & (
            (raw_min_dyn_dist < critical_distance)
            | ((raw_min_dyn_dist < 0.85) & (raw_min_ttc < stop_ttc))
        )
        emergency_trigger_distance = env_float("SEA_NAV_EMERGENCY_POLICY_TRIGGER_DISTANCE", critical_distance)
        emergency_trigger_ttc = env_float("SEA_NAV_EMERGENCY_POLICY_TRIGGER_TTC", stop_ttc)
        emergency_mask = front_dynamic & (
            (raw_min_dyn_dist < emergency_trigger_distance)
            | ((raw_min_dyn_dist < safe_distance) & (raw_min_ttc < emergency_trigger_ttc))
        )
        if emergency_policy_actions is not None:
            emergency_mask = emergency_mask | hard_risk
        linear_scale = torch.ones(env.num_envs, device=env.device)
        soft_scale = env_float("SEA_NAV_HYBRID_SOFT_SCALE", 0.65)
        linear_scale = torch.where(soft_risk, torch.ones_like(linear_scale) * soft_scale, linear_scale)
        if emergency_policy_actions is None:
            linear_scale = torch.where(hard_risk, torch.ones_like(linear_scale) * 0.05, linear_scale)
        action[:, :2] = action[:, :2] * linear_scale[:, None]
        if emergency_policy_actions is None:
            reverse_escape = torch.zeros_like(action)
            reverse_escape[:, 0] = -0.25
            reverse_escape[:, 2] = action[:, 2] * 0.25
            action = torch.where(hard_risk[:, None], reverse_escape, action)
        if emergency_policy_actions is not None:
            emergency = torch.clip(emergency_policy_actions.to(env.device), nav_min, nav_max)
            blend = float(np.clip(env_float("SEA_NAV_EMERGENCY_POLICY_BLEND", 1.0), 0.0, 1.0))
            mixed_emergency = blend * emergency + (1.0 - blend) * action
            action = torch.where(emergency_mask[:, None], mixed_emergency, action)
        env._hybrid_state = torch.zeros_like(env._hybrid_state)
        env._hybrid_state_time = torch.zeros_like(env._hybrid_state_time)
        env._hybrid_last_action = action
        if int(getattr(env, "_hybrid_filter_print_step", -1000000)) + 250 < int(env.common_step_counter):
            filtered_ratio = torch.mean(soft_risk.float()).item()
            hard_ratio = torch.mean(hard_risk.float()).item()
            emergency_ratio = torch.mean(emergency_mask.float()).item() if emergency_policy_actions is not None else 0.0
            env._hybrid_filter_print_step = int(env.common_step_counter)
            print(f"[N1 hybrid] speed_filter={filtered_ratio:.2f} hard_stop={hard_ratio:.2f} rl_emergency={emergency_ratio:.2f}")
        return action
    gap_open = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id in range(env.num_envs):
        if monitor_dynamic[env_id] and not near_goal[env_id]:
            gap_open[env_id] = has_safe_crossing_window(
                env,
                env_id,
                goal_dir,
                raw,
                horizon=dynamic_rollout_horizon,
                dt=rollout_dt,
            )

    HYBRID_GO = 0
    HYBRID_WAIT = 1
    HYBRID_CROSSING = 2
    reset_mask = env.episode_length_buf <= 1
    env._hybrid_state = torch.where(reset_mask, torch.zeros_like(env._hybrid_state), env._hybrid_state)
    env._hybrid_state_time = torch.where(reset_mask, torch.zeros_like(env._hybrid_state_time), env._hybrid_state_time)
    previous_state = env._hybrid_state.clone()
    state = env._hybrid_state.clone()
    for env_id in range(env.num_envs):
        if near_goal[env_id]:
            state[env_id] = HYBRID_GO
        elif previous_state[env_id] == HYBRID_WAIT:
            if gap_open[env_id]:
                state[env_id] = HYBRID_CROSSING
            elif blocked_hint[env_id] or stop_required[env_id] or (
                monitor_dynamic[env_id] and env._hybrid_state_time[env_id] < max_wait_time
            ):
                state[env_id] = HYBRID_WAIT
            else:
                state[env_id] = HYBRID_GO
        elif previous_state[env_id] == HYBRID_CROSSING:
            if env._hybrid_state_time[env_id] < crossing_commit_time and not critical_dynamic_risk[env_id]:
                state[env_id] = HYBRID_CROSSING
            elif (blocked_hint[env_id] or stop_required[env_id]) and not gap_open[env_id]:
                state[env_id] = HYBRID_WAIT
            else:
                state[env_id] = HYBRID_GO
        elif (blocked_hint[env_id] or stop_required[env_id]) and not gap_open[env_id]:
            state[env_id] = HYBRID_WAIT
        elif dynamic_risk[env_id] and gap_open[env_id]:
            state[env_id] = HYBRID_CROSSING
        else:
            state[env_id] = HYBRID_GO
    env._hybrid_state_time = torch.where(
        state == previous_state,
        env._hybrid_state_time + env.dt,
        torch.zeros_like(env._hybrid_state_time),
    )
    env._hybrid_state = state
    # Use the explicit wait/crossing state machine downstream in candidate
    # scoring. This keeps the robot at a holding point until a full crossing
    # window opens, instead of re-deciding every frame and dithering.
    waiting_state = state == HYBRID_WAIT
    crossing_state = state == HYBRID_CROSSING

    rel_y = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, "dynamic_nearest_rel_pos_local"):
        rel_y = env.dynamic_nearest_rel_pos_local[:, 1]
    side_sign = torch.where(rel_y >= 0.0, torch.ones_like(rel_y), -torch.ones_like(rel_y))
    side_away = -side_sign

    zero = torch.zeros_like(raw)
    candidates = [
        raw,
        raw * torch.tensor([0.75, 0.75, 0.75], device=env.device),
        raw * torch.tensor([0.50, 0.50, 0.65], device=env.device),
        raw * torch.tensor([0.25, 0.25, 0.45], device=env.device),
        torch.stack((torch.zeros(env.num_envs, device=env.device), torch.zeros(env.num_envs, device=env.device), raw[:, 2] * 0.5), dim=-1),
        zero,
    ]
    lateral_mag = torch.ones(env.num_envs, device=env.device) * max_lateral_speed
    candidates.extend(
        [
            torch.stack((torch.ones(env.num_envs, device=env.device) * crossing_speed, torch.zeros(env.num_envs, device=env.device), raw[:, 2] * 0.5), dim=-1),
            torch.stack((torch.ones(env.num_envs, device=env.device) * crossing_speed, side_away * lateral_mag * 0.35, raw[:, 2] * 0.5), dim=-1),
            torch.stack((torch.ones(env.num_envs, device=env.device) * crossing_speed, -side_away * lateral_mag * 0.35, raw[:, 2] * 0.5), dim=-1),
            torch.stack((raw[:, 0].clamp(max=0.25), side_away * lateral_mag, raw[:, 2]), dim=-1),
            torch.stack((raw[:, 0].clamp(max=0.18), -side_away * lateral_mag, raw[:, 2]), dim=-1),
            torch.stack((torch.zeros(env.num_envs, device=env.device), side_away * lateral_mag * 0.75, side_away * escape_yaw_rate), dim=-1),
            torch.stack((torch.zeros(env.num_envs, device=env.device), -side_away * lateral_mag * 0.75, -side_away * escape_yaw_rate), dim=-1),
        ]
    )

    best_score = torch.ones(env.num_envs, device=env.device) * -1.0e9
    best_action = zero.clone()
    last_action = env._hybrid_last_action

    for cand in candidates:
        action = torch.clip(cand, nav_min, nav_max)
        action = torch.where(near_goal[:, None], torch.zeros_like(action), action)
        waiting_action = torch.stack(
            (
                torch.zeros(env.num_envs, device=env.device),
                action[:, 1] * 0.15,
                action[:, 2] * 0.35,
            ),
            dim=-1,
        )
        action = torch.where(waiting_state[:, None], waiting_action, action)
        forward_floor = torch.ones(env.num_envs, device=env.device) * crossing_speed
        crossing_action = torch.stack(
            (
                torch.maximum(action[:, 0], forward_floor),
                action[:, 1],
                action[:, 2],
            ),
            dim=-1,
        )
        action = torch.where(crossing_state[:, None], crossing_action, action)
        vel_world = rotate_local_to_world(action[:, :2], yaw)
        min_ttc, min_dyn_dist = dynamic_ttc_for_velocity(env, vel_world)
        ray_clearance = static_clearance_for_velocity(env, action)
        front_clearance = forward_static_clearance(env)

        hard_safe = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for env_id in range(env.num_envs):
            dynamic_safe = True
            if dynamic_risk[env_id]:
                dynamic_safe = is_dynamic_trajectory_rollout_safe(
                    env,
                    env_id,
                    action[env_id],
                    horizon=dynamic_rollout_horizon,
                    dt=rollout_dt,
                    safe_distance=safe_distance,
                )
            static_safe = True
            action_changed = torch.norm(action[env_id] - raw[env_id]).item() > 0.08
            if action_changed or front_static_risk[env_id]:
                static_safe = is_static_footprint_rollout_safe(
                    env,
                    env_id,
                    action[env_id],
                    horizon=static_rollout_horizon,
                    dt=rollout_dt,
                )
            hard_safe[env_id] = dynamic_safe and static_safe

        speed = torch.norm(action[:, :2], dim=-1)
        serious_dynamic_risk = (min_dyn_dist < critical_distance) | ((min_ttc < stop_ttc) & (speed > 0.08))
        static_front_risk = (front_clearance < front_ray_clearance) & (action[:, 0] > 0.08)
        hard_safe = hard_safe & (~serious_dynamic_risk)

        closing_risk = torch.relu((ttc_horizon - min_ttc) / ttc_horizon)
        slow_zone = (min_ttc < slow_ttc) | dynamic_risk
        dyn_clearance_penalty = torch.relu((safe_distance - min_dyn_dist) / safe_distance)
        static_penalty = torch.relu((front_ray_clearance - ray_clearance) / front_ray_clearance)
        wait_score = (stop_required | waiting_state).float() * torch.exp(-torch.square(speed / 0.14))
        policy_deviation = torch.sum(torch.square(action - raw), dim=-1)
        progress_score = torch.sum(action[:, :2] * goal_dir, dim=-1)
        progress_score = torch.where(crossing_state, progress_score + 0.55 * torch.norm(action[:, :2], dim=-1), progress_score)
        progress_score = torch.where(waiting_state, progress_score - 0.8 * action[:, 0].clamp(min=0.0), progress_score)
        smoothness = torch.sum(torch.square(action - last_action), dim=-1)

        score = (
            -policy_weight * policy_deviation
            + progress_weight * progress_score
            - ttc_weight * closing_risk
            - clearance_weight * dyn_clearance_penalty
            - static_weight * static_penalty
            + wait_bonus * wait_score
            - smoothness_weight * smoothness
        )
        score = torch.where(hard_safe, score, torch.ones_like(score) * -1.0e8)
        improve = score > best_score
        best_score = torch.where(improve, score, best_score)
        best_action = torch.where(improve[:, None], action, best_action)

    no_safe_motion = best_score < -1.0e7
    emergency_action = torch.stack(
        (
            torch.zeros(env.num_envs, device=env.device),
            side_away * lateral_mag * 0.75,
            side_away * escape_yaw_rate,
        ),
        dim=-1,
    )
    emergency_reverse = torch.stack(
        (
            -torch.ones(env.num_envs, device=env.device) * 0.25,
            side_away * lateral_mag,
            side_away * escape_yaw_rate,
        ),
        dim=-1,
    )
    emergency_action = torch.clip(emergency_action, nav_min, nav_max)
    emergency_reverse = torch.clip(emergency_reverse, nav_min, nav_max)
    best_action = torch.where(no_safe_motion[:, None], emergency_action, best_action)
    best_action = torch.where((no_safe_motion & critical_dynamic_risk)[:, None], emergency_reverse, best_action)
    env._hybrid_last_action = best_action
    if int(getattr(env, "_hybrid_filter_print_step", -1000000)) + 250 < int(env.common_step_counter):
        filtered_ratio = torch.mean((torch.norm(best_action - raw, dim=-1) > 0.05).float()).item()
        env._hybrid_filter_print_step = int(env.common_step_counter)
        wait_ratio = torch.mean(waiting_state.float()).item()
        crossing_ratio = torch.mean(crossing_state.float()).item()
        print(f"[N1 hybrid] adjusted={filtered_ratio:.2f} wait={wait_ratio:.2f} crossing={crossing_ratio:.2f}")
    return best_action


def dwa_pipeline_actions(env):
    control_period_s = env_float("SEA_NAV_DWA_CONTROL_PERIOD", 0.12)
    control_steps = max(1, int(round(control_period_s / max(float(env.dt), 1e-6))))
    current_steps = env.episode_length_buf.clone()
    if not hasattr(env, "_pipeline_last_action"):
        env._pipeline_last_action = torch.zeros(env.num_envs, 3, device=env.device)
        env._pipeline_last_control_step = torch.ones(env.num_envs, dtype=torch.long, device=env.device) * -10**9
    due = (current_steps <= 1) | ((current_steps - env._pipeline_last_control_step) >= control_steps)
    if not torch.any(due):
        return env._pipeline_last_action

    preferred_speed = env_float("SEA_NAV_DWA_PREFERRED_SPEED", 0.50)
    wait_speed = env_float("SEA_NAV_DWA_WAIT_SPEED", 0.00)
    max_lateral_speed = env_float("SEA_NAV_DWA_MAX_LATERAL_SPEED", 0.45)
    max_yaw_rate = env_float("SEA_NAV_DWA_MAX_YAW_RATE", 0.85)
    ttc_horizon = env_float("SEA_NAV_DWA_TTC_HORIZON", 3.0)
    static_clearance = env_float("SEA_NAV_DWA_STATIC_CLEARANCE", 0.65)

    goal_weight = env_float("SEA_NAV_DWA_GOAL_WEIGHT", 4.0)
    velocity_weight = env_float("SEA_NAV_DWA_VELOCITY_WEIGHT", 1.2)
    ttc_weight = env_float("SEA_NAV_DWA_TTC_WEIGHT", 5.0)
    clearance_weight = env_float("SEA_NAV_DWA_CLEARANCE_WEIGHT", 3.5)
    yaw_weight = env_float("SEA_NAV_DWA_YAW_WEIGHT", 0.8)
    smoothness_weight = env_float("SEA_NAV_DWA_SMOOTHNESS_WEIGHT", 0.4)
    wait_bonus = env_float("SEA_NAV_DWA_WAIT_BONUS", 1.8)

    waypoint_world = torch.zeros(env.num_envs, 2, device=env.device)
    for env_id in range(env.num_envs):
        waypoint_world[env_id] = torch.tensor(get_local_waypoint_world(env, env_id), device=env.device)
    waypoint_vec_world = waypoint_world - env.root_states[:, :2]
    yaw_for_goal = yaw_from_xyzw(env.base_quat)
    c, s = torch.cos(-yaw_for_goal), torch.sin(-yaw_for_goal)
    goal_local = torch.stack(
        (
            c * waypoint_vec_world[:, 0] - s * waypoint_vec_world[:, 1],
            s * waypoint_vec_world[:, 0] + c * waypoint_vec_world[:, 1],
        ),
        dim=-1,
    )

    goal_dist = torch.norm(goal_local, dim=-1).clamp(min=1e-4)
    final_goal_dist = torch.norm(env.position_targets[:, :2] - env.root_states[:, :2], dim=-1)
    goal_dir_local = goal_local / goal_dist[:, None]
    near_goal = final_goal_dist < 0.55
    blocked = getattr(env, "dynamic_path_blocked", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    vx_values = torch.tensor([0.0, 0.22, preferred_speed], device=env.device)
    vy_values = torch.tensor([-max_lateral_speed, 0.0, max_lateral_speed], device=env.device)
    wz_values = torch.tensor([-max_yaw_rate, 0.0, max_yaw_rate], device=env.device)
    candidates = torch.cartesian_prod(vx_values, vy_values, wz_values).to(env.device)
    candidates = torch.cat((torch.tensor([[wait_speed, 0.0, 0.0]], device=env.device), candidates), dim=0)

    yaw = yaw_from_xyzw(env.base_quat)
    target_vel = goal_dir_local * torch.where(blocked, torch.ones_like(goal_dist) * wait_speed, torch.ones_like(goal_dist) * preferred_speed)[:, None]
    target_vel = torch.where(near_goal[:, None], torch.zeros_like(target_vel), target_vel)
    target_heading = torch.atan2(goal_local[:, 1], goal_local[:, 0])
    desired_yaw_rate = target_heading.clamp(min=-max_yaw_rate, max=max_yaw_rate)
    last_action = getattr(env, "nav_actions_after_clip", torch.zeros(env.num_envs, 3, device=env.device))

    best_score = torch.ones(env.num_envs, device=env.device) * -1.0e9
    best_action = torch.zeros(env.num_envs, 3, device=env.device)
    for cand in candidates:
        action = cand[None, :].repeat(env.num_envs, 1)
        action = torch.where(near_goal[:, None], torch.zeros_like(action), action)
        hard_safe = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for env_id in range(env.num_envs):
            hard_safe[env_id] = is_static_rollout_safe(env, env_id, action[env_id]) and is_dynamic_rollout_safe(env, env_id, action[env_id])
        vel_world = rotate_local_to_world(action[:, :2], yaw)
        min_ttc, min_dyn_dist = dynamic_ttc_for_velocity(env, vel_world)
        ray_clearance = static_clearance_for_velocity(env, action)

        progress_score = torch.sum(action[:, :2] * goal_dir_local, dim=-1)
        velocity_score = -torch.square(torch.norm(action[:, :2] - target_vel, dim=-1))
        ttc_penalty = torch.relu((ttc_horizon - min_ttc) / ttc_horizon)
        dyn_clearance_penalty = torch.relu((static_clearance - min_dyn_dist) / static_clearance)
        static_penalty = torch.relu((static_clearance - ray_clearance) / static_clearance)
        yaw_score = -torch.square(desired_yaw_rate - action[:, 2])
        smoothness = -torch.square(torch.norm(action - last_action, dim=-1))
        wait_score = (blocked | (min_ttc < 1.5)).float() * torch.exp(-torch.square(torch.norm(action[:, :2], dim=-1) / 0.12))

        score = (
            goal_weight * progress_score
            + velocity_weight * velocity_score
            - ttc_weight * ttc_penalty
            - clearance_weight * (dyn_clearance_penalty + static_penalty)
            + yaw_weight * yaw_score
            + smoothness_weight * smoothness
            + wait_bonus * wait_score
        )
        score = torch.where(hard_safe, score, torch.ones_like(score) * -1.0e8)
        improve = score > best_score
        best_score = torch.where(improve, score, best_score)
        best_action = torch.where(improve[:, None], action, best_action)

    no_safe_motion = best_score < -1.0e7
    best_action = torch.where(no_safe_motion[:, None], torch.zeros_like(best_action), best_action)
    env._pipeline_last_action = torch.where(due[:, None], best_action, env._pipeline_last_action)
    env._pipeline_last_control_step = torch.where(due, current_steps, env._pipeline_last_control_step)
    return env._pipeline_last_action


def set_recording_camera(env, camera_handle):
    """Set the recording camera.

    The default follows the robot from a low oblique angle.  For navigation
    demos we often want the same top-down view used for debugging dynamic
    obstacle interactions, controlled by SEA_NAV_RECORD_CAMERA_MODE=topdown.
    """
    robot_pos = env.root_states[0, :3].detach().cpu().numpy()
    goal_pos = env.position_targets[0, :3].detach().cpu().numpy()

    if os.environ.get("SEA_NAV_RECORD_CAMERA_MODE", "").lower() in {"topdown", "overhead"}:
        if hasattr(env, "terrain_levels") and hasattr(env, "terrain_types") and hasattr(env, "terrain"):
            row = int(env.terrain_levels[0].item())
            col = int(env.terrain_types[0].item())
            center_xy = np.array(
                [
                    (row + 0.5) * float(env.terrain.env_length),
                    (col + 0.5) * float(env.terrain.env_width),
                ],
                dtype=np.float32,
            )
        elif hasattr(env, "env_origins"):
            center_xy = env.env_origins[0, :2].detach().cpu().numpy()
        else:
            center_xy = 0.5 * (robot_pos[:2] + goal_pos[:2])
        camera_height = float(os.environ.get("SEA_NAV_RECORD_TOPDOWN_HEIGHT", "11.5"))
        camera_pos = gymapi.Vec3(float(center_xy[0]), float(center_xy[1]), camera_height)
        camera_target = gymapi.Vec3(float(center_xy[0]), float(center_xy[1] + 0.003), 0.0)
        env.gym.set_camera_location(camera_handle, env.envs[0], camera_pos, camera_target)
        return

    direction = goal_pos[:2] - robot_pos[:2]
    distance = np.linalg.norm(direction)
    if distance < 1e-3:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / distance

    side = np.array([-direction[1], direction[0]])
    camera_xy = robot_pos[:2] - 3.0 * direction + 1.2 * side
    lookahead = np.clip(distance * 0.45, 1.2, 4.5)
    lookat_xy = robot_pos[:2] + lookahead * direction

    camera_pos = gymapi.Vec3(float(camera_xy[0]), float(camera_xy[1]), float(robot_pos[2] + 2.0))
    camera_target = gymapi.Vec3(float(lookat_xy[0]), float(lookat_xy[1]), float(robot_pos[2] + 0.45))
    env.gym.set_camera_location(camera_handle, env.envs[0], camera_pos, camera_target)


def project_world_to_image(env, camera_handle, point_world, width, height):
    view = np.array(env.gym.get_camera_view_matrix(env.sim, env.envs[0], camera_handle), dtype=np.float32).reshape(4, 4)
    proj = np.array(env.gym.get_camera_proj_matrix(env.sim, env.envs[0], camera_handle), dtype=np.float32).reshape(4, 4)
    point = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float32)

    candidates = [
        proj @ view @ point,
        point @ view @ proj,
        proj @ view.T @ point,
        point @ view.T @ proj,
    ]
    projected = []
    for clip in candidates:
        if abs(float(clip[3])) < 1e-6:
            continue
        ndc = clip[:3] / clip[3]
        if not np.all(np.isfinite(ndc)):
            continue
        x = int((float(ndc[0]) + 1.0) * 0.5 * width)
        y = int((1.0 - float(ndc[1])) * 0.5 * height)
        if -width <= x <= 2 * width and -height <= y <= 2 * height:
            projected.append((x, y))
    if not projected:
        return None
    return min(projected, key=lambda item: (item[0] - width // 2) ** 2 + (item[1] - height // 2) ** 2)


def overlay_navigation_target(img_bgr, env, camera_handle, width, height):
    robot_pos = env.root_states[0, :3].detach().cpu().numpy()
    goal_pos = env.position_targets[0, :3].detach().cpu().numpy()
    distance = float(np.linalg.norm(goal_pos[:2] - robot_pos[:2]))
    marker = project_world_to_image(env, camera_handle, [goal_pos[0], goal_pos[1], goal_pos[2] + 0.45], width, height)
    if marker is None:
        marker = (width // 2, 72)
    x = int(np.clip(marker[0], 40, width - 40))
    y = int(np.clip(marker[1], 40, height - 40))
    color = (40, 255, 40)
    cv2.circle(img_bgr, (x, y), 24, color, 4)
    cv2.drawMarker(img_bgr, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=34, thickness=3)
    cv2.putText(img_bgr, f"TARGET {distance:.1f}m", (max(10, x - 92), max(32, y - 34)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def overlay_navigation_target_gui_style(img_bgr, env, camera_handle, width, height):
    """Draw a compact target marker without text for GUI-style recordings."""
    goal_pos = env.position_targets[0, :3].detach().cpu().numpy()
    marker = project_world_to_image(
        env,
        camera_handle,
        [float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2] + 0.45)],
        width,
        height,
    )
    if marker is None:
        return
    x = int(np.clip(marker[0], 28, width - 28))
    y = int(np.clip(marker[1], 28, height - 28))
    color = (40, 255, 40)
    cv2.circle(img_bgr, (x, y), 22, color, 4, cv2.LINE_AA)
    cv2.drawMarker(img_bgr, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=34, thickness=4, line_type=cv2.LINE_AA)


def overlay_dynamic_obstacles(img_bgr, env, camera_handle, width, height):
    if not hasattr(env, "dynamic_obstacle_pos"):
        return

    obstacle_pos = env.dynamic_obstacle_pos[0].detach().cpu().numpy()
    obstacle_vel = env.dynamic_obstacle_vel[0].detach().cpu().numpy()
    robot_z = float(env.root_states[0, 2].detach().cpu().item())
    colors = [(0, 165, 255), (0, 220, 255), (0, 100, 255)]
    motion_modes = []
    if hasattr(env, "cfg") and hasattr(env.cfg, "dynamic_obstacles"):
        motion_modes = list(getattr(env.cfg.dynamic_obstacles, "motion_modes", []))
    mode_alias = {
        "pedestrian_like": "PED",
        "back_and_forth": "BACK",
        "random_rigid_body": "RAND",
    }
    for idx, (pos_xy, vel_xy) in enumerate(zip(obstacle_pos, obstacle_vel)):
        marker = project_world_to_image(
            env,
            camera_handle,
            [float(pos_xy[0]), float(pos_xy[1]), robot_z + 0.55],
            width,
            height,
        )
        if marker is None:
            continue
        x = int(np.clip(marker[0], 32, width - 32))
        y = int(np.clip(marker[1], 32, height - 32))
        color = colors[idx % len(colors)]
        mode_name = motion_modes[idx % len(motion_modes)] if motion_modes else f"dyn_{idx + 1}"
        mode_text = mode_alias.get(mode_name, mode_name.upper())
        cv2.circle(img_bgr, (x, y), 18, color, -1)
        cv2.circle(img_bgr, (x, y), 24, (255, 255, 255), 2)
        cv2.putText(
            img_bgr,
            mode_text,
            (max(8, x - 42), max(24, y - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        speed = float(np.linalg.norm(vel_xy))
        cv2.putText(
            img_bgr,
            f"#{idx + 1} {speed:.1f}m/s",
            (max(8, x - 38), min(height - 12, y + 42)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )


def _project_polyline(env, camera_handle, points_world, width, height):
    projected = []
    for point in points_world:
        marker = project_world_to_image(env, camera_handle, point, width, height)
        if marker is None:
            continue
        x = int(np.clip(marker[0], -width, 2 * width))
        y = int(np.clip(marker[1], -height, 2 * height))
        projected.append((x, y))
    return projected


def _draw_projected_polyline(img_bgr, points, color, closed=False, thickness=3):
    if len(points) < 2:
        return
    if closed:
        points = points + [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        cv2.line(img_bgr, start, end, color, thickness, cv2.LINE_AA)


def overlay_dynamic_obstacle_gui_style(img_bgr, env, camera_handle, width, height):
    """Draw the same kind of trajectory cues used in the GUI debug view.

    Isaac Gym viewer debug lines are not captured by camera sensors in
    headless recording, so we redraw only clean trajectory geometry here:
    no text labels, no large annotations, just colored paths and current
    obstacle markers.
    """
    if not hasattr(env, "dynamic_obstacle_pos"):
        return

    env_idx = 0
    base_z = float(env.root_states[env_idx, 2].detach().cpu().item() + 0.42)
    motion_modes = []
    if hasattr(env, "cfg") and hasattr(env.cfg, "dynamic_obstacles"):
        motion_modes = list(getattr(env.cfg.dynamic_obstacles, "motion_modes", []))
    mode_colors = {
        "pedestrian_like": (242, 242, 0),
        "back_and_forth": (255, 90, 50),
        "random_rigid_body": (0, 115, 255),
    }
    default_color = (0, 180, 255)

    for obs_idx in range(int(getattr(env, "num_dynamic_obstacles", 0))):
        mode = motion_modes[obs_idx % len(motion_modes)] if motion_modes else "dynamic"
        color = mode_colors.get(mode, default_color)
        pos_xy = env.dynamic_obstacle_pos[env_idx, obs_idx].detach().cpu().numpy()
        base_xy = env.dynamic_obstacle_base[env_idx, obs_idx].detach().cpu().numpy()
        axis_xy = env.dynamic_obstacle_axis[env_idx, obs_idx].detach().cpu().numpy()
        perp_xy = env.dynamic_obstacle_perp[env_idx, obs_idx].detach().cpu().numpy()
        amp = float(env.dynamic_obstacle_amp[env_idx, obs_idx].detach().cpu().item())

        if mode == "pedestrian_like":
            path_xy = [base_xy - amp * perp_xy, base_xy + amp * perp_xy]
            path_world = [[float(x), float(y), base_z] for x, y in path_xy]
            _draw_projected_polyline(
                img_bgr,
                _project_polyline(env, camera_handle, path_world, width, height),
                color,
                closed=False,
                thickness=3,
            )
        elif mode == "back_and_forth":
            path_world = []
            for theta in np.linspace(0.0, 2.0 * math.pi, 49, endpoint=False):
                curve_xy = (
                    base_xy
                    + amp * math.sin(theta) * axis_xy
                    + float(getattr(env, "BACKTRACK_LATERAL_SCALE", 0.30)) * amp * math.sin(2.0 * theta) * perp_xy
                )
                path_world.append([float(curve_xy[0]), float(curve_xy[1]), base_z])
            _draw_projected_polyline(
                img_bgr,
                _project_polyline(env, camera_handle, path_world, width, height),
                color,
                closed=True,
                thickness=3,
            )
        else:
            path_world = []
            axis_scale = float(getattr(env, "RAND_AXIS_SCALE", 1.05))
            perp_scale = float(getattr(env, "RAND_PERP_SCALE", 0.65))
            for theta in np.linspace(0.0, 2.0 * math.pi, 65, endpoint=False):
                orbit_xy = (
                    base_xy
                    + axis_scale * amp * math.cos(theta) * axis_xy
                    + perp_scale * amp * math.sin(theta) * perp_xy
                )
                path_world.append([float(orbit_xy[0]), float(orbit_xy[1]), base_z])
            _draw_projected_polyline(
                img_bgr,
                _project_polyline(env, camera_handle, path_world, width, height),
                color,
                closed=True,
                thickness=3,
            )

        marker = project_world_to_image(
            env,
            camera_handle,
            [float(pos_xy[0]), float(pos_xy[1]), base_z + 0.25],
            width,
            height,
        )
        if marker is None:
            continue
        x = int(np.clip(marker[0], 24, width - 24))
        y = int(np.clip(marker[1], 24, height - 24))
        cv2.circle(img_bgr, (x, y), 13, color, 3, cv2.LINE_AA)
        cv2.circle(img_bgr, (x, y), 6, color, -1, cv2.LINE_AA)

    
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # overwrite some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    
    env_cfg.terrain.terrain_types = ['hard_room']  
    env_cfg.terrain.terrain_proportions = [1.0]
    env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description.urdf'
    env_cfg.replay.enable_collision_replay = False
    
    env_cfg.visualization.ray_groups = {
            # "all": [None, "ray_pink"],
            "guidance_navigation": ["guide", "guide_ray_marker"],
        }
    
    if env_cfg.env.num_envs == 1:
        env_cfg.terrain.num_rows = 1 # level  
        env_cfg.terrain.num_cols = 1 # type
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.max_init_terrain_level = 3
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0, 0]
    env_cfg.env.episode_length_s = float(os.environ.get("SEA_NAV_EPISODE_LENGTH_S", "40"))
    env_cfg.env.stay_time = 500
    env_cfg.env.debug_viz = True
    env_cfg.asset.terminate_after_contacts_on = [] # no termination

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    controller = os.environ.get("SEA_NAV_PLAY_CONTROLLER", "policy").lower()
    policy = None
    emergency_policy = None
    policy_uses_stripped_obs = False
    use_hybrid_safety_filter = False
    emergency_checkpoint = os.environ.get("SEA_NAV_EMERGENCY_POLICY_CHECKPOINT", "").strip()
    if controller in {"policy", "hybrid", "hybrid_safety", "hybrid_safety_filter"}:
        if emergency_checkpoint:
            if getattr(args, "experiment_name", None) is not None:
                train_cfg.runner.experiment_name = args.experiment_name
            baseline_path = resolve_policy_checkpoint(train_cfg, getattr(args, "load_run", None), getattr(args, "checkpoint", None))
            if baseline_path is None:
                raise ValueError("Emergency hybrid mode requires a baseline --load_run and --checkpoint.")
            if not os.path.exists(emergency_checkpoint):
                raise FileNotFoundError(f"Emergency policy checkpoint not found: {emergency_checkpoint}")
            policy = load_inference_policy_from_checkpoint(
                env,
                train_cfg,
                baseline_path,
                num_dynamic_obstacle_obs=0,
            )
            emergency_policy = load_inference_policy_from_checkpoint(
                env,
                train_cfg,
                emergency_checkpoint,
                num_dynamic_obstacle_obs=int(getattr(env.cfg.env, "num_dynamic_obstacle_obs", 0)),
            )
            policy_uses_stripped_obs = int(getattr(env.cfg.env, "num_dynamic_obstacle_obs", 0)) > 0
            train_cfg.runner.load_run = getattr(args, "load_run", train_cfg.runner.load_run)
            train_cfg.runner.checkpoint = getattr(args, "checkpoint", train_cfg.runner.checkpoint)
            print("Using baseline SEA-Nav policy + low-risk speed filter + RL emergency avoidance policy.")
        else:
            # load policy
            train_cfg.runner.resume = True
            if getattr(args, "load_run", None) is None:
                train_cfg.runner.load_run = -1
            if getattr(args, "checkpoint", None) is None:
                train_cfg.runner.checkpoint = -1

            ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
            policy = ppo_runner.get_inference_policy(device=env.device)
            if getattr(args, "load_run", None) is not None:
                train_cfg.runner.load_run = args.load_run
            if getattr(args, "checkpoint", None) is not None:
                train_cfg.runner.checkpoint = args.checkpoint
            print('Loaded policy from: ', task_registry.loaded_policy_path)
        if controller != "policy":
            use_hybrid_safety_filter = True
            print("Using SEA-Nav policy with TTC/VO safety filter for dynamic-obstacle playback.")
    elif controller in {"dwa", "dwa_pipeline", "pipeline"}:
        train_cfg.runner.load_run = "dwa_pipeline"
        train_cfg.runner.checkpoint = "n1"
        print("Using DWA/VO-style pipeline controller for navigation playback.")
    else:
        raise ValueError(f"Unsupported SEA_NAV_PLAY_CONTROLLER={controller!r}")

    # ---------------------------
    # Camera Setup for Recording
    # ---------------------------
    camera_props = gymapi.CameraProperties()
    camera_props.width = int(os.environ.get("SEA_NAV_RECORD_WIDTH", "1000"))
    camera_props.height = int(os.environ.get("SEA_NAV_RECORD_HEIGHT", "1000"))
    camera_props.horizontal_fov = float(os.environ.get("SEA_NAV_RECORD_FOV", "58.0"))
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    
    set_recording_camera(env, camera_handle)

    RECORD_VIDEO = os.environ.get("SEA_NAV_RECORD_VIDEO") == "1"
    SAVE_IMAGES = os.environ.get("SEA_NAV_SAVE_IMAGES") == "1"
    SEGMENT_VIDEO_BY_EPISODE = os.environ.get("SEA_NAV_SEGMENT_VIDEO_BY_EPISODE") == "1"
    GUI_STYLE_RECORDING = os.environ.get("SEA_NAV_RECORD_GUI_STYLE") == "1"
    TOTAL_EPISODES = int(os.environ.get("SEA_NAV_TOTAL_EPISODES", "10"))
    video = None
    active_video_episode = None
    current_frame = 0
    max_frames = int(os.environ.get("SEA_NAV_VIDEO_LENGTH", "20000"))

    env.reset()
    obs, _ = env.reset()
    episode_count = 0
    eval_stats = {
        "success": 0,
        "dynamic_collision": 0,
        "static_collision": 0,
        "timeout": 0,
        "stuck": 0,
        "fall": 0,
        "other": 0,
        "final_distance_sum": 0.0,
        "min_dynamic_distance_sum": 0.0,
    }

    max_steps = None if TOTAL_EPISODES <= 0 else 100 * int(env.max_episode_length)
    i = 0
    with torch.no_grad():
        while max_steps is None or i < max_steps:
            i += 1
            # Step the environment
            if policy is None:
                actions = dwa_pipeline_actions(env)
            else:
                policy_obs = strip_dynamic_obstacle_obs(env, obs.detach()) if policy_uses_stripped_obs else obs.detach()
                raw_actions = policy(policy_obs)
                emergency_actions = emergency_policy(obs.detach()) if emergency_policy is not None else None
                actions = (
                    hybrid_safety_filter_actions(env, raw_actions, emergency_actions)
                    if use_hybrid_safety_filter
                    else raw_actions
                )
            obs, _, rews, dones, infos = env.step(actions.detach())
            set_recording_camera(env, camera_handle)

            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).flatten()
                for env_id in done_ids:
                    episode_count += 1
                    success = bool(getattr(env, "last_episode_goal_reached", torch.zeros_like(dones))[env_id].item())
                    dynamic_collision = bool(getattr(env, "last_episode_dynamic_collision", torch.zeros_like(dones))[env_id].item())
                    timeout = bool(getattr(env, "last_episode_time_out", torch.zeros_like(dones))[env_id].item())
                    stuck = bool(getattr(env, "last_episode_stand_still", torch.zeros_like(dones))[env_id].item())
                    fall = bool(getattr(env, "last_episode_fall_down", torch.zeros_like(dones))[env_id].item())
                    static_collision = bool(getattr(env, "last_episode_static_collision", torch.zeros_like(dones))[env_id].item())
                    final_distance = float(getattr(env, "last_episode_distance", torch.zeros(env.num_envs, device=env.device))[env_id].item())
                    min_dyn = float(
                        getattr(
                            env,
                            "last_episode_dynamic_min_distance",
                            torch.ones(env.num_envs, device=env.device) * 100.0,
                        )[env_id].item()
                    )
                    reason = "success"
                    if success:
                        eval_stats["success"] += 1
                    elif dynamic_collision:
                        eval_stats["dynamic_collision"] += 1
                        reason = "dynamic_collision"
                    elif static_collision:
                        eval_stats["static_collision"] += 1
                        reason = "static_collision"
                    elif stuck:
                        eval_stats["stuck"] += 1
                        reason = "stuck"
                    elif fall:
                        eval_stats["fall"] += 1
                        reason = "fall"
                    elif timeout:
                        eval_stats["timeout"] += 1
                        reason = "timeout"
                    else:
                        eval_stats["other"] += 1
                        reason = "other"
                    eval_stats["final_distance_sum"] += final_distance
                    eval_stats["min_dynamic_distance_sum"] += min_dyn
                    print(
                        f"============== Episode {episode_count} Finished: {reason} "
                        f"dist={final_distance:.2f}m min_dyn={min_dyn:.2f}m ============== "
                    )
                    if SEGMENT_VIDEO_BY_EPISODE and video is not None:
                        video.release()
                        video = None
                        active_video_episode = None

            if TOTAL_EPISODES > 0 and episode_count >= TOTAL_EPISODES:
                print(f"Reached {TOTAL_EPISODES} episodes, stopping.")
                denom = max(1, episode_count)
                print(
                    "[N1 eval summary] "
                    f"episodes={episode_count} "
                    f"success_rate={eval_stats['success'] / denom:.3f} "
                    f"dynamic_collision_rate={eval_stats['dynamic_collision'] / denom:.3f} "
                    f"static_collision_rate={eval_stats['static_collision'] / denom:.3f} "
                    f"timeout_rate={eval_stats['timeout'] / denom:.3f} "
                    f"stuck_rate={eval_stats['stuck'] / denom:.3f} "
                    f"fall_rate={eval_stats['fall'] / denom:.3f} "
                    f"avg_final_distance={eval_stats['final_distance_sum'] / denom:.2f} "
                    f"avg_min_dynamic_distance={eval_stats['min_dynamic_distance_sum'] / denom:.2f}"
                )
                if video is not None:
                    video.release()  
                break           

            # Recording Logic
            if (RECORD_VIDEO or SAVE_IMAGES) and current_frame < max_frames:
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
                img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if GUI_STYLE_RECORDING:
                    overlay_navigation_target_gui_style(
                        img_bgr,
                        env,
                        camera_handle,
                        camera_props.width,
                        camera_props.height,
                    )
                    overlay_dynamic_obstacle_gui_style(
                        img_bgr,
                        env,
                        camera_handle,
                        camera_props.width,
                        camera_props.height,
                    )
                else:
                    overlay_navigation_target(img_bgr, env, camera_handle, camera_props.width, camera_props.height)
                    overlay_dynamic_obstacles(img_bgr, env, camera_handle, camera_props.width, camera_props.height)

                if RECORD_VIDEO:
                    if video is None:
                        fps = 50
                        if SEGMENT_VIDEO_BY_EPISODE:
                            active_video_episode = episode_count + 1
                            output_path = os.path.join(
                                LEGGED_GYM_ROOT_DIR,
                                'logs',
                                train_cfg.runner.experiment_name,
                                'exported',
                                'episodes',
                                f"{train_cfg.runner.load_run}_{train_cfg.runner.checkpoint}_episode_{active_video_episode:03d}.mp4",
                            )
                        else:
                            output_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', f"{train_cfg.runner.load_run}_{train_cfg.runner.checkpoint}.mp4")
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (camera_props.width, camera_props.height))
                        print(f"Recording video to {output_path}")
                    video.write(img_bgr)

                if SAVE_IMAGES:
                    img_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
                    os.makedirs(img_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(img_dir, f"frame_{current_frame:04d}.png"), img_bgr)

                current_frame += 1
                if current_frame % 100 == 0:
                    print(f"Recorded {current_frame}/{max_frames} frames")
            
            elif (RECORD_VIDEO or SAVE_IMAGES) and current_frame >= max_frames:
                if video is not None:
                    video.release()
                    video = None
                RECORD_VIDEO = False
                SAVE_IMAGES = False


if __name__ == '__main__':
    args = get_args()
    args.headless = os.environ.get("SEA_NAV_PLAY_HEADLESS") == "1"
    play(args)
