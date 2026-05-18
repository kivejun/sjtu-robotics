from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    backend: str
    description: str
    runner_hint: str


TASK_REGISTRY: dict[str, TaskSpec] = {
    "locomotion/humanoid_flat": TaskSpec(
        name="locomotion/humanoid_flat",
        family="locomotion",
        backend="IsaacLab or robot_lab",
        description="Humanoid flat-ground velocity tracking baseline.",
        runner_hint="tasks/locomotion/runners/robot_lab_velocity_runner.py",
    ),
    "locomotion/humanoid": TaskSpec(
        name="locomotion/humanoid",
        family="locomotion",
        backend="robot_lab",
        description="Unitree G1 rough-terrain velocity tracking baseline.",
        runner_hint="tasks/locomotion/runners/robot_lab_velocity_runner.py",
    ),
    "locomotion/quadruped": TaskSpec(
        name="locomotion/quadruped",
        family="locomotion",
        backend="robot_lab",
        description="Unitree Go2 rough-terrain velocity tracking baseline.",
        runner_hint="tasks/locomotion/runners/robot_lab_velocity_runner.py",
    ),
    "locomotion/wheel_legged": TaskSpec(
        name="locomotion/wheel_legged",
        family="locomotion",
        backend="robot_lab",
        description="Unitree Go2W rough-terrain velocity tracking baseline.",
        runner_hint="tasks/locomotion/runners/robot_lab_velocity_runner.py",
    ),
    "navigation/sea_nav_baseline": TaskSpec(
        name="navigation/sea_nav_baseline",
        family="navigation",
        backend="SEA-Nav",
        description="SEA-Nav baseline training/evaluation reproduction.",
        runner_hint="tasks/navigation/runners/sea_nav_runner.py",
    ),
    "navigation/dynamic_obstacles": TaskSpec(
        name="navigation/dynamic_obstacles",
        family="navigation",
        backend="SEA-Nav modified or IsaacLab",
        description="Navigation with at least three dynamic obstacles.",
        runner_hint="tasks/navigation/runners/dynamic_obstacles_runner.py",
    ),
    "navigation/stairs_depth": TaskSpec(
        name="navigation/stairs_depth",
        family="navigation",
        backend="SEA-Nav modified or IsaacLab",
        description="Dense stairs and low-obstacle navigation with depth input.",
        runner_hint="tasks/navigation/runners/stairs_depth_runner.py",
    ),
    "manipulation/franka_reach": TaskSpec(
        name="manipulation/franka_reach",
        family="manipulation",
        backend="IsaacLab official task",
        description="Official IsaacLab Franka Reach baseline.",
        runner_hint="tasks/manipulation/runners/franka_reach_runner.py",
    ),
    "manipulation/button_press": TaskSpec(
        name="manipulation/button_press",
        family="manipulation",
        backend="IsaacLab custom task",
        description="Press elevator-panel button with robot arm or mobile manipulator.",
        runner_hint="tasks/manipulation/runners/button_press_runner.py",
    ),
    "manipulation/bag_lift": TaskSpec(
        name="manipulation/bag_lift",
        family="manipulation",
        backend="IsaacLab custom task",
        description="Grasp and lift a basket/bag by the handle.",
        runner_hint="tasks/manipulation/runners/bag_lift_runner.py",
    ),
}
