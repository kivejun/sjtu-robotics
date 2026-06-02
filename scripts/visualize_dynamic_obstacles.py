from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _trajectory(mode: str, base: np.ndarray, axis: np.ndarray, perp: np.ndarray, amp: float, speed: float, t: np.ndarray):
    omega = speed / max(amp, 1e-3)
    if mode == "pedestrian_like":
        direction = perp
        pos = base[None, :] + amp * np.sin(omega * t)[:, None] * direction[None, :]
    elif mode == "back_and_forth":
        direction = axis
        pos = base[None, :] + amp * np.sin(omega * t)[:, None] * direction[None, :]
    else:
        pos = (
            base[None, :]
            + amp * np.sin(omega * t)[:, None] * (0.65 * axis)[None, :]
            + 0.65 * amp * np.cos(0.7 * omega * t)[:, None] * perp[None, :]
        )
    return pos


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview analytical dynamic obstacles for SEA-Nav N1.")
    parser.add_argument(
        "--config",
        default="tasks/navigation/configs/dynamic_obstacles.yaml",
        help="Dynamic obstacle task config.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    env_cfg = cfg.get("environment", {})
    out_cfg = cfg.get("outputs", {})

    modes = list(env_cfg.get("obstacle_motion_modes", ["pedestrian_like", "back_and_forth", "random_rigid_body"]))
    num_obstacles = int(env_cfg.get("num_dynamic_obstacles", len(modes)))
    radius = float(env_cfg.get("obstacle_radius", 0.35))
    speed = float(env_cfg.get("obstacle_speed", 0.45))
    output = Path(args.output or out_cfg.get("preview_path", "outputs/figures/navigation/dynamic_obstacles_preview.png"))
    output.parent.mkdir(parents=True, exist_ok=True)

    # A fixed top-down room preview. The real SEA-Nav env samples start/goal
    # positions, but this layout mirrors the tensor initialization strategy.
    robot = np.array([1.2, 1.4])
    goal = np.array([8.6, 8.0])
    axis = goal - robot
    axis = axis / np.linalg.norm(axis)
    perp = np.array([-axis[1], axis[0]])
    midpoint = robot + 0.45 * (goal - robot)
    t = np.linspace(0.0, 12.0, 240)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("N1 Dynamic Obstacles Preview")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.plot([robot[0], goal[0]], [robot[1], goal[1]], color="#808080", linestyle="--", label="nominal path")
    ax.scatter([robot[0]], [robot[1]], s=140, marker="o", color="#2c7be5", label="robot start")
    ax.scatter([goal[0]], [goal[1]], s=180, marker="*", color="#24a148", label="goal")

    colors = ["#ff8c00", "#ffbf00", "#d62728"]
    for idx in range(num_obstacles):
        mode = modes[idx % len(modes)]
        shift = idx - (num_obstacles - 1) * 0.5
        base = midpoint + shift * 1.1 * perp
        amp = 1.0 + 0.35 * idx
        traj = _trajectory(mode, base, axis, perp, amp, speed, t)
        color = colors[idx % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0, label=f"dyn-{idx + 1}: {mode}")
        for frame in [20, 90, 160, 230]:
            circle = plt.Circle((traj[frame, 0], traj[frame, 1]), radius, color=color, fill=False, linewidth=1.8)
            ax.add_patch(circle)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color=color, s=60)

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"Saved dynamic obstacle preview: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
