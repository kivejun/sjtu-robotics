from __future__ import annotations

import argparse
from pathlib import Path

from summer_camp_rl.common.config import load_config
from summer_camp_rl.common.logging import get_logger
from summer_camp_rl.registry import TASK_REGISTRY, TaskSpec

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summer-camp",
        description="Unified CLI for summer camp embodied AI tasks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["train", "eval", "play", "record"]:
        p = subparsers.add_parser(command)
        p.add_argument("--task", required=True, help="Task name, e.g. manipulation/franka_reach")
        p.add_argument("--config", required=True, help="Path to YAML config file")
        p.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
        p.add_argument("--dry-run", action="store_true", help="Only validate routing/config; do not launch simulator")

    list_parser = subparsers.add_parser("list-tasks")
    list_parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_task(task_name: str) -> TaskSpec:
    if task_name not in TASK_REGISTRY:
        available = "\n".join(f"  - {name}" for name in sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task: {task_name}\nAvailable tasks:\n{available}")
    return TASK_REGISTRY[task_name]


def run_command(args: argparse.Namespace) -> int:
    if args.command == "list-tasks":
        for name, spec in sorted(TASK_REGISTRY.items()):
            if args.verbose:
                print(f"{name}: {spec.description}")
            else:
                print(name)
        return 0

    task = _resolve_task(args.task)
    cfg = load_config(Path(args.config))

    logger.info("Command: %s", args.command)
    logger.info("Task: %s", args.task)
    logger.info("Config: %s", args.config)
    logger.info("Backend: %s", task.backend)

    if args.dry_run:
        logger.info("Dry-run passed. Config keys: %s", sorted(cfg.keys()))
        logger.info("Next step: implement %s runner at %s", args.command, task.runner_hint)
        return 0

    raise NotImplementedError(
        "Real simulator/training integration is not implemented yet. "
        "Use --dry-run first, then connect IsaacLab / SEA-Nav / robot_lab in the task runner."
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
