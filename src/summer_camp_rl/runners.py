from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from summer_camp_rl.registry import TaskSpec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runner_path_from_hint(runner_hint: str) -> Path:
    if not runner_hint.endswith(".py"):
        raise ValueError(f"Runner hint must point to a Python file: {runner_hint}")
    path = _repo_root() / runner_hint
    if not path.exists():
        raise FileNotFoundError(f"Runner file not found: {path}")
    return path


def _load_runner_module(runner_hint: str):
    path = _runner_path_from_hint(runner_hint)
    module_name = "summer_camp_rl_external_runner_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load runner module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_task_command(
    command: str,
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    task: TaskSpec,
) -> int:
    module = _load_runner_module(task.runner_hint)
    run = getattr(module, "run", None)
    if run is None:
        raise AttributeError(f"Runner module does not expose run(): {task.runner_hint}")
    return int(run(command=command, args=args, cfg=cfg, task=task))
