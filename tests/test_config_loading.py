from pathlib import Path

from summer_camp_rl.common.config import load_config
from summer_camp_rl.registry import TASK_REGISTRY


def test_franka_reach_config_loads():
    cfg = load_config(Path("tasks/manipulation/configs/franka_reach.yaml"))
    assert cfg["task"]["name"] == "manipulation/franka_reach"


def test_registry_contains_core_tasks():
    assert "manipulation/franka_reach" in TASK_REGISTRY
    assert "navigation/sea_nav_baseline" in TASK_REGISTRY
    assert "locomotion/quadruped" in TASK_REGISTRY
