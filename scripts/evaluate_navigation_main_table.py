from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_RE = re.compile(
    r"\[N1 eval summary\]\s+"
    r"episodes=(?P<episodes>\d+)\s+"
    r"success_rate=(?P<success_rate>[0-9.]+)\s+"
    r"dynamic_collision_rate=(?P<dynamic_collision_rate>[0-9.]+)\s+"
    r"static_collision_rate=(?P<static_collision_rate>[0-9.]+)\s+"
    r"timeout_rate=(?P<timeout_rate>[0-9.]+)\s+"
    r"stuck_rate=(?P<stuck_rate>[0-9.]+)\s+"
    r"fall_rate=(?P<fall_rate>[0-9.]+)\s+"
    r"avg_final_distance=(?P<avg_final_distance>[0-9.]+)\s+"
    r"avg_min_dynamic_distance=(?P<avg_min_dynamic_distance>[0-9.]+)"
)


def repo_path(path: str) -> Path:
    return REPO_ROOT / path


ROWS = [
    {
        "version": "SEA-Nav baseline",
        "scheme": "static_baseline",
        "task": "navigation/sea_nav_baseline",
        "config": "tasks/navigation/configs/sea_nav_baseline.yaml",
        "checkpoint": "external/SEA-Nav-Code/training/legged_gym/logs/Go2_pos_rough/05_18_21-16-22_/model_4000.pt",
        "note": "Original SEA-Nav static hard_room navigation baseline.",
    },
    {
        "version": "Dynamic obstacles as moving static obstacles",
        "scheme": "dynamic_no_extra_reward",
        "task": "navigation/dynamic_obstacles",
        "config": "tasks/navigation/configs/dynamic_obstacles_stage3.yaml",
        "checkpoint": "external/SEA-Nav-Code/training/legged_gym/logs/Go2_dynamic_obstacles/06_03_16-28-52_/model_7400.pt",
        "note": "Dynamic obstacles are fused as moving obstacles without explicit dynamic-obstacle reward shaping.",
    },
    {
        "version": "End-to-end RL + dynamic reward",
        "scheme": "dynamic_reward",
        "task": "navigation/dynamic_obstacles",
        "config": "tasks/navigation/configs/dynamic_obstacles_stage3.yaml",
        "checkpoint": "external/SEA-Nav-Code/training/legged_gym/logs/Go2_dynamic_obstacles/06_03_23-42-17_/model_6400.pt",
        "note": "Uses dynamic_ttc, clearance, wait, detour and preferred-velocity rewards.",
    },
    {
        "version": "Dynamic obstacle observation RL",
        "scheme": "dynamic_obs",
        "task": "navigation/dynamic_obstacles",
        "config": "tasks/navigation/configs/dynamic_obstacles_obs_stage3.yaml",
        "checkpoint": "external/SEA-Nav-Code/training/legged_gym/logs/Go2_dynamic_obstacles_obs/06_04_06-39-43_/model_10000.pt",
        "note": "Policy observation includes explicit dynamic obstacle state.",
    },
    {
        "version": "Traditional pipeline",
        "scheme": "traditional_pipeline",
        "task": "navigation/dynamic_obstacles",
        "config": "tasks/navigation/configs/dynamic_obstacles_pipeline.yaml",
        "checkpoint": None,
        "note": "No RL checkpoint. Uses the DWA/VO-style pipeline controller in play.py.",
    },
    {
        "version": "Baseline + TTC/VO speed filter",
        "scheme": "hybrid_filter",
        "task": "navigation/dynamic_obstacles",
        "config": "tasks/navigation/configs/dynamic_obstacles_mixed_policy_best.yaml",
        "checkpoint": "external/SEA-Nav-Code/training/legged_gym/logs/Go2_pos_rough/05_18_21-16-22_/model_4000.pt",
        "note": "Original SEA-Nav policy with a TTC/VO low-risk speed filter.",
    },
    {
        "version": "Baseline + TTC/VO + RL emergency",
        "scheme": "hybrid_filter_rl_emergency",
        "task": "navigation/dynamic_obstacles",
        "config": "tasks/navigation/configs/dynamic_obstacles_mixed_policy_rl_emergency.yaml",
        "checkpoint": "external/SEA-Nav-Code/training/legged_gym/logs/Go2_pos_rough/05_18_21-16-22_/model_4000.pt",
        "note": "Original SEA-Nav policy plus TTC/VO speed filter and high-risk RL emergency policy.",
    },
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def make_eval_config(row: dict, episodes: int, tmpdir: Path) -> Path:
    cfg = load_yaml(repo_path(row["config"]))
    cfg = deepcopy(cfg)
    evaluation = cfg.setdefault("evaluation", {})
    evaluation["num_envs"] = 1
    evaluation["headless"] = True
    evaluation["total_episodes"] = episodes
    tmp_path = tmpdir / f"{row['scheme']}.yaml"
    dump_yaml(cfg, tmp_path)
    return tmp_path


def parse_summary(output: str) -> dict[str, str]:
    matches = list(SUMMARY_RE.finditer(output))
    if not matches:
        raise RuntimeError("Could not find [N1 eval summary] in command output.")
    return matches[-1].groupdict()


def run_row(row: dict, episodes: int, tmpdir: Path, log_dir: Path) -> dict:
    cfg_path = make_eval_config(row, episodes, tmpdir)
    cmd = [
        sys.executable,
        "-m",
        "summer_camp_rl.cli",
        "eval",
        "--task",
        row["task"],
        "--config",
        str(cfg_path),
    ]
    checkpoint = row.get("checkpoint")
    if checkpoint:
        ckpt_path = repo_path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {row['version']}: {ckpt_path}")
        cmd.extend(["--checkpoint", str(ckpt_path)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    print(f"\n[main-table] Running: {row['version']}")
    print("[main-table] Command:", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = log_dir / f"{row['scheme']}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {row['version']}. See {log_path}")
    stats = parse_summary(completed.stdout)
    result = {
        "version": row["version"],
        "scheme": row["scheme"],
        "task": row["task"],
        "config": row["config"],
        "checkpoint": row.get("checkpoint") or "No RL checkpoint",
        "note": row["note"],
        **stats,
    }
    print(
        "[main-table] Result:",
        f"success={result['success_rate']}",
        f"dyn_col={result['dynamic_collision_rate']}",
        f"static_col={result['static_collision_rate']}",
        f"dist={result['avg_final_distance']}",
        flush=True,
    )
    return result


def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "navigation_main_table.csv"
    md_path = out_dir / "navigation_main_table.md"
    fields = [
        "version",
        "episodes",
        "success_rate",
        "dynamic_collision_rate",
        "static_collision_rate",
        "timeout_rate",
        "stuck_rate",
        "fall_rate",
        "avg_final_distance",
        "avg_min_dynamic_distance",
        "config",
        "checkpoint",
        "note",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})

    headers = [
        "版本",
        "Episode 数",
        "成功率",
        "动态碰撞率",
        "静态碰撞率",
        "超时率",
        "卡住率",
        "平均最终距离/m",
        "平均最小动态距离/m",
    ]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for result in results:
            f.write(
                "| "
                + " | ".join(
                    [
                        result["version"],
                        result["episodes"],
                        result["success_rate"],
                        result["dynamic_collision_rate"],
                        result["static_collision_rate"],
                        result["timeout_rate"],
                        result["stuck_rate"],
                        result["avg_final_distance"],
                        result["avg_min_dynamic_distance"],
                    ]
                )
                + " |\n"
            )
    print(f"\n[main-table] Wrote {csv_path}")
    print(f"[main-table] Wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs/metrics/navigation_main_table")
    parser.add_argument("--only", nargs="*", default=None, help="Optional scheme names to run.")
    args = parser.parse_args()

    rows = ROWS
    if args.only:
        requested = set(args.only)
        rows = [row for row in ROWS if row["scheme"] in requested]
        missing = requested.difference(row["scheme"] for row in rows)
        if missing:
            raise ValueError(f"Unknown scheme(s): {', '.join(sorted(missing))}")

    out_dir = repo_path(args.out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nav-main-table-") as tmp:
        tmpdir = Path(tmp)
        results = [run_row(row, args.episodes, tmpdir, log_dir) for row in rows]
    write_outputs(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
