#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEA_NAV_ROOT = REPO_ROOT / "external" / "SEA-Nav-Code"
LOG_ROOT = SEA_NAV_ROOT / "training" / "legged_gym" / "logs" / "Go2_dynamic_obstacles_obs"
RUN_RE = re.compile(r"^(\d{2}_\d{2}_\d{2}-\d{2}-\d{2}.*)$")
ITER_RE = re.compile(r"(?:\[timing\] iter=|Iteration:\s+)(\d+)")


@dataclass(frozen=True)
class Stage:
    name: str
    config: str
    max_iterations: int


STAGES = [
    Stage("stage0_static_obs", "tasks/navigation/configs/dynamic_obstacles_obs_stage0_static.yaml", 4000),
    Stage("stage1_single_pedestrian", "tasks/navigation/configs/dynamic_obstacles_obs_stage1.yaml", 6000),
    Stage("stage2_wait_single", "tasks/navigation/configs/dynamic_obstacles_obs_stage2_wait.yaml", 3000),
    Stage("stage2_two_obstacles", "tasks/navigation/configs/dynamic_obstacles_obs_stage2_two.yaml", 3000),
    Stage("stage3_three_obstacles", "tasks/navigation/configs/dynamic_obstacles_obs_stage3.yaml", 4000),
]


def list_runs() -> set[str]:
    if not LOG_ROOT.exists():
        return set()
    return {p.name for p in LOG_ROOT.iterdir() if p.is_dir() and RUN_RE.match(p.name)}


def latest_new_run(before: set[str]) -> Path:
    deadline = time.time() + 30.0
    while time.time() < deadline:
        after = sorted(list_runs() - before)
        if after:
            return LOG_ROOT / after[-1]
        time.sleep(0.5)
    runs = sorted(list_runs())
    if not runs:
        raise RuntimeError(f"No SEA-Nav runs found under {LOG_ROOT}")
    return LOG_ROOT / runs[-1]


def latest_checkpoint(run_dir: Path) -> Path:
    models = list(run_dir.glob("model_*.pt"))
    if not models:
        raise RuntimeError(f"No checkpoint found in {run_dir}")

    def model_num(path: Path) -> int:
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    return max(models, key=model_num)


def checkpoint_iteration(path: Path | None) -> int:
    if path is None:
        return 0
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else 0


def stage_command(stage: Stage, checkpoint: Path | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "summer_camp_rl.cli",
        "train",
        "--task",
        "navigation/dynamic_obstacles",
        "--config",
        stage.config,
    ]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    return cmd


def make_remaining_config(stage: Stage, start_iter: int, log_dir: Path) -> str:
    remaining = stage.max_iterations - start_iter
    if start_iter <= 0 or remaining <= 0:
        return stage.config

    src = REPO_ROOT / stage.config
    dst = log_dir / f"{stage.name}_resume_from_{start_iter}_to_{stage.max_iterations}.yaml"
    shutil.copyfile(src, dst)

    text = dst.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_training = False
    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith("training:"):
            in_training = True
            continue
        if in_training and line and not line.startswith(" "):
            in_training = False
        if in_training and re.match(r"^\s+max_iterations:\s+\d+\s*$", line):
            indent = line[: len(line) - len(line.lstrip())]
            lines[idx] = f"{indent}max_iterations: {remaining}"
            replaced = True
            break

    if not replaced:
        raise RuntimeError(f"Unable to override training.max_iterations in {dst}")

    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(dst.relative_to(REPO_ROOT))


def run_stage(stage: Stage, checkpoint: Path | None, log_dir: Path, resume_to_stage_target: bool = False) -> Path:
    before = list_runs()
    log_path = log_dir / f"{stage.name}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["SEA_NAV_ENABLE_WANDB"] = "1"
    env.pop("WANDB_DISABLED", None)
    if env.get("WANDB_MODE", "").lower() == "offline":
        env.pop("WANDB_MODE", None)

    start_iter = checkpoint_iteration(checkpoint)
    stage_to_run = stage
    if resume_to_stage_target and checkpoint is not None and start_iter < stage.max_iterations:
        stage_to_run = Stage(stage.name, make_remaining_config(stage, start_iter, log_dir), stage.max_iterations)

    cmd = stage_command(stage_to_run, checkpoint)
    print(f"\n[stage-start] {stage.name}")
    print(f"[stage-config] {stage_to_run.config}")
    if checkpoint is not None:
        print(f"[stage-init] {checkpoint}")
        print(f"[stage-resume-iter] {start_iter}")
        if stage_to_run.config != stage.config:
            print(f"[stage-remaining] target={stage.max_iterations} additional={stage.max_iterations - start_iter}")
    print(f"[stage-log] {log_path}")
    sys.stdout.flush()

    next_report = ((start_iter // 1000) + 1) * 1000
    last_iter = None
    last_timing = None
    last_mean_reward = None
    last_terrain = None
    last_goal = None
    last_dyn_collision = None

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            stripped = line.strip()
            match = ITER_RE.search(stripped)
            if match:
                last_iter = int(match.group(1))
                if stripped.startswith("[timing]"):
                    last_timing = stripped
                while last_iter is not None and last_iter >= next_report:
                    print(
                        f"[stage-monitor] {stage.name} iter={last_iter} "
                        f"reward={last_mean_reward or 'n/a'} terrain={last_terrain or 'n/a'} "
                        f"goal={last_goal or 'n/a'} dyn_collision={last_dyn_collision or 'n/a'}"
                    )
                    if last_timing:
                        print(f"[stage-timing] {last_timing}")
                    sys.stdout.flush()
                    next_report += 1000
            elif "Mean reward:" in stripped:
                last_mean_reward = stripped.split()[-1]
            elif "Mean episode terrain_level:" in stripped:
                last_terrain = stripped.split()[-1]
            elif "Mean episode goal_level:" in stripped:
                last_goal = stripped.split()[-1]
            elif "Mean episode dynamic_collision_count:" in stripped:
                last_dyn_collision = stripped.split()[-1]

        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"{stage.name} failed with exit code {code}. See {log_path}")

    run_dir = latest_new_run(before)
    checkpoint_out = latest_checkpoint(run_dir)
    print(f"[stage-done] {stage.name} run={run_dir.name} checkpoint={checkpoint_out}")
    sys.stdout.flush()
    return checkpoint_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train navigation dynamic-obstacle stages sequentially.")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--end-stage", type=int, default=len(STAGES) - 1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    log_dir = REPO_ROOT / "outputs" / "logs" / "navigation" / "dynamic_obstacles_multistage"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint
    completed: list[tuple[str, Path]] = []
    for idx, stage in enumerate(STAGES):
        if idx < args.start_stage or idx > args.end_stage:
            continue
        checkpoint = run_stage(
            stage,
            checkpoint,
            log_dir,
            resume_to_stage_target=(idx == args.start_stage),
        )
        completed.append((stage.name, checkpoint))

    print("\n[all-stages-done]")
    for name, path in completed:
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
