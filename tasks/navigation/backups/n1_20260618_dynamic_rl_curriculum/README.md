# N1 Dynamic Obstacle Policy Backup - 2026-06-18

This backup preserves the current N1 dynamic-obstacle navigation schemes before running the next staged RL curriculum.

## Backed-up Schemes

- `dynamic_obstacles_mixed_policy_best.yaml`
  - Previous best engineering fallback.
  - SEA-Nav baseline navigation policy + low-intervention TTC/VO-style DMA speed filter.
  - Uses the hand-tuned parameters:
    - `hybrid_simple_speed_filter: 1`
    - `hybrid_soft_scale: 0.85`
    - `hybrid_safe_distance: 0.62`
    - `hybrid_critical_distance: 0.48`
    - `hybrid_stop_ttc: 0.45`
    - `hybrid_slow_ttc: 0.85`

- `dynamic_obstacles_mixed_policy_rl_emergency.yaml`
  - Current mixed policy.
  - SEA-Nav baseline navigation policy + TTC/VO speed filter + RL emergency avoidance policy.

- `dynamic_obstacles_avoidance_stage1_encouraged.yaml`
  - Standalone high-risk dynamic avoidance RL skill training config.
  - Used as the emergency policy source for the mixed policy.

## Curriculum Configs

- `dynamic_obstacles_obs_stageA_three_low_speed.yaml`
  - Three dynamic obstacles, low speed, fixed phases.

- `dynamic_obstacles_obs_stageB_three_standard.yaml`
  - Three dynamic obstacles, standard speed, fixed phases.

- `dynamic_obstacles_obs_stageC_three_random.yaml`
  - Three dynamic obstacles, standard speed, randomized phases and wider trajectory variation.

## Planned Run

Warm start from:

`external/SEA-Nav-Code/training/legged_gym/logs/Go2_dynamic_obstacles_obs/06_04_06-39-43_/model_8000.pt`

Then train:

1. Stage A: low-speed fixed three-obstacle training.
2. Stage B: standard-speed fixed three-obstacle training.
3. Stage C: randomized three-obstacle training.

