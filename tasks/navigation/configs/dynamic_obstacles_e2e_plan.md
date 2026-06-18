# N1 End-to-End Dynamic Avoidance Plan

This plan keeps SEA-Nav as the navigation prior and adds dynamic-obstacle
learning in stages. It is separate from the earlier `obs_full` and
`three_obstacle_curriculum` training plans.

## Executable Stages

1. `dynamic_obstacles_e2e_stage1_avoidance.yaml`
   - Goal: learn a lightweight dynamic-avoidance response.
   - Main signal: successful avoidance, risk reduction, and static free-space
     safety.
   - Navigation progress is intentionally weak because the skill target is
     "enter a safe free space" under dynamic risk, not full navigation.

2. `dynamic_obstacles_e2e_stage2_recovery.yaml`
   - Goal: recover from avoidance and resume goal-directed navigation.
   - This is a config-level REBot-inspired stage, not a separate recovery
     network yet.

3. `dynamic_obstacles_e2e_stage3_failure_replay.yaml`
   - Goal: revisit failure states more often.
   - Uses SEA-Nav's existing collision replay hooks exposed through config.
   - This is the first engineering version of failure replay. It does not yet
     implement a full offline buffer for dynamic collision, stuck, and detour
     failures.

4. Stage-4 curriculum:
   - `dynamic_obstacles_e2e_stage4a_curriculum_low_speed.yaml`
   - `dynamic_obstacles_e2e_stage4b_curriculum_standard.yaml`
   - `dynamic_obstacles_e2e_stage4c_curriculum_random.yaml`
   - `dynamic_obstacles_e2e_stage4d_curriculum_mild_adversarial.yaml`
   - Goal: move from low-speed fixed trajectories to random phases and
     near-path focused obstacle sampling.
   - The last stage is only mildly adversarial. Current code does not implement
     a learned adversary.

## Stage 5 Placeholder

`dynamic_obstacles_e2e_stage5_attention_placeholder.yaml` documents the next
architecture direction: replace flat concatenated obstacle observation with an
attention or graph encoder so the policy can focus on the most dangerous
obstacle. This is not added to the automatic training plan because it requires
policy-network changes and checkpoint conversion.

## How to Run Later

Do not run this during config review. When ready, start the executable stages
with:

```bash
cd ~/sjtu-robotics
conda activate sea_nav
PYTHONPATH=src python scripts/train_navigation_dynamic_obs_stages.py --plan e2e_dynamic_avoidance
```

The script automatically uses each completed stage checkpoint as the next
stage's initialization model.

