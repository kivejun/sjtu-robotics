# Navigation Dynamic Obstacle Legacy Backup

This directory keeps runnable legacy configs for the dynamic obstacle task.

The legacy behavior means:

- Dynamic obstacles are still analytical moving obstacles.
- Dynamic obstacle rays are still fused into the normal ray observation.
- Dynamic obstacle collision still uses the original distance threshold reset.
- The new VO/DWA-style reward terms are disabled.
- The `stuck` reward uses the original SEA-Nav logic.
- Dynamic obstacle state observation is disabled, so old checkpoints remain loadable.

Use these configs when you want to compare against or return to the earlier logic:

```bash
PYTHONPATH=src python -m summer_camp_rl.cli play \
  --task navigation/dynamic_obstacles \
  --config backups/navigation_dynamic_legacy/dynamic_obstacles_stage3_legacy.yaml \
  --checkpoint external/SEA-Nav-Code/training/legged_gym/logs/Go2_dynamic_obstacles/warm_start_static_4000/model_4000.pt
```

For training:

```bash
PYTHONPATH=src python -m summer_camp_rl.cli train \
  --task navigation/dynamic_obstacles \
  --config backups/navigation_dynamic_legacy/dynamic_obstacles_stage1_legacy.yaml \
  --checkpoint external/SEA-Nav-Code/training/legged_gym/logs/Go2_dynamic_obstacles/warm_start_static_4000/model_4000.pt
```

The switch is controlled by:

```yaml
environment:
  use_legacy_dynamic_reward: true
  include_dynamic_obstacle_state: false
```

