# Codex 提示词：Navigation Baseline

请在当前仓库中为 Navigation 方向实现 SEA-Nav baseline 的接入骨架。

要求：

1. 检查 `tasks/navigation/` 目录结构。
2. 补全 `sea_nav_baseline.yaml` 配置字段：environment、robot、sensor、planner/policy、training、checkpoint、evaluation。
3. 新增 runner stub：`tasks/navigation/runners/sea_nav_runner.py`。
4. 设计如何通过 wrapper 调用外部 SEA-Nav 仓库，而不是把 SEA-Nav 官方源码复制进本仓库。
5. 增加动态障碍物任务的 TODO，包括至少 3 个动态障碍物、成功率、碰撞次数、三组运动模式测试。
6. 保持 dry-run 可运行。

输出时请说明：当前只是接入骨架，真实训练需要配置 SEA-Nav 外部路径和依赖。
