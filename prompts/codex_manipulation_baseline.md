# Codex 提示词：Manipulation Franka Reach Baseline

请在当前仓库中优先接入 Manipulation 方向的 `manipulation/franka_reach` baseline。该任务基于 IsaacLab 官方 Franka Reach，目标是完成机械臂末端到达随机目标点的训练、保存 checkpoint、加载策略并录制 demo。

要求：

1. 检查 `tasks/manipulation/` 目录结构。
2. 补全 `franka_reach.yaml`，至少包含：task name、backend、robot、target randomization、training steps、seed、checkpoint dir、video dir。
3. 新增 runner stub：`tasks/manipulation/runners/franka_reach_runner.py`。
4. 设计 IsaacLab 导入逻辑：如果未安装 IsaacLab，要给出清晰错误提示；不要让用户看到一长串不可读 traceback。
5. 打通 `summer-camp train/eval/play/record --task manipulation/franka_reach ...` 的路由。
6. 先保证 dry-run 可运行，再准备真实训练入口。
7. README 里写明后续真实 IsaacLab 命令应该放在哪里。

注意：不要伪造训练结果；不要生成假的 checkpoint 或假的视频。
