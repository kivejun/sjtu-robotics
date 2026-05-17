# Codex 提示词：Locomotion Baseline

请在当前仓库中为 Locomotion 方向实现基础接入方案。优先选择 `locomotion/quadruped`，目标是后续接入 IsaacLab 或 robot_lab 的四足机器人速度跟踪训练。

要求：

1. 检查 `tasks/locomotion/` 目录结构。
2. 补全 quadruped 配置字段：robot、terrain、command range、reward、termination、training、evaluation。
3. 新增 runner stub：`tasks/locomotion/runners/quadruped_runner.py`。
4. 设计后续接入真实 IsaacLab terrain 和 robot_lab 的 wrapper 位置。
5. 保持 dry-run 可运行。
6. 不要实现虚假的训练结果，不要生成假的曲线。

最低目标：命令能完成参数检查，并输出后续应调用哪个外部框架、哪个任务配置、结果保存到哪里。
