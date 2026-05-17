# Codex 提示词：基础仓库搭建与维护

你是一名熟悉 IsaacLab、Isaac Sim、机器人强化学习、导航和机械臂操作任务的资深工程师。现在我有一个 2026 夏令营具身智能练习项目，任务方向包括 Locomotion、Navigation 和 Manipulation。当前仓库处于起步阶段，目标不是一次性完成所有算法，而是先搭建一个清晰、可扩展、方便后续逐步开发的基础工程仓库。

## 项目背景

项目有三类方向：

1. Locomotion：人形、四足、轮足机器人基础移动训练，重点包括速度跟踪、稳定站立、复杂地形移动、训练曲线和 demo 视频。
2. Navigation：SEA-Nav baseline，要求能运行环境、完成训练流程、保存/加载策略，并提交 checkpoint、训练曲线和测试视频；后续可扩展动态障碍物导航或楼梯 + depth sensor 导航。
3. Manipulation：IsaacLab 官方 Franka Reach baseline，要求完成末端到达随机目标点训练、保存/加载策略，并提交训练曲线和 demo 视频；后续可扩展按按钮和提袋子任务。

## 你的任务

请你阅读当前仓库结构，然后完成以下工作：

1. 检查仓库结构是否清晰，是否适合后续接入 IsaacLab、SEA-Nav 或 robot_lab。
2. 保留现有三大方向结构，不要删除已有目录。
3. 完善统一命令入口：`train`、`eval`、`play`、`record`。
4. 为每个任务保留独立配置文件、README、runner stub 和 TODO。
5. 所有真实仿真框架调用先做成 wrapper，不要把外部框架源码直接塞进本仓库。
6. 先保证 dry-run 可运行，再逐步接入真实训练流程。
7. 代码要简洁、模块化、可读，重要位置加中文注释。
8. 不要大改无关文件，不要提交大文件，不要删除已有 prompt 和 docs。

## 当前优先目标

先完成一个最小可用闭环：

```bash
summer-camp list-tasks
summer-camp train --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run
summer-camp eval  --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run
summer-camp play  --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run
```

如果 dry-run 已经通过，请继续为 `manipulation/franka_reach` 创建 runner 文件，并设计后续接入 IsaacLab 官方 Franka Reach 的代码位置和步骤。不要假设 IsaacLab 已安装；需要写出清晰的错误提示和安装指引。

## 输出要求

完成修改后，请给出：

1. 修改了哪些文件；
2. 当前能运行的命令；
3. 还没实现的 TODO；
4. 下一步建议先接哪个 baseline。
