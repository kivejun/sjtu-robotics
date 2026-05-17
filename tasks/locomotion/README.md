# Locomotion Tasks

本目录用于管理人形、四足、轮足机器人 locomotion 任务。

## Baseline

可选：

- `locomotion/humanoid_flat`：人形机器人平地移动。
- `locomotion/quadruped`：四足机器人复杂地形速度跟踪。
- `locomotion/wheel_legged`：轮足机器人复杂地形速度跟踪。

## 选做

- L1：机器人站在球上。
- L2：双腿 Carry。

## 推荐开发顺序

1. 先跑通一个官方/参考 locomotion baseline。
2. 接入速度指令范围和复杂地形配置。
3. 保存训练曲线、checkpoint 和 demo 视频。
4. 再做 L1 或 L2。
