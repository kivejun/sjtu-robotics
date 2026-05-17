# Navigation Tasks

本目录用于管理 SEA-Nav baseline、动态障碍物导航和楼梯 + depth sensor 导航任务。

## Baseline

- `navigation/sea_nav_baseline`

## 选做

- `navigation/dynamic_obstacles`
- `navigation/stairs_depth`

## 推荐开发顺序

1. 外部安装并验证 SEA-Nav。
2. 使用本仓库 wrapper 调用 SEA-Nav 训练。
3. 保存 checkpoint、训练曲线和测试视频。
4. 再魔改动态障碍物或高密度楼梯场景。
