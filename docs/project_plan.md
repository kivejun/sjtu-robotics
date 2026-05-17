# 项目推进计划

## 阶段 0：仓库搭建

目标：形成可被 Codex 和人工共同维护的基础工程结构。

交付物：

- 统一 CLI：`summer-camp train/eval/play/record`
- 三大方向任务目录：`locomotion/`、`navigation/`、`manipulation/`
- 配置模板、README、验收清单、prompt 模板

## 阶段 1：跑通一个官方 baseline

优先建议：`manipulation/franka_reach` 或 `navigation/sea_nav_baseline`。

最小闭环：

1. 环境能加载。
2. 训练能启动。
3. checkpoint 能保存。
4. checkpoint 能加载测试。
5. 能导出训练曲线和 demo 视频。

## 阶段 2：做一个轻量改造任务

可选：

- Locomotion：四足复杂地形 / 轮足复杂地形。
- Navigation：动态障碍物导航。
- Manipulation：固定底座 Franka 按按钮。

## 阶段 3：整理展示材料

- 训练曲线
- 成功率 / 碰撞次数 / 末端误差等指标
- demo 视频
- 方法说明文档
- 遇到的问题与解决方案
