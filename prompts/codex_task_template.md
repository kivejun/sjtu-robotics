# Codex 单任务开发提示词模板

你是一名机器人强化学习工程师。请基于当前仓库完成下面这个具体任务。

## 任务名称

`<填写任务名，例如 manipulation/franka_reach>`

## 目标

`<填写目标，例如接入 IsaacLab 官方 Franka Reach 训练流程>`

## 约束

- 不要破坏现有 CLI。
- 不要删除已有目录和文档。
- 不要提交 checkpoint、视频、日志等大文件。
- 外部框架通过 wrapper 调用，不要直接复制官方源码。
- 先写最小可运行版本，再做复杂功能。

## 需要实现

1. 配置文件字段补全。
2. runner 文件实现。
3. train/eval/play/record 四个入口打通。
4. README 增加运行命令。
5. 如果依赖缺失，给出清晰报错。
6. 增加至少一个 dry-run 或轻量测试。

## 验收命令

```bash
summer-camp train --task <task> --config <config> --dry-run
summer-camp eval  --task <task> --config <config> --dry-run
summer-camp play  --task <task> --config <config> --dry-run
```

## 输出

请说明修改文件、运行结果、未完成项和下一步。
