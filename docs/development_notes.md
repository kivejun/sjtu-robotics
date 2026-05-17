# 开发记录

## 约定

- 外部框架不要直接复制到本仓库；推荐使用 git submodule、外部路径配置或 wrapper。
- 大文件不要提交到 Git，包括 `.pt`、`.pth`、`.ckpt`、`.mp4`、`.usd` 大资产等。
- 所有实验都用 config 文件记录关键参数。
- 每次实验记录：日期、任务、命令、seed、checkpoint 路径、主要结果、失败原因。

## 实验记录模板

```text
日期：
任务：
环境：
命令：
seed：
训练步数：
checkpoint：
结果：
问题：
下一步：
```
