# 2026 Summer Camp Embodied AI Project

面向 2026 夏令营具身智能练习项目的基础仓库。当前仓库优先解决三个问题：

1. **统一项目结构**：Locomotion / Navigation / Manipulation 三条任务线分目录管理。
2. **统一运行入口**：训练、评估、播放、录制 demo 使用统一 CLI 和脚本入口。
3. **方便 Codex 接手**：每个模块保留清晰 TODO、配置模板、验收清单和开发提示词。

> 当前阶段是 scaffold / bootstrap，不包含完整 IsaacLab、SEA-Nav 或 robot_lab 训练实现。后续应在本仓库基础上逐步接入真实环境、任务配置、奖励函数、训练器与评估脚本。

---

## 1. 项目方向

项目要求覆盖三类任务：

- **Locomotion**：人形 / 四足 / 轮足机器人速度跟踪、复杂地形移动。
- **Navigation**：SEA-Nav baseline，后续可扩展动态障碍物导航或楼梯 + depth sensor 导航。
- **Manipulation**：IsaacLab 官方 Franka Reach baseline，后续可扩展按按钮或提袋子任务。

建议起步优先级：

1. `manipulation/franka_reach`：最适合作为 IsaacLab 入门 baseline。
2. `navigation/sea_nav_baseline`：适合做导航方向复现。
3. `locomotion/quadruped`：适合具身运动控制方向，但环境配置和训练成本更高。

---

## 2. 快速开始

```bash
# 1. 创建 Python 环境，建议 Python 3.10+
conda create -n sjtu-summer-camp python=3.10 -y
conda activate sjtu-summer-camp

# 2. 安装本仓库基础依赖
pip install -e .

# 3. 查看统一命令入口
summer-camp --help

# 4. dry-run 测试，不调用重型仿真环境
summer-camp train --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run
summer-camp eval  --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run
summer-camp play  --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run
```

### Ubuntu 20.04 上配置 Franka Reach baseline

当前推荐从 `manipulation/franka_reach` 开始。Ubuntu 20.04 默认 GLIBC 较旧，不建议使用 Isaac Sim pip 安装；请使用 Isaac Sim binary，并让 IsaacLab 通过 `_isaac_sim` 软链接找到它。

```bash
# 1. 将 Isaac Sim binary 解压到 ~/isaacsim
# 目录中应包含 isaac-sim.sh、python.sh、kit/、exts/ 等文件

# 2. 克隆 IsaacLab
cd ~/sjtu-robotics
mkdir -p external
git clone https://github.com/isaac-sim/IsaacLab.git external/IsaacLab
cd external/IsaacLab
git checkout v2.2.1

# 3. 安装依赖和 RSL-RL
sudo apt install cmake build-essential ffmpeg -y
./isaaclab.sh --install rsl_rl

# 4. 回到本仓库，检查 Isaac Sim / IsaacLab / Franka 任务
cd ~/sjtu-robotics
bash scripts/check_isaaclab_franka_env.sh

# 5. 小规模验证 Franka Reach 训练
cd external/IsaacLab
export OMNI_KIT_ACCEPT_EULA=yes
unset DISPLAY
unset WAYLAND_DISPLAY
unset XAUTHORITY
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-Franka-v0 \
  --headless \
  --num_envs 16 \
  --max_iterations 20
```

在 Ubuntu 20.04 桌面环境中，即使使用 `--headless`，Isaac Sim 4.5 仍可能因为连接到当前 X server 而触发 `xcb` 段错误。运行 headless 训练和环境检查前清理 `DISPLAY`、`WAYLAND_DISPLAY`、`XAUTHORITY` 可以避免这个问题。

也可以使用脚本：

```bash
bash scripts/train.sh manipulation/franka_reach tasks/manipulation/configs/franka_reach.yaml
bash scripts/eval.sh  manipulation/franka_reach tasks/manipulation/configs/franka_reach.yaml
bash scripts/play.sh  manipulation/franka_reach tasks/manipulation/configs/franka_reach.yaml
```

---

## 3. 仓库结构

```text
summer_camp_embodied_ai_repo/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .gitignore
├── docs/
│   ├── project_plan.md
│   ├── acceptance_checklist.md
│   └── development_notes.md
├── prompts/
│   ├── codex_bootstrap_prompt.md
│   ├── codex_task_template.md
│   ├── codex_locomotion_baseline.md
│   ├── codex_navigation_baseline.md
│   └── codex_manipulation_baseline.md
├── scripts/
│   ├── setup_env.sh
│   ├── train.sh
│   ├── eval.sh
│   ├── play.sh
│   └── record_demo.sh
├── src/summer_camp_rl/
│   ├── cli.py
│   ├── registry.py
│   └── common/
│       ├── config.py
│       ├── logging.py
│       ├── paths.py
│       └── seed.py
├── tasks/
│   ├── locomotion/
│   ├── navigation/
│   └── manipulation/
├── experiments/
├── assets/
├── checkpoints/
├── logs/
├── outputs/
│   ├── videos/
│   ├── figures/
│   └── metrics/
└── tests/
```

---

## 4. 开发原则

- 不直接修改官方 IsaacLab / SEA-Nav / robot_lab 源码；优先通过 wrapper、配置继承、注册任务实现扩展。
- 每个任务必须同时保留：训练脚本、评估脚本、播放脚本、配置文件、README、验收指标记录。
- 所有实验结果保存到 `logs/`、`checkpoints/`、`outputs/`，不要提交大文件到 Git。
- 每次实现一个最小可运行闭环：`env loads -> train starts -> checkpoint saves -> policy loads -> demo records`。

---

## 5. 当前 TODO

- [ ] 确定优先方向：Locomotion / Navigation / Manipulation。
- [ ] 安装并验证 IsaacLab / SEA-Nav / robot_lab 中至少一个外部环境。
- [ ] 将官方 baseline 接入本仓库统一 CLI。
- [ ] 完成 dry-run 到真实训练的切换。
- [ ] 保存训练曲线、checkpoint、测试视频。
- [ ] 整理夏令营提交材料。
