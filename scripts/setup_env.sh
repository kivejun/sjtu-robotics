#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-sjtu-summer-camp}

echo "[setup] Create conda env: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
pip install -e .

echo "[setup] Done."
echo "[setup] For Ubuntu 20.04 + Franka Reach, use Isaac Sim binary at ~/isaacsim,"
echo "[setup] clone IsaacLab into external/IsaacLab, then run:"
echo "        bash scripts/check_isaaclab_franka_env.sh"
