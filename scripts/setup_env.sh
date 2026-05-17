#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-summer-camp-embodied-ai}

echo "[setup] Create conda env: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
pip install -e .

echo "[setup] Done. Next: install IsaacLab / SEA-Nav / robot_lab according to selected task."
