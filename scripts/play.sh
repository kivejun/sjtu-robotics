#!/usr/bin/env bash
set -euo pipefail

TASK=${1:-manipulation/franka_reach}
CONFIG=${2:-tasks/manipulation/configs/franka_reach.yaml}
CHECKPOINT=${3:-}

if [[ -n "${CHECKPOINT}" ]]; then
  summer-camp play --task "${TASK}" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --dry-run
else
  summer-camp play --task "${TASK}" --config "${CONFIG}" --dry-run
fi
