#!/usr/bin/env bash
set -euo pipefail

TASK=${1:-manipulation/franka_reach}
CONFIG=${2:-tasks/manipulation/configs/franka_reach.yaml}

summer-camp train --task "${TASK}" --config "${CONFIG}" --dry-run
