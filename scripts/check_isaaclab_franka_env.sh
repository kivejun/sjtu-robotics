#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-${REPO_ROOT}/external/IsaacLab}"
ISAACSIM_PATH="${ISAACSIM_PATH:-${HOME}/isaacsim}"
ISAACSIM_LINK="${ISAACLAB_PATH}/_isaac_sim"

echo "[check] Repository: ${REPO_ROOT}"
echo "[check] IsaacLab:   ${ISAACLAB_PATH}"
echo "[check] Isaac Sim:  ${ISAACSIM_PATH}"

if [[ ! -d "${ISAACLAB_PATH}" ]]; then
  echo "[error] IsaacLab directory not found: ${ISAACLAB_PATH}"
  echo "        Clone it with: git clone https://github.com/isaac-sim/IsaacLab.git external/IsaacLab"
  exit 1
fi

if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
  echo "[error] isaaclab.sh is missing or not executable under: ${ISAACLAB_PATH}"
  exit 1
fi

if [[ ! -d "${ISAACSIM_PATH}" ]]; then
  echo "[error] Isaac Sim binary directory not found: ${ISAACSIM_PATH}"
  echo "        On Ubuntu 20.04, download/extract Isaac Sim binary to ~/isaacsim instead of using pip."
  exit 1
fi

if [[ ! -x "${ISAACSIM_PATH}/python.sh" ]]; then
  echo "[error] Isaac Sim python.sh not found: ${ISAACSIM_PATH}/python.sh"
  exit 1
fi

if [[ ! -e "${ISAACSIM_LINK}" ]]; then
  echo "[check] Creating IsaacLab -> Isaac Sim symlink:"
  echo "        ${ISAACSIM_LINK} -> ${ISAACSIM_PATH}"
  ln -s "${ISAACSIM_PATH}" "${ISAACSIM_LINK}"
fi

if [[ ! -L "${ISAACSIM_LINK}" ]]; then
  echo "[error] ${ISAACSIM_LINK} exists but is not a symlink."
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-yes}"

# Ubuntu 20.04 + Isaac Sim 4.5 can crash in headless mode if the process still
# connects to the desktop X server. Keep validation truly headless.
unset DISPLAY
unset WAYLAND_DISPLAY
unset XAUTHORITY

echo "[check] Verifying Isaac Sim python..."
"${ISAACSIM_PATH}/python.sh" -c "print('Isaac Sim python OK')"

echo "[check] Listing Franka environments..."
cd "${ISAACLAB_PATH}"
./isaaclab.sh -p scripts/environments/list_envs.py | grep -E "Isaac-Reach-Franka-v0|Franka" || {
  echo "[error] Could not find Franka environments. Check IsaacLab/IsaacSim versions."
  exit 1
}

echo "[check] Environment looks ready for Franka Reach baseline."
