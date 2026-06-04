#!/bin/bash
# Install the soccer kick open-loop config into rl_sar.
#
# This script copies the keyframe YAML config to the rl_sar policy directory
# and builds the rl_real_go2 binary. No neural-network policy is needed —
# the kick is an open-loop keyframe animation.
#
# Usage:
#   bash /home/yczhang/hrl_soccer_shooting/scripts/install_soccer_kick_to_rl_sar.sh

set -e

RL_SAR_DIR="/home/yczhang/rl_sar"
SRC_CONFIG="/home/yczhang/hrl_soccer_shooting/../rl_sar/policy/go2/soccer_kick/config.yaml"
# The config was created directly in rl_sar/policy/go2/soccer_kick/ during setup.
# This script verifies it exists and builds the target.

echo "[1] Checking config file..."
POLICY_DIR="${RL_SAR_DIR}/policy/go2/soccer_kick"
if [ ! -f "${POLICY_DIR}/config.yaml" ]; then
    echo "ERROR: config.yaml not found at ${POLICY_DIR}/config.yaml"
    echo "Create it first, then run this script."
    exit 1
fi
echo "OK: config.yaml found at ${POLICY_DIR}/config.yaml"

echo "[2] Checking FSM source..."
FSM_FILE="${RL_SAR_DIR}/src/rl_sar/fsm_robot/fsm_go2.hpp"
if [ ! -f "${FSM_FILE}" ]; then
    echo "ERROR: fsm_go2.hpp not found at ${FSM_FILE}"
    exit 1
fi

# Check that SoccerKick state exists in FSM
if ! grep -q "RLFSMStateSoccerKick" "${FSM_FILE}"; then
    echo "ERROR: RLFSMStateSoccerKick not found in fsm_go2.hpp"
    exit 1
fi
echo "OK: SoccerKick FSM state found"

# Check that it reads soccer_kick config
if ! grep -q "soccer_kick" "${FSM_FILE}"; then
    echo "WARNING: fsm_go2.hpp does not reference 'soccer_kick' config."
    echo "The FSM may be using hardcoded parameters. Check the source."
fi

echo "[3] Building rl_real_go2..."
cd "${RL_SAR_DIR}"

# CMake configure if needed
if [ ! -d "cmake_build" ]; then
    echo "Running CMake configure..."
    cmake src/rl_sar/ -B cmake_build \
        -DUSE_CMAKE=ON \
        -DTBB_DIR=/usr/lib/x86_64-linux-gnu/cmake/TBB
fi

cmake --build cmake_build --target rl_real_go2 -j4
echo "OK: Build successful"

echo ""
echo "==========================================="
echo "  Soccer Kick Sim2Real Install Complete"
echo "==========================================="
echo ""
echo "Config: ${POLICY_DIR}/config.yaml"
echo "Binary: ${RL_SAR_DIR}/cmake_build/bin/rl_real_go2"
echo ""
echo "To run on the real Go2:"
echo "  cd ${RL_SAR_DIR}"
echo "  ./cmake_build/bin/rl_real_go2 enp4s0"
echo ""
echo "Operation:"
echo "  0 = GetUp (stand up)"
echo "  2 = SoccerKick (after standing)"
echo "  P = Emergency Passive"
echo ""
echo "FIRST TEST: set safety_scale=0.5 in config.yaml"
echo "  before running on the real robot."