#!/usr/bin/env bash
# 사용법: bash B_room.sh <GPU> <start_steps> <lambda> [start_steps lambda ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=$1; shift
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
while [ $# -gt 0 ]; do
    python B_room.py --start_steps="$1" --lambda_anchor="$2" || echo "FAILED $1 $2"
    shift 2
done
