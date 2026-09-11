#!/usr/bin/env bash
# R13 — 가우시안 샘플 좌표 level 앵커. 사용법: bash R13.sh [GPU] [추가인자...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-0}; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p results/R13
python -u R13.py "$@" 2>&1 | tee -a results/R13/train.log
