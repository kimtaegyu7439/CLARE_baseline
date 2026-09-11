#!/usr/bin/env bash
#
# R10 — 수송 좌표 위의 level+structure 앵커. 사용법: bash R10.sh [GPU] [추가인자...]
#
#   bash R10.sh 0 --smoke          연기시험
#   bash R10.sh 0                  본 실행
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-0}; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p results/R10
python -u R10.py "$@" 2>&1 | tee -a results/R10/train.log
