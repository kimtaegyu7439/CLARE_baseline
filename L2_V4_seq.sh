#!/usr/bin/env bash
# 순차(CLARE) 셋팅 — v4 / v3 / Seq-FT. CUDA_VISIBLE_DEVICES 는 torch import 전에.
#
#   bash L2_V4_seq.sh <gpu> --arm seq_ft --suite libero_spatial --num_tasks 4 ...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-0}; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
mkdir -p logs results outputs
exec python -u L2_V4_seq.py "$@"
