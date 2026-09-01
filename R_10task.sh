#!/usr/bin/env bash
# R10 / R11 을 libero_spatial 10 태스크로. 사용법: bash R_10task.sh <ARM> <GPU>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
ARM=${1:?R10 / R11 / R12}; GPU=${2:?GPU}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
OUT="results/${ARM}_10task"
mkdir -p "${OUT}" logs/mod0
echo "[$(date '+%F %T')] ${ARM} 10 태스크 시작 (gpu ${GPU})"
python -u "${ARM}.py" --out "${OUT}" --chunk_backward \
    --passthru --num_tasks 10 \
    --ckpt_root "outputs/${ARM}_10task" \
    2>&1 | tee -a "${OUT}/train.log"
echo "[$(date '+%F %T')] ${ARM} 10 태스크 종료"
