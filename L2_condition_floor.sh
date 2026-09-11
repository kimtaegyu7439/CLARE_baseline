#!/usr/bin/env bash
# 조건 붕괴 바닥(floor) 측정. CUDA_VISIBLE_DEVICES 는 torch import 전에 잡혀야 한다.
#
#   bash L2_condition_floor.sh [gpu] [추가 인자...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
# 데이터셋/모델 루트. 이걸 안 잡으면 HF_LEROBOT_HOME 이 ~/.cache 로 떨어져
# task_0~3 만 있는 낡은 사본을 읽는다(2026-08 잔재).
source "${HERE}/bash/clare/env.sh"
GPU=${1:-0}; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${GPU}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p logs results
exec python -u L2_condition_floor.py "$@"
