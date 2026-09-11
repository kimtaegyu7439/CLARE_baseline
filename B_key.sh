#!/usr/bin/env bash
# GPU 가 빌 때까지 기다렸다가 명령어 교체 실험을 돌린다.
# 사용법: bash B_key.sh <GPU> [기다릴 pid]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-3}; WAIT_PID=${2:-}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p logs/mod
if [ -n "${WAIT_PID}" ]; then
    echo "[$(date '+%F %T')] pid ${WAIT_PID} 종료 대기 (gpu ${GPU})"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] 대기 종료 — B_key 시작"
fi
python -u B_key.py "${@:3}" 2>&1 | tee -a logs/mod/B_key.log
echo "[$(date '+%F %T')] B_key 완료"
