#!/usr/bin/env bash
# 지정한 큐가 끝나면 이어서 한 런을 돌린다.
# 사용법: bash B_mod_tail.sh <GPU> <대기할 pid> <이름> <스크립트> <인자...>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=$1; WAIT_PID=$2; NAME=$3; SCRIPT=$4; shift 4
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p logs/mod results/mod
echo "[$(date '+%F %T')] ${NAME}: pid ${WAIT_PID} (gpu ${GPU} 큐) 종료 대기"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] === ${NAME} 시작 (gpu ${GPU})"
python -u "${SCRIPT}" "$@" \
    --anchor_agg sum --out_dir "results/mod/${NAME}" --ckpt_root "outputs/mod/${NAME}" \
    >"logs/mod/${NAME}.log" 2>&1 \
    && echo "[$(date '+%F %T')] === ${NAME} 완료" \
    || echo "[$(date '+%F %T')] === ${NAME} 실패"
