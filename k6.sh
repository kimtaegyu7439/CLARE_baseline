#!/usr/bin/env bash
# K6 — DiT(teacher) 출력 기하 관찰. U-신호의 생사 판정. 분석 전용, 기본 GPU 1.
#   bash k6.sh            /  bash k6.sh 1 --bins 2,5,8
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-1}; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
mkdir -p results/K6
echo "[$(date '+%F %T')] K6 시작  gpu=${GPU}"
python -u k6_probe.py "$@" 2>&1 | tee results/K6/k6.log
echo "[$(date '+%F %T')] K6 종료"
