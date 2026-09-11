#!/usr/bin/env bash
# K5b — witness 판별력 벤치. 학습 없음, 판정 전용. 기본 GPU 1.
#   bash k5b.sh          /  bash k5b.sh 1 --bins 2,5,8
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-1}; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
mkdir -p results/K5b
echo "[$(date '+%F %T')] K5b 시작  gpu=${GPU}"
python -u k5b_bench.py "$@" 2>&1 | tee results/K5b/k5b.log
echo "[$(date '+%F %T')] K5b 종료"
