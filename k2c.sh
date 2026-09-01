#!/usr/bin/env bash
# K2c — w-공간에서 task1 고유방향의 task6 커버율. 분석 전용, GPU 불필요.
#   bash k2c.sh              기본 (src=6, tgt=1)
#   bash k2c.sh --src 9 --tgt 1
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} \
       MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
mkdir -p results/K2c
python -u k2c_coverage_w.py "$@" 2>&1 | tee results/K2c/k2c.log
