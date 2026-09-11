#!/usr/bin/env bash
# K2d — 실제 수송 출력 b_1 의 방향별 공급률. 분석 전용, GPU 불필요.
#   bash k2d.sh                       기본 (stage 6 source=task6 -> tgt=task1)
#   bash k2d.sh --stage 9 --tgt 1
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} \
       MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
mkdir -p results/K2d
python -u k2d_supply_curve.py "$@" 2>&1 | tee results/K2d/k2d.log
