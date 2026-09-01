#!/usr/bin/env bash
# 적분 후 액션 청크 오차 측정. 사용법: bash B_chunk.sh [GPU]
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clare
source bash/clare/env.sh
export CUDA_VISIBLE_DEVICES="${1:-0}"
cd "$(dirname "$0")"
python B_chunk.py "${@:2}"
