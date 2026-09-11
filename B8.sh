#!/usr/bin/env bash
#
# B8 — 정답 충돌량 G-hat 로 앵커를 가중 (B1 + 가중 하나)
# 세팅은 B1 기본값 그대로: libero_spatial 태스크 0..3, 5000 steps, 45 에피소드, 롤아웃 20, seed 42
#
#   bash B8.sh            # GPU 3
#   bash B8.sh smoke
#   bash B8.sh 3          # GPU 지정
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export MUJOCO_GL=${MUJOCO_GL:-egl}
EXTRA=(); GPU=3
for a in "$@"; do case "${a}" in
    smoke) EXTRA+=(--smoke) ;;
    [0-9]*) GPU="${a}" ;;
    *) EXTRA+=("${a}") ;;
esac; done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES}}
mkdir -p "${HERE}/results/B8"
LOG="${HERE}/results/B8/train.log"
echo "══ B8  gpu=${CUDA_VISIBLE_DEVICES}  extra=${EXTRA[*]:-none}  $(date '+%F %T %Z')" | tee -a "${LOG}"
python "${HERE}/B8.py" "${EXTRA[@]}" 2>&1 | tee -a "${LOG}"
