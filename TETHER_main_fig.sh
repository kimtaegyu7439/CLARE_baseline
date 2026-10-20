#!/usr/bin/env bash
# TETHER 메인 2x4 그림. CUDA_VISIBLE_DEVICES 는 torch import 전에 잡혀야 한다.
#
#   bash TETHER_main_fig.sh                 GPU 1 에서 롤아웃 + 그림
#   GPU=0 bash TETHER_main_fig.sh
#   bash TETHER_main_fig.sh --plot_only     캐시로 그림만 다시
#   bash TETHER_main_fig.sh --axes xz       세로축을 z 로 (높이 방향 차이를 볼 때)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
mkdir -p logs results
LOG=logs/TETHER_main_fig.log
echo "[$(date '+%F %T')] ▶ GPU ${GPU} -> results/TETHER_main_fig" | tee "${LOG}"
python -u TETHER_main_fig.py "$@" 2>&1 | tee -a "${LOG}"
echo "[$(date '+%F %T')] ◀ 완료 (로그 ${LOG})" | tee -a "${LOG}"
