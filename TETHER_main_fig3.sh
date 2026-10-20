#!/usr/bin/env bash
# TETHER 메인 그림 3 — 행동 + velocity + 보존 곡선을 한 장에.
# CUDA_VISIBLE_DEVICES 는 torch import 전에 잡혀야 한다.
#
#   bash TETHER_main_fig3.sh                 GPU 3 에서 측정 + 그림
#   GPU=1 bash TETHER_main_fig3.sh
#   bash TETHER_main_fig3.sh --plot_only     캐시로 그림만 다시
#   bash TETHER_main_fig3.sh --cond_mode lang_only   관측 고정, 명령어만 바꾼 통제
#
# 왼쪽 열은 TETHER_main_fig2 의 롤아웃 캐시를, 아래 곡선은 TETHER_main_fig 의
# 캐시를 그대로 쓴다. 새로 도는 것은 velocity 측정뿐이라 롤아웃이 필요 없다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${GPU:-3}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results
LOG=logs/TETHER_main_fig3.log
echo "[$(date '+%F %T')] ▶ GPU ${GPU} -> results/TETHER_main_fig3" | tee "${LOG}"
python -u TETHER_main_fig3.py "$@" 2>&1 | tee -a "${LOG}"
echo "[$(date '+%F %T')] ◀ 완료 (로그 ${LOG})" | tee -a "${LOG}"
