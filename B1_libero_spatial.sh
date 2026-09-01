#!/usr/bin/env bash
#
# B1 — libero_spatial 10태스크 전체. 기본 세팅은 CLARE/ER libero_spatial 표와 같다.
#
#   steps/task 20000, 학습 에피소드 50(전체), 롤아웃 100/칸, batch 32, seed 42
#   백본 dit_flow_mt_libero_90_pretrain (CLARE/ER 와 동일)
#
# 사용법
#   bash B1_libero_spatial.sh                # GPU 0, full run
#   bash B1_libero_spatial.sh smoke          # 2태스크 x 100스텝 x 2에피소드
#   bash B1_libero_spatial.sh 1              # GPU 1
#   bash B1_libero_spatial.sh baseline       # p_drop=0, lambda_anchor=0
#   bash B1_libero_spatial.sh report         # 학습 없이 표/리포트만 다시
#   bash B1_libero_spatial.sh mirror_e0      # E0/B1 4태스크 실행과 같은 세팅(빠름)
#
# 출력은 results/B1_libero_spatial/train.log 에 tee 된다.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    # shellcheck disable=SC1091
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh
    conda activate clare
fi

# shellcheck disable=SC1091
source "${HERE}/bash/clare/env.sh"

export MUJOCO_GL=${MUJOCO_GL:-egl}

EXTRA=()
MODE=ours
GPU=0
for a in "$@"; do
    case "${a}" in
        smoke)      EXTRA+=(--smoke) ;;
        baseline)   MODE=baseline ;;
        ours)       MODE=ours ;;
        report)     EXTRA+=(--report_only) ;;
        mirror_e0)  EXTRA+=(--mirror_e0) ;;
        [0-9]*)     GPU="${a}" ;;
        *)          EXTRA+=("${a}") ;;
    esac
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES}}

mkdir -p "${HERE}/results/B1_libero_spatial"
LOG="${HERE}/results/B1_libero_spatial/train.log"

echo "══ B1 libero_spatial  mode=${MODE}  gpu=${CUDA_VISIBLE_DEVICES}  extra=${EXTRA[*]:-none}" | tee -a "${LOG}"
echo "══ $(date '+%F %T %Z')  PRETRAIN_PATH=${PRETRAIN_PATH}" | tee -a "${LOG}"

python "${HERE}/B1_libero_spatial.py" --mode "${MODE}" "${EXTRA[@]}" 2>&1 | tee -a "${LOG}"
