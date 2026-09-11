#!/usr/bin/env bash
#
# B1 — condition dropout + counterfactual conditional anchoring
#
# 사용법
#   bash B1.sh smoke            # 2태스크 x 100스텝 x 2에피소드 (검증용)
#   bash B1.sh                  # full run (mode=ours)
#   bash B1.sh baseline         # full run (mode=baseline, seq-FT 등가)
#   bash B1.sh smoke baseline   # baseline smoke
#   CUDA_VISIBLE_DEVICES=2 bash B1.sh
#   bash B1.sh 2               # 첫 인자가 숫자면 GPU 번호로 본다
#
# 모든 출력은 results/B1/train.log 에 tee 된다.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# 이전 실험 스크립트들과 같은 환경. bash/*/*.sh 는 conda clare 가 활성화돼 있다고
# 가정하지만, 여기서는 비대화형 실행까지 되도록 직접 활성화한다.
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    # shellcheck disable=SC1091
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh
    conda activate clare
fi

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH — 기존 스크립트와 같은 경로 설정
# shellcheck disable=SC1091
source "${HERE}/bash/clare/env.sh"

export MUJOCO_GL=${MUJOCO_GL:-egl}

EXTRA=()
MODE=ours
for a in "$@"; do
    case "${a}" in
        smoke)     EXTRA+=(--smoke) ;;
        baseline)  MODE=baseline ;;
        ours)      MODE=ours ;;
        [0-9]*)    export CUDA_VISIBLE_DEVICES="${a}" ;;
        *)         EXTRA+=("${a}") ;;          # 그 밖의 인자는 B1.py 로 그대로 전달
    esac
done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES}}

mkdir -p "${HERE}/results/B1"
LOG="${HERE}/results/B1/train.log"

echo "══ B1  mode=${MODE}  gpu=${CUDA_VISIBLE_DEVICES}  extra=${EXTRA[*]:-none}" | tee -a "${LOG}"
echo "══ $(date '+%F %T')  PRETRAIN_PATH=${PRETRAIN_PATH}" | tee -a "${LOG}"

python "${HERE}/B1.py" --mode "${MODE}" "${EXTRA[@]}" 2>&1 | tee -a "${LOG}"
