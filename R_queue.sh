#!/usr/bin/env bash
#
# 앞선 작업이 끝나면 이어서 한 팔을 돌린다.
# 사용법: bash R_queue.sh <GPU> <대기 패턴> <ARM> <태스크수> [추가인자...]
#   대기 패턴은 pgrep -f 로 찾는다. 빈 문자열이면 즉시 시작.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=$1; WAITPAT=$2; ARM=$3; NT=$4; shift 4
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
SUF=$([ "${NT}" -eq 4 ] && echo "" || echo "_${NT}task")
OUT="results/${ARM}${SUF}"
mkdir -p logs/mod0

if [ -n "${WAITPAT}" ]; then
    echo "[$(date '+%F %T')] '${WAITPAT}' 종료 대기 (gpu ${GPU}, 다음: ${ARM} ${NT}태스크)"
    # ★ 자기 자신을 잡지 않도록 제외한다. 대기 패턴이 이 스크립트의 argv 에도
    #   들어 있어서, 그냥 pgrep -f 하면 영원히 자기를 찾는다.
    while pgrep -f "${WAITPAT}" | grep -vx "$$" | grep -q .; do sleep 60; done
    echo "[$(date '+%F %T')] 대기 종료"
fi

echo "[$(date '+%F %T')] === ${ARM} ${NT}태스크 시작 (gpu ${GPU})"
rm -rf "${OUT}" "outputs/${ARM}${SUF}"; mkdir -p "${OUT}"
if python -u "${ARM}.py" --out "${OUT}" --chunk_backward \
        --passthru --num_tasks "${NT}" \
        --ckpt_root "outputs/${ARM}${SUF}" "$@" > "${OUT}/train.log" 2>&1; then
    echo "[$(date '+%F %T')] === ${ARM} ${NT}태스크 완료"
else
    echo "[$(date '+%F %T')] === ${ARM} ${NT}태스크 실패 -> ${OUT}/train.log"
fi
