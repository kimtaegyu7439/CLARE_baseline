#!/usr/bin/env bash
#
# AICP baseline 순차 큐 — GPU 하나에 한 번에 하나씩, 끝나면 다음 스위트.
#
#   bash bash/aicp_basline/aicp_queue.sh <GPU> <SEED> <suite> [suite ...]
#   예) bash bash/aicp_basline/aicp_queue.sh 1 7 spatial object
#       LOG_TAG=seed42 bash .../aicp_queue.sh 1 42 object spatial
#
# LOG_TAG 를 주면 로그가 results/aicp/libero_<suite>_<TAG>.log 로 나간다.
# 같은 스위트를 다른 시드로 다시 돌릴 때 기존 로그를 덮어쓰지 않기 위한 것이다.
# (SR 은 aicp_sr.py 가 libero_<suite>*.log 를 전부 훑어 태그별로 표를 낸다.)
#
# 각 스위트의 stdout 을 results/aicp/<suite>.log 로 남긴다. SR 은 그 로그에서
# aicp_sr.py 가 뽑는다(clare.py 는 SR 을 json 으로 안 남기고 logging 으로만 찍는다).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${HERE}/../.." && pwd)"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
GPU=$1; SEED=$2; shift 2
mkdir -p results/aicp
TAG=${LOG_TAG:+_${LOG_TAG}}
Q=results/aicp/queue_gpu${GPU}${TAG}.log
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${Q}"; }

ts "GPU ${GPU} 큐 시작  seed=${SEED}  EVAL_SEED=${EVAL_SEED:-(env)}  tag=${LOG_TAG:-없음}  순서: $*"
for s in "$@"; do
    ts "  ▶ libero_${s} 시작"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
        bash "${HERE}/aicp_clare_libero_${s}.sh" "${SEED}" \
        > "results/aicp/libero_${s}${TAG}.log" 2>&1
    rc=$?
    ts "  ◀ libero_${s} 종료 rc=${rc}"
    python -u aicp_sr.py >> "${Q}" 2>&1 || true
done
ts "GPU ${GPU} 큐 완료"
