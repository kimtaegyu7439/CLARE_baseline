#!/usr/bin/env bash
#
# K0 — 공유 기저 전이 검사. 학습 없음, 분석 전용.
#
#   bash k0.sh                       기본: GPU 0, libero_spatial 4 태스크
#   bash k0.sh 0 libero_spatial 4
#   bash k0.sh 2 libero_goal 10
#
# 임베딩은 results/K0/emb_cache/ 에 캐시되어 재실행 시 재사용된다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-0}; SUITE=${2:-libero_spatial}; NTASK=${3:-4}
OUT=${4:-results/K0}
# 임베딩 캐시는 실행끼리 공유한다 (같은 스위트면 같은 파일).
CACHE=${CACHE:-results/K0/emb_cache}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
# ER 3 팔이 같이 도는 중이라 스레드를 제한한다 (32 코어).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}

mkdir -p "${OUT}" "${CACHE}"
echo "[$(date '+%F %T')] K0 시작  gpu=${GPU}  suite=${SUITE}  tasks=0..$((NTASK-1))"
python -u k0_basis_check.py --suite "${SUITE}" --num_tasks "${NTASK}" \
    --out "${OUT}" --cache "${CACHE}" 2>&1 | tee "${OUT}/k0.log"
echo "[$(date '+%F %T')] K0 종료"
