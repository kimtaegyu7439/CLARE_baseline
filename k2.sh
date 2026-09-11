#!/usr/bin/env bash
#
# K2 — task1 붕괴 원인 판정 (원료 커버리지 H1 vs 좌표 무관 누적 H2). 분석 전용.
#
#   bash k2.sh                 기본
#   bash k2.sh 1               보충 추출이 필요할 때 쓸 GPU (기본 1)
#
# 임베딩은 results/K0/emb_cache/ 를 재사용한다. 없는 태스크가 있으면
# k0_basis_check.py 로 먼저 채운 뒤 분석한다(그 경우에만 GPU 사용).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-1}; SUITE=${SUITE:-libero_spatial}; NTASK=${NTASK:-10}
CACHE=${CACHE:-results/K0/emb_cache}
OUT=${OUT:-results/K2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
mkdir -p "${OUT}"

# ── 캐시 확인 ────────────────────────────────────────────────────────────────
MISSING=0
for k in $(seq 0 $((NTASK-1))); do
    [ -f "${CACHE}/${SUITE}_task${k}.pt" ] || MISSING=1
done
if [ "${MISSING}" = "1" ]; then
    echo "[K2] 캐시 누락 -> k0_basis_check.py 로 보충 (gpu ${GPU})"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl} \
      python -u k0_basis_check.py --suite "${SUITE}" --num_tasks "${NTASK}" \
        --out "${OUT}/_fill" --cache "${CACHE}" >"${OUT}/fill.log" 2>&1 \
        || { echo "[K2] 보충 추출 실패 — ${OUT}/fill.log 확인"; exit 1; }
else
    echo "[K2] 임베딩 캐시 ${NTASK}/${NTASK} 확인 — GPU 불필요"
fi

echo "[$(date '+%F %T')] K2 분석 시작"
python -u k2_coverage.py --suite "${SUITE}" --num_tasks "${NTASK}" \
    --cache "${CACHE}" --out "${OUT}" 2>&1 | tee "${OUT}/k2.log"
echo "[$(date '+%F %T')] K2 종료"
