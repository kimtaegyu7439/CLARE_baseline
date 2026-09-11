#!/usr/bin/env bash
#
# K5 — 가우시안 샘플 + witness 유도 manifold 정련.
#
#   bash k5.sh                                       기본: GPU 1, spatial 10 task, M=8
#   bash k5.sh 1 K5_spatial_10task 10 libero_spatial --M 8
#   SMOKE_ONLY=1 bash k5.sh 1                        연기시험(M=0 / M=4 / M=8)만
#   SKIP_SMOKE=1 bash k5.sh 1                        본 실행만
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-1}; NAME=${2:-K5_spatial_10task}; NTASK=${3:-10}; SUITE=${4:-libero_spatial}
for _ in 1 2 3 4; do [ $# -gt 0 ] && shift; done
M=${M:-8}; RHO=${RHO:-0.3}; BEVERY=${BEVERY:-4}

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
NUM_WORKERS=${NUM_WORKERS:-6}

OUT="results/${NAME}"; mkdir -p "${OUT}"

# ── (1) 연기시험: M = 0 / 4 / 8 ──────────────────────────────────────────────
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  for m in ${SMOKE_MS:-0 4 8}; do
    S="results/${NAME}_smoke_M${m}"; mkdir -p "${S}"
    echo "[$(date '+%F %T')] K5 연기시험 M=${m}  gpu=${GPU}  suite=${SUITE}"
    python -u k5_train.py --out "${S}" --chunk_backward --smoke --suite "${SUITE}" \
        --M "${m}" --rho_clip "${RHO}" --blocks_every "${BEVERY}" \
        --stats_workers 4 "$@" \
        --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
        --num_workers "${NUM_WORKERS}" 2>&1 | tee "${S}/smoke.log"
    if ! grep -qE "\[K5\]\[sanity\] task[0-9]" "${S}/smoke.log"; then
        echo "[K5] 연기시험 M=${m} 실패 — sanity 로그 없음. 중단." >&2; exit 1
    fi
    echo; echo "── M=${m} sanity ──"
    grep -E "\[K5\]|판별력|η 자동" "${S}/smoke.log" | tail -12
    echo "──────────────────"; echo
  done
fi
[ "${SMOKE_ONLY:-0}" = "1" ] && { echo "[K5] SMOKE_ONLY — 본 실행 생략"; exit 0; }

# ── (2) 본 실행 ──────────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] K5 본 실행  gpu=${GPU} out=${OUT} suite=${SUITE} tasks=${NTASK} M=${M}"
nohup python -u k5_train.py --out "${OUT}" --chunk_backward --suite "${SUITE}" \
    --M "${M}" --rho_clip "${RHO}" --blocks_every "${BEVERY}" --stats_workers 4 "$@" \
    --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
    --num_workers "${NUM_WORKERS}" > "${OUT}/train.log" 2>&1 &
echo $! > "${OUT}/run.pid"
echo "  pid $(cat ${OUT}/run.pid)   로그 ${OUT}/train.log"
