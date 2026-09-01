#!/usr/bin/env bash
#
# K3 — 공유기저 사영 + 분위수 사상만 (K1 에서 잔여 성분 제거).
#
#   bash k3.sh                                   기본: GPU 0, libero_spatial 10 task
#   bash k3.sh 0 K3_spatial_10task 10 libero_spatial
#   bash k3.sh 1 K3 4 libero_spatial             4 태스크판
#   SKIP_SMOKE=1 bash k3.sh ...
#
# 순서: (1) 연기시험 2 스테이지 -> sanity 확인  (2) 본 실행 nohup 백그라운드
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-0}; NAME=${2:-K3_spatial_10task}; NTASK=${3:-10}; SUITE=${4:-libero_spatial}
for _ in 1 2 3 4; do [ $# -gt 0 ] && shift; done

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
# 여러 팔이 같은 32 코어를 나눠 쓸 때의 스레싱 방지. 알고리즘과 무관.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
NUM_WORKERS=${NUM_WORKERS:-6}; STATS_WORKERS=${STATS_WORKERS:-4}

OUT="results/${NAME}"; SMOKE="results/${NAME}_smoke"
mkdir -p "${OUT}" "${SMOKE}"

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "[$(date '+%F %T')] K3 연기시험  gpu=${GPU}  suite=${SUITE}"
    python -u k3.py --out "${SMOKE}" --chunk_backward --smoke --suite "${SUITE}" \
        --stats_workers "${STATS_WORKERS}" "$@" \
        --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
        --num_workers "${NUM_WORKERS}" 2>&1 | tee "${SMOKE}/smoke.log"
    if ! grep -qE "\[K1\]\[sanity\] task[0-9]" "${SMOKE}/smoke.log"; then
        echo "[K3] 연기시험 실패 — sanity 로그가 없다. 중단한다." >&2; exit 1
    fi
    echo; echo "── 연기시험 sanity ──"
    grep -E "\[K1\]|단조위반|공유기저" "${SMOKE}/smoke.log" | tail -12
    echo "─────────────────────"; echo
fi

echo "[$(date '+%F %T')] K3 본 실행  gpu=${GPU}  out=${OUT}  suite=${SUITE}  tasks=${NTASK}"
nohup python -u k3.py --out "${OUT}" --chunk_backward --suite "${SUITE}" \
    --stats_workers "${STATS_WORKERS}" "$@" \
    --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
    --num_workers "${NUM_WORKERS}" > "${OUT}/train.log" 2>&1 &
echo $! > "${OUT}/run.pid"
echo "  pid $(cat "${OUT}/run.pid")   로그 ${OUT}/train.log"
echo "진행 확인:  tail -f ${OUT}/train.log   /   cat ${OUT}/sr_matrix.csv"
