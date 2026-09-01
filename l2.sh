#!/usr/bin/env bash
#
# L2 — teacher-부트스트랩 x_t (R13 의 앵커 x_t 를 task-j 행동으로 보간).
#
#   bash l2.sh                                  기본: GPU 0, libero_spatial 10 task
#   bash l2.sh 0 L2 10 libero_spatial
#   SKIP_SMOKE=1 bash l2.sh ...                 연기시험 없이 바로 본 실행
#
# 연기시험은 결과용이 아니라 **크래시 검사**다. 2 태스크 x 100 스텝이라
# 스테이지 1 에서 부트스트랩 경로 + sanity + 내장 진단이 전부 돈다.
# 밤새 도는 본 실행이 stage 1 에서 죽는 것을 막는 용도다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-0}; NAME=${2:-L2}; NTASK=${3:-10}; SUITE=${4:-libero_spatial}
for _ in 1 2 3 4; do [ $# -gt 0 ] && shift; done

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
NUM_WORKERS=${NUM_WORKERS:-6}
XT_MODE=${XT_MODE:-teacher}

OUT="results/${NAME}"; SMOKE="results/${NAME}_smoke"
mkdir -p "${OUT}" "${SMOKE}"

COMMON=(--chunk_backward --xt_mode "${XT_MODE}")

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "[$(date '+%F %T')] L2 연기시험(크래시 검사)  gpu=${GPU}  suite=${SUITE}"
    python -u l2.py --out "${SMOKE}" "${COMMON[@]}" --smoke "$@" \
        --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
        --num_workers "${NUM_WORKERS}" 2>&1 | tee "${SMOKE}/smoke.log"
    if ! grep -q "\[L2\]\[sanity\] task1" "${SMOKE}/smoke.log"; then
        echo "[L2] 연기시험 실패 — sanity 로그가 없다. 중단한다." >&2; exit 1
    fi
    if ! grep -q "\[L2\]\[diag\]" "${SMOKE}/smoke.log"; then
        echo "[L2] 연기시험 실패 — 내장 진단(diag)이 안 돌았다. 중단한다." >&2; exit 1
    fi
    echo; echo "── 연기시험 sanity ──"
    grep -E "\[L2\]" "${SMOKE}/smoke.log" | tail -14
    echo "─────────────────────"; echo
fi

echo "[$(date '+%F %T')] L2 본 실행  gpu=${GPU}  out=${OUT}  suite=${SUITE}  tasks=${NTASK}"
nohup python -u l2.py --out "${OUT}" "${COMMON[@]}" "$@" \
    --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
    --num_workers "${NUM_WORKERS}" > "${OUT}/train.log" 2>&1 &
echo $! > "${OUT}/run.pid"
echo "  pid $(cat "${OUT}/run.pid")   로그 ${OUT}/train.log"
echo "진행 확인:  tail -f ${OUT}/train.log   /   cat ${OUT}/sr_matrix.csv"
