#!/usr/bin/env bash
#
# K1 실행 공통부. 각 스위트 스크립트가 source 해서 k1_run 을 부른다.
#
#   k1_run <GPU> <NAME> <NTASK> <SUITE> [k1.py 추가인자...]
#
# 하는 일: (1) 연기시험 2 스테이지 -> sanity 확인  (2) 본 실행 nohup 백그라운드.
# SKIP_SMOKE=1 이면 (1) 을 건너뛴다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

# 4 팔을 한 장비(32 코어)에서 동시에 돌리면 프로세스마다 torch 가 코어 전부를
# 잡아 스레싱한다. 실측: 통계 패스가 단독 116s -> 4 병렬 983s (8.5배). 스레드를
# 나눠 주고 워커 수도 낮춘다. 알고리즘에는 영향이 없다.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export K1_NUM_WORKERS=${K1_NUM_WORKERS:-6}
export K1_STATS_WORKERS=${K1_STATS_WORKERS:-4}

k1_run() {
    local GPU=$1 NAME=$2 NTASK=$3 SUITE=$4; shift 4
    export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
           MUJOCO_GL=${MUJOCO_GL:-egl}
    local OUT="results/${NAME}" SMOKE="results/${NAME}_smoke"
    mkdir -p "${OUT}" "${SMOKE}"

    # ── (1) 연기시험 ─────────────────────────────────────────────────────────
    # B1 의 --smoke 는 2 스테이지 x 100 스텝이다(B1.py:170-174). 스테이지 0 에서
    # 공유기저+분위수표를 만들고 스테이지 1 에서 수송/앵커가 도므로 sanity 는
    # 전부 탄다. 4 스테이지 전체를 도는 건 아니다.
    if [ "${SKIP_SMOKE:-0}" != "1" ]; then
        echo "[$(date '+%F %T')] K1 연기시험  gpu=${GPU}  suite=${SUITE}"
        python -u k1.py --out "${SMOKE}" --chunk_backward --smoke --suite "${SUITE}" \
            --stats_workers "${K1_STATS_WORKERS}" "$@" \
            --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
            --num_workers "${K1_NUM_WORKERS}" \
            2>&1 | tee "${SMOKE}/smoke.log"
        if ! grep -qE "\[K1\]\[sanity\] task[0-9]" "${SMOKE}/smoke.log"; then
            echo "[K1] 연기시험 실패 — sanity 로그가 없다. 중단한다." >&2; return 1
        fi
        echo; echo "── 연기시험 sanity ──"
        grep -E "\[K1\]|단조위반|공유기저" "${SMOKE}/smoke.log" | tail -12
        grep -q "⚠5%초과" "${SMOKE}/smoke.log" && echo "[K1] 경고: 범위밖 clamp 5% 초과" >&2
        echo "─────────────────────"; echo
    fi

    # ── (2) 본 실행 ──────────────────────────────────────────────────────────
    echo "[$(date '+%F %T')] K1 본 실행  gpu=${GPU}  out=${OUT}  suite=${SUITE}  tasks=${NTASK}"
    nohup python -u k1.py --out "${OUT}" --chunk_backward --suite "${SUITE}" \
        --stats_workers "${K1_STATS_WORKERS}" "$@" \
        --passthru --num_tasks "${NTASK}" --suite "${SUITE}" \
        --num_workers "${K1_NUM_WORKERS}" \
        > "${OUT}/train.log" 2>&1 &
    echo $! > "${OUT}/run.pid"
    echo "  pid $(cat "${OUT}/run.pid")   로그 ${OUT}/train.log"
    echo
    echo "진행 확인:"
    echo "  tail -f ${OUT}/train.log"
    echo "  cat ${OUT}/sr_matrix.csv"
    echo "  kill \$(cat ${OUT}/run.pid)"
}
