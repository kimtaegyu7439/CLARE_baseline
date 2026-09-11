#!/usr/bin/env bash
#
# GPU 0 예약 큐 — GPU 1 큐가 아직 못 간 후속 팔을 먼저 가져간다.
#
#   bash scripts/queue_gpu1.sh [기다릴PID]
#
# 순서
#   1. GPU 1 이 빌 때까지 대기 (60초 간격, memory<1000MiB & util<5% 가 5회 연속).
#      인자로 PID 를 주면 그 PID 종료도 조건에 넣는다.
#   2. 스모크 — task 2 stage, 60 step, 배치 축소로 세 플래그를 켠 채 끝까지.
#      실패하면 본 run 을 띄우지 않고 중단한다.
#   3. 본 run  l2_codebook_v4_hard_sf_k96   (hard_linear + speedinv + speedinv_ema)
#   4. 정상 종료하면 후속 팔  l2_codebook_v4_hard_only_k96  (샘플러 변경만)
#      — GPU 0 이 먼저 비면 queue_gpu0.sh 가 가져가므로 여기서는 락으로 중복을 막는다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${GPU:-0}
WAIT_PID=${1:-}
SUITE=${SUITE:-goal}
STEPS=${STEPS:-10000}
LOG=logs/queue_gpu${GPU}.log
mkdir -p logs scripts results
ts(){ echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ── 1. GPU 대기 ──────────────────────────────────────────────────────────────
ts "대기 시작 — GPU ${GPU} (memory<1000MiB & util<5% 5회 연속)${WAIT_PID:+, pid ${WAIT_PID} 종료도 조건}"
ok=0
while true; do
    read -r mem util < <(nvidia-smi -i "${GPU}" \
        --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
    busy_pid=0
    [ -n "${WAIT_PID}" ] && kill -0 "${WAIT_PID}" 2>/dev/null && busy_pid=1
    if [ "${mem:-9999}" -lt 1000 ] && [ "${util:-100}" -lt 5 ] && [ "${busy_pid}" -eq 0 ]; then
        ok=$((ok+1))
    else
        [ $((ok)) -gt 0 ] && ts "  리셋 (mem=${mem} util=${util} pid_alive=${busy_pid})"
        ok=0
    fi
    [ "${ok}" -ge 5 ] && break
    sleep 60
done
ts "GPU ${GPU} 비었음 확인 (${ok}회 연속)"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}

# ── 2. 스모크는 GPU 1 큐가 담당한다. 여기서는 후속 팔만 가져간다.
#     락(results/<이름>.lock)으로 두 큐가 같은 팔을 두 번 돌리지 않게 막는다.

# ── 3·4. 본 run 과 후속 팔 ───────────────────────────────────────────────────
run_arm() {   # run_arm <이름> <추가 인자...>
    local NAME=$1; shift
    local OUT="results/${NAME}"
    if ! mkdir "${OUT}.lock" 2>/dev/null; then
        ts "  ${NAME} 은 다른 큐가 이미 가져갔다 — 건너뜀"; return 0
    fi
    mkdir -p "${OUT}"
    ts "▶ ${NAME} 시작 (gpu ${GPU})"
    python -u Lotus_l2cb_cl.py --suite "${SUITE}" --steps "${STEPS}" \
        --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \
        --num_workers "${NUM_WORKERS:-6}" --out "${OUT}" "$@" \
        > "logs/${NAME}.log" 2>&1 &
    local PID=$!
    {
        echo "# ${NAME}"; echo; echo "- 시작: $(date '+%F %T')"; echo "- PID: ${PID}"
        echo "- GPU: ${GPU}"; echo "- config: ${OUT}/config.json"
        echo "- 로그: logs/${NAME}.log"; echo
        echo '```'; echo "python -u Lotus_l2cb_cl.py --suite ${SUITE} --steps ${STEPS} \\"
        echo "    --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \\"
        echo "    --num_workers ${NUM_WORKERS:-6} --out ${OUT} $*"; echo '```'
    } > "${OUT}/RUN_INFO.md"
    sleep 120
    if ! kill -0 "${PID}" 2>/dev/null; then
        ts "★ ${NAME} 이 120초 안에 죽었다"
        { echo; echo "## 크래시 (120초 내)"; echo '```'; tail -40 "logs/${NAME}.log"; echo '```'; } \
            >> "${OUT}/RUN_INFO.md"
        return 1
    fi
    ts "  ${NAME} pid ${PID} 생존 확인, 학습 진행 중"
    wait "${PID}"; local r=$?
    ts "◀ ${NAME} 종료 rc=${r}"
    echo "- 종료: $(date '+%F %T') rc=${r}" >> "${OUT}/RUN_INFO.md"
    return "${r}"
}

# 본 run 이 아직 안 끝났으면 후속 팔만 먼저 가져간다 (락으로 중복 방지)
run_arm l2_codebook_v4_hard_only_k96 --sampler_mode hard_linear

ts "큐 완료"
