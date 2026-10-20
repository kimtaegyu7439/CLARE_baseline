#!/usr/bin/env bash
# GPU 3 에서 지금 도는 런이 끝나면 이어서 lotus_seq_50_seed42 를 처음부터 돌린다.
#
#   PID=858303 bash LOTUS_seq_after_current.sh &
#
# ★ 처음부터 돈다. L2_seq_cl.py 에는 재개 경로가 없어(stage 루프가 항상 0 부터)
#   죽은 런의 stage1~4 체크포인트를 이어받을 수 없다. 중간 재개는 옵티마이저·
#   스케줄러·RNG 스트림이 끊겨 seed 대조의 의미가 흐려지므로, 기준선 실험에서는
#   처음부터 돌리는 쪽이 맞다.
#
# 끝나면 results/LOTUS_Seq_CL_SR.txt 가 갱신된다(LOTUS_seq_cl_queue.sh 안에서).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"

PID=${PID:?기다릴 PID 를 지정해라}
GPU=${GPU:-3}
SUITE=${SUITE:-50}
SEED=${SEED:-42}
QUEUE="${HERE}/logs/LOTUS_seq_cl.queue"

echo "[$(date '+%F %T')] ⏳ PID ${PID} 종료를 기다린다 — 끝나면 ${SUITE} seed${SEED} 시작 (GPU ${GPU})" \
    | tee -a "${QUEUE}"
while kill -0 "${PID}" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] PID ${PID} 종료 확인 — 이어서 시작" | tee -a "${QUEUE}"

# 죽은 런이 남긴 체크포인트/진행 기록을 치운다. 그대로 두면 완주 검사가 헷갈리고
# diag.csv 에 옛 stage 기록이 섞인다.
OLD="results/lotus_seq_${SUITE}_seed${SEED}"
if [ -d "${OLD}" ]; then
    BAK="${OLD}.killed_$(date '+%m%d_%H%M')"
    mv "${OLD}" "${BAK}"
    echo "[$(date '+%F %T')] 이전 잔여물 -> ${BAK}" | tee -a "${QUEUE}"
fi

SUITES="${SUITE}" SEEDS="${SEED}" GPU="${GPU}" bash LOTUS_seq_cl_queue.sh
