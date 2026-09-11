#!/usr/bin/env bash
# GPU 3 순차 큐 — libero_goal 이 끝나면 이어서 libero_object.
#   L2_codebook v3(--grad_enable), 92.5 arm 과 같은 설정.
# 사용: nohup bash bash/queue_l2cb_gpu3.sh <앞 잡의 pid> &
set -uo pipefail
cd /home/sa090180/clare
Q=logs/queue_l2cb_gpu3.log
ts(){ echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

WAIT_PID=${1:-}
if [ -n "${WAIT_PID}" ]; then
    ts "대기 — pid ${WAIT_PID} (libero_goal) 종료를 기다린다"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 120; done
    ts "libero_goal 종료 확인"
fi
sleep 60                     # GPU 메모리 반납 여유

NAME=l2_codebook_k96_grad_libero_object
if [ -d "outputs/L2_codebook/libero_object_seed42_ours" ]; then
    ts "★ ckpt_root 가 이미 있다 — 중복 실행 방지로 중단"; exit 1
fi
ts "▶ ${NAME} 시작 (gpu 3, suite libero_object)"
bash l2_codebook.sh 3 "${NAME}" 10 libero_object --grad_enable
sleep 30
ts "  pid $(cat results/${NAME}/run.pid 2>/dev/null) — 로그 results/${NAME}/train.log"

# 이 실행이 끝날 때까지 지켜보며 리포트를 갱신한다
while pgrep -u sa090180 -f "l2_codebook.py --out results/${NAME}" > /dev/null; do
    python -u L2_codebook_grad_report.py >> "$Q" 2>&1
    sleep 1800
done
python -u L2_codebook_grad_report.py >> "$Q" 2>&1
ts "◀ ${NAME} 종료 — 리포트 갱신 완료"
