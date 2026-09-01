#!/usr/bin/env bash
# ER libero_40 평가 820칸이 다 찰 때까지 기다렸다가 Fig.4 스타일 2패널 그림을 그린다.
# 30분마다 중간 그림도 갱신하므로, 다 끝나기 전에도 figs/libero40_sr.png를 열어보면 된다.
#
#   setsid nohup bash tools/watch_and_plot_libero40.sh > /tmp/plot_libero40.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ER_DIR=outputs/ER_eval/libero_40/seed42
CLARE_SR=outputs/CLARE_eval/libero_40/seed42/libero_40_SR.txt
OUT=figs/libero40_sr.pdf
TOTAL=820

plot() { python tools/plot_sr_matrix.py --panel CLARE "${CLARE_SR}" --panel ER "${ER_DIR}" -o "${OUT}"; }

while :; do
    n=$(find "${ER_DIR}" -name eval_info.json 2>/dev/null | wc -l)
    echo "[$(date +%F\ %T)] ER ${n}/${TOTAL}"
    plot
    [ "${n}" -ge "${TOTAL}" ] && break
    # 평가 프로세스가 다 죽었으면 더 기다려도 늘지 않는다.
    if ! pgrep -f "eval_libero_40.sh" > /dev/null; then
        echo "[$(date +%F\ %T)] eval_libero_40.sh 프로세스가 없다 — ${n}/${TOTAL}에서 멈춤. 마지막 그림만 남기고 종료."
        break
    fi
    sleep 1800
done

echo "[$(date +%F\ %T)] done -> ${OUT}"
