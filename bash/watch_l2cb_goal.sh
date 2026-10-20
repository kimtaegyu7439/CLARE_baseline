#!/usr/bin/env bash
# l2_codebook_k96_grad_libero_goal 이 끝나면 L2_codebook_grad_SR_matrix.txt 를 갱신한다.
# 진행 중에도 30분마다 갱신해서 부분 집계를 볼 수 있게 한다.
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
W=logs/watch_l2cb_goal.log
CSV=results/l2_codebook_k96_grad_libero_goal/sr_matrix.csv
ts(){ echo "[$(date '+%F %T')] $*" >> "$W"; }
ts "감시 시작 — l2_codebook_k96_grad_libero_goal (55칸 대기)"
while true; do
    n=$(grep -o "[0-9]\+\.[0-9]" "$CSV" 2>/dev/null | wc -l)
    run=$(pgrep -u sa090180 -f "l2_codebook.py --out results/l2_codebook_k96_grad_libero_goal" | wc -l)
    python -u L2_codebook_grad_report.py >> "$W" 2>&1
    if [ "$n" -ge 55 ]; then ts "55칸 완료 — 최종 갱신"; break; fi
    if [ "$run" -eq 0 ]; then ts "프로세스 없음, 칸 ${n}/55 — 중단으로 보고 종료"; break; fi
    sleep 1800
done
python -u L2_codebook_grad_report.py >> "$W" 2>&1
ts "완료 -> results/L2_codebook_grad_SR_matrix.txt"
