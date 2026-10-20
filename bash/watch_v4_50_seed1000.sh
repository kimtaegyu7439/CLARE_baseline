#!/usr/bin/env bash
# LOTUS v4 suite 50 / train seed 1000 이 끝나면 LOTUS_v4_SR_matrix.txt 를 갱신한다.
# 진행 중에도 30분마다 갱신해서 부분 집계(n/225칸)를 볼 수 있게 한다.
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
W=logs/watch_v4_50_seed1000.log
OUT=results/l2_codebook_v4_50_seed1000
ts(){ echo "[$(date '+%F %T')] $*" >> "$W"; }
ts "감시 시작 — l2_codebook_v4_50_seed1000 (225칸 대기, GPU 1)"
while true; do
    run=$(pgrep -u sa090180 -f "Lotus_l2cb_cl.py --suite 50 .*--out ${OUT}" | wc -l)
    python -u lotus_v4_report.py >> "$W" 2>&1
    # metrics.json 은 학습 루프가 전부 끝난 뒤에만 쓰인다 = 정상 완료 신호.
    if [ -f "${OUT}/metrics.json" ]; then ts "metrics.json 확인 — 정상 완료"; break; fi
    if [ "$run" -eq 0 ]; then ts "프로세스 없음 + metrics.json 없음 — 중단으로 보고 종료"; break; fi
    sleep 1800
done
python -u lotus_v4_report.py >> "$W" 2>&1
ts "완료 -> results/LOTUS_v4_SR_matrix.txt"
