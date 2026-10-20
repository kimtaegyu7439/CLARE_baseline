#!/usr/bin/env bash
# seed41 두 실행(spatial, libero_10)이 도는 동안 30분마다 리포트를 갱신한다.
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
W=logs/watch_l2cb_seed41.log
ts(){ echo "[$(date '+%F %T')] $*" >> "$W"; }
ts "감시 시작 — seed41 spatial + libero_10"
while true; do
    python -u L2_codebook_grad_report.py >> "$W" 2>&1
    n=$(pgrep -u sa090180 -f "l2_codebook.py --out results/l2_codebook_k96_grad.*seed41" | wc -l)
    [ "$n" -eq 0 ] && { ts "두 실행 모두 종료"; break; }
    sleep 1800
done
python -u L2_codebook_grad_report.py >> "$W" 2>&1
ts "완료 -> results/L2_codebook_grad_SR_matrix.txt"
