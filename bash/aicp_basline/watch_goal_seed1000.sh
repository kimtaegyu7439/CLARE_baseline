#!/usr/bin/env bash
# libero_goal (seed1000) 큐가 끝나면 SR 표를 갱신한다.
# aicp_queue.sh 도 끝에 aicp_sr.py 를 부르지만, 큐가 죽어도 표가 갱신되도록 이중으로 건다.
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
W=logs/watch_goal_seed1000.log
ts(){ echo "[$(date '+%F %T')] $*" >> "$W"; }
ts "감시 시작 — libero_goal seed1000 (55칸 대기)"
while true; do
    n=$(grep -o "success_Libero_Goal_Task_[0-9]*:[0-9.]*" results/aicp/libero_goal_seed1000.log 2>/dev/null | wc -l)
    running=$(pgrep -u sa090180 -f "clare.py.*job_name=dit_flow_mt_cl_seed_1000_libero_goal" | wc -l)
    if [ "$n" -ge 55 ]; then ts "55칸 완료 (${n}) — SR 표 갱신"; break; fi
    if [ "$running" -eq 0 ]; then ts "프로세스 없음, 칸 ${n}/55 — 중단된 것으로 보고 표만 갱신"; break; fi
    sleep 300
done
python -u aicp_sr.py >> "$W" 2>&1
ts "완료 -> results/aicp_libero_SR.txt"
