#!/usr/bin/env bash
# K 스윕이 도는 동안 results/Clare_setting_libero_goal.txt 를 주기적으로 갱신한다.
#
#   setsid nohup bash CLARE_goal_K_watch.sh > logs/CLARE_goal_K_watch.log 2>&1 &
#   INTERVAL=180 bash CLARE_goal_K_watch.sh      갱신 주기(초, 기본 300)
#   REPORT_ARGS="--ks 70,100,120,140 --seeds 42 --out results/Clare_setting_libero_goal2.txt" \
#     setsid nohup bash CLARE_goal_K_watch.sh &   다른 스윕을 같은 형식으로
#
# sr_matrix.csv 는 칸마다 갱신되므로 읽기만 하면 실시간 표가 된다. 도는 런은
# 건드리지 않는다. 큐도 러너도 사라지면 한 번 더 쓰고 빠진다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
INTERVAL=${INTERVAL:-300}
# 비워 두면 리포트 기본값(K 90~120, 시드 3 개)이라 기존 동작이 그대로다.
REPORT_ARGS=${REPORT_ARGS:-}

while true; do
    python -u CLARE_goal_K_report.py ${REPORT_ARGS} 2>&1 | tail -1
    # ★ 학습(L2_V4_seq.py)뿐 아니라 칸 워커도 살아 있으면 계속 돈다. 학습을
    #   --eval_after_each_task false 로 떼어내고 롤아웃을 별도 프로세스에 돌리면
    #   학습이 먼저 끝나므로, 학습만 보고 빠지면 칸이 덜 찬 표가 최종본이 된다.
    if ! pgrep -f "L2_V4_seq.py" > /dev/null 2>&1 \
       && ! pgrep -f "CLARE_goal_K_queue.sh" > /dev/null 2>&1 \
       && ! pgrep -f "dispatch.py --gpu" > /dev/null 2>&1 \
       && ! pgrep -f "cell_worker.py" > /dev/null 2>&1; then
        sleep 60
        if ! pgrep -f "L2_V4_seq.py" > /dev/null 2>&1 \
           && ! pgrep -f "dispatch.py --gpu" > /dev/null 2>&1 \
           && ! pgrep -f "cell_worker.py" > /dev/null 2>&1; then
            python -u CLARE_goal_K_report.py ${REPORT_ARGS} 2>&1 | tail -1
            echo "[$(date '+%F %T')] K 스윕이 모두 끝났다 — 최종본 쓰고 종료"
            break
        fi
    fi
    sleep "${INTERVAL}"
done
