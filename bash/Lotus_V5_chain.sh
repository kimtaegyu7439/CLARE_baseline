#!/usr/bin/env bash
# 지금 돌고 있는 V5 런이 **전부 끝나면** 다음 시드를 자동으로 띄운다.
#
#   NEXT_SEEDS=42 bash bash/Lotus_V5_chain.sh start
#   bash bash/Lotus_V5_chain.sh status
#   bash bash/Lotus_V5_chain.sh stop
#
# 왜 체인이 필요한가: 런처의 run_one 은 ${SEEDS} 를 서브셸에 구워 넣고 떠난다.
# 이미 뜬 런의 SEEDS 는 바꿀 수 없고, SEEDS=42 로 런처를 또 띄우면 같은 GPU 에
# 두 번째 런이 **동시에** 올라간다(already_running 은 seed42 태그를 보므로 막지
# 못한다). 그래서 기존 런이 사라질 때까지 기다린 뒤 띄운다.
#
# ★ 판정은 "L2_V5.py 파이썬 프로세스가 없다" 로 한다. 패턴에 --setting lotus 를
#   붙여 이 스크립트 자신의 명령줄과 겹치지 않게 한다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

NEXT_SEEDS=${NEXT_SEEDS:-42}
POLL=${POLL:-300}
SCOPE=${SCOPE:-lifelong}
JOBS=${JOBS-"goal:0.1:0 object:0.1:1 goal:0.01:2 object:0.01:3"}
PIDF="${HERE}/logs/.v5chain.pid"
LOGF="${HERE}/logs/v5_chain.log"
mkdir -p logs

running() { pgrep -f 'L2_V5\.py --setting lotus' >/dev/null 2>&1; }
alive()   { [ -f "${PIDF}" ] && kill -0 "$(cat "${PIDF}" 2>/dev/null)" 2>/dev/null; }

case "${1:-status}" in
start)
    alive && { echo "이미 예약돼 있다 (pid $(cat "${PIDF}"))"; exit 1; }
    setsid nohup "$0" _wait >> "${LOGF}" 2>&1 < /dev/null &
    sleep 2
    if alive; then
        echo "예약했다  pid $(cat "${PIDF}")"
        echo "  현재 런이 끝나면 seed ${NEXT_SEEDS} 로 같은 구성을 띄운다"
        echo "  확인   bash bash/Lotus_V5_chain.sh status"
        echo "  취소   bash bash/Lotus_V5_chain.sh stop"
        echo "  로그   ${LOGF#${HERE}/}"
    else
        echo "✗ 바로 죽었다"; tail -5 "${LOGF}"; exit 1
    fi
    ;;
_wait)
    echo $$ > "${PIDF}"
    trap 'rm -f "${PIDF}"; exit 0' TERM INT
    echo "[$(date '+%F %T')] 예약 시작 — 현재 런 종료 대기 (${POLL}s 주기), 다음 시드 ${NEXT_SEEDS}"
    while running; do sleep "${POLL}"; done
    echo "[$(date '+%F %T')] 현재 런 전부 종료됨. 완료 상태:"
    for j in ${JOBS}; do
        IFS=: read -r SU C _ <<< "${j}"
        t="lotus_v5act_c${C}_k25_scopeLL_${SU}_seed7"
        if grep -q '\[LOTUS\] 완료' "results/${t}/train.log" 2>/dev/null; then
            echo "    ✓ ${t}"
        else
            echo "    ✗ ${t}  — 미완(중단되었을 수 있다). 그래도 다음 시드로 간다"
        fi
    done
    echo "[$(date '+%F %T')] seed ${NEXT_SEEDS} 띄운다"
    SEEDS="${NEXT_SEEDS}" SCOPE="${SCOPE}" JOBS="${JOBS}" bash "${HERE}/bash/Lotus_V5_run.sh"
    echo "[$(date '+%F %T')] 예약 완료 — 체인 종료"
    rm -f "${PIDF}"
    ;;
status)
    if alive; then
        echo "● 예약 중  pid $(cat "${PIDF}")   다음 시드 ${NEXT_SEEDS}"
        echo -n "  현재 런: "; running && echo "진행 중 ($(pgrep -cf 'L2_V5\.py --setting lotus')개)" || echo "없음 — 곧 띄운다"
        tail -4 "${LOGF}" 2>/dev/null | sed 's/^/  /'
    else
        echo "○ 예약 없음"
    fi
    ;;
stop)
    alive && { kill -TERM "$(cat "${PIDF}")" 2>/dev/null; rm -f "${PIDF}"; echo "예약 취소"; } \
           || echo "예약 없음"
    ;;
*) echo "사용법: bash bash/Lotus_V5_chain.sh {start|status|stop}"; exit 1;;
esac
