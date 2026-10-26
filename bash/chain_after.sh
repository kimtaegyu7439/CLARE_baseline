#!/usr/bin/env bash
# 지정한 런들이 **전부 완료되면** 명령을 하나 실행하는 범용 예약기.
#
#   DONE_LOGS="results/A/train.log results/B/train.log" \
#   LAUNCH_CMD='SCOPE=lifelong JOBS="goal:4.0:0 object:4.0:1" bash bash/Lotus_V13_run.sh' \
#   NAME=v13b4 bash bash/chain_after.sh start
#   NAME=v13b4 bash bash/chain_after.sh status | stop
#
# 완료 판정은 **양성 마커**("[LOTUS] 완료" 가 train.log 에 찍힘)로 한다.
# "프로세스가 없다" 로 판정하면 시드와 시드 사이의 짧은 틈(파이썬이 끝나고
# 다음 시드가 뜨기 전)에 폴링이 걸려 **거짓 완료**가 난다. 마커는 그 틈에
# 안 흔들린다. 마커 파일이 아직 없는 경우(그 시드가 아직 시작도 안 함)도
# "미완"으로 본다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

NAME=${NAME:-chain}
DONE_LOGS=${DONE_LOGS:-}
LAUNCH_CMD=${LAUNCH_CMD:-}
POLL=${POLL:-300}
PIDF="${HERE}/logs/.chain_${NAME}.pid"
LOGF="${HERE}/logs/chain_${NAME}.log"
mkdir -p logs

all_done() {
    local f
    for f in ${DONE_LOGS}; do
        grep -q '\[LOTUS\] 완료' "${f}" 2>/dev/null || return 1
    done
    return 0
}
alive() { [ -f "${PIDF}" ] && kill -0 "$(cat "${PIDF}" 2>/dev/null)" 2>/dev/null; }

case "${1:-status}" in
start)
    [ -n "${DONE_LOGS}" ] && [ -n "${LAUNCH_CMD}" ] || { echo "DONE_LOGS 와 LAUNCH_CMD 가 필요하다"; exit 1; }
    alive && { echo "이미 예약돼 있다 (pid $(cat "${PIDF}"))"; exit 1; }
    # 서브셸에 환경으로 넘긴다 — 인자로 넘기면 명령줄에 패턴이 들어가 pgrep 류와 꼬인다
    DONE_LOGS="${DONE_LOGS}" LAUNCH_CMD="${LAUNCH_CMD}" POLL="${POLL}" NAME="${NAME}" \
        setsid nohup "$0" _wait >> "${LOGF}" 2>&1 < /dev/null &
    sleep 2
    if alive; then
        echo "예약했다  pid $(cat "${PIDF}")  [${NAME}]"
        echo "  기다리는 것: $(echo ${DONE_LOGS} | wc -w)개 런의 '[LOTUS] 완료'"
        echo "  실행할 것  : ${LAUNCH_CMD}"
        echo "  확인 NAME=${NAME} bash bash/chain_after.sh status   취소 ... stop"
    else
        echo "✗ 바로 죽었다"; tail -5 "${LOGF}"; exit 1
    fi
    ;;
_wait)
    echo $$ > "${PIDF}"
    trap 'rm -f "${PIDF}"; exit 0' TERM INT
    echo "[$(date '+%F %T')] [${NAME}] 예약 시작 — ${POLL}s 주기"
    for f in ${DONE_LOGS}; do echo "    기다림: ${f}"; done
    while ! all_done; do sleep "${POLL}"; done
    echo "[$(date '+%F %T')] [${NAME}] 전부 완료 확인. 실행:"
    echo "    ${LAUNCH_CMD}"
    bash -c "cd '${HERE}'; ${LAUNCH_CMD}"
    echo "[$(date '+%F %T')] [${NAME}] 예약 완료 — 체인 종료"
    rm -f "${PIDF}"
    ;;
status)
    if alive; then
        echo "● 예약 중 [${NAME}]  pid $(cat "${PIDF}")"
        for f in ${DONE_LOGS}; do
            printf '  %-62s %s\n' "$(basename "$(dirname "${f}")")" \
                "$(grep -q '\[LOTUS\] 완료' "${f}" 2>/dev/null && echo 완료 || echo 미완)"
        done
        tail -3 "${LOGF}" 2>/dev/null | sed 's/^/  /'
    else
        echo "○ 예약 없음 [${NAME}]"
    fi
    ;;
stop)
    alive && { kill -TERM "$(cat "${PIDF}")" 2>/dev/null; rm -f "${PIDF}"; echo "예약 취소 [${NAME}]"; } \
           || echo "예약 없음 [${NAME}]"
    ;;
*) echo "사용법: NAME=.. DONE_LOGS=.. LAUNCH_CMD=.. bash bash/chain_after.sh {start|status|stop}"; exit 1;;
esac
