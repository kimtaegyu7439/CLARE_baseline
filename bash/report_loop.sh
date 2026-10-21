#!/usr/bin/env bash
# 리포트를 주기적으로 다시 써서 txt 가 계속 갱신되게 한다.
#
#   bash bash/report_loop.sh start            # 10 분마다
#   INTERVAL=300 bash bash/report_loop.sh start
#   REPORT=TETHER_forward_24sample_report.py bash bash/report_loop.sh start
#   bash bash/report_loop.sh status
#   bash bash/report_loop.sh stop
#
# 왜 필요한가
#   리포트는 런 하나가 끝날 때(55/55 칸)만 자동으로 쓰인다. 런 하나가 하루
#   걸리므로 그때까지 txt 가 아예 없거나 옛날 것이다. 이 루프가 주기적으로
#   같은 리포트를 다시 돌려 학습 진행률까지 포함해 갱신한다.
#
# 읽기 전용이다 — 결과 폴더를 읽어 txt 를 쓸 뿐, 학습에 영향을 주지 않는다.
# 스윕이 런 종료 때 쓰는 것과 겹치지 않게 같은 flock 을 쓴다.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

REPORT=${REPORT:-CLARE_V67_sweep_report.py}
INTERVAL=${INTERVAL:-600}
NAME="reportloop_$(basename "${REPORT}" .py)"
PIDF="${HERE}/logs/.${NAME}.pid"
LOGF="${HERE}/logs/${NAME}.log"
LOCK="${HERE}/logs/.v67_report.lock"      # 스윕 드라이버가 쓰는 것과 같은 락

mkdir -p logs
[ -f "${REPORT}" ] || { echo "✗ 리포트가 없다: ${REPORT}"; exit 1; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; }

# python3 로 돈다 — 리포트는 표준 라이브러리만 쓰므로 venv 가 없어도 된다.
PY=$(command -v python3 || command -v python)

case "${1:-status}" in

start)
    alive "${PIDF}" && { echo "이미 돌고 있다 (pid $(cat "${PIDF}"))"; exit 1; }
    if command -v setsid >/dev/null 2>&1; then
        setsid nohup "$0" _loop >> "${LOGF}" 2>&1 < /dev/null &
    else
        nohup "$0" _loop >> "${LOGF}" 2>&1 < /dev/null &
    fi
    sleep 2
    if alive "${PIDF}"; then
        echo "시작했다  pid $(cat "${PIDF}")   ${REPORT}  ${INTERVAL}초마다"
        echo "  결과   results/ 아래 txt 가 ${INTERVAL}초마다 갱신된다"
        echo "  중지   bash bash/report_loop.sh stop"
    else
        echo "✗ 바로 죽었다"; tail -10 "${LOGF}"; exit 1
    fi
    ;;

_loop)
    echo $$ > "${PIDF}"
    trap 'rm -f "${PIDF}"; exit 0' TERM INT
    while true; do
        if command -v flock >/dev/null 2>&1; then
            flock "${LOCK}" "${PY}" -u "${REPORT}" > /dev/null 2>&1 \
                || echo "[$(date '+%F %T')] 리포트 실패 (아직 결과가 없을 수 있다)"
        else
            "${PY}" -u "${REPORT}" > /dev/null 2>&1 || true
        fi
        sleep "${INTERVAL}"
    done
    ;;

status)
    if alive "${PIDF}"; then
        echo "● 갱신 중  pid $(cat "${PIDF}")   ${REPORT}  ${INTERVAL}초마다"
    else
        echo "○ 안 돌고 있다   ${REPORT}"
    fi
    for f in results/CLARE_V67_beta_sweep_SR.txt results/TETHER_forward_24sample.txt; do
        [ -f "${f}" ] && echo "  ${f}  ($(date -r "${f}" '+%F %T') 갱신)"
    done
    ;;

stop)
    if alive "${PIDF}"; then
        kill "$(cat "${PIDF}")" 2>/dev/null; rm -f "${PIDF}"; echo "중지했다"
    else
        echo "안 돌고 있다"
    fi
    ;;

*)
    sed -n '2,16p' "$0" | sed 's|^# \?||'
    ;;
esac
