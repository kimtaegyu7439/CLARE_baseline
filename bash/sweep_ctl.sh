#!/usr/bin/env bash
# 스윕을 백그라운드로 돌리고 상태를 보는 컨트롤러.
#
#   bash bash/sweep_ctl.sh start                 # 떼어내서 실행 (세션 끊겨도 산다)
#   bash bash/sweep_ctl.sh status                # 살아있나 / 몇 런 끝났나
#   bash bash/sweep_ctl.sh log                   # 실시간 로그
#   bash bash/sweep_ctl.sh stop                  # 중지 (자식까지)
#
# 환경변수는 그대로 대상 스크립트에 넘어간다:
#   VERS="v7 v6" MAXJOBS=8 bash bash/sweep_ctl.sh start
#   SEEDS="7 42" bash bash/sweep_ctl.sh start
#
# 대상 바꾸기 (기본은 β 스윕):
#   TARGET=bash/TETHER_forward_24sample.sh bash/sweep_ctl.sh start
#
# ★ 다시 띄워도 안전하다
#   대상 스크립트들이 55/55 칸을 채운 런은 건너뛰고, 중간에 끊긴 런은
#   CLARE_V6_queue.sh 의 --resume 이 이어받는다. 그래서 멈춘 뒤 start 를
#   다시 해도 같은 일을 두 번 하지 않는다.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

TARGET=${TARGET:-bash/CLARE_V67_beta_sweep.sh}
NAME=$(basename "${TARGET}" .sh)
PIDF="${HERE}/logs/.${NAME}.pid"
LOGF="${HERE}/logs/${NAME}.out"

mkdir -p logs results
[ -f "${TARGET}" ] || { echo "✗ 대상이 없다: ${TARGET}"; exit 1; }

alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; }

# 대상 스크립트가 띄운 학습 프로세스들
train_pids() { pgrep -u "$(id -un)" -f 'L2_V[0-9]+\.py' 2>/dev/null; }

progress() {
    python3 - "${HERE}/results" <<'PY'
import sys, pathlib, re
root = pathlib.Path(sys.argv[1])
pat = re.compile(r"^clare_v[67]_\w+?(_b[\d.]+)?(_b\d+bs)?_goal_seed\d+$")
done = part = 0
rows = []
for d in sorted(root.glob("clare_v[67]_*_goal_seed*")):
    p = d / "sr_matrix.csv"
    if not p.exists():
        continue
    body = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    n = sum(1 for l in body[1:] for c in l.split(",")[1:] if c.strip())
    if n >= 55: done += 1
    elif n > 0: part += 1; rows.append((d.name, n))
print(f"  완주 {done} 런, 진행 중 {part} 런")
for name, n in rows[:8]:
    print(f"    {name:44s} {n}/55칸")
PY
}

case "${1:-status}" in

start)
    if alive "${PIDF}"; then
        echo "이미 돌고 있다 (pid $(cat "${PIDF}")).  상태: bash bash/sweep_ctl.sh status"; exit 1
    fi
    # setsid 로 세션에서 완전히 떼어낸다 — 터미널이 닫혀도 산다.
    if command -v setsid >/dev/null 2>&1; then
        setsid nohup bash "${TARGET}" >> "${LOGF}" 2>&1 < /dev/null &
    else
        nohup bash "${TARGET}" >> "${LOGF}" 2>&1 < /dev/null &
    fi
    echo $! > "${PIDF}"
    sleep 2
    if alive "${PIDF}"; then
        echo "시작했다  pid $(cat "${PIDF}")   대상 ${TARGET}"
        echo "  로그   bash bash/sweep_ctl.sh log"
        echo "  상태   bash bash/sweep_ctl.sh status"
    else
        echo "✗ 바로 죽었다 — ${LOGF} 확인"; tail -20 "${LOGF}"; exit 1
    fi
    ;;

status)
    if alive "${PIDF}"; then
        echo "● 실행 중  pid $(cat "${PIDF}")   대상 ${TARGET}"
    elif [ -n "$(train_pids)" ]; then
        echo "○ 이 컨트롤러로 띄운 게 아니다 (pid 파일 없음)   대상 ${TARGET}"
        echo "  — 학습은 돌고 있다. 아래 프로세스 수를 보라."
    else
        echo "○ 멈춰 있다   대상 ${TARGET}"
    fi
    n=$(train_pids | wc -l)
    echo "  학습 프로세스 ${n} 개"
    [ "${n}" -gt 0 ] && nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader 2>/dev/null | sed 's|^|    GPU |'
    progress
    echo "  최근 로그:"
    tail -4 "${LOGF}" 2>/dev/null | sed 's|^|    |' || echo "    (없음)"
    ;;

log)
    tail -f "${LOGF}"
    ;;

stop)
    if alive "${PIDF}"; then
        P=$(cat "${PIDF}")
        # 프로세스 그룹째 보낸다 (setsid 로 띄웠으면 그룹 리더다)
        kill -TERM -- "-${P}" 2>/dev/null || kill -TERM "${P}" 2>/dev/null
        echo "중지 요청 pid ${P}"
    else
        echo "컨트롤러는 안 돌고 있다"
    fi
    sleep 3
    L=$(train_pids)
    if [ -n "${L}" ]; then
        echo "학습 프로세스가 남아 있다:"; echo "${L}" | sed 's|^|  pid |'
        echo "  전부 종료하려면: pkill -u $(id -un) -f 'L2_V[0-9]+\\.py'"
    fi
    rm -f "${PIDF}"
    ;;

*)
    sed -n '2,25p' "$0" | sed 's|^# \?||'
    ;;
esac
