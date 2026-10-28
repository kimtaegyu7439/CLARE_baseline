#!/usr/bin/env bash
# 4090_2 에서 도는 V13 β1/β2 결과(체크포인트 제외)를 로컬로 당겨 와서
# 로컬 results/LOTUS_act_V5_SR.txt 의 V13 절에 β1·2(4090_2)·3·4(로컬)가 한 표에 모이게 한다.
#
#   setsid nohup bash bash/sync_v13_from_4090_2.sh > logs/sync_v13.log 2>&1 < /dev/null &
#
# 12 런이 전부 "[LOTUS] 완료" 면 마지막 리포트를 쓰고 끝난다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
HOST=${HOST:-sa090180@10.74.12.95}
INTERVAL=${INTERVAL:-1200}
BETAS=${BETAS:-"1.0 2.0"}
LOCK="${HERE}/logs/.v67_report.lock"
ts() { echo "[$(date '+%F %T')] $*"; }
done_n() {
    local n=0
    for b in ${BETAS}; do for s in goal object; do for sd in 7 42 1000; do
        grep -q '\[LOTUS\] 완료' "results/lotus_v13_b${b}_lam3.0_k25_scopeLL_${s}_seed${sd}/train.log" 2>/dev/null && n=$((n+1))
    done; done; done
    echo "$n"
}
while true; do
    for b in ${BETAS}; do
        rsync -a --timeout=120 --exclude 'checkpoints/' --exclude '*.pt' --exclude '*.safetensors' \
            "${HOST}:clare/results/lotus_v13_b${b}_lam3.0_k25_scopeLL_*" results/ 2>&1 | tail -2
    done
    V13_BETAS="1.0 2.0 3.0 4.0" flock "${LOCK}" python3 -u LOTUS_act_V5_report.py > /dev/null 2>&1 \
        && ts "리포트 갱신  완료 런 $(done_n)/12" || ts "리포트 실패"
    [ "$(done_n)" -ge 12 ] && { ts "전부 완료 — 종료"; exit 0; }
    sleep "${INTERVAL}"
done
