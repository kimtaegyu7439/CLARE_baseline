#!/usr/bin/env bash
# 4090_2 의 V13 object 런(체크포인트 제외)을 로컬로 당겨 와서 V13 스윕 리포트를 다시 쓴다.
#   setsid nohup bash bash/sync_v13_object_from_4090_2.sh > logs/sync_v13_object.log 2>&1 < /dev/null &
# → results/LOTUS_V13_sweep.txt  (FAMILY=v13 LOTUS_MAR_sweep_report.py)
# CONFIGS 의 모든 (λ,β) × 3 시드가 "[LOTUS] 완료" 면 마지막 리포트를 쓰고 끝난다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
HOST=${HOST:-sa090180@10.74.12.95}
INTERVAL=${INTERVAL:-1200}
CONFIGS=${CONFIGS:-"1.0:1.5 1.0:1.0 1.0:2.0 0.8:1.5 0.7:2.0 1.2:1.0"}   # λ:β
LOCK="${HERE}/logs/.v67_report.lock"
ts() { echo "[$(date '+%F %T')] $*"; }
done_n() {
    local n=0 lb
    for lb in ${CONFIGS}; do for sd in 7 42 1000; do
        grep -q '\[LOTUS\] 완료' "results/lotus_v13_b${lb#*:}_lam${lb%%:*}_k25_scopeLL_object_seed${sd}/train.log" 2>/dev/null && n=$((n+1))
    done; done
    echo "$n"
}
TOTAL=$(( $(echo ${CONFIGS} | wc -w) * 3 ))
while true; do
    rsync -a --timeout=120 --exclude 'checkpoints/' --exclude '*.pt' --exclude '*.safetensors' \
        --include 'lotus_v13_*_object_seed*' --include 'lotus_v13_*_object_seed*/**' --exclude '*' \
        "${HOST}:clare/results/" results/ 2>&1 | tail -2
    FAMILY=v13 flock "${LOCK}" python3 -u LOTUS_MAR_sweep_report.py > /dev/null 2>&1 \
        && ts "리포트 갱신  완료 런 $(done_n)/${TOTAL}" || ts "리포트 실패"
    [ "$(done_n)" -ge "${TOTAL}" ] && { ts "전부 완료 — 종료"; exit 0; }
    sleep "${INTERVAL}"
done
