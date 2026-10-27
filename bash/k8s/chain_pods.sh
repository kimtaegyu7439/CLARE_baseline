#!/usr/bin/env bash
# A6000 로그인 노드에서 돈다: 지정한 파드들이 끝나면(Running/Pending 이 아니게 되면)
# object V5 λ 스윕 파드들을 만든다 — β 하나당 1-GPU 파드 하나, 파드 안에서 시드 순차.
#
#   setsid nohup bash bash/k8s/chain_pods.sh > logs/chain_pods.out 2>&1 &
#   WAIT_PODS="l3b10-goal l3b10-object" SUITE=object LAM=0.5 BETAS="1.0 0.5 2.0 4.0" SEEDS="7 42 1000"
#   DRY=1 bash bash/k8s/chain_pods.sh     # 계획만
#
# 파드 이름: v5<suite첫글자>-lam<λ>-b<β> (점 제거: 0.5 -> 05)
# 결과: results/lotus_v5act_c0.1_lam<λ>[_b<β>]_k25_scopeLL_<suite>_seed<N>  (PVC 공유)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${HERE}"
export KUBECONFIG=${KUBECONFIG:-/home/asdl/asdl-kubeconfig}
NS=${NS:-asdl}
WAIT_PODS=${WAIT_PODS:-"l3b10-goal l3b10-object"}
SUITE=${SUITE:-object}
LAM=${LAM:-0.5}
BETAS=${BETAS:-"1.0 0.5 2.0 4.0"}
SEEDS=${SEEDS:-"7 42 1000"}
C=${C:-0.1}
POLL=${POLL:-300}
mkdir -p logs

pod_name() { local l=${LAM//./} b=${1//./}; echo "v5${SUITE:0:1}-lam${l}-b${b}"; }
phase_of() { kubectl -n "${NS}" get pod "$1" -o jsonpath='{.status.phase}' 2>/dev/null; }
busy() {   # 하나라도 Running/Pending 이면 0
    local p ph
    for p in ${WAIT_PODS}; do
        ph=$(phase_of "${p}")
        case "${ph}" in Running|Pending) return 0;; esac
    done
    return 1
}
ts() { echo "[$(date '+%F %T')] $*"; }

if [ -n "${DRY:-}" ]; then
    echo "기다릴 파드: ${WAIT_PODS}  (지금: $(for p in ${WAIT_PODS}; do echo -n "${p}=$(phase_of "${p}") "; done))"
    for B in ${BETAS}; do
        echo "  파드 $(pod_name "${B}")  ->  SUITE=${SUITE} LAM=${LAM} BETA=${B} C=${C} SEEDS='${SEEDS}' bash bash/k8s/v5_seed_loop.sh"
    done
    exit 0
fi

ts "════ chain_pods: ${WAIT_PODS} 종료 대기 → ${SUITE} λ=${LAM} β∈{${BETAS}} 시드 {${SEEDS}} ════"
while busy; do
    [ $(( $(date +%s) % 3600 )) -lt "${POLL}" ] && ts "대기 중 — $(for p in ${WAIT_PODS}; do echo -n "${p}=$(phase_of "${p}") "; done)"
    sleep "${POLL}"
done
ts "기다리던 파드 종료: $(for p in ${WAIT_PODS}; do echo -n "${p}=$(phase_of "${p}") "; done)"

for B in ${BETAS}; do
    N=$(pod_name "${B}")
    ts "▶ 파드 ${N}  (β=${B})"
    NAME="${N}" GPUS=1 \
    CMD="SUITE=${SUITE} LAM=${LAM} BETA=${B} C=${C} SEEDS='${SEEDS}' bash bash/k8s/v5_seed_loop.sh" \
        bash bash/k8s/pod.sh 2>&1 | sed 's/^/    /'
    sleep 5
done
REPORT=LOTUS_MAR_sweep_report.py INTERVAL=300 bash bash/report_loop.sh start 2>&1 | head -1
ts "════ 파드 생성 끝 ════"
kubectl -n "${NS}" get pods
