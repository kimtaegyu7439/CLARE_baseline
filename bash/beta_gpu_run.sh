#!/usr/bin/env bash
# β 를 GPU 에 하나씩 꽂아 돌린다. GPU 가 여러 장인 서버용.
#
#   bash bash/beta_gpu_run.sh                      # 기본: β 4.0/4.5/5.5 -> GPU 0/1/2
#   DRY=1 bash bash/beta_gpu_run.sh
#   BETAS="4.0 4.5" GPUS="0 1" bash bash/beta_gpu_run.sh
#   SEED=42 bash bash/beta_gpu_run.sh
#   VER=v6 ENTRY=L2_V6.py bash bash/beta_gpu_run.sh   # 속도 공간으로
#   nohup bash bash/beta_gpu_run.sh > logs/beta_gpu.out 2>&1 &
#
# CLARE_V67_beta_sweep.sh 와 다른 점
#   저쪽은 GPU 한 장에 MAXJOBS 개를 몰아넣는다(H200 처럼 큰 카드용).
#   이쪽은 **GPU 한 장에 런 하나**다. 4090 24G 처럼 롤아웃 1 개(약 22G)로
#   꽉 차는 카드에서 쓴다.
#
# 나머지 설정은 seed7 β 커브와 같다 — batch 32, 10,000 step/task, 50 demo,
# 10 task, 칸당 50 rollout, eval seed 7, K=96, λ_lvl=3.0.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

BETAS=${BETAS:-"4.0 4.5 5.5"}
GPUS=${GPUS:-"0 1 2"}
SEED=${SEED:-7}
VER=${VER:-v7}
ENTRY=${ENTRY:-L2_V7.py}
FORM=${FORM:-dea}
STEPS=${STEPS:-10000}
NW=${NW:-6}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
MIN_FREE_GB=${MIN_FREE_GB:-40}
GAP=${GAP:-20}
# GPU 가 서로 다르므로 롤아웃이 겹쳐도 메모리가 충돌하지 않는다.
# 기본을 GPU 수와 같게 둬서 괜히 직렬화하지 않는다.
read -r -a _G <<< "${GPUS}"
ROLLOUT_SLOTS=${ROLLOUT_SLOTS:-${#_G[@]}}

Q="${HERE}/logs/beta_gpu_run.queue"
mkdir -p logs results
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${Q}"; }

read -r -a BETA_A <<< "${BETAS}"
read -r -a GPU_A  <<< "${GPUS}"
[ "${#BETA_A[@]}" -eq "${#GPU_A[@]}" ] || {
    echo "✗ BETAS(${#BETA_A[@]}) 와 GPUS(${#GPU_A[@]}) 개수가 다르다"; exit 1; }

NEED=55
filled() {
    python3 - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists(): print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
}

if [ -n "${DRY:-}" ]; then
    echo "${VER} ${FORM}  seed ${SEED}  batch 32  steps ${STEPS}  롤아웃 슬롯 ${ROLLOUT_SLOTS}"
    echo "여유 $(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')G"
    for i in "${!BETA_A[@]}"; do
        TAG="clare_${VER}_${FORM}_b${BETA_A[$i]}_goal_seed${SEED}"
        printf '  GPU %-3s β%-5s %-42s %s/%s칸\n' \
            "${GPU_A[$i]}" "${BETA_A[$i]}" "${TAG}" "$(filled "results/${TAG}")" "${NEED}"
    done
    exit 0
fi

ts "════ β x GPU — ${VER} ${FORM}, seed ${SEED}, steps ${STEPS} ════"
ts "    β ${BETAS} -> GPU ${GPUS}   롤아웃 슬롯 ${ROLLOUT_SLOTS}"

for i in "${!BETA_A[@]}"; do
    BETA="${BETA_A[$i]}"; GPU="${GPU_A[$i]}"
    TAG="clare_${VER}_${FORM}_b${BETA}_goal_seed${SEED}"
    f=$(filled "results/${TAG}")
    if [ "${f}" -ge "${NEED}" ]; then
        ts "⏭ ${TAG} — 이미 완주 (${f}/${NEED})"; continue
    fi
    ts "▶ GPU ${GPU}  β${BETA}  ${TAG}  (${f}/${NEED} 에서 시작)"
    (
        GPU="${GPU}" FORM="${FORM}" SEED="${SEED}" BETA="${BETA}" \
        ENTRY="${ENTRY}" VER="${VER}" TAGSUF="_b${BETA}" \
        STEPS="${STEPS}" NW="${NW}" N_EVAL="${N_EVAL}" EVAL_SEED="${EVAL_SEED}" \
        MIN_FREE_GB="${MIN_FREE_GB}" ROLLOUT_SLOTS="${ROLLOUT_SLOTS}" \
        bash "${HERE}/CLARE_V6_queue.sh"
        rc=$?
        ts "◀ ${TAG} exit=${rc}  ($(filled "results/${TAG}")/${NEED}칸)"
    ) &
    sleep "${GAP}"
done

wait
ts "════ 전부 종료 ════"
for i in "${!BETA_A[@]}"; do
    TAG="clare_${VER}_${FORM}_b${BETA_A[$i]}_goal_seed${SEED}"
    printf '  %-46s %s/%s칸\n' "${TAG}" "$(filled "results/${TAG}")" "${NEED}" | tee -a "${Q}"
done
python3 -u "${HERE}/CLARE_V6_report.py" > /dev/null 2>&1 \
    && ts "리포트 -> results/direction error aware anchor_V6_SR.txt"
