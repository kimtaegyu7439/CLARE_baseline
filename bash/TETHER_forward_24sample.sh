#!/usr/bin/env bash
# V7 dea β2.0 을 **배치 24** 로 돌린다. GPU 0/1/2 에 시드 7/42/1000 하나씩.
#
#   bash bash/TETHER_forward_24sample.sh
#   DRY=1 bash bash/TETHER_forward_24sample.sh
#   SEEDS="7 42" GPUS="0 1" bash bash/TETHER_forward_24sample.sh
#   nohup bash bash/TETHER_forward_24sample.sh > logs/fwd24.out 2>&1 &
#
# 왜 24 인가
#   ER 은 배치 32 를 현재 24 + 버퍼 8 로 쓴다(L2_er_cl.py:181). 즉 **새 태스크를
#   배우는 데 쓰는 표본이 24 개**다. TETHER 는 32 개를 전부 새 태스크에 쓰므로
#   둘을 비교하면 "새 태스크 표본 수"가 다르다. 그 축을 맞춘 대조군이다.
#
# ★ 주의 — 이 설정은 두 가지를 동시에 바꾼다
#   앵커가 과거 태스크당 n=B 개를 뽑으므로(L2_V6.py:341, L2_V4_seq.py:1261),
#   B=24 면 **앵커 표본도 32 -> 24 로 줄어든다**. 새 태스크 표본만 줄인 게
#   아니다. 앵커 몬테카를로 분산이 32/24 배 커진다(약 1.33 배, sd 로는 1.15 배).
#   ER 과의 표본 수 대조로는 맞지만, 순수하게 "새 태스크 표본 수만"의 효과로
#   읽으면 안 된다.
#
# 나머지는 전부 batch 32 런과 같다 — 10,000 step/task, 50 demo, 10 task,
# 칸당 50 rollout, eval seed 7, K=96, λ_lvl=3.0.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

SEEDS=${SEEDS:-"7 42 1000"}
GPUS=${GPUS:-"0 1 2"}
BATCH=${BATCH:-24}
BETA=${BETA:-2.0}
FORM=${FORM:-dea}
VER=${VER:-v7}
ENTRY=${ENTRY:-L2_V7.py}
STEPS=${STEPS:-10000}          # batch 32 런과 동일하게 유지
NW=${NW:-6}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
MIN_FREE_GB=${MIN_FREE_GB:-40}
GAP=${GAP:-20}

Q="${HERE}/logs/TETHER_forward_24sample.queue"
mkdir -p logs results
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${Q}"; }

read -r -a SEED_A <<< "${SEEDS}"
read -r -a GPU_A  <<< "${GPUS}"
[ "${#SEED_A[@]}" -eq "${#GPU_A[@]}" ] || {
    echo "✗ SEEDS(${#SEED_A[@]}) 와 GPUS(${#GPU_A[@]}) 개수가 다르다"; exit 1; }

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
    echo "V7 ${FORM} β${BETA}  batch ${BATCH}  steps ${STEPS}  (batch 32 런과 steps 동일)"
    echo "여유 $(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')G"
    for i in "${!SEED_A[@]}"; do
        TAG="clare_${VER}_${FORM}_b${BATCH}bs_goal_seed${SEED_A[$i]}"
        printf '  GPU %-3s %-40s %s/%s칸\n' "${GPU_A[$i]}" "${TAG}" "$(filled "results/${TAG}")" "${NEED}"
    done
    exit 0
fi

ts "════ TETHER forward 24-sample — V7 ${FORM} β${BETA}, batch ${BATCH}, steps ${STEPS} ════"
ts "    시드 ${SEEDS} -> GPU ${GPUS}"

for i in "${!SEED_A[@]}"; do
    SEED="${SEED_A[$i]}"; GPU="${GPU_A[$i]}"
    TAG="clare_${VER}_${FORM}_b${BATCH}bs_goal_seed${SEED}"
    f=$(filled "results/${TAG}")
    if [ "${f}" -ge "${NEED}" ]; then
        ts "⏭ ${TAG} — 이미 완주 (${f}/${NEED})"; continue
    fi
    ts "▶ GPU ${GPU}  ${TAG}  (${f}/${NEED} 에서 시작)"
    (
        GPU="${GPU}" FORM="${FORM}" SEED="${SEED}" BETA="${BETA}" \
        ENTRY="${ENTRY}" VER="${VER}" BATCH="${BATCH}" \
        STEPS="${STEPS}" NW="${NW}" N_EVAL="${N_EVAL}" EVAL_SEED="${EVAL_SEED}" \
        MIN_FREE_GB="${MIN_FREE_GB}" \
        bash "${HERE}/CLARE_V6_queue.sh"
        rc=$?
        ts "◀ ${TAG} exit=${rc}  ($(filled "results/${TAG}")/${NEED}칸)"
        flock "${HERE}/logs/.fwd24_report.lock" \
            python -u "${HERE}/TETHER_forward_24sample_report.py" > /dev/null 2>&1 || true
    ) &
    sleep "${GAP}"
done

wait
ts "════ 전부 종료 ════"
for i in "${!SEED_A[@]}"; do
    TAG="clare_${VER}_${FORM}_b${BATCH}bs_goal_seed${SEED_A[$i]}"
    printf '  %-44s %s/%s칸\n' "${TAG}" "$(filled "results/${TAG}")" "${NEED}" | tee -a "${Q}"
done
python -u "${HERE}/TETHER_forward_24sample_report.py" > /dev/null 2>&1 \
    && ts "리포트 -> results/TETHER_forward_24sample.txt"
