#!/usr/bin/env bash
# L2_V12 — 앵커 표본을 과거 태스크에 무작위 배분. 시드별로 GPU 에 꽂아 돌린다.
#
#   bash bash/V12_run.sh                       # seed 7/42/1000 -> GPU 0,1 (2 개씩)
#   DRY=1 bash bash/V12_run.sh
#   SEEDS="7 42" GPUS="0 1" bash bash/V12_run.sh
#   nohup bash bash/V12_run.sh > logs/v12.out 2>&1 &
#
# 배치 분할은 ER 과 같다 — 새 태스크 BATCH(24) + 앵커 ANCHOR_ROWS(8) = 32.
# 나머지는 V7 β 커브와 같은 프로토콜이다: 10,000 step/task, 50 demo, 10 task,
# 칸당 50 rollout, eval seed 7, K=96, λ_lvl=3.0.
#
# GPU 가 시드보다 적으면 슬롯이 빌 때까지 기다렸다 다음 시드를 띄운다.
# 이 서버는 GPU 2·3 을 다른 사용자가 쓰고 있어 0·1 만 쓴다.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

SEEDS=${SEEDS:-"7 42 1000"}
GPUS=${GPUS:-"0 1"}
BETA=${BETA:-2.0}
BATCH=${BATCH:-24}               # 새 태스크 몫 (FM)
ANCHOR_ROWS=${ANCHOR_ROWS:-8}    # 앵커 몫
FORM=${FORM:-dea}
VER=${VER:-v12}
ENTRY=${ENTRY:-L2_V12.py}
STEPS=${STEPS:-10000}
NW=${NW:-6}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
MIN_FREE_GB=${MIN_FREE_GB:-40}
GAP=${GAP:-20}

Q="${HERE}/logs/V12_run.queue"
mkdir -p logs results
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${Q}"; }

read -r -a SEED_A <<< "${SEEDS}"
read -r -a GPU_A  <<< "${GPUS}"
NG=${#GPU_A[@]}
NEED=55
free_gb() { df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9'; }
filled() {
    python3 - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists(): print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
}
# 태그에 배치 분할을 적어 둔다 — 32 분할 런과 섞이면 안 된다
# V12 는 24+8 분할이 버전 정의라 태그에 따로 적지 않는다.
# 리포트가 clare_v12_dea_b<β>_goal_seed<N> 로 자동 탐지한다.
TAGSUF="_b${BETA}"

if [ -n "${DRY:-}" ]; then
    echo "${VER} ${FORM} β${BETA}  배치 ${BATCH}(새) + ${ANCHOR_ROWS}(앵커) = $(( BATCH + ANCHOR_ROWS ))"
    echo "steps ${STEPS}  GPU ${GPUS} (${NG}개)  여유 $(free_gb)G"
    for s in "${SEED_A[@]}"; do
        TAG="clare_${VER}_${FORM}${TAGSUF}_goal_seed${s}"
        printf '  seed %-6s %-46s %s/%s칸\n' "${s}" "${TAG}" "$(filled "results/${TAG}")" "${NEED}"
    done
    exit 0
fi

ts "════ V12 — β${BETA}, 배치 ${BATCH}+${ANCHOR_ROWS}, steps ${STEPS} ════"
ts "    시드 ${SEEDS} -> GPU ${GPUS} (동시 ${NG}개)"

running=0; gi=0
for SEED in "${SEED_A[@]}"; do
    TAG="clare_${VER}_${FORM}${TAGSUF}_goal_seed${SEED}"
    f=$(filled "results/${TAG}")
    if [ "${f}" -ge "${NEED}" ]; then ts "⏭ ${TAG} 이미 완주"; continue; fi
    while [ "${running}" -ge "${NG}" ]; do
        wait -n 2>/dev/null || true; running=$((running-1))
    done
    FREE=$(free_gb)
    if [ "${FREE}" -lt "${MIN_FREE_GB}" ]; then
        ts "✗ 디스크 여유 ${FREE}G < ${MIN_FREE_GB}G — 중단"; exit 1
    fi
    GPU="${GPU_A[$(( gi % NG ))]}"; gi=$((gi+1))
    ts "▶ GPU ${GPU}  ${TAG}  (${f}/${NEED}, 여유 ${FREE}G)"
    (
        GPU="${GPU}" FORM="${FORM}" SEED="${SEED}" BETA="${BETA}" \
        ENTRY="${ENTRY}" VER="${VER}" TAGSUF="${TAGSUF}" BATCH="${BATCH}" \
        STEPS="${STEPS}" NW="${NW}" N_EVAL="${N_EVAL}" EVAL_SEED="${EVAL_SEED}" \
        MIN_FREE_GB="${MIN_FREE_GB}" ROLLOUT_SLOTS="${NG}" TAG_BATCH_SUFFIX=0 \
        EXTRA_ARGS="--anchor_rows ${ANCHOR_ROWS}" \
        bash "${HERE}/CLARE_V6_queue.sh"
        ts "◀ ${TAG} exit=$?  ($(filled "results/${TAG}")/${NEED}칸)"
        flock "${HERE}/logs/.v6report.lock" \
            python3 -u "${HERE}/CLARE_V6_report.py" > /dev/null 2>&1 || true
    ) &
    running=$((running+1)); sleep "${GAP}"
done
while [ "${running}" -gt 0 ]; do wait -n 2>/dev/null || true; running=$((running-1)); done

ts "════ 전부 종료 ════"
for s in "${SEED_A[@]}"; do
    TAG="clare_${VER}_${FORM}${TAGSUF}_goal_seed${s}"
    printf '  %-48s %s/%s칸\n' "${TAG}" "$(filled "results/${TAG}")" "${NEED}" | tee -a "${Q}"
done
python3 -u "${HERE}/CLARE_V6_report.py" > /dev/null 2>&1 \
    && ts "리포트 -> results/direction error aware anchor_V6_SR.txt"
