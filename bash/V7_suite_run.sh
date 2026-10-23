#!/usr/bin/env bash
# V7 dea 를 **스위트별로 GPU 하나씩** 꽂고, 그 안에서 시드를 순차로 돌린다.
#
#   bash bash/V7_suite_run.sh                          # goal->GPU0, object->GPU1, seed 7,42
#   DRY=1 bash bash/V7_suite_run.sh
#   SUITES="libero_goal" GPUS="0" SEEDS="7 42 1000" bash bash/V7_suite_run.sh
#   nohup bash bash/V7_suite_run.sh > logs/v7_suite.out 2>&1 &
#
# 다른 런처와의 차이
#   beta_gpu_run.sh   β 를 GPU 에 꽂는다 (스위트 하나)
#   V12_run.sh        시드를 GPU 에 꽂는다 (β 하나)
#   이것            **스위트**를 GPU 에 꽂고 그 안에서 시드를 순차 처리한다
#
# CLARE_V6_queue.sh 는 SUITE 가 libero_goal 로 하드코딩돼 있어서 여기서 넘긴다.
# 프로토콜은 β 커브와 같다: 10,000 step/task, 50 demo, 10 task, 칸당 50 rollout.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

SUITES=${SUITES:-"libero_goal libero_object"}
GPUS=${GPUS:-"0 1"}
SEEDS=${SEEDS:-"7 42"}
BETA=${BETA:-2.0}
FORM=${FORM:-dea}
VER=${VER:-v7}
ENTRY=${ENTRY:-L2_V7.py}
STEPS=${STEPS:-10000}
NW=${NW:-6}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
NTASK=${NTASK:-10}
MIN_FREE_GB=${MIN_FREE_GB:-40}
GAP=${GAP:-20}
REPORT_PY=${REPORT_PY:-V7_suite_report.py}

Q="${HERE}/logs/V7_suite_run.queue"
mkdir -p logs results
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${Q}"; }

read -r -a SUITE_A <<< "${SUITES}"
read -r -a GPU_A   <<< "${GPUS}"
read -r -a SEED_A  <<< "${SEEDS}"
[ "${#SUITE_A[@]}" -eq "${#GPU_A[@]}" ] || {
    echo "✗ SUITES(${#SUITE_A[@]}) 와 GPUS(${#GPU_A[@]}) 개수가 다르다"; exit 1; }

NEED=$(( NTASK * (NTASK + 1) / 2 ))
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
# 태그에 스위트를 넣는다 — goal 전용이던 기존 태그 규칙과 섞이면 안 된다
tag_of() { echo "clare_${VER}_${FORM}_b${BETA}_${1#libero_}_seed${2}"; }

if [ -n "${DRY:-}" ]; then
    echo "${VER} ${FORM} β${BETA}  steps ${STEPS}  여유 $(free_gb)G"
    for i in "${!SUITE_A[@]}"; do
        echo "  GPU ${GPU_A[$i]}  ${SUITE_A[$i]}"
        for s in "${SEED_A[@]}"; do
            t=$(tag_of "${SUITE_A[$i]}" "${s}")
            printf '     seed %-5s %-44s %s/%s칸\n' "${s}" "${t}" "$(filled "results/${t}")" "${NEED}"
        done
    done
    exit 0
fi

ts "════ V7 ${FORM} β${BETA} — 스위트별 GPU, 시드 순차 ════"
ts "    ${SUITES} -> GPU ${GPUS},  시드 ${SEEDS}"

for i in "${!SUITE_A[@]}"; do
    SUITE="${SUITE_A[$i]}"; GPU="${GPU_A[$i]}"
    (
        for SEED in "${SEED_A[@]}"; do
            TAG=$(tag_of "${SUITE}" "${SEED}")
            f=$(filled "results/${TAG}")
            if [ "${f}" -ge "${NEED}" ]; then ts "⏭ ${TAG} 이미 완주"; continue; fi
            FREE=$(free_gb)
            if [ "${FREE}" -lt "${MIN_FREE_GB}" ]; then
                ts "✗ ${SUITE}: 디스크 여유 ${FREE}G < ${MIN_FREE_GB}G — 중단"; break
            fi
            ts "▶ GPU ${GPU}  ${TAG}  (${f}/${NEED}, 여유 ${FREE}G)"
            GPU="${GPU}" FORM="${FORM}" SEED="${SEED}" BETA="${BETA}" \
            ENTRY="${ENTRY}" VER="${VER}" SUITE="${SUITE}" \
            TAGSUF="_b${BETA}_${SUITE#libero_}" TAG_SUITE=1 \
            STEPS="${STEPS}" NW="${NW}" N_EVAL="${N_EVAL}" EVAL_SEED="${EVAL_SEED}" \
            NTASK="${NTASK}" MIN_FREE_GB="${MIN_FREE_GB}" ROLLOUT_SLOTS=1 \
            bash "${HERE}/CLARE_V6_queue.sh"
            ts "◀ ${TAG} exit=$?  ($(filled "results/${TAG}")/${NEED}칸)"
            flock "${HERE}/logs/.$(basename "${REPORT_PY}" .py).lock" \
                python3 -u "${HERE}/${REPORT_PY}" > /dev/null 2>&1 || true
        done
    ) &
    sleep "${GAP}"
done
wait

ts "════ 전부 종료 ════"
for i in "${!SUITE_A[@]}"; do
    for s in "${SEED_A[@]}"; do
        t=$(tag_of "${SUITE_A[$i]}" "${s}")
        printf '  %-46s %s/%s칸\n' "${t}" "$(filled "results/${t}")" "${NEED}" | tee -a "${Q}"
    done
done
python3 -u "${HERE}/${REPORT_PY}" > /dev/null 2>&1 \
    && ts "리포트 -> results/V7_suite_study.txt"
