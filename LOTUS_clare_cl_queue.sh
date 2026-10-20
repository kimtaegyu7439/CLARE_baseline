#!/usr/bin/env bash
# LOTUS 셋팅 CLARE(어댑터 라우팅 기준선) — GPU 하나가 시드 하나를 맡아 object -> goal -> 50.
#
#   SEED=7    GPU=0 bash LOTUS_clare_cl_queue.sh
#   SEED=42   GPU=1 bash LOTUS_clare_cl_queue.sh
#   SEED=1000 GPU=2 bash LOTUS_clare_cl_queue.sh
#   DRY=1 SEED=7 GPU=0 bash LOTUS_clare_cl_queue.sh      큐 내용만 출력
#
# ★ 시드로 가른다. 스위트로 가르면 object/goal 이 6 시간에 끝나고 50 만 20 시간 넘게
#   남아 카드 두 장이 논다. seq-FT/ER 큐와 같은 이유, 같은 배치다.
#
# ★ 한 카드에서 하나씩만 돈다. 롤아웃이 n_envs=20 벡터 env 를 띄운다.
#
# 런 하나가 끝날 때마다 results/LOTUS_CLARE_SR.txt 를 다시 만든다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
# ★ env.sh 가 HF_LEROBOT_HOME 을 잡는다. 빠뜨리면 로컬 데이터셋을 못 찾고 새로 받는다.
source "${HERE}/bash/clare/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라 (예: GPU=0)}
SEED=${SEED:?SEED 를 지정해라 (예: SEED=7)}
SUITES=${SUITES:-"object goal 50"}
STEPS=${STEPS:-10000}           # seq-FT / ER 과 같은 어댑터 학습 예산
DISC_STEPS=${DISC_STEPS:-2000}  # 판별기 — CLARE 고유 오버헤드
DETECT_STEPS=${DETECT_STEPS:-200}
# ★ CLARE 태스크 1개 = LIBERO 태스크 1개는 L2_clare_cl.py 의 build_units 에 박혀 있다
#   (확장 판정이 태스크마다 일어나야 해서 스테이지를 뭉치지 않는다). 옵션이 아니다.
N_EVAL=${N_EVAL:-20}
BS_EVAL=${BS_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
BATCH=${BATCH:-32}
NW=${NW:-6}
QUEUE="${HERE}/logs/LOTUS_clare_cl.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results

# 완주 검사 — metrics.json 의 R 에서 아래삼각형 칸이 다 찼는지 본다.
# 파일 존재만으로는 완주의 증거가 못 된다.
is_done() {
    python - "$1" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "metrics.json"
if not p.exists():
    print("0"); raise SystemExit
try:
    d = json.loads(p.read_text())
except Exception:
    print("0"); raise SystemExit
R = d.get("R") or []
la = d.get("learned_at") or []
if not R or not la:
    print("0"); raise SystemExit
need = sum(1 for k in range(len(R)) for t in range(len(la)) if la[t] <= k)
have = sum(1 for k in range(len(R)) for t in range(len(R[k])) if R[k][t] is not None)
print("1" if have >= need and need > 0 else "0")
PY
}

report() {
    python LOTUS_seq_cl_report.py \
        --prefix "lotus_clare_" \
        --out "${HERE}/results/LOTUS_CLARE_SR.txt" \
        --title "LOTUS 셋팅 lifelong — CLARE (어댑터 라우팅 기준선, 저자 구현)" \
        --loss_note "  손실      어댑터만 학습(베이스 동결). 태스크당 어댑터 ${STEPS} step + 판별기 ${DISC_STEPS} step." \
        --tail_note "  · base 태스크는 lifelong 시작 전에 판별기만 보정했다(어댑터는 항등). 원 논문에 없는 우리 추가분이다." \
        > /dev/null 2>&1
}

for SUITE in ${SUITES}; do
    TAG="lotus_clare_${SUITE}_seed${SEED}"
    OUT="results/${TAG}"
    LOG="logs/${TAG}.log"
    if [ -n "${DRY:-}" ]; then
        echo "GPU ${GPU}  ${SUITE}  seed ${SEED}  ${STEPS}+${DISC_STEPS} step  -> ${OUT}"; continue
    fi
    if [ "$(is_done "${OUT}")" = "1" ]; then
        echo "[$(date '+%F %T')] skip ${TAG} — 이미 완주" | tee -a "${QUEUE}"; continue
    fi
    # ★ clare.py 는 output_dir 이 이미 있으면 FileExistsError 를 낸다. 중단된 런의
    #   잔여물을 치우고 처음부터 간다 (재개 경로는 어디에도 없다).
    if [ -d "${OUT}" ]; then
        BAK="${OUT}.killed_$(date '+%m%d_%H%M')"
        mv "${OUT}" "${BAK}"
        echo "[$(date '+%F %T')] 이전 잔여물 -> ${BAK}" | tee -a "${QUEUE}"
    fi
    echo "[$(date '+%F %T')] ▶ ${TAG}  GPU ${GPU}  태스크당 ${STEPS}+${DISC_STEPS} step  칸당 ${N_EVAL} rollout" \
        | tee -a "${QUEUE}"
    echo "════ $(date '+%F %T') 시작 ════" >> "${LOG}"
    python -u L2_clare_cl.py \
        --suite "${SUITE}" --steps "${STEPS}" \
        --disc_steps "${DISC_STEPS}" --detect_steps "${DETECT_STEPS}" \
        --batch_size "${BATCH}" --num_workers "${NW}" \
        --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
        --out "${OUT}" >> "${LOG}" 2>&1
    rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다
    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}" | tee -a "${QUEUE}"
    [ "${rc}" -ne 0 ] && echo "   -> ${LOG} 확인" | tee -a "${QUEUE}"
    report && echo "   SR 기록 -> results/LOTUS_CLARE_SR.txt" | tee -a "${QUEUE}"
done
echo "[$(date '+%F %T')] seed ${SEED} 전체 종료 (GPU ${GPU})" | tee -a "${QUEUE}"
