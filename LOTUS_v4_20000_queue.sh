#!/usr/bin/env bash
# LOTUS 셋팅 v4(TETHER) — 태스크당 20,000 step, 셀 30개, lifelong 구간만 평가.
#
#   GPU=0 JOBS="goal:7 goal:42 goal:1000"          bash LOTUS_v4_20000_queue.sh
#   GPU=1 JOBS="object:7 object:42 object:1000"    bash LOTUS_v4_20000_queue.sh
#   GPU=2 JOBS="50:7 50:42"                        bash LOTUS_v4_20000_queue.sh
#   GPU=3 JOBS="50:1000"                           bash LOTUS_v4_20000_queue.sh
#   DRY=1 GPU=0 JOBS="goal:7" bash LOTUS_v4_20000_queue.sh
#
# 기존 LOTUS_v4_SR_matrix.txt 실행과 다른 점
#   · stage 당 10,000 -> **20,000 step**
#   · 셀 수 K = k_per_demo x demo (lifelong 20) -> **전 태스크 30 고정**
#   · base 태스크 롤아웃을 **건너뛴다**. base 는 이미 사전학습으로 92~94% 인 게
#     확인돼 있고, libero_50 은 base 가 25 태스크라 stage 마다 25 칸을 더 도느라
#     전체 시간의 대부분을 거기 쓴다. lifelong 만 재면 훨씬 빠르다.
#   · 결과는 results/LOTUS_v4_SR_matrix_20000.txt
#
# ★ v4 = hard_linear 샘플러 + speedinv 가중. 이 스크립트의 기본값은 v3 경로라
#   둘 다 명시해야 한다(안 주면 조용히 v3 가 돈다).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"          # HF_LEROBOT_HOME — 빠뜨리면 데이터를 새로 받는다
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라}
JOBS=${JOBS:?JOBS 를 지정해라 (예: "goal:7 goal:42")}
STEPS=${STEPS:-20000}
CODEBOOK_K=${CODEBOOK_K:-30}
N_EVAL=${N_EVAL:-20}
BS_EVAL=${BS_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
BATCH=${BATCH:-32}
NW=${NW:-6}
PREFIX=${PREFIX:-lotus_v4k30_}
QUEUE="${HERE}/logs/LOTUS_v4_20000.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results

# 완주 검사 — lifelong 칸만 센다(base 는 애초에 비워 두므로 세면 안 된다).
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
R, la = d.get("R") or [], d.get("learned_at") or []
if not R or not la:
    print("0"); raise SystemExit
need = sum(1 for k in range(len(R)) for t in range(len(la)) if la[t] > 0 and la[t] <= k)
have = sum(1 for k in range(len(R)) for t in range(len(R[k]))
           if la[t] > 0 and R[k][t] is not None)
print("1" if have >= need and need > 0 else "0")
PY
}

report() {
    python LOTUS_seq_cl_report.py \
        --prefix "${PREFIX}" \
        --out "${HERE}/results/LOTUS_v4_SR_matrix_20000.txt" \
        --title "LOTUS 셋팅 lifelong — v4/TETHER (셀 30, 태스크당 ${STEPS} step, lifelong 만 평가)" \
        --loss_note "  손실      L = L_FM(현재) + λ_a·λ_lvl·L_anchor. 코드북 셀 ${CODEBOOK_K}개(전 태스크 고정), hard_linear + speedinv." \
        --tail_note "  · base 태스크는 롤아웃하지 않았다(사전학습 92~94%). 표의 지표는 lifelong 태스크만으로 낸 값이다." \
        > /dev/null 2>&1
}

for J in ${JOBS}; do
    SUITE="${J%%:*}"; SEED="${J##*:}"
    TAG="${PREFIX}${SUITE}_seed${SEED}"
    OUT="results/${TAG}"
    LOG="logs/${TAG}.log"
    if [ -n "${DRY:-}" ]; then
        echo "GPU ${GPU}  ${SUITE}  seed ${SEED}  ${STEPS} step  K=${CODEBOOK_K}  lifelong만 -> ${OUT}"; continue
    fi
    if [ "$(is_done "${OUT}")" = "1" ]; then
        echo "[$(date '+%F %T')] skip ${TAG} — 이미 완주" | tee -a "${QUEUE}"; continue
    fi
    if [ -d "${OUT}" ]; then
        BAK="${OUT}.killed_$(date '+%m%d_%H%M')"
        mv "${OUT}" "${BAK}"
        echo "[$(date '+%F %T')] 이전 잔여물 -> ${BAK}" | tee -a "${QUEUE}"
    fi
    echo "[$(date '+%F %T')] ▶ ${TAG}  GPU ${GPU}  ${STEPS} step  K=${CODEBOOK_K}  칸당 ${N_EVAL} rollout" \
        | tee -a "${QUEUE}"
    echo "════ $(date '+%F %T') 시작 ════" >> "${LOG}"
    python -u Lotus_l2cb_cl.py \
        --suite "${SUITE}" --steps "${STEPS}" \
        --codebook_k "${CODEBOOK_K}" --skip_base_eval \
        --sampler_mode hard_linear --anchor_weight speedinv --lambda_level 3.0 \
        --batch_size "${BATCH}" --num_workers "${NW}" \
        --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
        --out "${OUT}" >> "${LOG}" 2>&1
    rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다
    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}" | tee -a "${QUEUE}"
    [ "${rc}" -ne 0 ] && echo "   -> ${LOG} 확인" | tee -a "${QUEUE}"
    report && echo "   SR 기록 -> results/LOTUS_v4_SR_matrix_20000.txt" | tee -a "${QUEUE}"
done
echo "[$(date '+%F %T')] GPU ${GPU} 전체 종료 (${JOBS})" | tee -a "${QUEUE}"
