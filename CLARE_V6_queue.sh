#!/usr/bin/env bash
# L2_V6 — Directional-Error-Aware 앵커 3 팔. CLARE 셋팅 libero_goal.
#
#   GPU=3 FORM=dea    bash CLARE_V6_queue.sh
#   GPU=0 FORM=sqerr  bash CLARE_V6_queue.sh
#   GPU=1 FORM=unifmd bash CLARE_V6_queue.sh
#   DRY=1 GPU=3 FORM=dea bash CLARE_V6_queue.sh
#
# 세 팔은 방향 항 e_j 의 계수만 다르다 (method note 2절 표):
#   sqerr   2·r_s·r_t / N   -> ‖v_s−v̄‖² 와 항등. 즉 기존 MSE 앵커
#   unifmd  β / N           -> 크기·방향을 분리하고 균일 배분
#   dea     β·w_j (detach)  -> 방향 예산을 오차 큰 과거 태스크로
#
# 프로토콜은 V5 시드 런과 맞춘다 — 태스크당 10,000 step, 50 demo, 10 task = 55칸,
# 칸당 **50 rollout**(eval seed 7~56), train seed 7, 코드북 K=96.
# 학습과 롤아웃을 분리한다(--eval_after_each_task false -> CLARE_rollout_only.py).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라}
FORM=${FORM:?FORM 을 지정해라 (dea|sqerr|unifmd)}
SEED=${SEED:-7}
SUITE=libero_goal
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
EPS=${EPS:-50}
NTASK=${NTASK:-10}
NW=${NW:-6}
CODEBOOK_K=${CODEBOOK_K:-96}
BETA=${BETA:-2.0}
EPS_W=${EPS_W:-1e-3}
# 같은 form 을 다른 하이퍼로 또 돌릴 때 결과가 섞이지 않게 태그에 붙인다
TAGSUF=${TAGSUF:-}
# L2_V6.py(속도 공간) 또는 L2_V7.py(액션 공간)
ENTRY=${ENTRY:-L2_V6.py}
VER=${VER:-v6}
REDUCE=${REDUCE:-mean}
MIN_FREE_GB=${MIN_FREE_GB:-30}
PRETRAIN=${PRETRAIN_PATH:-/home/sa090180/Models/dit_flow_mt_libero_90_pretrain}
QUEUE="${HERE}/logs/CLARE_V6.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

TAG="clare_${VER}_${FORM}${TAGSUF}_goal_seed${SEED}"
OUT="results/${TAG}"
CKPT="${HERE}/outputs/CLARE_V6/${VER}_${FORM}${TAGSUF}_seed${SEED}"
LOG="logs/${TAG}.log"
NEED=$(( NTASK * (NTASK + 1) / 2 ))

ts() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }
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
    echo "GPU ${GPU}  form=${FORM} β=${BETA}  seed ${SEED}  학습 -> 롤아웃 ${N_EVAL}회 -> ${OUT}"
    exit 0
fi

FREE=$(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')
[ "${FREE}" -lt "${MIN_FREE_GB}" ] && { ts "✗ ${TAG} — 디스크 ${FREE}G < ${MIN_FREE_GB}G"; exit 1; }

# ── 1단계: 학습 (롤아웃 없이) ───────────────────────────────────────────
ts "▶ ${TAG} 학습  GPU ${GPU}  form=${FORM} β=${BETA} ε_w=${EPS_W} reduce=${REDUCE}"
echo "════ $(date '+%F %T') 학습 시작 ${FORM} ════" >> "${LOG}"
PRETRAIN_PATH="${PRETRAIN}" python -u "${ENTRY}" \
    --anchor_form "${FORM}" --beta "${BETA}" --eps_w "${EPS_W}" --anchor_reduce "${REDUCE}" \
    --arm v4 --suite "${SUITE}" --num_tasks "${NTASK}" \
    --episodes_per_task "${EPS}" --steps_per_task "${STEPS}" \
    --batch_size 32 --num_workers "${NW}" \
    --codebook_k "${CODEBOOK_K}" --lambda_level 3.0 \
    --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
    --eval_after_each_task false \
    --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
    --ckpt_root "${CKPT}" --out "${OUT}" --resume >> "${LOG}" 2>&1
rc=$?
ts "◀ ${TAG} 학습 exit=${rc}"
[ "${rc}" -ne 0 ] && { ts "   -> ${LOG} 확인"; exit "${rc}"; }

# ── 2단계: 롤아웃 50회 ──────────────────────────────────────────────────
CKDIR=$(ls -d "${CKPT}"/*_ours 2>/dev/null | head -1)
[ -z "${CKDIR}" ] && { ts "   ★ 체크포인트 폴더 없음"; exit 1; }
ts "▶ ${TAG} 롤아웃 ${N_EVAL}회"
python -u CLARE_rollout_only.py --ckpt_dir "${CKDIR}" --out "${OUT}" \
    --suite "${SUITE}" --rows all \
    --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
    --eval_seed "${EVAL_SEED}" >> "${LOG}" 2>&1
rc=$?
ts "◀ ${TAG} 롤아웃 exit=${rc}  ($(filled "${OUT}")/${NEED}칸)"
python -u CLARE_V6_report.py > /dev/null 2>&1 && ts "   기록 -> results/direction error aware anchor_V6_SR.txt"
