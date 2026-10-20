#!/usr/bin/env bash
# L2_V8 — OMP 의 Directional Alignment 손실(L_DA)을 앵커에 붙인 2 팔.
#          CLARE 셋팅 libero_goal.
#
#   GPU=0 FORM=mse_da bash CLARE_V8_queue.sh     # 팔 A
#   GPU=3 FORM=mag_da bash CLARE_V8_queue.sh     # 팔 B
#   DRY=1 GPU=0 FORM=mse_da bash CLARE_V8_queue.sh
#
# 두 팔은 "유지 항" 한 가지만 다르다 (방향 항 λ_DA·L_DA 는 공통):
#   mse_da   ‖v_s − v_T‖²            + λ_DA·L_DA   (= V6 sqerr + L_DA)
#   mag_da   (‖v_s‖₂ − ‖v_T‖₂)²      + λ_DA·L_DA   (크기만 남기고 방향은 L_DA 에)
#
#   L_DA = −log((cos α + 1)/2),  cos α = (v_T·v_s)/((‖v_T‖+ε_dir)·‖v_s‖)
#   각도는 **속도** v 로 잰다 (V7 의 액션 공간이 아니다).
#
# 프로토콜은 V6/V7 과 똑같다 — 태스크당 10,000 step, 50 demo, 10 task = 55칸,
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
FORM=${FORM:?FORM 을 지정해라 (mse_da|mag_da)}
SEED=${SEED:-7}
SUITE=libero_goal
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
EPS=${EPS:-50}
NTASK=${NTASK:-10}
NW=${NW:-6}
CODEBOOK_K=${CODEBOOK_K:-96}
LAMBDA_DA=${LAMBDA_DA:-1.0}
EPS_DIR=${EPS_DIR:-1e-6}
TAGSUF=${TAGSUF:-}
ENTRY=${ENTRY:-L2_V8.py}        # L2_V8.py(속도) 또는 L2_V9.py(액션)
VER=${VER:-v8}
CKROOT=${CKROOT:-CLARE_V8}
REDUCE=${REDUCE:-mean}
MIN_FREE_GB=${MIN_FREE_GB:-30}
PRETRAIN=${PRETRAIN_PATH:-/home/sa090180/Models/dit_flow_mt_libero_90_pretrain}
QUEUE="${HERE}/logs/${CKROOT}.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

TAG="clare_${VER}_${FORM}${TAGSUF}_goal_seed${SEED}"
OUT="results/${TAG}"
CKPT="${HERE}/outputs/${CKROOT}/${VER}_${FORM}${TAGSUF}_seed${SEED}"
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
    echo "GPU ${GPU}  form=${FORM} λ_DA=${LAMBDA_DA}  seed ${SEED}  학습 -> 롤아웃 ${N_EVAL}회 -> ${OUT}"
    exit 0
fi

FREE=$(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')
[ "${FREE}" -lt "${MIN_FREE_GB}" ] && { ts "✗ ${TAG} — 디스크 ${FREE}G < ${MIN_FREE_GB}G"; exit 1; }

# ── 1단계: 학습 (롤아웃 없이) ───────────────────────────────────────────
ts "▶ ${TAG} 학습  GPU ${GPU}  form=${FORM} λ_DA=${LAMBDA_DA} ε_dir=${EPS_DIR} reduce=${REDUCE}"
echo "════ $(date '+%F %T') 학습 시작 ${FORM} ════" >> "${LOG}"
PRETRAIN_PATH="${PRETRAIN}" python -u "${ENTRY}" \
    --anchor_form "${FORM}" --lambda_da "${LAMBDA_DA}" --eps_dir "${EPS_DIR}" \
    --anchor_reduce "${REDUCE}" \
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
