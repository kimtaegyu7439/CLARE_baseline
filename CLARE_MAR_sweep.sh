#!/usr/bin/env bash
# CLARE(순차) 셋팅 libero_goal — 앵커 가중 분모 상수 c 스윕.
#
#   GPU=0 C=0.1    bash CLARE_MAR_sweep.sh
#   GPU=1 C=0.01   bash CLARE_MAR_sweep.sh
#   GPU=2 C=median bash CLARE_MAR_sweep.sh      기존 규칙 (c = kappa x speed_med2)
#   DRY=1 GPU=0 C=0.1 bash CLARE_MAR_sweep.sh
#
# 무엇을 바꾸는가
#   v4 앵커 가중은  w = 1 / (||v~_T[0:6]||^2 + c)  이고, 기존에는
#     c = kappa x speed_med2_j,  speed_med2_j = median_{데모·스텝} ||a[0:6]||^2
#   로 **태스크마다** 정해졌다. libero_goal 에서 그 값이 0.61~0.79 다.
#   이 스윕은 c 를 0.1 / 0.01 로 **전 태스크 고정**해서 비교한다. 둘 다 median
#   규칙보다 훨씬 작으므로 w 가 커지고, 느린 프레임에 실리는 가중이 날카로워진다.
#
# 프로토콜은 CLARE_v4_sr.sh 와 맞추되 **칸당 rollout 만 20** 이다(요청).
#   태스크당 10,000 step / 50 demo / 10 task = 55 칸 / eval seed 7 고정
#   libero_90 사전학습 체크포인트에서 출발
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"          # HF_LEROBOT_HOME — 빠뜨리면 데이터를 새로 받는다
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라}
C=${C:?C 를 지정해라 (0.1 | 0.01 | median)}
SUITE=libero_goal
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}               # ★ 요청대로 칸당 20 rollout
BS_EVAL=${BS_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
EPS=${EPS:-50}
NTASK=${NTASK:-10}
NW=${NW:-6}
# true 면 매 스테이지 끝에 롤아웃까지 한다. false 면 학습만 하고 SR 은
# CLARE_rollout_only.py 가 다른 GPU 에서 채운다(학습이 훨씬 빨라진다).
EVAL_AFTER=${EVAL_AFTER:-true}
# mean (v4) 가중을 배치 평균 1 로 정규화 — 앵커 "크기"가 c 와 무관.
# none (v5) w~_h 를 그대로 사용 — 크기도 c 에 따라 변한다.
WNORM=${WNORM:-mean}
ARMTAG=${ARMTAG:-mar}              # 결과 폴더 접두사: clare_<ARMTAG>_goal_<tag>_seed<seed>
SEEDS=${SEEDS:-"7"}
CODEBOOK_K=${CODEBOOK_K:-96}
MIN_FREE_GB=${MIN_FREE_GB:-25}
PRETRAIN=${PRETRAIN_PATH:-/home/sa090180/Models/dit_flow_mt_libero_90_pretrain}
QUEUE="${HERE}/logs/CLARE_MAR.queue"

if [ "${C}" = "median" ]; then
    C_ARG=""                       # --anchor_c 0 == 기존 median 규칙
    TAGC="median"
else
    C_ARG="--anchor_c ${C}"
    TAGC="c${C}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

filled_cells() {
    python - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists():
    print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
}

for SEED in ${SEEDS}; do
    TAG="clare_${ARMTAG}_goal_${TAGC}_seed${SEED}"
    OUT="results/${TAG}"
    LOG="logs/${TAG}.log"
    CKPT="${HERE}/outputs/CLARE_MAR/${ARMTAG}_goal_${TAGC}_seed${SEED}"
    NEED=$(( NTASK * (NTASK + 1) / 2 ))

    if [ -n "${DRY:-}" ]; then
        echo "GPU ${GPU}  c=${C}  seed ${SEED}  칸당 ${N_EVAL} rollout -> ${OUT}"; continue
    fi

    DONE=$(filled_cells "${OUT}")
    if [ "${DONE}" -ge "${NEED}" ]; then
        echo "[$(date '+%F %T')] skip ${TAG} — 이미 ${DONE}/${NEED}칸 완주" | tee -a "${QUEUE}"; continue
    fi

    FREE=$(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')
    if [ "${FREE}" -lt "${MIN_FREE_GB}" ]; then
        echo "[$(date '+%F %T')] ✗ ${TAG} 시작 안 함 — 디스크 여유 ${FREE}G < ${MIN_FREE_GB}G" | tee -a "${QUEUE}"
        break
    fi

    echo "[$(date '+%F %T')] ▶ ${TAG}  GPU ${GPU}  c=${C} wnorm=${WNORM}  (${DONE}/${NEED}칸, 여유 ${FREE}G)" | tee -a "${QUEUE}"
    echo "════ $(date '+%F %T') 시작  c=${C} ════" >> "${LOG}"

    PRETRAIN_PATH="${PRETRAIN}" python -u L2_V4_seq.py \
        --arm v4 --suite "${SUITE}" --num_tasks "${NTASK}" \
        --episodes_per_task "${EPS}" --steps_per_task "${STEPS}" \
        --batch_size 32 --num_workers "${NW}" \
        --codebook_k "${CODEBOOK_K}" --lambda_level 3.0 \
        ${C_ARG} --anchor_wnorm "${WNORM}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
        --eval_after_each_task "${EVAL_AFTER}" \
        --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --ckpt_root "${CKPT}" \
        --out "${OUT}" --resume >> "${LOG}" 2>&1
    rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다

    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}  ($(filled_cells "${OUT}")/${NEED}칸)" | tee -a "${QUEUE}"
    [ "${rc}" -ne 0 ] && echo "   -> ${LOG} 확인" | tee -a "${QUEUE}"
done
echo "[$(date '+%F %T')] GPU ${GPU} c=${C} 큐 종료" | tee -a "${QUEUE}"
