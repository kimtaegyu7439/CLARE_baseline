#!/usr/bin/env bash
# V5 (앵커 가중을 배치평균으로 나누지 않음) 를 추가 시드로 돌린다.
#
#   GPU=0 C=0.1  bash CLARE_v5_seeds.sh
#   GPU=1 C=0.05 bash CLARE_v5_seeds.sh
#   GPU=2 C=0.01 bash CLARE_v5_seeds.sh
#   DRY=1 GPU=0 C=0.1 bash CLARE_v5_seeds.sh
#
# ★ GPU 하나당 프로세스 하나. 시드를 순서대로, 각 시드마다
#      학습(롤아웃 없이) -> 롤아웃 50회 -> 중간 체크포인트 정리
#   를 끝내고 다음 시드로 간다. 학습과 롤아웃이 같은 카드에서 겹치지 않는다.
#
# 롤아웃 50회는 eval_seed=7 로 **시드 7~56** 을 쓴다. seed7 런의 n50
# (20회 7~26 + 30회 27~56 을 합친 것)과 같은 초기 상태 집합이다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라}
C=${C:?C 를 지정해라 (0.1 | 0.05 | 0.01)}
SEEDS=${SEEDS:-"42 1000"}
SUITE=libero_goal
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
EPS=${EPS:-50}
NTASK=${NTASK:-10}
NW=${NW:-6}
CODEBOOK_K=${CODEBOOK_K:-96}
KEEP_CKPT=${KEEP_CKPT:-1}
MIN_FREE_GB=${MIN_FREE_GB:-30}
PRETRAIN=${PRETRAIN_PATH:-/home/sa090180/Models/dit_flow_mt_libero_90_pretrain}
QUEUE="${HERE}/logs/CLARE_v5_seeds.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

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

prune() {   # 롤아웃까지 끝난 런의 중간 체크포인트를 지운다(마지막 태스크는 남긴다)
    local ROOT="$1"; [ "${KEEP_CKPT}" = "0" ] && return 0
    local LAST=$(( NTASK - 1 )) N=0
    for D in "${ROOT}"/*/task_*; do
        [ -d "${D}" ] || continue
        [ "$(basename "${D}")" = "task_${LAST}" ] && continue
        rm -rf "${D}" && N=$(( N + 1 ))
    done
    [ "${N}" -gt 0 ] && ts "   중간 체크포인트 ${N}개 정리"
    return 0
}

for SEED in ${SEEDS}; do
    TAG="clare_v5_goal_c${C}_seed${SEED}"
    OUT="results/${TAG}"
    CKPT="${HERE}/outputs/CLARE_MAR/v5_goal_c${C}_seed${SEED}"
    LOG="logs/${TAG}.log"
    NEED=$(( NTASK * (NTASK + 1) / 2 ))

    if [ -n "${DRY:-}" ]; then
        echo "GPU ${GPU}  c=${C}  seed ${SEED}  학습 -> 롤아웃 ${N_EVAL}회 -> ${OUT}"; continue
    fi

    FREE=$(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')
    if [ "${FREE}" -lt "${MIN_FREE_GB}" ]; then
        ts "✗ ${TAG} 시작 안 함 — 디스크 여유 ${FREE}G < ${MIN_FREE_GB}G"; break
    fi

    # ── 1단계: 학습 (롤아웃 없이) ────────────────────────────────────────
    ts "▶ ${TAG} 학습  GPU ${GPU}  (여유 ${FREE}G)"
    echo "════ $(date '+%F %T') 학습 시작 c=${C} seed=${SEED} ════" >> "${LOG}"
    PRETRAIN_PATH="${PRETRAIN}" python -u L2_V4_seq.py \
        --arm v4 --suite "${SUITE}" --num_tasks "${NTASK}" \
        --episodes_per_task "${EPS}" --steps_per_task "${STEPS}" \
        --batch_size 32 --num_workers "${NW}" \
        --codebook_k "${CODEBOOK_K}" --lambda_level 3.0 \
        --anchor_c "${C}" --anchor_wnorm none \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
        --eval_after_each_task false \
        --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --ckpt_root "${CKPT}" --out "${OUT}" --resume >> "${LOG}" 2>&1
    rc=$?
    ts "◀ ${TAG} 학습 exit=${rc}"
    [ "${rc}" -ne 0 ] && { ts "   -> ${LOG} 확인. 다음 시드로"; continue; }

    # ── 2단계: 롤아웃 50회 (전 행) ───────────────────────────────────────
    CKDIR=$(ls -d "${CKPT}"/*_ours 2>/dev/null | head -1)
    if [ -z "${CKDIR}" ]; then ts "   ★ 체크포인트 폴더 없음 — 롤아웃 건너뜀"; continue; fi
    ts "▶ ${TAG} 롤아웃 ${N_EVAL}회"
    python -u CLARE_rollout_only.py --ckpt_dir "${CKDIR}" --out "${OUT}" \
        --suite "${SUITE}" --rows all \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
        --eval_seed "${EVAL_SEED}" >> "${LOG}" 2>&1
    rc=$?
    ts "◀ ${TAG} 롤아웃 exit=${rc}  ($(filled "${OUT}")/${NEED}칸)"

    [ "$(filled "${OUT}")" -ge "${NEED}" ] && prune "${CKDIR}"
done
ts "GPU ${GPU} c=${C} 큐 종료"
