#!/usr/bin/env bash
# CLARE(순차) 셋팅 libero_goal — 코드북 셀 수 K 스윕. GPU 한 장이 K 하나를 맡고,
# 그 안에서 시드 7 -> 42 -> 1000 을 순차로 돈다.
#
#   GPU=0 K=90  bash CLARE_goal_K_queue.sh
#   GPU=1 K=100 bash CLARE_goal_K_queue.sh
#   GPU=2 K=110 bash CLARE_goal_K_queue.sh
#   GPU=3 K=120 bash CLARE_goal_K_queue.sh
#   DRY=1 GPU=0 K=90 bash CLARE_goal_K_queue.sh      큐 내용만 출력
#
# 프로토콜은 CLARE_v4_sr.sh 와 **한 글자도 다르지 않게** 맞췄다 — 바꾸는 건 K 하나뿐이다.
# 그래야 기존 K=96 런(results/clare_v4_goal_seed*)이 이 표의 기준점으로 그대로 들어온다.
#   태스크당 10,000 step / 50 demo / 칸당 100 rollout(n_envs=50 x 2) / eval seed 7 고정
#   libero_90 사전학습 체크포인트에서 출발, task 0 부터 10 개 순차 = 55 칸
#
# ★ 디스크. 런 하나가 태스크별 체크포인트 1.1G x 10 = 11G 를 쓴다. 12 런이면 132G 라
#   남은 공간(123G)을 넘긴다. 그래서 **런이 끝나면 중간 태스크 체크포인트를 지우고
#   마지막 task_9 만 남긴다**(런당 11G -> 1.1G). 도는 중에는 손대지 않으므로
#   --resume 은 언제든 안전하다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"          # HF_LEROBOT_HOME — 빠뜨리면 데이터를 새로 받는다
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라}
K=${K:?K(코드북 셀 수)를 지정해라}
# ★ 기본값이 libero_goal 이라 SUITE 를 안 주면 기존 동작 그대로다.
SUITE=${SUITE:-libero_goal}
SHORT=${SUITE#libero_}          # libero_spatial -> spatial. 폴더 이름에 쓴다.
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-100}
BS_EVAL=${BS_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
EPS=${EPS:-50}
NTASK=${NTASK:-10}
NW=${NW:-5}                        # 카드당 잡 1개 x 4카드 = 워커 20개 (32코어)
SEEDS=${SEEDS:-"7 42 1000"}
KEEP_CKPT=${KEEP_CKPT:-1}          # 0 이면 정리하지 않는다(디스크가 넉넉할 때)
MIN_FREE_GB=${MIN_FREE_GB:-25}
PRETRAIN=${PRETRAIN_PATH:-/home/sa090180/Models/dit_flow_mt_libero_90_pretrain}
QUEUE="${HERE}/logs/CLARE_goal_K.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

# 채운 칸 수. metrics.json 은 칸마다 다시 쓰이므로 존재만으로는 완주의 증거가 못 된다.
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

# 완주한 런의 중간 체크포인트를 지운다. 마지막 태스크는 남긴다(시퀀스 끝 모델).
prune_ckpt() {
    local ROOT="$1"
    [ "${KEEP_CKPT}" = "0" ] && return 0
    [ -d "${ROOT}" ] || return 0
    local LAST=$(( NTASK - 1 ))
    local N=0
    for D in "${ROOT}"/*/task_*; do
        [ -d "${D}" ] || continue
        [ "$(basename "${D}")" = "task_${LAST}" ] && continue
        rm -rf "${D}" && N=$(( N + 1 ))
    done
    [ "${N}" -gt 0 ] && echo "[$(date '+%F %T')]   중간 체크포인트 ${N}개 정리 (task_${LAST} 보존)" \
        | tee -a "${QUEUE}"
    return 0
}

for SEED in ${SEEDS}; do
    TAG="clare_v4_${SHORT}_k${K}_seed${SEED}"
    OUT="results/${TAG}"
    LOG="logs/${TAG}.log"
    CKPT="${HERE}/outputs/CLARE_V4_K/${SHORT}_k${K}_seed${SEED}"
    NEED=$(( NTASK * (NTASK + 1) / 2 ))

    if [ -n "${DRY:-}" ]; then
        echo "GPU ${GPU}  K=${K}  seed ${SEED}  ${STEPS} step  칸당 ${N_EVAL} rollout -> ${OUT}"
        continue
    fi

    DONE=$(filled_cells "${OUT}")
    if [ "${DONE}" -ge "${NEED}" ]; then
        echo "[$(date '+%F %T')] skip ${TAG} — 이미 ${DONE}/${NEED}칸 완주" | tee -a "${QUEUE}"
        prune_ckpt "${CKPT}"; continue
    fi

    # 디스크 여유가 없으면 시작하지 않는다. 런 도중에 차면 체크포인트 저장이 깨지고
    # --resume 도 못 하게 된다 — 시작 전에 막는 편이 낫다.
    FREE=$(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')
    if [ "${FREE}" -lt "${MIN_FREE_GB}" ]; then
        echo "[$(date '+%F %T')] ✗ ${TAG} 시작 안 함 — 디스크 여유 ${FREE}G < ${MIN_FREE_GB}G" \
            | tee -a "${QUEUE}"
        break
    fi

    echo "[$(date '+%F %T')] ▶ ${TAG}  GPU ${GPU}  K=${K}  (${DONE}/${NEED}칸에서 이어감, 여유 ${FREE}G)" \
        | tee -a "${QUEUE}"
    echo "════ $(date '+%F %T') 시작 (K=${K}, resume, ${DONE}/${NEED}칸) ════" >> "${LOG}"

    PRETRAIN_PATH="${PRETRAIN}" python -u L2_V4_seq.py \
        --arm v4 --suite "${SUITE}" --num_tasks "${NTASK}" \
        --episodes_per_task "${EPS}" --steps_per_task "${STEPS}" \
        --batch_size 32 --num_workers "${NW}" \
        --codebook_k "${K}" --lambda_level 3.0 \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
        --eval_after_each_task true \
        --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --ckpt_root "${CKPT}" \
        --out "${OUT}" --resume >> "${LOG}" 2>&1
    rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다

    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}  ($(filled_cells "${OUT}")/${NEED}칸)" | tee -a "${QUEUE}"
    if [ "${rc}" -ne 0 ]; then
        echo "   -> ${LOG} 확인. 체크포인트는 남겨 둔다(--resume 으로 이어감)" | tee -a "${QUEUE}"
    else
        prune_ckpt "${CKPT}"
    fi
done
echo "[$(date '+%F %T')] GPU ${GPU} K=${K} 큐 종료" | tee -a "${QUEUE}"
