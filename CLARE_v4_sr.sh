#!/usr/bin/env bash
# CLARE 셋팅 v4(TETHER) SR 매트릭스 — libero_10 / goal / spatial x train seed 7,42,1000.
#
# LOTUS 셋팅(L2_V4.py)이 아니라 **원본 CLARE 순차 셋팅**이다. task 0 부터 하나씩
# 배우고, task 0 의 초기값은 libero_90 사전학습 체크포인트다(L2_V4_seq.py 가
# PRETRAIN_PATH 또는 ~/Models/dit_flow_mt_libero_90_pretrain 를 쓴다).
#
# 배치: GPU 한 장에 **스위트 하나**, 그 안에서 시드 7 -> 42 -> 1000 을 **순차**로.
#   GPU 0  libero_10       GPU 1  libero_goal       GPU 2  libero_spatial
#
# ★ 카드당 잡 하나만 올리는 이유. 학습만 보면 5GB 라 3개도 들어가지만, SR 칸마다
#   n_envs=50 짜리 벡터 env 를 띄운다. MuJoCo EGL 렌더 컨텍스트 50개 + 배치 50
#   정책 forward 가 한꺼번에 올라가서 롤아웃 구간의 피크가 학습 구간보다 훨씬 높다.
#   같은 카드의 잡들이 우연히 동시에 롤아웃에 들어가면 24GB 를 넘긴다.
#
#   bash CLARE_v4_sr.sh              GPU 0/1/2 큐 3개를 띄운다
#   bash CLARE_v4_sr.sh spatial      한 스위트 큐만
#   DRY=1 bash CLARE_v4_sr.sh        큐 내용만 출력
#
# 이미 끝난 조합(results/<tag>/metrics.json 존재)은 건너뛴다 — 중간에 끊겨도
# 같은 명령으로 이어서 돌리면 된다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-100}
BS_EVAL=${BS_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}          # train seed 만 바꿔 변동을 잰다 — 평가 조건은 고정
EPS=${EPS:-50}                     # CLARE 원본과 동일: hold-out 없이 전량 학습
NTASK=${NTASK:-10}
NW=${NW:-8}                        # 카드당 잡 1개 x 3카드 = 워커 24개. 32코어에 맞다.
SEEDS=${SEEDS:-"7 42 1000"}
PRETRAIN=${PRETRAIN_PATH:-/home/sa090180/Models/dit_flow_mt_libero_90_pretrain}

mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

QUEUES=("libero_10:0" "libero_goal:1" "libero_spatial:2")

# 큐 하나 = GPU 하나. 시드를 순서대로, 앞이 끝나야 다음이 시작한다.
run_queue() {
    local SUITE="$1" GPU="$2"
    for SEED in ${SEEDS}; do
        local TAG="clare_v4_${SUITE#libero_}_seed${SEED}"
        local OUT="results/${TAG}"
        local LOG="logs/${TAG}.log"
        # 완주(55칸)한 것만 건너뛴다. metrics.json 은 SR 칸마다 다시 쓰이므로 존재만으로는
        # 완주의 증거가 못 된다 — 예전엔 이걸로 판단해 중간에 죽은 seed7 을 끝난 걸로 봤다.
        local DONE NEED=$(( NTASK * (NTASK + 1) / 2 ))
        DONE=$(python - "${OUT}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists():
    print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
)
        if [ "${DONE}" -ge "${NEED}" ]; then
            echo "[queue ${SUITE}] skip ${TAG} — 이미 ${DONE}/${NEED}칸 완주"; continue
        fi
        # 죽은 실행은 지우지 않고 --resume 으로 끊긴 지점부터 잇는다. L2_V4_seq.py 가
        # 체크포인트·코드북·sr_matrix.csv 를 보고 롤아웃부터인지 학습부터인지 정한다.
        echo "[$(date '+%F %T')] [queue ${SUITE}] ▶ GPU ${GPU} ${TAG}  (log ${LOG}, ${DONE}/${NEED}칸에서 이어감)"
        echo "════ $(date '+%F %T') 시작 (resume, ${DONE}/${NEED}칸) ════" >> "${LOG}"
        CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PRETRAIN_PATH="${PRETRAIN}" \
            python -u L2_V4_seq.py \
                --arm v4 --suite "${SUITE}" --num_tasks "${NTASK}" \
                --episodes_per_task "${EPS}" --steps_per_task "${STEPS}" \
                --batch_size 32 --num_workers "${NW}" \
                --codebook_k 96 --lambda_level 3.0 \
                --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
                --eval_after_each_task true \
                --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
                --ckpt_root "${HERE}/outputs/CLARE_V4/${SUITE}_seed${SEED}" \
                --out "${OUT}" --resume >> "${LOG}" 2>&1
        local rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다
        echo "[$(date '+%F %T')] [queue ${SUITE}] ◀ ${TAG} exit=${rc}"
    done
    echo "[$(date '+%F %T')] [queue ${SUITE}] 큐 완료"
}

FILTER="${1:-}"
for Q in "${QUEUES[@]}"; do
    IFS=':' read -r SUITE GPU <<< "${Q}"
    [ -n "${FILTER}" ] && [[ "${SUITE}" != *"${FILTER}"* ]] && continue
    if [ -n "${DRY:-}" ]; then
        echo "GPU ${GPU}  ${SUITE}  seeds: ${SEEDS}"; continue
    fi
    run_queue "${SUITE}" "${GPU}" &
done
wait
echo "[$(date '+%F %T')] 전 큐 종료"
