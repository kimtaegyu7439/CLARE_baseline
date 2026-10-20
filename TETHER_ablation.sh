#!/usr/bin/env bash
# TETHER ablation — 추정 조건 좌표 (s̃,õ) 가 바뀌면 성능이 어떻게 변하는가.
#
#   1 indep    Independent Gaussian  + 정규화 없음 (w=1)
#   2 shuffle  Shuffled pairing      + 정규화 없음 (w=1)
#   3 joint    Joint reconstruction  + 정규화 없음 (w=1)
#   4 tether   Joint reconstruction  + 크기 정규화  ← Full TETHER
#
# 1~3 은 가중을 고정한 채 **조건 좌표 생성 절차만** 바꾼 비교다. 4 는 거기에
# 정규화를 더해 그 항이 따로 무엇을 보태는지 본다. 3 -> 4 가 정규화의 순수 효과다.
#
# 주장: 같은 teacher-matching objective 와 같은 query 예산을 써도, joint
#       reconstruction 이 독립 샘플링·shuffled pairing 보다 과거 task 를 더 잘
#       보존하고 신규 task 학습도 유지한다.
#
# 고정되는 것 — 코드에서 _sample() 한 군데만 갈리므로 자동으로 고정된다.
#   · 시드별 같은 출발 체크포인트(libero_90 사전학습), task 순서와 demo
#   · 태스크당 step 수, optimizer, lr 스케줄
#   · 과거 task 당 anchor query 수(n = batch 32)와 task 샘플링 비율(매 스텝 전체 과거)
#   · 앵커 합산 방식(청크 backward = 과거에 대한 합)과 λ_lvl = 3.0
#   · action 좌표 생성 절차(teacher 1-step 오일러 â 로 x_t 를 다시 깖)와 추론 설정
#   ※ 1~3 은 가중까지 같다. 4 만 가중이 다르다.
#
#   bash TETHER_ablation.sh                12런(4팔 x 3시드)을 GPU 3 에서 순차로
#   DRY=1 bash TETHER_ablation.sh          큐 내용만
#   ARMS="joint tether" bash TETHER_ablation.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${GPU:-3}
SUITE=${SUITE:-libero_spatial}
NTASK=${NTASK:-4}                  # ★ 아래 "태스크 수" 주석 참고
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}
BS_EVAL=${BS_EVAL:-20}
EPS=${EPS:-50}
NW=${NW:-4}
EVAL_SEED=${EVAL_SEED:-7}
SEEDS=${SEEDS:-"7 42 1000"}
ARMS=${ARMS:-"indep shuffle joint tether"}
PRETRAIN=${PRETRAIN_PATH:-/home/sa090180/Models/dit_flow_mt_libero_90_pretrain}

# 태스크 수: 지시에 없어서 4(t0~t3)로 잡았다. 이 저장소의 순차-셋팅 실행
# (v4_spatial_t0123 / seqft_spatial_t0123, TETHER_main_fig2)이 쓰는 길이와 같아
# 그림과 직접 대조된다. 10 태스크면 런당 18시간이라 12런이 GPU 한 장에서 9일이
# 된다. NTASK=10 으로 덮어쓸 수는 있다.

arm_spec() {   # 팔 -> "sampler anchor_weight 설명"
    case "$1" in
        indep)   echo "indep_gauss mse      IndependentGaussian (정규화 없음)" ;;
        shuffle) echo "shuffled    mse      ShuffledPairing (정규화 없음)" ;;
        joint)   echo "hard_linear mse      JointReconstruction (정규화 없음)" ;;
        tether)  echo "hard_linear speedinv JointReconstruction+정규화 (FullTETHER)" ;;
        *) return 1 ;;
    esac
}

mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

# 팔 x 시드가 아니라 **시드 바깥 / 팔 안쪽** 으로 돌린다. 중간에 끊겨도 네 팔이
# 고르게 채워져 있어야 비교가 되기 때문이다 — 팔 하나만 3시드 다 끝나 있으면
# 중간 리포트로는 아무것도 말할 수 없다.
for SEED in ${SEEDS}; do
    for ARM in ${ARMS}; do
        read -r SM AW DESC <<< "$(arm_spec "${ARM}")" || { echo "모르는 팔 ${ARM}"; exit 1; }
        TAG="tether_ab_${ARM}_seed${SEED}"
        OUT="results/${TAG}"
        LOG="logs/${TAG}.log"
        if [ -n "${DRY:-}" ]; then
            printf "GPU %s  %-8s sampler=%-12s weight=%-9s seed=%-5s -> %s\n" \
                "${GPU}" "${ARM}" "${SM}" "${AW}" "${SEED}" "${OUT}"
            continue
        fi
        # 완주(10칸)한 것만 건너뛴다. metrics.json 은 태스크마다 다시 쓰이므로
        # 존재만으로는 완주의 증거가 못 된다 — 중간에 죽은 런을 그냥 두면 리포트가
        # 반쯤 찬 행렬을 완성본으로 읽는다.
        DONE=$(python - "${OUT}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists():
    print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
)
        NEED=$(( NTASK * (NTASK + 1) / 2 ))
        if [ "${DONE}" -ge "${NEED}" ]; then
            echo "[skip] ${TAG} — 이미 ${DONE}/${NEED}칸 완주"; continue
        fi
        # 죽은 런은 지우지 않고 --resume 으로 끊긴 지점부터 잇는다(L2_V4_seq.py 가
        # 체크포인트·코드북·sr_matrix.csv 를 보고 롤아웃부터인지 학습부터인지 정한다).
        echo "[$(date '+%F %T')] ▶ ${TAG}  ${DESC}  (log ${LOG}, ${DONE}/${NEED}칸에서 이어감)"
        echo "════ $(date '+%F %T') 시작 (resume, ${DONE}/${NEED}칸) ════" >> "${LOG}"
        CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PRETRAIN_PATH="${PRETRAIN}" \
            python -u L2_V4_seq.py \
                --arm v4 --sampler_mode "${SM}" --anchor_weight "${AW}" \
                --suite "${SUITE}" --num_tasks "${NTASK}" \
                --episodes_per_task "${EPS}" --steps_per_task "${STEPS}" \
                --batch_size 32 --num_workers "${NW}" \
                --codebook_k 96 --lambda_level 3.0 \
                --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
                --eval_after_each_task true \
                --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
                --ckpt_root "${HERE}/outputs/TETHER_AB/${ARM}_seed${SEED}" \
                --out "${OUT}" --resume >> "${LOG}" 2>&1
        rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다
        echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}"
        python clare_ablation_report.py > /dev/null 2>&1 || true
    done
done
echo "[$(date '+%F %T')] ablation 전체 종료"
python clare_ablation_report.py | tail -40
