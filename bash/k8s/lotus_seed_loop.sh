#!/usr/bin/env bash
# 파드 **안에서** 도는 드라이버. 한 스위트의 시드를 순차로 돌린다.
#
#   SUITE=goal   SEEDS="7 42 1000" BETA=3.0 bash bash/k8s/lotus_seed_loop.sh
#   SUITE=50     SKIP_EVAL=1 bash bash/k8s/lotus_seed_loop.sh    # 학습만
#
# 쿠버네티스에서는 GPU 번호를 고르지 않는다 — 파드에 할당된 GPU 가 항상 0 이다.
# 그래서 "GPU 0/1/2/3" 배치는 1-GPU 파드 4개로 만든다 (bash/k8s/pod.sh 주석).
#
# ★ Lotus_l2cb_cl.py 에는 resume 이 없다. 그래서 끝난 시드는 metrics.json
#   (런 맨 끝에만 쓰인다) 으로 판별해 건너뛴다. 중간에 끊긴 시드는 **처음부터**
#   다시 돈다 — 스테이지 체크포인트는 남으니 롤아웃만 따로 돌릴 수는 있다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${HERE}"

SUITE=${SUITE:?SUITE (goal|object|50)}
SEEDS=${SEEDS:-"7 42 1000"}
BETA=${BETA:-3.0}
LIFELONG_K=${LIFELONG_K:-25}
SCOPE=${SCOPE:-lifelong}
# scope 를 태그에 짧게: lifelong -> LL, all -> ALL
SCOPE_TAG=$([ "${SCOPE}" = "all" ] && echo ALL || echo LL)     # L3: lifelong 코드북 셀 수를 태스크당 고정
ENTRY=${ENTRY:-Lotus_L3_V7dea.py}
FORM=${FORM:-dea}
SPACE=${SPACE:-act}
REDUCE=${REDUCE:-mean}
EPS_W=${EPS_W:-1e-3}
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_BASE_EVAL=${SKIP_BASE_EVAL:-1}
NW=${NW:-8}
VENV=${VENV:-/home/asdl/tether-venv}
EXTRA=${EXTRA:-}

# ── 환경 ────────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
. "${VENV}/bin/activate"
source "${HERE}/bash/clare/env.sh"          # ~/nvegl 가 있으면 EGL 을 NVIDIA 로 돌린다
export MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
# 128 코어 노드다. 캡을 안 씌우면 BLAS 가 코어 수만큼 스레드를 띄워
# 파드 4개가 서로 CPU 를 빼앗는다 (H200 에서 420 스레드까지 갔던 그 문제).
T=${OMP_THREADS:-8}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T

ts() { echo "[$(date '+%F %T')] $*"; }
ts "════ LOTUS L3 ${FORM} β${BETA} lifelong K=${LIFELONG_K} — suite=${SUITE}, 시드 ${SEEDS} ════"
python -c "import torch;print(f'    GPU {torch.cuda.device_count()}개  {torch.cuda.get_device_name(0)}')" 2>/dev/null

for SEED in ${SEEDS}; do
    # ★ 태그에 K 를 넣는다 — K=20 으로 돌린 기존 lotus_v7_dea_* 과 섞이면 안 된다.
    TAG="lotus_l3_${FORM}_b${BETA}_k${LIFELONG_K}_scope${SCOPE_TAG}_${SUITE}_seed${SEED}"
    OUT="results/${TAG}"
    # ★ metrics.json 으로 판단하면 안 된다 — 롤아웃 전담 파드가 **첫 칸을 잴 때**
    #   이미 그 파일을 만든다. 그걸 "완주"로 읽으면 학습이 안 끝난 시드를
    #   건너뛴다. skip_eval 런은 metrics_train.json, 보통 런은 "[LOTUS] 완료".
    if [ -f "${OUT}/metrics_train.json" ] \
       || grep -q '\[LOTUS\] 완료' "${OUT}/train.log" 2>/dev/null; then
        ts "⏭ ${TAG} 학습 이미 완료"; continue
    fi
    mkdir -p "${OUT}"
    ARGS=(--suite "${SUITE}" --steps "${STEPS}"
          --lifelong_k "${LIFELONG_K}" --anchor_scope "${SCOPE}"
          --anchor_form "${FORM}" --beta "${BETA}" --eps_w "${EPS_W}"
          --dir_space "${SPACE}" --anchor_reduce "${REDUCE}"
          --sampler_mode hard_linear
          --seed "${SEED}" --eval_seed "${EVAL_SEED}"
          --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}"
          --num_workers "${NW}" --out "${OUT}")
    [ "${SKIP_BASE_EVAL}" = "1" ] && ARGS+=(--skip_base_eval)
    [ "${SKIP_EVAL}" = "1" ] && ARGS+=(--skip_eval)
    # shellcheck disable=SC2206
    [ -n "${EXTRA}" ] && ARGS+=(${EXTRA})

    ts "▶ ${TAG}  (로그 ${OUT}/train.log)"
    python -u "${ENTRY}" "${ARGS[@]}" >> "${OUT}/train.log" 2>&1
    rc=$?
    ts "◀ ${TAG} exit=${rc}"
    if [ "${rc}" -ne 0 ]; then
        ts "   ✗ 실패 — ${OUT}/train.log 의 마지막 30줄:"
        tail -30 "${OUT}/train.log" | sed 's/^/     /'
        ts "   다음 시드로 넘어간다"
    fi
done

ts "════ ${SUITE} 전부 종료 ════"
for SEED in ${SEEDS}; do
    TAG="lotus_l3_${FORM}_b${BETA}_k${LIFELONG_K}_scope${SCOPE_TAG}_${SUITE}_seed${SEED}"
    printf '  %-52s %s\n' "${TAG}" \
        "$([ -f "results/${TAG}/metrics_train.json" ] \
            || grep -q '\[LOTUS\] 완료' "results/${TAG}/train.log" 2>/dev/null \
            && echo 학습완료 || echo 미완)"
done
