#!/usr/bin/env bash
# 파드 **안에서** 도는 L2_V5 드라이버 (LOTUS 셋팅). 한 스위트의 시드를 순차로.
#
#   SUITE=goal C=0.1 WARM=0.25 SEEDS="7 42 1000" bash bash/k8s/v5_seed_loop.sh
#   SUITE=object LAM=0.5 BETA=2.0 SEEDS="7 42 1000" bash bash/k8s/v5_seed_loop.sh
#
# bash/Lotus_V5_run.sh 의 run_one 과 같은 명령이고, 파드 환경에 맞춘 점만 다르다:
#   · venv 로 켠다 (파드 이미지의 torch 2.8.0+cu128 위에 얹은 /home/asdl/tether-venv)
#   · CUDA_VISIBLE_DEVICES 를 안 건드린다 — k8s 가 GPU 1 장을 주면 항상 0 번이다
#   · BLAS 스레드를 캡한다 — 128 코어 노드에서 파드끼리 CPU 를 빼앗는다
#
# 태그: lotus_v5act_c<c>[_lam<λ>][_b<β>][_warm<w>]_k<K>_scope<LL|ALL>_<suite>_seed<N>
#   λ=1.0 / β=1.0 / warm=0 은 기본값이라 생략 — bash/Lotus_V5_run.sh 와 같은 규칙이다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${HERE}"

SUITE=${SUITE:?SUITE (goal|object|50)}
SEEDS=${SEEDS:-"7 42 1000"}
C=${C:-0.1}
LAM=${LAM:-1.0}
WARM=${WARM:-0}
BETA=${BETA:-1.0}        # 방향 항 배율 (1.0 = 기존 V5)
SPACE=${SPACE:-act}
SCOPE=${SCOPE:-lifelong}
LIFELONG_K=${LIFELONG_K:-25}
REDUCE=${REDUCE:-mean}
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
NW=${NW:-8}
VENV=${VENV:-/home/asdl/tether-venv}
EXTRA=${EXTRA:-}

# shellcheck disable=SC1091
. "${VENV}/bin/activate"
source "${HERE}/bash/clare/env.sh"          # ~/nvegl 가 있으면 EGL 을 NVIDIA 로
export MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
T=${OMP_THREADS:-8}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T

SCOPE_TAG=$([ "${SCOPE}" = "all" ] && echo ALL || echo LL)
LAM_TAG=$([ "${LAM}" = "1.0" ] && echo "" || echo "_lam${LAM}")
WARM_TAG=$([ "${WARM}" = "0" ] && echo "" || echo "_warm${WARM}")
BETA_TAG=$([ "${BETA}" = "1.0" ] && echo "" || echo "_b${BETA}")
tag_of() { echo "lotus_v5${SPACE}_c${C}${LAM_TAG}${BETA_TAG}${WARM_TAG}_k${LIFELONG_K}_scope${SCOPE_TAG}_${1}_seed${2}"; }
train_done() { grep -q '\[LOTUS\] 완료' "results/$1/train.log" 2>/dev/null; }

ts() { echo "[$(date '+%F %T')] $*"; }
ts "════ L2_V5 (${SPACE}) LOTUS — c=${C} λ=${LAM} β=${BETA} warmup=${WARM} K=${LIFELONG_K} scope=${SCOPE} — suite=${SUITE}, 시드 ${SEEDS} ════"
python -c "import torch;print(f'    GPU {torch.cuda.device_count()}개  {torch.cuda.get_device_name(0)}')" 2>/dev/null

for SEED in ${SEEDS}; do
    TAG=$(tag_of "${SUITE}" "${SEED}")
    OUT="results/${TAG}"
    if train_done "${TAG}"; then ts "⏭ ${TAG} 이미 완료"; continue; fi
    mkdir -p "${OUT}"
    ts "▶ ${TAG}  (로그 ${OUT}/train.log)"
    python -u L2_V5.py --setting lotus \
        --anchor_c "${C}" --dir_space "${SPACE}" --anchor_reduce "${REDUCE}" \
        --lambda_level "${LAM}" --lifelong_k "${LIFELONG_K}" \
        --anchor_scope "${SCOPE}" --anchor_warmup "${WARM}" --beta "${BETA}" \
        --suite "${SUITE}" --steps "${STEPS}" --sampler_mode hard_linear \
        --skip_base_eval --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
        --num_workers "${NW}" --out "${OUT}" ${EXTRA} \
        >> "${OUT}/train.log" 2>&1
    rc=$?    # ★ echo 안의 $(date) 가 $? 를 덮으므로 먼저 잡는다
    ts "◀ ${TAG} exit=${rc}"
    if [ "${rc}" -ne 0 ]; then
        ts "   ✗ 실패 — train.log 마지막 20줄:"; tail -20 "${OUT}/train.log" | sed 's/^/     /'
    fi
done
ts "════ ${SUITE} 전부 종료 ════"
for SEED in ${SEEDS}; do
    TAG=$(tag_of "${SUITE}" "${SEED}")
    printf '  %-60s %s\n' "${TAG}" "$(train_done "${TAG}" && echo 완료 || echo 미완)"
done
