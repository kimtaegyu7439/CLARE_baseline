#!/usr/bin/env bash
# H200 (GPU 1장) 에서 Lotus_eval_seeds.py 를 런 × 평가시드 로 MAXJOBS 개씩 병렬 실행.
#   (venv 켠 상태에서)
#   SEEDS="$(seq -s ' ' 8 27)" MAXJOBS=4 setsid nohup bash bash/Lotus_eval_seeds_h200_queue.sh > /dev/null 2>&1 &
#   RUNS="..."   기본 세 런 (λ0.7 β2 object seed 7/42/1000)
#   SEEDS        기본 8..107 (100개; 0..7 은 서버1/4090_2 에서 이미 쟀다). 런 3개 × 100 = 300 잡, 4병렬 ≈ 2.3일
#   MAXJOBS      기본 4 (잡당 VRAM ~3G, CPU 가 병목)
# 결과: results/<run>/eval_seeds/seed<N>.json + results/LOTUS_eval_seed_sweep.txt (한 줄/시드)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
source "${HERE}/bash/clare/env.sh"
PY=${PY:-$(command -v python || command -v python3)}
_chk='from lerobot.envs.configs import LIBERO_TASK_PREFIX_TO_BENCHMARK; from lerobot.datasets.factory import resolve_delta_timestamps'
if ! "${PY}" -c "${_chk}" 2>/dev/null; then
    export PYTHONPATH="${HERE}/lerobot_lsy/src${PYTHONPATH:+:${PYTHONPATH}}"
    "${PY}" -c "${_chk}" 2>/dev/null || { echo "✗ lerobot import 실패"; exit 1; }
fi
RUNS=${RUNS:-"lotus_v13_b2.0_lam0.7_k25_scopeLL_object_seed7 lotus_v13_b2.0_lam0.7_k25_scopeLL_object_seed42 lotus_v13_b2.0_lam0.7_k25_scopeLL_object_seed1000"}
SEEDS=${SEEDS:-"$(seq -s ' ' 8 107)"}
MAXJOBS=${MAXJOBS:-4}; GPU=${GPU:-0}
_per=$(( $(nproc) / MAXJOBS )); [ "${_per}" -lt 2 ] && _per=2; [ "${_per}" -gt 6 ] && _per=6
mkdir -p logs; QLOG=logs/h200_eval_seeds_queue.log
{
echo "[$(date '+%F %T')] ════ 평가시드 큐  seeds {${SEEDS}}  ${MAXJOBS} 병렬 ════"
for R in ${RUNS}; do
    [ -s "results/${R}/checkpoints/stage4/checkpoints/last/pretrained_model/model.safetensors" ] \
        || { echo "✗ ${R} 체크포인트 없음 → bash bash/h200_eval_seeds_assets.sh"; continue; }
    for S in ${SEEDS}; do
        [ -f "results/${R}/eval_seeds/seed${S}.json" ] && { echo "⏭ ${R} seed${S} 완료"; continue; }
        while [ "$(jobs -rp | wc -l)" -ge "${MAXJOBS}" ]; do wait -n; done
        echo "[$(date '+%F %T')] ▶ ${R} seed${S}"
        CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
        OMP_NUM_THREADS="${_per}" MKL_NUM_THREADS="${_per}" \
        "${PY}" -u Lotus_eval_seeds.py --run "results/${R}" --seeds "${S}" \
            > "logs/eval_seeds_${R##*_}_s${S}.log" 2>&1 &
        sleep 20
    done
done
wait
echo "[$(date '+%F %T')] ════ 큐 완료 ════"
} >> "${QLOG}" 2>&1
echo "큐를 띄웠다 — 진행: tail -f ${QLOG}   결과: results/LOTUS_eval_seed_sweep.txt"
