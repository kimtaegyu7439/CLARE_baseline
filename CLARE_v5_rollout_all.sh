#!/usr/bin/env bash
# V5 세 팔의 완료된 행을 한 GPU 가 돌아가며 채운다. 학습이 끝나고 165칸이
# 전부 차면 종료. (팔당 55칸 x 3)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source bash/clare/env.sh; export HF_HUB_OFFLINE=1
GPU=${GPU:?}; INTERVAL=${INTERVAL:-240}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ARMS=${ARMS:-"v5_goal_c0.01 v5_goal_c0.05 v5_goal_c0.1"}
while true; do
    TOTAL=0
    for A in ${ARMS}; do
        CK=$(ls -d outputs/CLARE_MAR/${A}_seed7/*_ours 2>/dev/null | head -1)
        [ -z "${CK}" ] && continue
        ROWS=""
        for k in $(seq 0 9); do
            [ -f "${CK}/task_${k}/checkpoints/last/pretrained_model/model.safetensors" ] && ROWS="${ROWS}${ROWS:+,}${k}"
        done
        [ -z "${ROWS}" ] && continue
        python -u CLARE_rollout_only.py --ckpt_dir "${CK}" --out "results/clare_${A}_seed7" \
            --suite libero_goal --rows "${ROWS}" --eval_episodes 20 --eval_batch_size 20 \
            >> "logs/rollout_${A}.log" 2>&1
        N=$(python3 -c "
import pathlib
p=pathlib.Path('results/clare_${A}_seed7/sr_matrix.csv')
b=[l for l in p.read_text().splitlines() if l.strip() and not l.startswith('#')] if p.exists() else []
print(sum(1 for l in b[1:] for c in l.split(',')[1:] if c.strip()))")
        TOTAL=$((TOTAL+N))
    done
    echo "[$(date '+%F %T')] 누적 ${TOTAL}/165 칸"
    [ "${TOTAL}" -ge 165 ] && { echo "전부 완료"; break; }
    pgrep -f "[L]2_V4_seq.py" >/dev/null || echo "  (학습 종료됨)"
    sleep "${INTERVAL}"
done
