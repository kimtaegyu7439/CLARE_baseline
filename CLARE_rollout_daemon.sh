#!/usr/bin/env bash
# 학습이 도는 동안, 끝난 스테이지의 SR 칸을 다른 GPU 에서 채운다.
#
#   GPU=2 ARM=c0.1  bash CLARE_rollout_daemon.sh
#   GPU=3 ARM=c0.01 bash CLARE_rollout_daemon.sh
#
# 한 바퀴마다 체크포인트가 생긴 행(task_k/checkpoints/last)을 찾아 빈 칸을 채우고,
# 더 할 게 없으면 잠시 쉬었다 다시 본다. 학습이 끝나고 모든 칸이 차면 빠진다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"; export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
GPU=${GPU:?}; ARM=${ARM:?}          # ARM = 폴더 접미 (예: v5_goal_c0.01)
N_EVAL=${N_EVAL:-20}; NTASK=${NTASK:-10}; INTERVAL=${INTERVAL:-300}
CK="${HERE}/outputs/CLARE_MAR/${ARM}_seed7"
OUT="results/clare_${ARM}_seed7"
LOG="logs/rollout_daemon_${ARM}.log"
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ckpt_root 는 <ckpt_root>/<suite>_seed<seed>_ours 아래에 task_* 가 있다
CKDIR=$(ls -d "${CK}"/*_ours 2>/dev/null | head -1)
[ -z "${CKDIR}" ] && { echo "체크포인트 폴더 없음: ${CK}"; exit 1; }

echo "[$(date '+%F %T')] 데몬 시작  ARM=${ARM} GPU=${GPU}  ck=${CKDIR}" | tee -a "${LOG}"
while true; do
    # 체크포인트가 있는 행을 전부 넘긴다. 이미 찬 칸은 드라이버가 건너뛴다.
    ROWS=""
    for k in $(seq 0 $((NTASK-1))); do
        [ -f "${CKDIR}/task_${k}/checkpoints/last/pretrained_model/model.safetensors" ] && ROWS="${ROWS}${ROWS:+,}${k}"
    done
    if [ -n "${ROWS}" ]; then
        python -u CLARE_rollout_only.py --ckpt_dir "${CKDIR}" --out "${OUT}" \
            --suite libero_goal --rows "${ROWS}" \
            --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" >> "${LOG}" 2>&1
    fi
    # 학습이 끝났고 칸도 다 찼으면 종료
    FILLED=$(python3 - "${OUT}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists(): print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
)
    NEED=$(( NTASK * (NTASK + 1) / 2 ))
    if [ "${FILLED}" -ge "${NEED}" ]; then
        echo "[$(date '+%F %T')] ${ARM} 전부 완료 (${FILLED}/${NEED}) — 데몬 종료" | tee -a "${LOG}"; break
    fi
    if ! pgrep -f "[L]2_V4_seq.py" > /dev/null 2>&1; then
        echo "[$(date '+%F %T')] 학습 프로세스 없음. ${FILLED}/${NEED} 칸 — 한 바퀴 더 돌고 끝낸다" | tee -a "${LOG}"
    fi
    sleep "${INTERVAL}"
done
