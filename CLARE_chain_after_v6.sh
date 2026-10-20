#!/usr/bin/env bash
# V6 가 전부 끝나면 미뤄 둔 V5 작업을 되살린다.
#   GPU 0  V5 c=0.1  seed1000  (학습 stage3 부터 + 롤아웃)
#   GPU 1  V5 c=0.05 seed1000  (〃)
#   GPU 2  V5 c=0.01 seed1000  **롤아웃만** — 학습은 이미 끝났고 32/55칸 차 있다
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
while pgrep -f "[C]LARE_V6_queue.sh" > /dev/null 2>&1; do sleep 120; done
echo "[$(date '+%F %T')] V6 전부 종료 — V5 재개"
setsid nohup env GPU=0 C=0.1  SEEDS=1000 bash CLARE_v5_seeds.sh > logs/launch_v5resume_gpu0.log 2>&1 &
setsid nohup env GPU=1 C=0.05 SEEDS=1000 bash CLARE_v5_seeds.sh > logs/launch_v5resume_gpu1.log 2>&1 &
# c0.01 은 학습이 끝났으므로 롤아웃만 이어 붙인다(찬 칸은 드라이버가 건너뛴다)
CK=$(ls -d outputs/CLARE_MAR/v5_goal_c0.01_seed1000/*_ours 2>/dev/null | head -1)
[ -n "${CK}" ] && setsid nohup env GPU=2 ROWS=all CK="${CK}" \
    OUT=results/clare_v5_goal_c0.01_seed1000 N_EVAL=50 \
    bash CLARE_rollout_queue.sh > logs/launch_v5resume_gpu2.log 2>&1 &
echo "[$(date '+%F %T')] 3 건 재개 요청 완료"
