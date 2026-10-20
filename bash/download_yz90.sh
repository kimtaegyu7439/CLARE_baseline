#!/usr/bin/env bash
# yzembodied/libero_90_image (LeRobot v2.1, 74.6 GB, 4500 에피소드 = 90 task x 50 demo)
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
DEST=${DEST:-/home/sa090180/Datasets/lerobot/yzembodied_libero_90_dataset}
LOG=logs/download_yz90.log
mkdir -p "${DEST}" logs
echo "[$(date '+%F %T')] 시작 -> ${DEST}" >> "${LOG}"
hf download yzembodied/libero_90_image --repo-type dataset \
    --local-dir "${DEST}" --max-workers 8 >> "${LOG}" 2>&1
echo "[$(date '+%F %T')] 종료 rc=$?  $(du -sh "${DEST}" 2>/dev/null | cut -f1)" >> "${LOG}"
