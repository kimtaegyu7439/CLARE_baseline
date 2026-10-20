#!/usr/bin/env bash
# yifengzhu-hf/LIBERO-datasets (원본 LIBERO HDF5 데모, 100.4 GB) 내려받기.
#   libero_90 66.7GB · libero_10 13.7GB · object 7.4GB · goal 6.4GB · spatial 6.2GB
# --local-dir 로 받으므로 hub 캐시에 사본이 또 생기지 않는다(huggingface_hub>=0.23).
# 중단돼도 다시 실행하면 이어받는다.
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
DEST=${DEST:-/home/sa090180/Datasets/lerobot/yifengzhu-hf}
LOG=logs/download_libero_raw.log
mkdir -p "${DEST}" logs
echo "[$(date '+%F %T')] 다운로드 시작 -> ${DEST}" >> "${LOG}"
hf download yifengzhu-hf/LIBERO-datasets \
    --repo-type dataset \
    --local-dir "${DEST}" \
    --max-workers 8 >> "${LOG}" 2>&1
rc=$?
echo "[$(date '+%F %T')] 종료 rc=${rc}  용량 $(du -sh "${DEST}" 2>/dev/null | cut -f1)" >> "${LOG}"
