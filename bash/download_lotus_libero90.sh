#!/usr/bin/env bash
# LIBERO libero_90 원본 HDF5 (66.7 GB, 90 파일) 내려받기.
# LIBERO 공식 download_libero_datasets.py 와 같은 소스(yifengzhu-hf/LIBERO-datasets)다.
# 파일은 ${DEST}/libero_90/*.hdf5 로 떨어진다.
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
DEST=${DEST:-/home/sa090180/Datasets/Lotus_libero_90}
LOG=logs/download_lotus_libero90.log
mkdir -p "${DEST}" logs
echo "[$(date '+%F %T')] libero_90 다운로드 시작 -> ${DEST}" >> "${LOG}"
hf download yifengzhu-hf/LIBERO-datasets \
    --repo-type dataset --include "libero_90/*" \
    --local-dir "${DEST}" --max-workers 8 >> "${LOG}" 2>&1
rc=$?
echo "[$(date '+%F %T')] 종료 rc=${rc}  파일 $(find "${DEST}" -name '*.hdf5' | wc -l)개  $(du -sh "${DEST}" 2>/dev/null | cut -f1)" >> "${LOG}"
