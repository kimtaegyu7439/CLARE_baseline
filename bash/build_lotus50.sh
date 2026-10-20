#!/usr/bin/env bash
# yz90 다운로드가 끝나면 LIBERO-50 을 생성한다.
# OBJECT/GOAL 은 로컬 데이터만 쓰므로 이미 만들어져 있다.
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
D=/home/sa090180/Datasets/lerobot/yzembodied_libero_90_dataset
W=logs/build_lotus50.log
ts(){ echo "[$(date '+%F %T')] $*" >> "$W"; }
ts "대기 — yz90 다운로드 (4500 parquet + meta)"
while true; do
    n=$(find "$D" -name '*.parquet' 2>/dev/null | wc -l)
    run=$(pgrep -u sa090180 -f "hf download yzembodied" | wc -l)
    if [ "$n" -ge 4500 ] && [ -f "$D/meta/info.json" ]; then ts "다운로드 완료 (${n} parquet)"; break; fi
    if [ "$run" -eq 0 ]; then ts "★ 다운로드 프로세스가 없다 (${n}/4500) — 중단"; exit 1; fi
    sleep 300
done
ts "LIBERO-50 생성 시작"
python -u lotus_build_setting.py --only LIBERO-50 >> "$W" 2>&1
ts "생성 완료 rc=$?  $(du -sh /home/sa090180/Datasets/lerobot/LOTUS_setting 2>/dev/null | cut -f1)"
