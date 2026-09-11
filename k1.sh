#!/usr/bin/env bash
# K1 — libero_spatial task 0,1,2,3 (기존 4 태스크 세팅). 기본 GPU 0.
#   bash k1.sh          / bash k1.sh <GPU>
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/k1_common.sh"
k1_run "${1:-0}" K1 4 libero_spatial "${@:2}"
