#!/usr/bin/env bash
# K1 — libero_spatial 10 태스크. 기본 GPU 1.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/k1_common.sh"
k1_run "${1:-1}" K1_spatial_10task 10 libero_spatial "${@:2}"
