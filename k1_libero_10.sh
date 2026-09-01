#!/usr/bin/env bash
# K1 — libero_10 10 태스크. 기본 GPU 3.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/k1_common.sh"
k1_run "${1:-3}" K1_l10_10task 10 libero_10 "${@:2}"
