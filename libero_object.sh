#!/usr/bin/env bash
# K1 — libero_object 10 태스크. 기본 GPU 0 (k1.sh 가 끝난 뒤 이어서 돈다).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/k1_common.sh"
k1_run "${1:-0}" K1_object_10task 10 libero_object "${@:2}"
