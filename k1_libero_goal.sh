#!/usr/bin/env bash
# K1 — libero_goal 10 태스크. 기본 GPU 2.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/k1_common.sh"
k1_run "${1:-2}" K1_goal_10task 10 libero_goal "${@:2}"
