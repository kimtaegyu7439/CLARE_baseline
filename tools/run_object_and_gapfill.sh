#!/usr/bin/env bash
# 한 번에 세 가지를 돌린다.
#
#   1) ER libero_40 빈 칸 4개 채우기  (stage19/t19, stage24/t0, stage31/t10,11)
#      ★ N_EVAL=20 — 이 표의 나머지 816칸이 20 에피소드로 잰 값이라 반드시 맞춰야 한다.
#   2) ER  libero_object 10×10  (55칸, 100 에피소드)
#   3) CLARE libero_object 10×10 (어댑터가 있는 stage0..2 = 6칸만. 나머지는 학습이 안 됐다)
#
# 2·3은 다른 단일 스위트 표와 같게 100 에피소드다. BS_EVAL=20은 E0 프로브에서 검증된 값.
# 이미 끝난 칸은 eval_info.json 존재로 건너뛰므로 중단 후 재실행해도 안전하다.
#
#   setsid nohup bash tools/run_object_and_gapfill.sh > <log> 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR=${LOGDIR:-/tmp/obj_eval_logs}
mkdir -p "${LOGDIR}"

# ── 1) libero_40 빈 칸. 스테이지 19/24/31만 훑으면 끝난 칸은 건너뛰고 4칸만 돈다.
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 N_EVAL=20 BS_EVAL=10 \
    STAGES="19 24 31" bash bash/er/eval_libero_40.sh > "${LOGDIR}/gapfill40.log" 2>&1 &

# ── 2) ER libero_object. 55칸을 4샤드로 쪼갠다 (칸수 15/14/14/12로 균형).
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 N_EVAL=100 BS_EVAL=20 \
    STAGES="9 2 1" bash bash/er/eval_libero_object.sh > "${LOGDIR}/er_obj_a.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 N_EVAL=100 BS_EVAL=20 \
    STAGES="8 3 0" bash bash/er/eval_libero_object.sh > "${LOGDIR}/er_obj_b.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 N_EVAL=100 BS_EVAL=20 \
    STAGES="7 5"   bash bash/er/eval_libero_object.sh > "${LOGDIR}/er_obj_c.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 N_EVAL=100 BS_EVAL=20 \
    STAGES="6 4"   bash bash/er/eval_libero_object.sh > "${LOGDIR}/er_obj_d.log" 2>&1 &

# ── 3) CLARE libero_object. 어댑터 없는 스테이지는 스크립트가 알아서 SKIP한다.
#      COLLECT=0 — 여기서 모으지 않고 맨 끝에 한 번만 모은다.
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 N_EVAL=100 BS_EVAL=20 COLLECT=0 \
    bash bash/clare/eval_libero_object_clare.sh > "${LOGDIR}/clare_obj.log" 2>&1 &

wait
echo "[run] 모든 평가 종료 $(date +%F\ %T)"

# ── 표로 모은다 ──────────────────────────────────────────────────────────────
bash bash/er/collect_er_sr.sh libero_40 libero_object
bash bash/clare/collect_clare_sr.sh libero_object
echo "[run] 완료 $(date +%F\ %T)"
