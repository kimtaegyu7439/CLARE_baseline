#!/usr/bin/env bash
# 롤아웃을 **먼저** 끝내고, 학습이 안 된 것만 그 다음에 돌린다.
#
#   bash bash/rollout_first.sh                 # 전체 (스윕과 같은 계획)
#   DRY=1 bash bash/rollout_first.sh           # 무엇을 어느 단계로 보낼지만 출력
#   ROLLOUT_PAR=4 MAXJOBS=8 bash bash/rollout_first.sh
#   PHASE=rollout bash bash/rollout_first.sh   # 1 단계만
#   PHASE=train   bash bash/rollout_first.sh   # 2 단계만
#   bash bash/sweep_ctl.sh start  (TARGET=bash/rollout_first.sh)
#
# 왜 필요한가
#   gym-libero 가 libero_goal 을 거부해 롤아웃이 55 칸 전부 0 초 만에 None 으로
#   끝난 적이 있다. 학습(런당 약 40 시간)은 멀쩡했고 체크포인트도 온전했다.
#   그래서 **재학습 없이 롤아웃만** 다시 돌리면 된다. 학습이 남은 런보다
#   이쪽이 훨씬 싸므로 먼저 치운다.
#
# 판정
#   · 55/55 칸      -> 아무것도 안 한다
#   · 체크포인트 완비(task_0..task_{NTASK-1}) -> 1 단계(롤아웃)
#   · 그 외          -> 2 단계(학습 + 롤아웃), 원래 스윕 순서 그대로
#
# 1 단계는 ROLLOUT_PAR 개씩, 2 단계는 MAXJOBS 개씩 병행한다.
# 롤아웃 VRAM 이 피크 약 22G 라 H200(143G) 에서 4 개가 안전선이다.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

GPU=${GPU:-0}
FORM=${FORM:-dea}
VERS=${VERS:-"v7 v6"}
SEEDS=${SEEDS:-"42 1000"}
BETAS=${BETAS:-"0.5 1.0 1.5 2.0 2.5 3.0"}
NTASK=${NTASK:-10}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
NW=${NW:-6}
STEPS=${STEPS:-10000}
ROLLOUT_PAR=${ROLLOUT_PAR:-4}      # 1 단계 동시 롤아웃 수
# ★ 롤아웃 VRAM 은 환경 50 개를 띄우는 **처음 1 분에만** 약 22G 로 치솟고
#   그 뒤에는 5.9G 로 떨어진다(실측). 그래서 시작을 충분히 어긋나게 하면
#   피크가 겹치지 않아 동시 실행 수를 크게 올릴 수 있다.
#     겹칠 때   8 x 22G = 176G  -> H200 143G 초과
#     어긋나면  8 x 5.9G = 47G  -> 여유
#   호스트 RAM 은 롤아웃당 32G 라 1T 서버에서 28 개까지 들어간다(제약 아님).
#   CPU 는 롤아웃당 1.35 코어뿐이라 192 코어 서버에서 전혀 제약이 아니다.
ROLLOUT_GAP=${ROLLOUT_GAP:-90}     # 1 단계 롤아웃 시작 간격(초). 피크 지속이 약 60s
MAXJOBS=${MAXJOBS:-8}              # 2 단계 동시 학습 런 수
MIN_FREE_GB=${MIN_FREE_GB:-40}
GAP=${GAP:-20}
PHASE=${PHASE:-all}                # all | rollout | train
SUITE=${SUITE:-libero_goal}

Q="${HERE}/logs/rollout_first.queue"
mkdir -p logs results
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${Q}"; }

NEED=$(( NTASK * (NTASK + 1) / 2 ))
entry_of() { [ "$1" = "v7" ] && echo L2_V7.py || echo L2_V6.py; }
free_gb() { df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9'; }

filled() {
    python3 - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists(): print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
}

# 체크포인트가 10 개 태스크까지 다 있는가
ckdir_of() { ls -d "${HERE}/outputs/CLARE_V6/$1"/*_ours 2>/dev/null | head -1; }
ckpt_ready() {
    # $1 = ckpt 폴더명, $2 = 런 태그(로그 이름)
    local d; d=$(ckdir_of "$1")
    [ -n "${d}" ] || return 1
    local n; n=$(ls "${d}" 2>/dev/null | grep -c '^task_[0-9]\+$')
    [ "${n}" -ge "${NTASK}" ] || return 1
    # ★ 디렉토리 수만 보면 "마지막 태스크 학습 중"인 런도 완비로 보인다.
    #   학습이 실제로 끝났을 때만 찍히는 마커를 같이 확인한다.
    grep -qa '\[B1\] 완료  wall' "${HERE}/logs/$2.log" 2>/dev/null
}

# ── 계획 ────────────────────────────────────────────────────────────────
ROLL=(); TRAIN=(); DONE=0
for VER in ${VERS}; do
  for SEED in ${SEEDS}; do
    for BETA in ${BETAS}; do
      TAG="clare_${VER}_${FORM}_b${BETA}_goal_seed${SEED}"
      CKNAME="${VER}_${FORM}_b${BETA}_seed${SEED}"
      f=$(filled "results/${TAG}")
      if [ "${f}" -ge "${NEED}" ]; then DONE=$((DONE+1)); continue; fi
      if ckpt_ready "${CKNAME}" "${TAG}"; then ROLL+=("${VER}|${SEED}|${BETA}|${f}")
      else TRAIN+=("${VER}|${SEED}|${BETA}|${f}"); fi
    done
  done
done

ts "════ 롤아웃 우선 — 완주 ${DONE} / 롤아웃만 ${#ROLL[@]} / 학습필요 ${#TRAIN[@]} ════"
ts "    PHASE=${PHASE}  ROLLOUT_PAR=${ROLLOUT_PAR}  MAXJOBS=${MAXJOBS}  GPU ${GPU}"

if [ -n "${DRY:-}" ]; then
    echo "완주 ${DONE} 런 — 건드리지 않음"
    echo "[1단계] 롤아웃만 (${#ROLL[@]}) — 체크포인트 완비, 재학습 불필요"
    for it in "${ROLL[@]:-}"; do [ -n "${it}" ] && IFS='|' read -r v s b f <<< "${it}" \
        && printf '    %s β%-5s seed%-5s %s/%s칸\n' "${v}" "${b}" "${s}" "${f}" "${NEED}"; done
    echo "[2단계] 학습+롤아웃 (${#TRAIN[@]}) — 스윕 순서 그대로"
    for it in "${TRAIN[@]:-}"; do [ -n "${it}" ] && IFS='|' read -r v s b f <<< "${it}" \
        && printf '    %s β%-5s seed%-5s %s/%s칸\n' "${v}" "${b}" "${s}" "${f}" "${NEED}"; done
    echo "여유 $(free_gb)G"
    exit 0
fi

report() {
    flock "${HERE}/logs/.v6report.lock" \
        python3 -u "${HERE}/CLARE_V6_report.py" > /dev/null 2>&1 || true
}

# ── 1단계: 롤아웃만 ─────────────────────────────────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "rollout" ]; then
    running=0
    for it in "${ROLL[@]:-}"; do
        [ -n "${it}" ] || continue
        IFS='|' read -r VER SEED BETA _f <<< "${it}"
        TAG="clare_${VER}_${FORM}_b${BETA}_goal_seed${SEED}"
        CKDIR=$(ckdir_of "${VER}_${FORM}_b${BETA}_seed${SEED}")
        while [ "${running}" -ge "${ROLLOUT_PAR}" ]; do
            wait -n 2>/dev/null || true; running=$((running-1))
        done
        ts "▶ [롤아웃] ${TAG}"
        (
            export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
                   MUJOCO_GL=${MUJOCO_GL:-egl} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
            python -u "${HERE}/CLARE_rollout_only.py" --ckpt_dir "${CKDIR}" \
                --out "${HERE}/results/${TAG}" --suite "${SUITE}" --rows all \
                --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
                --eval_seed "${EVAL_SEED}" >> "${HERE}/logs/${TAG}.log" 2>&1
            rc=$?
            n=$(filled "results/${TAG}")
            if [ "${rc}" -ne 0 ] || [ "${n}" -eq 0 ]; then
                ts "✗ [롤아웃] ${TAG} 실패 — ${n}/${NEED}칸, exit=${rc}  (logs/${TAG}.log 확인)"
            else
                ts "◀ [롤아웃] ${TAG}  ${n}/${NEED}칸"
            fi
            report
        ) &
        running=$((running+1)); sleep "${ROLLOUT_GAP}"
    done
    while [ "${running}" -gt 0 ]; do wait -n 2>/dev/null || true; running=$((running-1)); done
    ts "──── 1단계 끝 ────"
fi

# ── 2단계: 학습 + 롤아웃 (스윕 순서) ───────────────────────────────────
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "train" ]; then
    running=0
    for it in "${TRAIN[@]:-}"; do
        [ -n "${it}" ] || continue
        IFS='|' read -r VER SEED BETA _f <<< "${it}"
        TAG="clare_${VER}_${FORM}_b${BETA}_goal_seed${SEED}"
        while [ "${running}" -ge "${MAXJOBS}" ]; do
            wait -n 2>/dev/null || true; running=$((running-1))
        done
        FREE=$(free_gb)
        while [ "${FREE}" -lt "${MIN_FREE_GB}" ]; do
            if [ "${running}" -gt 0 ]; then
                ts "⏸ 여유 ${FREE}G < ${MIN_FREE_GB}G — 대기"
                wait -n 2>/dev/null || true; running=$((running-1)); FREE=$(free_gb)
            else
                ts "✗ 여유 ${FREE}G 부족, 진행 중인 런 없음 — 중단"; exit 1
            fi
        done
        ts "▶ [학습] ${TAG}  (여유 ${FREE}G)"
        (
            GPU="${GPU}" FORM="${FORM}" SEED="${SEED}" BETA="${BETA}" \
            ENTRY="$(entry_of "${VER}")" VER="${VER}" TAGSUF="_b${BETA}" \
            STEPS="${STEPS}" NW="${NW}" N_EVAL="${N_EVAL}" EVAL_SEED="${EVAL_SEED}" \
            MIN_FREE_GB="${MIN_FREE_GB}" ROLLOUT_SLOTS="${ROLLOUT_PAR}" \
            bash "${HERE}/CLARE_V6_queue.sh"
            ts "◀ [학습] ${TAG} exit=$?  ($(filled "results/${TAG}")/${NEED}칸)"
            report
        ) &
        running=$((running+1)); sleep "${GAP}"
    done
    while [ "${running}" -gt 0 ]; do wait -n 2>/dev/null || true; running=$((running-1)); done
    ts "──── 2단계 끝 ────"
fi

report
ts "════ 전부 종료 — 리포트: results/direction error aware anchor_V6_SR.txt ════"
