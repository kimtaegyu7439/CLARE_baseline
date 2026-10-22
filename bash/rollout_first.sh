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

# ★ 데이터셋/모델 경로. 이게 없으면 LeRobotDatasetMetadata 가 HF_LEROBOT_HOME
#   아래에서 로컬 메타데이터를 못 찾고(기본 캐시를 본다) 허브로 가는데,
#   HF_HUB_OFFLINE=1 이라 OfflineModeIsEnabled 로 죽는다.
#   2 단계는 CLARE_V6_queue.sh 가 이걸 source 해서 멀쩡했고, 1 단계만 빠져 있었다.
# shellcheck disable=SC1091
source "${HERE}/bash/clare/env.sh" > /dev/null 2>&1 || true

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
ROLLOUT_RSS_GB=${ROLLOUT_RSS_GB:-26}   # 롤아웃 1 개의 호스트 RAM(실측 24G + 여유)

# ★ 진짜 상한은 **컨테이너 메모리 제한**이지 free -g 가 아니다.
#   쿠버네티스 파드에서 free 는 호스트 전체(예: 1007G)를 보여주지만 cgroup
#   제한은 그보다 훨씬 작다. 롤아웃 하나가 RSS 24G 라, 제한을 모르고 8 개를
#   띄우면 OOM 킬러가 exit=137 로 죽인다(실제로 5/8 이 그렇게 날아갔다).
#   제한을 읽어 ROLLOUT_PAR 을 자동으로 깎는다. 못 읽으면 건드리지 않는다.
cgroup_limit_gb() {
    local v=""
    for f in /sys/fs/cgroup/memory.max \
             /sys/fs/cgroup/memory/memory.limit_in_bytes; do
        [ -r "${f}" ] && v=$(cat "${f}" 2>/dev/null) && break
    done
    case "${v}" in
        ''|max|*[!0-9]*) echo ""; return;;
    esac
    # 호스트 전체와 같으면 제한이 없는 것으로 본다
    local tot; tot=$(awk '/MemTotal/{print $2*1024}' /proc/meminfo 2>/dev/null)
    [ -n "${tot}" ] && [ "${v}" -ge "${tot}" ] && { echo ""; return; }
    echo $(( v / 1073741824 ))
}
TRAIN_RSS_GB=${TRAIN_RSS_GB:-12}       # 학습 런 1 개의 호스트 RAM(워커 포함 추정)
MEMGB=$(cgroup_limit_gb)
if [ -n "${MEMGB}" ]; then
    BUDGET=$(( MEMGB * 85 / 100 ))     # 시스템 몫 15% 는 남긴다

    # ── 1 단계: 롤아웃만 도므로 롤아웃 RSS 로만 나눈다 ──────────────────
    CAP=$(( BUDGET / ROLLOUT_RSS_GB )); [ "${CAP}" -lt 1 ] && CAP=1
    if [ "${ROLLOUT_PAR}" -gt "${CAP}" ]; then
        echo "[rollout_first] 메모리 제한 ${MEMGB}G / 롤아웃당 ${ROLLOUT_RSS_GB}G" \
             "-> ROLLOUT_PAR ${ROLLOUT_PAR} 는 OOM 위험. ${CAP} 로 낮춘다." >&2
        ROLLOUT_PAR="${CAP}"
    fi

    # ── 2 단계: 학습 (M-S) 개 + 롤아웃 S 개가 **같이** 떠 있을 수 있다 ──
    #   런이 학습을 끝내면 바로 롤아웃으로 넘어가므로 둘이 공존한다.
    #   최악 = (M-S)*TRAIN + S*ROLL <= BUDGET 를 만족하는 최대 M 을 찾는다.
    #   (이걸 안 보고 M=8 로 뒀다가 롤아웃 5/8 이 OOM 으로 날아갔다.)
    S="${ROLLOUT_PAR}"
    while [ "${S}" -gt 1 ] && [ $(( S * ROLLOUT_RSS_GB )) -gt "${BUDGET}" ]; do
        S=$(( S - 1 ))
    done
    MCAP="${S}"
    while [ $(( (MCAP + 1 - S) * TRAIN_RSS_GB + S * ROLLOUT_RSS_GB )) -le "${BUDGET}" ]; do
        MCAP=$(( MCAP + 1 ))
        [ "${MCAP}" -ge 64 ] && break
    done
    [ "${MCAP}" -lt 1 ] && MCAP=1
    if [ "${MAXJOBS}" -gt "${MCAP}" ]; then
        echo "[rollout_first] 2 단계는 학습과 롤아웃이 공존한다." \
             "최악 $(( (MAXJOBS - S) * TRAIN_RSS_GB + S * ROLLOUT_RSS_GB ))G > ${BUDGET}G" \
             "-> MAXJOBS ${MAXJOBS} 를 ${MCAP} 로 낮춘다." >&2
        MAXJOBS="${MCAP}"
    fi
    echo "[rollout_first] 메모리 ${MEMGB}G -> 예산 ${BUDGET}G," \
         "ROLLOUT_PAR=${ROLLOUT_PAR}, MAXJOBS=${MAXJOBS}" \
         "(롤아웃 ${ROLLOUT_RSS_GB}G/개, 학습 ${TRAIN_RSS_GB}G/개)" >&2
fi
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
            # env.sh 가 export 한 값들(HF_LEROBOT_HOME, HF_HUB_CACHE, PRETRAIN_PATH)은
            # 상속되지만, 서브셸에서 한 번 더 확인해 둔다.
            [ -n "${HF_LEROBOT_HOME:-}" ] || source "${HERE}/bash/clare/env.sh" > /dev/null 2>&1
            python -u "${HERE}/CLARE_rollout_only.py" --ckpt_dir "${CKDIR}" \
                --out "${HERE}/results/${TAG}" --suite "${SUITE}" --rows all \
                --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
                --eval_seed "${EVAL_SEED}" >> "${HERE}/logs/${TAG}.log" 2>&1
            rc=$?
            n=$(filled "results/${TAG}")
            if [ "${rc}" -ne 0 ] || [ "${n}" -eq 0 ]; then
                if [ "${rc}" -eq 137 ]; then
                    ts "✗ [롤아웃] ${TAG} OOM 으로 강제종료(exit=137) — 동시 실행 수를 줄여라"
                    ts "   롤아웃 1 개가 RSS 약 ${ROLLOUT_RSS_GB}G 다. 현재 ROLLOUT_PAR=${ROLLOUT_PAR}."
                else
                    ts "✗ [롤아웃] ${TAG} 실패 — ${n}/${NEED}칸, exit=${rc}  (logs/${TAG}.log 확인)"
                fi
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
