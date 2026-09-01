#!/usr/bin/env bash
#
# ER libero_object용 리플레이 버퍼 데이터셋을 만든다.
#
# ER_libero_object.sh는 스테이지 k(≥1)에서 "그때까지 배운 태스크 0..k-1"을 담은 버퍼를
# 재생한다. spatial/goal과 같은 규칙으로 **태스크당 5에피소드**를 뽑아 이어 붙인다:
#
#   libero_object_image_task_0_er_new   태스크 0            ->  5 에피소드
#   libero_object_image_task_0_1_new    태스크 0..1         -> 10
#   libero_object_image_task_0_2_new    태스크 0..2         -> 15
#   ...
#   libero_object_image_task_0_8_new    태스크 0..8         -> 45
#
# spatial 쪽 기존 버퍼로 규칙을 확인했다(_er_new 5 eps, _0_1_new 10 eps, _0_8_new 45 eps).
# object 버퍼는 로컬에도 HF에도 없어서 새로 만든다. 결과는 $HF_LEROBOT_HOME 아래에
# 로컬로만 저장되고 허브에 올리지 않는다.
#
# 사용법
#   bash bash/er/make_object_er_buffers.sh
#
# 이미 있는 버퍼는 건너뛴다. 중단했다 다시 돌려도 안전하다.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED=${SEED:-42}
EPS_PER_TASK=${EPS_PER_TASK:-5}
HF_ORG=${HF_ORG:-continuallearning}
PYTHON=${PYTHON:-python}
MAKE_PY=./lerobot_lsy/src/lerobot/scripts/util/create_er_dataset.py

n_made=0; n_skip=0; n_fail=0
for k in $(seq 0 8); do
    if [ "${k}" -eq 0 ]; then
        out="${HF_ORG}/libero_object_image_task_0_er_new"
    else
        out="${HF_ORG}/libero_object_image_task_0_${k}_new"
    fi

    if [ -d "${HF_LEROBOT_HOME}/${out}" ]; then
        echo "[buffer] SKIP ${out} (이미 있음)"
        n_skip=$((n_skip + 1)); continue
    fi

    # 태스크 0..k를 모두 넣는다. 스테이지 k+1 학습이 이 버퍼를 쓴다.
    repos=""
    for t in $(seq 0 "${k}"); do
        repos="${repos}${repos:+,}${HF_ORG}/libero_object_image_task_${t}"
    done

    echo "── ${out}  <-  태스크 0..${k} 에서 각 ${EPS_PER_TASK}개"
    "${PYTHON}" "${MAKE_PY}" \
        --repo_ids="${repos}" \
        --num_episodes="${EPS_PER_TASK}" \
        --merged_repo_id="${out}" \
        --seed="${SEED}" \
        || { echo "[buffer] FAILED ${out}"; n_fail=$((n_fail + 1)); continue; }
    n_made=$((n_made + 1))
done

echo ""
echo "[buffer] 생성 ${n_made}  건너뜀 ${n_skip}  실패 ${n_fail}"
echo "[buffer] 검증:"
for k in $(seq 0 8); do
    if [ "${k}" -eq 0 ]; then p="${HF_LEROBOT_HOME}/${HF_ORG}/libero_object_image_task_0_er_new"
    else p="${HF_LEROBOT_HOME}/${HF_ORG}/libero_object_image_task_0_${k}_new"; fi
    want=$(( (k + 1) * EPS_PER_TASK ))
    got=$("${PYTHON}" -c "import json;print(json.load(open('${p}/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "-")
    flag=$([ "${got}" = "${want}" ] && echo OK || echo "MISMATCH(기대 ${want})")
    echo "   $(basename ${p}) : ${got} eps  ${flag}"
done
