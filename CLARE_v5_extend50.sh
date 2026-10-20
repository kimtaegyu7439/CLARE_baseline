#!/usr/bin/env bash
# V5 세 팔의 SR 을 칸당 20 -> 50 rollout 으로 늘린다.
#
#   bash CLARE_v5_extend50.sh            GPU 0~3 에 균형 배분해 띄운다
#   DRY=1 bash CLARE_v5_extend50.sh      배분만 출력
#
# 어떻게 50 을 만드나
#   기존 20 회는 eval_seed=7 로 **시드 7~26** 을 썼다
#   (eval.py:333  seeds = range(start_seed + batch*n_envs, ...)).
#   여기서는 eval_seed=27 로 **시드 27~56** 의 30 회를 새로 돌려
#   별도 폴더(..._extra30)에 적고, CLARE_v5_merge50.py 가 가중평균으로 합친다:
#
#       SR_50 = (SR_20 x 20 + SR_30 x 30) / 50
#
#   ★ 기존 20 회 결과를 덮어쓰지 않는다. 두 파일이 다 남아야 합치기를 다시 할
#     수 있고, 20 회 기준 수치도 그대로 인용할 수 있다.
#
# ★★ GPU 하나당 **프로세스 하나**. 한 롤아웃이 n_envs=30 짜리 MuJoCo 벡터 env 를
#    띄우므로 같은 카드에 둘 이상 겹치면 24GB 를 넘긴다(실제로 21GB 를 찍었다).
#    각 GPU 는 자기 작업 목록을 **순차로** 처리한다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi

ARMS=${ARMS:-"v5_goal_c0.01 v5_goal_c0.05 v5_goal_c0.1"}
N_EXTRA=${N_EXTRA:-30}
SEED_EXTRA=${SEED_EXTRA:-27}        # 기존이 7~26 을 썼으므로 27 부터
NGPU=${NGPU:-4}
NTASK=${NTASK:-10}
mkdir -p logs

# 행 단위 작업을 칸 수 기준으로 가장 한가한 GPU 에 붙인다(그리디 균형).
PLAN=$(python3 - "${ARMS}" "${NGPU}" "${NTASK}" <<'PY'
import sys
arms = sys.argv[1].split(); ngpu = int(sys.argv[2]); ntask = int(sys.argv[3])
jobs = [(a, k, k + 1) for a in arms for k in range(ntask)]   # (팔, 행, 칸수)
jobs.sort(key=lambda j: -j[2])                               # 큰 것부터
load = [0] * ngpu; plan = [[] for _ in range(ngpu)]
for a, k, n in jobs:
    g = load.index(min(load)); load[g] += n; plan[g].append((a, k))
for g in range(ngpu):
    rows = {}
    for a, k in plan[g]:
        rows.setdefault(a, []).append(k)
    for a, ks in rows.items():
        print(f"{g}\t{a}\t{','.join(str(x) for x in sorted(ks))}\t{sum(k+1 for k in ks)}")
PY
)

echo "=== 배분 (GPU / 팔 / 행 / 칸수)"
echo "${PLAN}" | awk -F'\t' '{printf "  GPU %s  %-16s rows %-12s %s칸\n", $1, $2, $3, $4}'
echo "${PLAN}" | awk -F'\t' '{s[$1]+=$4} END {for (g in s) printf "  -> GPU %s 합계 %s칸\n", g, s[g]}' | sort

[ -n "${DRY:-}" ] && exit 0

for G in $(seq 0 $((NGPU - 1))); do
    JOBS=$(echo "${PLAN}" | awk -F'\t' -v g="${G}" '$1==g {print $2"\t"$3}')
    [ -z "${JOBS}" ] && continue
    QLOG="logs/extend50_queue_gpu${G}.log"
    (
        echo "[$(date '+%F %T')] GPU ${G} 큐 시작"
        while IFS=$'\t' read -r ARM ROWS; do
            [ -z "${ARM}" ] && continue
            CK=$(ls -d "outputs/CLARE_MAR/${ARM}_seed7"/*_ours 2>/dev/null | head -1)
            if [ -z "${CK}" ]; then echo "  ★ ${ARM} 체크포인트 없음 — 건너뜀"; continue; fi
            echo "[$(date '+%F %T')] ▶ ${ARM} rows ${ROWS}"
            source "${HERE}/bash/clare/env.sh"
            export HF_HUB_OFFLINE=1
            CUDA_VISIBLE_DEVICES="${G}" MUJOCO_EGL_DEVICE_ID="${G}" MUJOCO_GL=egl \
            OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
            python -u CLARE_rollout_only.py --ckpt_dir "${CK}" \
                --out "results/clare_${ARM}_seed7_extra30" \
                --suite libero_goal --rows "${ROWS}" \
                --eval_episodes "${N_EXTRA}" --eval_batch_size "${N_EXTRA}" \
                --eval_seed "${SEED_EXTRA}" >> "logs/extend50_run_gpu${G}.log" 2>&1
            rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다
            echo "[$(date '+%F %T')] ◀ ${ARM} exit=${rc}"
        done <<< "${JOBS}"
        echo "[$(date '+%F %T')] GPU ${G} 전체 종료"
    ) >> "${QLOG}" 2>&1 &
    echo "  GPU ${G} 큐 시작 (pid $!)"
done
echo "[$(date '+%F %T')] ${NGPU} 개 큐 띄움 — GPU 당 프로세스 1 개"
