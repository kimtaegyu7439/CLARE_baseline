#!/usr/bin/env bash
# LOTUS 셋팅 순차 CL (seq-FT 기준선) — goal, 50 을 GPU 3 에서 하나씩.
#
#   bash LOTUS_seq_cl_queue.sh                 goal -> 50, 각 seed 7/42/1000
#   SUITES="goal" bash LOTUS_seq_cl_queue.sh   한 스위트만
#   SEEDS=42 GPU=0 bash LOTUS_seq_cl_queue.sh
#   DRY=1 bash LOTUS_seq_cl_queue.sh           큐 내용만 출력
#
# ★ 한 번에 하나씩 돈다. 롤아웃 칸마다 n_envs=20 짜리 벡터 env 를 띄우므로 같은 카드에
#   둘을 올리면 롤아웃 구간에서 VRAM 이 겹친다.
#
# ★ object 는 이미 seed 7/42/1000 이 완주해 있어 기본 SUITES 에서 뺐다. 다시 돌리려면
#   SUITES="object goal 50" 로 주면 되고, 완주한 런은 아래 완주 검사에서 자동으로 걸러진다.
#
# 매 런이 끝날 때마다 results/LOTUS_Seq_CL_SR.txt 를 다시 만든다. 중간에 끊겨도
# 그때까지의 결과는 그 파일에 남는다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
# ★ env.sh 가 HF_LEROBOT_HOME 을 잡는다. 빠뜨리면 로컬 데이터셋을 못 찾고
#   허깅페이스에서 새로 내려받는다(실제로 그렇게 40GB 를 받은 적이 있다).
source "${HERE}/bash/clare/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}     # 로컬에 없으면 받지 말고 바로 실패

GPU=${GPU:-3}
SUITES=${SUITES:-"goal 50"}
SEEDS=${SEEDS:-"7 42 1000"}
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}
BS_EVAL=${BS_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
NW=${NW:-6}
QUEUE="${HERE}/logs/LOTUS_seq_cl.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results

# 완주 검사 — metrics.json 의 R 에서 아래삼각형 칸이 다 찼는지 본다. 파일 존재만으로는
# 완주의 증거가 못 된다(스테이지마다 다시 쓰인다).
is_done() {
    python - "$1" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "metrics.json"
if not p.exists():
    print("0"); raise SystemExit
try:
    d = json.loads(p.read_text())
except Exception:
    print("0"); raise SystemExit
R = d.get("R") or []
la = d.get("learned_at") or []
if not R or not la:
    print("0"); raise SystemExit
need = sum(1 for k in range(len(R)) for t in range(len(la)) if la[t] <= k)
have = sum(1 for k in range(len(R)) for t in range(len(R[k])) if R[k][t] is not None)
print("1" if have >= need and need > 0 else "0")
PY
}

for SUITE in ${SUITES}; do
    for SEED in ${SEEDS}; do
        TAG="lotus_seq_${SUITE}_seed${SEED}"
        OUT="results/${TAG}"
        LOG="logs/${TAG}.log"
        if [ -n "${DRY:-}" ]; then
            echo "GPU ${GPU}  ${SUITE}  seed ${SEED}  rollout ${N_EVAL}  -> ${OUT}"; continue
        fi
        if [ "$(is_done "${OUT}")" = "1" ]; then
            echo "[$(date '+%F %T')] skip ${TAG} — 이미 완주" | tee -a "${QUEUE}"; continue
        fi
        echo "[$(date '+%F %T')] ▶ ${TAG}  GPU ${GPU}  stage당 ${STEPS} step  칸당 ${N_EVAL} rollout" \
            | tee -a "${QUEUE}"
        echo "════ $(date '+%F %T') 시작 ════" >> "${LOG}"
        python -u L2_seq_cl.py \
            --suite "${SUITE}" --steps "${STEPS}" \
            --batch_size 32 --num_workers "${NW}" \
            --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
            --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
            --out "${OUT}" >> "${LOG}" 2>&1
        rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다
        echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}" | tee -a "${QUEUE}"
        [ "${rc}" -ne 0 ] && echo "   -> ${LOG} 확인" | tee -a "${QUEUE}"
        # 런 하나 끝날 때마다 SR 파일 갱신 — 중간에 끊겨도 여기까지는 남는다
        python LOTUS_seq_cl_report.py > /dev/null 2>&1 \
            && echo "   SR 기록 -> results/LOTUS_Seq_CL_SR.txt" | tee -a "${QUEUE}"
    done
done
echo "[$(date '+%F %T')] 전체 종료" | tee -a "${QUEUE}"
python LOTUS_seq_cl_report.py | tail -30
