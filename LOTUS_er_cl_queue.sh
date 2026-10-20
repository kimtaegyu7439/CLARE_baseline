#!/usr/bin/env bash
# LOTUS 셋팅 ER(경험 재생) — GPU 하나가 시드 하나를 맡아 object -> goal -> 50 을 순차로.
#
#   SEED=7    GPU=0 bash LOTUS_er_cl_queue.sh
#   SEED=42   GPU=1 bash LOTUS_er_cl_queue.sh
#   SEED=1000 GPU=2 bash LOTUS_er_cl_queue.sh
#   DRY=1 SEED=7 GPU=0 bash LOTUS_er_cl_queue.sh      큐 내용만 출력
#
# ★ GPU 하나가 시드 하나를 전담한다. 스위트를 GPU 로 가르면 goal/object(10 task)가
#   먼저 끝나고 50(50 task)만 남아 카드가 논다. 시드로 가르면 셋 다 같은 일을 하므로
#   끝나는 시각이 비슷하다.
#
# ★ 한 카드에서 하나씩만 돈다. 롤아웃이 n_envs=20 벡터 env 를 띄워 VRAM 을 많이 쓴다.
#
# 런 하나가 끝날 때마다 results/LOTUS_ER_SR.txt 를 다시 만든다. 중간에 끊겨도
# 그때까지의 결과는 남는다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
# ★ env.sh 가 HF_LEROBOT_HOME 을 잡는다. 빠뜨리면 로컬 데이터셋을 못 찾고 새로 받는다.
source "${HERE}/bash/clare/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라 (예: GPU=0)}
SEED=${SEED:?SEED 를 지정해라 (예: SEED=7)}
SUITES=${SUITES:-"object goal 50"}
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}
BS_EVAL=${BS_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
BATCH=${BATCH:-32}          # 현재 24 + 버퍼 8 = seq-FT 와 같은 총 배치
REPLAY_BS=${REPLAY_BS:-8}
BUFFER_FRAC=${BUFFER_FRAC:-0.10}
NW=${NW:-6}
RNW=${RNW:-3}
QUEUE="${HERE}/logs/LOTUS_er_cl.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results

# 완주 검사 — metrics.json 의 R 에서 아래삼각형 칸이 다 찼는지 본다.
# 파일 존재만으로는 완주의 증거가 못 된다(스테이지마다 다시 쓰인다).
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

report() {
    python LOTUS_seq_cl_report.py \
        --prefix "lotus_er_" \
        --out "${HERE}/results/LOTUS_ER_SR.txt" \
        --title "LOTUS 셋팅 lifelong — ER (경험 재생 기준선)" \
        --loss_note "  손실      L = L_FM(현재 24 + 버퍼 8). 버퍼는 base + 지난 스테이지에서 태스크당 에피소드의 10%." \
        --tail_note "  · ER 은 기준선이다. seq-FT 와 총 배치(32)가 같고 그중 8 개가 과거에서 온다." \
        > /dev/null 2>&1
}

for SUITE in ${SUITES}; do
    TAG="lotus_er_${SUITE}_seed${SEED}"
    OUT="results/${TAG}"
    LOG="logs/${TAG}.log"
    if [ -n "${DRY:-}" ]; then
        echo "GPU ${GPU}  ${SUITE}  seed ${SEED}  배치 ${BATCH}(재생 ${REPLAY_BS})  -> ${OUT}"; continue
    fi
    if [ "$(is_done "${OUT}")" = "1" ]; then
        echo "[$(date '+%F %T')] skip ${TAG} — 이미 완주" | tee -a "${QUEUE}"; continue
    fi
    echo "[$(date '+%F %T')] ▶ ${TAG}  GPU ${GPU}  stage당 ${STEPS} step  칸당 ${N_EVAL} rollout" \
        | tee -a "${QUEUE}"
    echo "════ $(date '+%F %T') 시작 ════" >> "${LOG}"
    python -u L2_er_cl.py \
        --suite "${SUITE}" --steps "${STEPS}" \
        --batch_size "${BATCH}" --replay_batch_size "${REPLAY_BS}" \
        --buffer_frac "${BUFFER_FRAC}" \
        --num_workers "${NW}" --replay_num_workers "${RNW}" \
        --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${BS_EVAL}" \
        --out "${OUT}" >> "${LOG}" 2>&1
    rc=$?   # ★ echo 안의 $(date) 가 $? 를 덮어쓰므로 바로 받아 둔다
    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}" | tee -a "${QUEUE}"
    [ "${rc}" -ne 0 ] && echo "   -> ${LOG} 확인" | tee -a "${QUEUE}"
    report && echo "   SR 기록 -> results/LOTUS_ER_SR.txt" | tee -a "${QUEUE}"
done
echo "[$(date '+%F %T')] seed ${SEED} 전체 종료 (GPU ${GPU})" | tee -a "${QUEUE}"
