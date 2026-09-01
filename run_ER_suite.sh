#!/usr/bin/env bash
#
# ER — 임의 LIBERO 스위트 10 태스크 학습 + SR 표 생성.
#
# run_ER_task0123.sh 의 스위트 일반화판. 원본은 libero_spatial 전용이라 그대로 둔다.
# 학습은 bash/er/ER_suite.sh, SR 프로브는 E0.py --reprobe 로 K1 과 같은 자로 잰다
# (칸당 20 롤아웃, start_seed 42, hold-out 5).
#
# 사용법
#   bash run_ER_suite.sh <SUITE> <GPU> [NUM_TASKS]
#   bash run_ER_suite.sh libero_goal 1 10
#   PROBE_ONLY=1 bash run_ER_suite.sh libero_goal 1 10
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

SUITE=${1:?SUITE}; GPU=${2:-0}; NUM_TASKS=${3:-10}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}

# K1 을 4 병렬로 돌렸을 때 32 코어에서 스레싱이 났다(통계 패스 116s -> 983s).
# ER 도 3 병렬이므로 같은 제한을 건다. 알고리즘에는 영향이 없다.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

SEED=${SEED:-42}
OUT_ROOT=${OUT_ROOT:-./outputs/ER/${SUITE}/seed${SEED}}
RES_DIR=${RES_DIR:-${HERE}/results/ER_${SUITE}_10task}
RESULTS=${RESULTS:-${RES_DIR}/er_results.jsonl}
LOG=${RES_DIR}/run.log

HOLDOUT_EP=${HOLDOUT_EP:-5}
PROBE_BATCHES=${PROBE_BATCHES:-16}
PROBE_N_EP=${PROBE_N_EP:-20}
PROBE_EVAL_BS=${PROBE_EVAL_BS:-20}

PYTHON=${PYTHON:-python}
DATASET_PREFIX="continuallearning/${SUITE}_image_task_"
ENV_TASK_PREFIX=$(${PYTHON} -c "print('_'.join(w.capitalize() for w in '${SUITE}'.split('_')) + '_Task_')")
E0_PY=./lerobot_lsy/src/lerobot/scripts/E0.py

mkdir -p "${RES_DIR}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
log "ER ${SUITE} 10task  gpu=${GPU}  out=${OUT_ROOT}"

# ── 1. 학습 ──────────────────────────────────────────────────────────────────
if [ "${PROBE_ONLY:-0}" != "1" ]; then
    log "학습 시작 — bash/er/ER_suite.sh (SUITE=${SUITE})"
    SUITE="${SUITE}" SEED="${SEED}" NUM_TASKS="${NUM_TASKS}" OUT_ROOT="${OUT_ROOT}" \
        bash bash/er/ER_suite.sh >>"${LOG}" 2>&1
    rc=$?
    log "학습 종료 rc=${rc}"
    [ $rc -ne 0 ] && { log "학습 실패 — 중단"; exit 1; }
fi
[ "${TRAIN_ONLY:-0}" = "1" ] && { log "TRAIN_ONLY — 프로브 생략"; exit 0; }

# ── 2. 스테이지마다 SR 프로브 ────────────────────────────────────────────────
[ -s "${RESULTS}" ] && mv "${RESULTS}" "${RESULTS}.bak" && log "이전 결과 -> ${RESULTS}.bak"

for k in $(seq 0 $((NUM_TASKS - 1))); do
    ckpt="${OUT_ROOT}/task_${k}/checkpoints/last/pretrained_model"
    if [ ! -d "${ckpt}" ]; then log "SKIP task ${k}: 체크포인트 없음"; continue; fi
    log "── probe stage ${k}  (태스크 0..${k}, 칸당 ${PROBE_N_EP} 롤아웃)"
    "${PYTHON}" "${E0_PY}" \
        --seed="${SEED}" --job_name="ER_${SUITE}_probe_task_${k}" \
        --output_dir="${OUT_ROOT}/task_${k}" \
        --dataset.repo_id="${DATASET_PREFIX}${k}" \
        --policy.path="${ckpt}" --policy.push_to_hub=false \
        --reprobe=true --eval_freq=0 \
        --env.type=libero --env.benchmark="${SUITE}" --env.task="${ENV_TASK_PREFIX}${k}" \
        --ewc_lambda=0 --run_tag="er" --current_task="${k}" \
        --task_ids="$(seq -s, 0 "${k}")" \
        --dataset_prefix="${DATASET_PREFIX}" --env_task_prefix="${ENV_TASK_PREFIX}" \
        --results_path="${RESULTS}" --holdout_episodes="${HOLDOUT_EP}" \
        --probe_batches="${PROBE_BATCHES}" --probe_sr=true \
        --probe_n_episodes="${PROBE_N_EP}" --probe_eval_batch_size="${PROBE_EVAL_BS}" \
        --wandb.enable=false >>"${LOG}" 2>&1 || log "FAILED probe stage ${k}"
done

# ── 3. 표 ────────────────────────────────────────────────────────────────────
"${PYTHON}" - "${RESULTS}" "${NUM_TASKS}" "${RES_DIR}" "${SUITE}" <<'PYEOF' 2>&1 | tee -a "${LOG}"
import json, sys
from pathlib import Path
res, n, out_dir, suite = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]

cells = {}
if res.exists():
    for line in res.read_text().splitlines():
        r = json.loads(line)
        if r.get("run_tag") == "er" and r.get("sr") is not None:
            cells[(r["stage"], r["probe_task"])] = float(r["sr"])

# B1 계열과 같은 형식의 sr_matrix.csv 도 남긴다 (K_report / B_mod 정리용)
with (out_dir / "sr_matrix.csv").open("w") as f:
    f.write(f"# task_order: {','.join(str(t) for t in range(n))}\n")
    f.write("after_stage," + ",".join(f"stage{t}(task{t})" for t in range(n)) + "\n")
    for k in range(n):
        f.write(f"{k}," + ",".join(
            f"{cells[(k,t)]:.1f}" if (k, t) in cells else "" for t in range(n)) + "\n")

last = [cells.get((n - 1, t)) for t in range(n)]
diag = [cells.get((t, t)) for t in range(n)]
L = ["=" * 70, f"ER  {suite}  태스크 0..{n-1}   seed 42, 칸당 20 롤아웃", "=" * 70, "",
     "5000 steps/task, 45 에피소드(뒤 5개 hold-out), 배치 32 = 현재 24 + 버퍼 8,",
     "과거 태스크당 버퍼 5 에피소드.  K1 과 같은 프로토콜이다.", ""]
if all(v is not None for v in last + diag):
    avg = sum(last) / n
    bwt = sum(last[t] - diag[t] for t in range(n - 1)) / (n - 1)
    L += [f"AvgSR {avg:.1f}   BWT {bwt:+.1f}   습득 {sum(diag)/n:.1f}", ""]
else:
    L += [f"미완 ({len(cells)}/{n*(n+1)//2} 칸)", ""]
L += ["행 = 스테이지, 열 = 태스크",
      "after\\task " + "".join(f"{t:>6d}" for t in range(n))]
for k in range(n):
    L.append(f"{k:>10d} " + "".join(
        f"{cells[(k,t)]:6.0f}" if (k, t) in cells else "     ." for t in range(n)))
txt = "\n".join(L)
(out_dir / "SR.txt").write_text(txt)
print(txt)
print(f"\nsaved -> {out_dir/'SR.txt'}, {out_dir/'sr_matrix.csv'}")
PYEOF

log "완료"
