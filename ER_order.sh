#!/usr/bin/env bash
#
# ER 을 임의의 태스크 순서로 학습 + SR 표 생성.
#
# 기존 bash/er/ER_task0123.sh 는 순서가 0,1,..,K-1 로 고정이라 B9(순서 실험)와
# 대조할 수 없다. 이 스크립트는 순서만 바꾸고 나머지는 전부 같게 유지한다.
#
#   * 버퍼는 stage k 에서 order[0..k-1] 을 모아 만든다(순서를 따라간다).
#   * stage 0 은 사전학습에서 새로 학습한다. 원본은 E0 lam0/task_0 을 심링크했지만
#     그건 "task 0 이 첫 번째"일 때만 맞는다.
#   * 프로브는 실제 태스크 번호로 기록한다 -> 표가 곧 stage x task 다.
#
# 기존 outputs/ results/ 는 건드리지 않는다.
#
# 사용법:  ORDER=3,2,1,0 bash ER_order.sh 3
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-3}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}

ORDER=${ORDER:-3,2,1,0}
IFS=',' read -ra ORD <<< "${ORDER}"
NUM_TASKS=${#ORD[@]}
TAG=$(echo "${ORDER}" | tr -d ',')

SEED=${SEED:-42}
STEPS=${STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-24}          # 현재 태스크
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-8}
BATCH_SIZE_FIRST=${BATCH_SIZE_FIRST:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
REPLAY_NUM_WORKERS=${REPLAY_NUM_WORKERS:-4}
HOLDOUT_EP=${HOLDOUT_EP:-5}
BUFFER_EP=${BUFFER_EP:-5}
TOTAL_EP=${TOTAL_EP:-50}
LOG_FREQ=${LOG_FREQ:-100}

OUT_ROOT=${OUT_ROOT:-./outputs/ER_order/libero_spatial/seed${SEED}_${TAG}}
RES_DIR=${RES_DIR:-${HERE}/results/ER_order_${TAG}}
RESULTS=${RES_DIR}/er_results.jsonl
LOG=${RES_DIR}/run.log
mkdir -p "${OUT_ROOT}" "${RES_DIR}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_
ER_PY=./lerobot_lsy/src/lerobot/scripts/er.py
TRAIN_PY=./lerobot_lsy/src/lerobot/scripts/train.py
BUFFER_PY=./lerobot_lsy/src/lerobot/scripts/util/create_er_dataset.py
E0_PY=./lerobot_lsy/src/lerobot/scripts/E0.py
PYTHON=${PYTHON:-python}
EPISODES="[$(${PYTHON} -c "print(','.join(str(i) for i in range(${TOTAL_EP}-${HOLDOUT_EP})))")]"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
log "ER order=${ORDER} gpu=${GPU} out=${OUT_ROOT}"

# ── 버퍼: order[0..k-1] 을 합친다 ─────────────────────────────────────────────
build_buffer() {   # build_buffer <k> -> repo_id
    local k=$1 repo_id="er_buffer/libero_spatial_seed${SEED}_ep${BUFFER_EP}_ord${TAG}_k$((k - 1))"
    local dir="${HF_LEROBOT_HOME}/${repo_id}"
    if [ -f "${dir}/meta/info.json" ]; then echo "${repo_id}"; return 0; fi
    local ids="" j
    for ((j = 0; j < k; j++)); do
        [ -n "${ids}" ] && ids="${ids},"
        ids="${ids}${DATASET_PREFIX}${ORD[$j]}"
    done
    echo "[ER] building buffer ${repo_id} <- ${ids}" >&2
    rm -rf "${dir}"
    ${PYTHON} "${BUFFER_PY}" --repo_ids="${ids}" --num_episodes="${BUFFER_EP}" \
        --merged_repo_id="${repo_id}" --holdout_episodes="${HOLDOUT_EP}" \
        --seed="${SEED}" >&2 || return 1
    echo "${repo_id}"
}

# ── 학습 ─────────────────────────────────────────────────────────────────────
if [ "${PROBE_ONLY:-0}" != "1" ]; then
for ((k = 0; k < NUM_TASKS; k++)); do
    t=${ORD[$k]}
    out_dir="${OUT_ROOT}/task_${k}"; prev_dir="${OUT_ROOT}/task_$((k - 1))"
    if [ -f "${out_dir}/.done" ]; then log "skip stage ${k} (task ${t})"; continue; fi
    [ -d "${out_dir}" ] && { log "미완 스테이지 제거: ${out_dir}"; rm -rf "${out_dir}"; }

    if [ "${k}" -eq 0 ]; then
        log "══ stage 0 (task ${t})  버퍼 없음, init=${PRETRAIN_PATH}"
        "${PYTHON}" "${TRAIN_PY}" --seed="${SEED}" --job_name="ERord_s0" \
            --output_dir="${out_dir}" --dataset.repo_id="${DATASET_PREFIX}${t}" \
            --dataset.episodes="${EPISODES}" --policy.path="${PRETRAIN_PATH}" \
            --policy.push_to_hub=false --batch_size="${BATCH_SIZE_FIRST}" \
            --num_workers="${NUM_WORKERS}" --steps="${STEPS}" --save_freq="${STEPS}" \
            --log_freq="${LOG_FREQ}" --eval_freq=0 --env.type=libero \
            --env.benchmark=libero_spatial --env.task="${ENV_TASK_PREFIX}${t}" \
            --wandb.enable=false >>"${LOG}" 2>&1 \
            || { log "FAILED stage 0"; exit 1; }
    else
        buf=$(build_buffer "${k}") || { log "FAILED buffer stage ${k}"; exit 1; }
        log "══ stage ${k} (task ${t})  replay=${buf}  init=task_$((k-1))"
        "${PYTHON}" "${ER_PY}" --seed="${SEED}" --job_name="ERord_s${k}" \
            --output_dir="${out_dir}" --dataset.repo_id="${DATASET_PREFIX}${t}" \
            --dataset.episodes="${EPISODES}" --replay_dataset.repo_id="${buf}" \
            --policy.path="${prev_dir}/checkpoints/last/pretrained_model" \
            --policy.push_to_hub=false --batch_size="${BATCH_SIZE}" \
            --num_workers="${NUM_WORKERS}" --replay_batch_size="${REPLAY_BATCH_SIZE}" \
            --replay_num_workers="${REPLAY_NUM_WORKERS}" --steps="${STEPS}" \
            --log_freq="${LOG_FREQ}" --save_freq="${STEPS}" --eval_freq=0 \
            --env.type=libero --env.benchmark=libero_spatial \
            --env.task="${ENV_TASK_PREFIX}${t}" --wandb.enable=false >>"${LOG}" 2>&1 \
            || { log "FAILED stage ${k}"; exit 1; }
    fi
    touch "${out_dir}/.done"
done
fi
[ "${TRAIN_ONLY:-0}" = "1" ] && { log "TRAIN_ONLY"; exit 0; }

# ── 프로브: stage k 에서 order[0..k] 를 잰다. 기록은 실제 태스크 번호로. ──────
[ -s "${RESULTS}" ] && mv "${RESULTS}" "${RESULTS}.bak"
for ((k = 0; k < NUM_TASKS; k++)); do
    ck="${OUT_ROOT}/task_${k}/checkpoints/last/pretrained_model"
    [ -d "${ck}" ] || { log "SKIP probe stage ${k}: 체크포인트 없음"; continue; }
    ids=""; for ((j = 0; j <= k; j++)); do [ -n "${ids}" ] && ids="${ids},"; ids="${ids}${ORD[$j]}"; done
    log "── probe stage ${k}  대상 task ${ids}"
    "${PYTHON}" "${E0_PY}" --seed="${SEED}" --job_name="ERord_probe_${k}" \
        --output_dir="${OUT_ROOT}/probe_${k}" \
        --dataset.repo_id="${DATASET_PREFIX}${ORD[$k]}" --policy.path="${ck}" \
        --policy.push_to_hub=false --reprobe=true --eval_freq=0 \
        --env.type=libero --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${ORD[$k]}" --ewc_lambda=0 --run_tag="er" \
        --current_task="${k}" --task_ids="${ids}" \
        --dataset_prefix="${DATASET_PREFIX}" --env_task_prefix="${ENV_TASK_PREFIX}" \
        --results_path="${RESULTS}" --holdout_episodes="${HOLDOUT_EP}" \
        --probe_batches=16 --probe_sr=true --probe_n_episodes=20 \
        --probe_eval_batch_size=20 --wandb.enable=false >>"${LOG}" 2>&1 \
        || log "FAILED probe stage ${k}"
done

ORDER="${ORDER}" RES_DIR="${RES_DIR}" ${PYTHON} - <<'PYEOF' 2>&1 | tee -a "${LOG}"
import json, os
from pathlib import Path
order = [int(x) for x in os.environ["ORDER"].split(",")]
K = len(order)
res = Path(os.environ["RES_DIR"]) / "er_results.jsonl"
c = {}
for line in res.read_text().splitlines():
    r = json.loads(line)
    if r.get("run_tag") == "er" and r.get("sr") is not None:
        c[(r["stage"], r["probe_task"])] = float(r["sr"])
last = [c.get((K - 1, t)) for t in range(K)]
diag = [c.get((k, order[k])) for k in range(K)]
L = ["=" * 62, f"ER  task_order: {','.join(map(str, order))}   seed 42, 칸당 20 롤아웃",
     "=" * 62, "", "행 = 스테이지, 열 = 실제 task",
     "after\\task " + "".join(f"{t:>7d}" for t in range(K))]
for k in range(K):
    L.append(f"{k:>10d} " + "".join(
        f"{c[(k,t)]:7.0f}" if (k, t) in c else "      ." for t in range(K)))
if all(v is not None for v in last + diag):
    avg = sum(last) / K
    bwt = sum(c[(K-1, order[i])] - diag[i] for i in range(K - 1)) / (K - 1)
    L += ["", f"AvgSR_final  {avg:.1f}", f"BWT          {bwt:+.1f}",
          f"습득(대각)    {sum(diag)/K:.1f}",
          "", "최종 행: " + "  ".join(f"task{t}={last[t]:.0f}" for t in range(K))]
    json.dump({"AvgSR_final": avg, "BWT": bwt, "task_order": order,
               "final_row": {f"task{t}": last[t] for t in range(K)},
               "learning_sr": {f"stage{k}": diag[k] for k in range(K)}},
              (Path(os.environ["RES_DIR"]) / "metrics.json").open("w"), indent=2)
else:
    L += ["", "미완 — SR 칸에 결측이 있다"]
rep = "\n".join(L)
(Path(os.environ["RES_DIR"]) / "SR.txt").write_text(rep)
print(rep)
PYEOF
log "완료 -> ${RES_DIR}"
