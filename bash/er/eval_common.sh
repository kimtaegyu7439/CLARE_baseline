#!/usr/bin/env bash
#
# ER 체크포인트 성능 평가 — 공통 루프. 직접 실행하지 않는다.
# eval_libero_{spatial,goal,10,40}.sh 가 아래 변수를 채운 뒤 이 파일을 source 한다.
#
#   BENCH_NAME          출력 경로에 쓸 이름 (예: libero_spatial)
#   NUM_TASKS           태스크 수
#   STAGE_CKPT[k]       스테이지 k 체크포인트 디렉터리 (…/checkpoints/last/pretrained_model)
#   TASK_BENCH[t]       태스크 t의 --env.benchmark
#   TASK_HANDLE[t]      태스크 t의 --env.task     (예: Libero_Spatial_Task_3)
#   TASK_REPO[t]        태스크 t의 --dataset.repo_id
#
# ── 왜 학습과 분리했는가 ──────────────────────────────────────────────────────
# 평가는 시뮬레이터 롤아웃이라 LIBERO 환경을 eval.batch_size개 **동시에** 띄운다.
# 환경 하나가 MuJoCo + EGL 렌더링 컨텍스트를 GPU에 잡으므로, 학습용 정책과 같은 카드에서
# 50개를 띄우면 정책이 올라갈 자리가 없다(실측: "Creating env" 70초 뒤
# LanguageEncoder.to(cuda)에서 CUDA out of memory). 학습 스크립트는 EVAL_FREQ=0으로
# 두어 make_env 호출 자체를 건너뛰고, 성능은 학습이 끝난 뒤 이 스크립트로 잰다.
#
# ── 무엇을 재는가 ─────────────────────────────────────────────────────────────
# 스테이지 k 체크포인트로 **그때까지 배운 태스크 0..k**를 평가한다(총 N(N+1)/2회).
# 행=체크포인트, 열=태스크인 하삼각 행렬이 나오고, 열을 세로로 읽으면 그 태스크가
# 이후 스테이지에서 얼마나 잊히는지가 보인다.

set -uo pipefail

: "${BENCH_NAME:?eval_common.sh는 직접 실행하지 않는다 — eval_libero_*.sh를 써라}"

SEED=${SEED:-42}
N_EVAL=${N_EVAL:-100}
# ★ 기본 20. 학습 스크립트의 50은 OOM을 낸 값이다. 평가만 단독으로 도는 지금은 여유가
#   있지만, E0 프로브가 20으로 문제없이 돈 실적이 있어 그 값을 기본으로 둔다.
#   카드가 넉넉하면 BS_EVAL=50으로 올려 시간을 줄여도 된다.
BS_EVAL=${BS_EVAL:-20}
RENDER=${RENDER:-0}                 # 저장할 비디오 수. 0이면 인코딩 비용이 사라진다.

# 무엇을 돌 것인가. 서버/GPU를 나눌 때 이 둘만 바꾸면 된다.
#   STAGES="0 1 2"   특정 체크포인트만
#   TASKS="3 4"      특정 태스크만 (스테이지보다 나중 태스크는 자동으로 건너뛴다)
STAGES=${STAGES:-$(seq 0 $((NUM_TASKS - 1)))}
TASKS=${TASKS:-""}

OUT_ROOT=${OUT_ROOT:-./outputs/ER_eval/${BENCH_NAME}/seed${SEED}}
PYTHON=${PYTHON:-python}
EVAL_PY=./lerobot_lsy/src/lerobot/scripts/eval.py
REDO=${REDO:-0}                     # 1이면 이미 끝난 조합도 다시 돈다

mkdir -p "${OUT_ROOT}"
SUMMARY=${OUT_ROOT}/summary.csv

echo "══ ER eval  ${BENCH_NAME}  seed=${SEED}"
echo "   stages   : ${STAGES}"
echo "   tasks    : ${TASKS:-<스테이지마다 0..k>}"
echo "   episodes : ${N_EVAL}  (환경 ${BS_EVAL}개 동시)"
echo "   out      : ${OUT_ROOT}"
echo ""

n_run=0; n_skip=0; n_fail=0
for k in ${STAGES}; do
    ckpt="${STAGE_CKPT[$k]}"
    if [ ! -d "${ckpt}" ]; then
        echo "[eval] SKIP stage ${k}: 체크포인트 없음 (${ckpt})"
        continue
    fi
    # 기본은 "그때까지 배운 태스크". TASKS로 좁히더라도 스테이지보다 나중 태스크는
    # 아직 배우지 않았으므로 제외한다(평가해도 의미가 없다).
    task_list=${TASKS:-$(seq 0 "${k}")}
    for t in ${task_list}; do
        [ "${t}" -le "${k}" ] || continue
        out="${OUT_ROOT}/stage${k}/task${t}"
        if [ -f "${out}/eval_info.json" ] && [ "${REDO}" != "1" ]; then
            n_skip=$((n_skip + 1)); continue
        fi
        mkdir -p "${out}"
        echo "── stage ${k} ckpt  ×  task ${t} (${TASK_HANDLE[$t]})"
        "${PYTHON}" "${EVAL_PY}" \
            --policy.path="${ckpt}" \
            --policy.device=cuda \
            --dataset.repo_id="${TASK_REPO[$t]}" \
            --env.type=libero \
            --env.benchmark="${TASK_BENCH[$t]}" \
            --env.task="${TASK_HANDLE[$t]}" \
            --eval.batch_size="${BS_EVAL}" \
            --eval.n_episodes="${N_EVAL}" \
            --eval.max_episodes_rendered="${RENDER}" \
            --output_dir="${out}" \
            --seed="${SEED}" \
            || { echo "[eval] FAILED stage ${k} task ${t}"; n_fail=$((n_fail + 1)); continue; }
        n_run=$((n_run + 1))
    done
done

echo ""
echo "[eval] 실행 ${n_run}  건너뜀 ${n_skip}  실패 ${n_fail}"

# ── 결과를 하나의 표로 모은다 ────────────────────────────────────────────────
"${PYTHON}" - "${OUT_ROOT}" "${SUMMARY}" "${NUM_TASKS}" <<'PYEOF'
import json, sys
from pathlib import Path

root, summary, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cells = {}
for p in sorted(root.glob("stage*/task*/eval_info.json")):
    k = int(p.parent.parent.name.removeprefix("stage"))
    t = int(p.parent.name.removeprefix("task"))
    try:
        agg = json.loads(p.read_text())["aggregated"]
    except Exception as e:                       # 중간에 죽어 잘린 파일은 건너뛴다
        print(f"  WARN 읽기 실패 {p}: {e}")
        continue
    cells[(k, t)] = agg.get("pc_success")

if not cells:
    print("[eval] 아직 결과가 없다.")
    raise SystemExit(0)

with open(summary, "w") as f:
    f.write("checkpoint," + ",".join(f"task{t}" for t in range(n)) + ",avg_seen\n")
    for k in range(n):
        row = [cells.get((k, t)) for t in range(n)]
        seen = [v for v in row[: k + 1] if v is not None]
        f.write(f"stage{k}," + ",".join("" if v is None else f"{v:.1f}" for v in row)
                + "," + (f"{sum(seen) / len(seen):.1f}" if seen else "") + "\n")
print(f"[eval] saved -> {summary}")

# 화면용 하삼각 행렬. 행=체크포인트, 열=태스크.
w = max(6, len(str(n)) + 5)
print("\n     SR(%)  " + "".join(f"{'t' + str(t):>{w}}" for t in range(n)) + f"{'avg':>{w}}")
for k in range(n):
    row = [cells.get((k, t)) for t in range(n)]
    seen = [v for v in row[: k + 1] if v is not None]
    print(f"  stage{k:<3}  " + "".join(f"{'' if v is None else f'{v:.0f}':>{w}}" for v in row)
          + f"{(f'{sum(seen) / len(seen):.1f}' if seen else ''):>{w}}")
PYEOF
