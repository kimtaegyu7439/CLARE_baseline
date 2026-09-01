#!/usr/bin/env bash
#
# ER 평가 결과(outputs/ER_eval/<bench>/seed<N>/stage*/task*/eval_info.json)를
# CLARE 쪽과 같은 형식의 SR 행렬 txt로 모은다.
#
#   행 = 스테이지 k 체크포인트, 열 = 태스크 j, 값 = 성공률(%). 하삼각.
#   탭 구분, 첫 줄은 "LIBERO_<NAME>\t0\t1\t..." — 기존 *_SR.txt와 같은 형식이라
#   기존 파싱 스크립트가 그대로 동작한다.
#
# 사용법
#   bash bash/er/collect_er_sr.sh                       # 넷 다 (있는 것만)
#   bash bash/er/collect_er_sr.sh libero_40             # 하나만
#
# 평가가 도는 중에 실행해도 된다. 아직 없는 칸은 빈칸으로 남는다.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED=${SEED:-42}
EVAL_ROOT=${EVAL_ROOT:-./outputs/ER_eval}
PYTHON=${PYTHON:-python3}

BENCHES=${*:-"libero_10 libero_goal libero_spatial libero_40"}

for bench in ${BENCHES}; do
    case "${bench}" in
        libero_40) n=40 ;;
        *)         n=10 ;;
    esac
    root="${EVAL_ROOT}/${bench}/seed${SEED}"
    if [ ! -d "${root}" ]; then
        echo "[collect] SKIP ${bench}: 결과 없음 (${root})"
        continue
    fi
    "${PYTHON}" - "${root}" "${n}" "${bench}" <<'PYEOF'
import json, sys
from pathlib import Path

root, n, bench = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
cells, broken, eps = {}, 0, set()
for p in sorted(root.glob("stage*/task*/eval_info.json")):
    k = int(p.parent.parent.name.removeprefix("stage"))
    t = int(p.parent.name.removeprefix("task"))
    try:
        d = json.loads(p.read_text())
        cells[(k, t)] = d["aggregated"]["pc_success"]
        eps.add(len(d.get("per_episode", [])))
    except Exception as e:                       # 중간에 죽어 잘린 파일은 건너뛴다
        print(f"  WARN 읽기 실패 {p}: {e}"); broken += 1

want = n * (n + 1) // 2
tag = f"LIBERO_{bench.removeprefix('libero_').upper()}"
print(f"[collect] {bench}: {len(cells)}/{want} 칸" +
      (f", {sorted(eps)} 에피소드" if eps else "") +
      (f" (손상 {broken})" if broken else ""))
if not cells:
    raise SystemExit(0)

out = root / f"{bench}_SR.txt"
with open(out, "w") as f:
    f.write(f"{tag}\t" + "\t".join(str(t) for t in range(n)) + "\n")
    for k in range(n):
        row = [cells.get((k, t)) for t in range(k + 1)]
        f.write(f"{k}\t" + "\t".join("" if v is None else f"{v:.0f}" for v in row) + "\n")
print(f"[collect] saved -> {out}")

# 마지막 스테이지가 다 찼을 때만 지표를 낸다(부분 집계는 오해를 부른다).
last = [cells.get((n - 1, t)) for t in range(n)]
if all(v is not None for v in last):
    acc = sum(last) / n
    diag = [cells.get((t, t)) for t in range(n - 1)]
    bwt = (sum(last[t] - diag[t] for t in range(n - 1) if diag[t] is not None)
           / max(1, sum(1 for d in diag if d is not None)))
    print(f"[metric] {bench}: ACC(최종 평균 SR) {acc:.1f}%   BWT(역방향 전이) {bwt:+.1f}%")
else:
    done = sum(v is not None for v in last)
    print(f"[metric] {bench}: 마지막 스테이지 {done}/{n}칸 — 지표는 전부 채워진 뒤에 낸다")

# libero_40에서 옛 benchmark/task_id 짝 버그가 되살아났는지 감시.
if n == 40:
    lo = [v for (k, t), v in cells.items() if t % 10 > k % 10]
    hi = [v for (k, t), v in cells.items() if t % 10 <= k % 10]
    if lo and hi:
        print(f"[check] (j%10)>(k%10) 칸 평균 {sum(lo)/len(lo):.1f} (n={len(lo)})   "
              f"나머지 {sum(hi)/len(hi):.1f} (n={len(hi)}) — 앞쪽이 ~0이면 짝이 또 깨진 것이다")
PYEOF
done
