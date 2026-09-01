#!/usr/bin/env python
"""B 시리즈 전체 비교표 — libero_spatial task 0..3, 칸당 20 롤아웃, seed 42.

모든 팔이 같은 자를 쓴다: 5000 steps/task, 45 에피소드(뒤 5개 hold-out), 배치 32,
백본 dit_flow_mt_libero_90_pretrain, 스테이지 k>0 은 k-1 체크포인트에서 이어받음.
아직 끝나지 않은 팔은 있는 칸까지만 표시한다.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
N = 4

# (표시명, 경로, 종류, 저장물 설명)
ARMS = [
    ("seq-FT", "outputs/E0/libero_spatial/seed_42/e0_results.jsonl", "jsonl:0", "없음"),
    ("B1",     "results/B1/sr_matrix.csv",  "csv", "모델 1개 (rolling teacher)"),
    ("B2",     "results/B2/sr_matrix.csv",  "csv", "모델 N개 (frozen teachers, 3.1GB)"),
    ("B3",     "results/B3/sr_matrix.csv",  "csv", "조건벡터 캐시 92MB"),
    ("B4",     "results/B4/sr_matrix.csv",  "csv", "모델 1개 + 질의점 4.1MB"),
    ("B5",     "results/B5/sr_matrix.csv",  "csv", "자기튜브 캐시 2.4MB (teacher 없음)"),
    ("B6",     "results/B6/sr_matrix.csv",  "csv", "자기튜브 캐시 38MB, N=512 [좌표 고정]"),
    ("B4N512", "results/B4_N512/sr_matrix.csv", "csv", "질의점 N=512 [좌표 열림]"),
    ("B7",     "results/B7/sr_matrix.csv",  "csv", "역적분 소환, 저장 0 [좌표 열림·on-tube]"),
    ("B8",     "results/B8/sr_matrix.csv",  "csv", "충돌량 Ĝ 가중 앵커, 저장 0"),
    ("B2λ3",   "results/B2_lam3/sr_matrix.csv",  "csv", "B2 + lambda 3"),
    ("B2λ10",  "results/B2_lam10/sr_matrix.csv", "csv", "B2 + lambda 10"),
    ("B2λ30",  "results/B2_lam30/sr_matrix.csv", "csv", "B2 + lambda 30"),
    ("B8λ3",   "results/B8_lam3/sr_matrix.csv",  "csv", "B8 + lambda 3"),
    ("B8λ10",  "results/B8_lam10/sr_matrix.csv", "csv", "B8 + lambda 10"),
    ("B1λ3",   "results/B1_lam3/sr_matrix.csv",  "csv", "B1 + lambda 3"),
    ("B1λ10",  "results/B1_lam10/sr_matrix.csv", "csv", "B1 + lambda 10"),
    ("B1λ30",  "results/B1_lam30/sr_matrix.csv", "csv", "B1 + lambda 30"),
    ("B9-1023", "results/B9_1023/sr_matrix_bytask.csv", "csv",
     "순서 1,0,2,3  (B2λ3 와 순서만 다름)"),
    ("B9-0321", "results/B9_0321/sr_matrix_bytask.csv", "csv", "순서 0,3,2,1  (task1↔task3)"),
    ("B9-2103", "results/B9_2103/sr_matrix_bytask.csv", "csv", "순서 2,1,0,3"),
    ("B9-3210", "results/B9_3210/sr_matrix_bytask.csv", "csv", "순서 3,2,1,0  (완전 역순)"),
    ("ER",     "results/ER_task0123/er_results.jsonl", "jsonl:er", "과거 관측 + 정답 액션"),
]


def load(path: str, kind: str) -> dict:
    p = REPO / path
    cells: dict[tuple[int, int], float] = {}
    if not p.exists():
        return cells
    if kind.startswith("jsonl"):
        tag = kind.split(":", 1)[1]
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("run_tag") != tag or r.get("sr") is None:
                continue
            cells[(r["stage"], r["probe_task"])] = float(r["sr"])
    else:
        for line in p.read_text().splitlines():
            # 첫 줄이 '# task_order: ...' 주석일 수 있고 그다음이 헤더다
            if not line or line.startswith("#") or not line.split(",")[0].strip().isdigit():
                continue
            f = line.split(",")
            for t, v in enumerate(f[1:]):
                if v.strip():
                    cells[(int(f[0]), t)] = float(v)
    return {kv: vv for kv, vv in cells.items() if kv[0] < N and kv[1] < N}


def _order_of(path: str) -> list[int]:
    """'# task_order: ...' 주석에서 학습 순서를 읽는다. 없으면 항등."""
    p = REPO / path
    if p.exists():
        first = p.read_text().splitlines()[0]
        if first.startswith("#") and "task_order:" in first:
            tok = first.split("task_order:")[1].split()[0]
            return [int(x) for x in tok.split(",")]
    return list(range(N))


def metrics(c: dict, order=None):
    order = order or list(range(N))
    pos = {t: i for i, t in enumerate(order)}     # task t 를 배운 스테이지
    last = [c.get((N - 1, t)) for t in range(N)]
    diag = [c.get((pos[t], t)) for t in range(N)]
    if any(v is None for v in last) or any(v is None for v in diag):
        return None
    return (sum(last) / N,
            sum(last[t] - diag[t] for t in range(N - 1)) / (N - 1),
            sum(diag) / N, last, diag)


def main() -> None:
    loaded = [(name, load(path, kind), note,
               _order_of(path) if kind == "csv" else list(range(N)))
              for name, path, kind, note in ARMS]
    want = N * (N + 1) // 2

    L = ["=" * 84,
         "B 시리즈 비교 — libero_spatial task 0..3, 칸당 20 롤아웃, seed 42",
         "=" * 84, "",
         "공통: 5000 steps/task, 45 에피소드, 배치 32, 백본 dit_flow_mt_libero_90_pretrain",
         "B1~B4 차이는 앵커 타깃을 어디서 얻는가 뿐이다:",
         "  B1  rolling teacher      직전 스냅샷 1개 — 세대마다 타깃이 표류",
         "  B2  frozen teachers      태스크별 스냅샷 영구 보관 — 표류 없음",
         "  B3  cached targets       태스크 종료 시 타깃을 얼림 — 표류 없음, Δ 성장 금지",
         "  B4  query-point anchor   rolling teacher + 과거 관측 임베딩 위에서 앵커",
         "  B5  self-rollout        전성기 모델이 자기 ODE 로 만든 튜브 위에 앵커 (N=32)",
         "  B6  self-rollout N=512  B5 와 N 만 다름 — 고정 좌표에서 개수 축",
         "  B4N512  질의점 N=512      B4 와 N 만 다름 — 열린 좌표에서 커버리지 축",
         "  B7  fresh inversion    매 스텝 a_k 를 teacher ℓ_j-field 로 역적분해 앵커점 소환",
         "",
         "축 분해:  B6 vs B4N512 = 좌표 개방성 |  B4 vs B4N512 = 커버리지 |  B1 vs B7 = on-tube",
         "",
         "-" * 84,
         f"{'팔':>8}{'칸':>9}{'AvgSR':>9}{'BWT':>9}{'습득':>8}   저장물",
         "-" * 84]
    for name, c, note, _o in loaded:
        m = metrics(c, _o)
        got = f"{len(c)}/{want}"
        if m:
            L.append(f"{name:>8}{got:>9}{m[0]:>9.1f}{m[1]:>+9.1f}{m[2]:>8.1f}   {note}")
        else:
            L.append(f"{name:>8}{got:>9}{'-':>9}{'-':>9}{'-':>8}   {note} (진행 중)")
    L.append("")

    # 태스크별 최종 행
    L += ["-" * 84, "최종 행 (stage 3) — 태스크별", "-" * 84,
          f"{'팔':>8}" + "".join(f"{'task'+str(t):>9}" for t in range(N))]
    for name, c, _, _o in loaded:
        row = [c.get((N - 1, t)) for t in range(N)]
        if all(v is None for v in row):
            continue
        L.append(f"{name:>8}" + "".join(
            f"{v:>9.0f}" if v is not None else f"{'.':>9}" for v in row))
    L.append("")

    # 스테이지별 평균
    L += ["-" * 84, "스테이지별 평균 SR (본 태스크들만)", "-" * 84,
          f"{'팔':>8}" + "".join(f"{'stage'+str(k):>10}" for k in range(N))]
    for name, c, _, _o in loaded:
        vals = []
        for k in range(N):
            row = [c.get((k, t)) for t in range(k + 1)]
            vals.append(sum(row) / len(row) if all(v is not None for v in row) else None)
        if all(v is None for v in vals):
            continue
        L.append(f"{name:>8}" + "".join(
            f"{v:>10.1f}" if v is not None else f"{'.':>10}" for v in vals))
    L.append("")

    for name, c, _, _o in loaded:
        if not c:
            continue
        note = ""
        for _n, _path, _k, _d in ARMS:
            if _n == name and _k == "csv":
                _p = REPO / _path
                if _p.exists():
                    _first = _p.read_text().splitlines()[0]
                    if _first.startswith("#"):
                        note = "   " + _first.lstrip("# ").strip()
        L += ["=" * 84, f"{name}   행 = 태스크 k 학습 후, 열 = 평가 태스크" + note, "=" * 84,
              "after\\task " + "".join(f"{t:>8d}" for t in range(N))]
        # ★ 열을 range(k+1) 로 자르면 안 된다. 순서를 바꾼 실행에서는 스테이지 k 까지
        #   본 태스크가 order[:k+1] 이라 task 인덱스가 연속이 아니다(열 누락 발생).
        #   전 열을 그리고 값이 없는 칸만 '.' 로 둔다. 빈 행도 건너뛰지 않는다.
        for k in range(N):
            L.append(f"{k:>10d} " + "".join(
                f"{c[(k, t)]:8.0f}" if (k, t) in c else f"{'.':>8}" for t in range(N)))
        L.append("")

    # ── 추론 시 guidance 증폭 w 스윕 (재학습 없음, stage 3 체크포인트 롤아웃) ──
    import glob
    wfiles = sorted(glob.glob(str(REPO / "results" / "B_wsweep" / "*" / "sr.json")))
    if wfiles:
        L += ["=" * 84,
              "추론 시 guidance 증폭 스윕   v = v(∅) + w·(v(ℓ) − v(∅))",
              "=" * 84,
              "학습 없음. stage 3 체크포인트를 w 마다 다시 롤아웃했다(칸당 20 에피소드).",
              "w=1 은 표준 조건부 추론이며, 같은 체크포인트라도 재측정 시 ±11%p 흔들리므로",
              "학습 시 기록된 SR 과 직접 비교하지 말고 이 표 안에서만 상대 비교할 것.",
              ""]
        for wf in wfiles:
            arm = Path(wf).parent.name
            d = json.loads(Path(wf).read_text())
            ws = sorted(d, key=float)
            tasks = sorted({int(t) for r in d.values() for t in r})
            L += ["-" * 84, f"{arm}", "-" * 84,
                  f"{'w':>6}" + "".join(f"{'task'+str(t):>9}" for t in tasks) + f"{'평균':>9}"]
            for w in ws:
                r = d[w]
                vals = [r[str(t)] for t in tasks if r.get(str(t)) is not None]
                L.append(f"{float(w):>6.2f}" + "".join(
                    f"{r[str(t)]:>9.0f}" if r.get(str(t)) is not None else f"{'—':>9}"
                    for t in tasks)
                    + (f"{sum(vals)/len(vals):>9.1f}" if vals else f"{'—':>9}"))
            L.append("")

    out = REPO / "results" / "B_compare.txt"
    rep = "\n".join(L)
    out.write_text(rep)
    print(rep)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
