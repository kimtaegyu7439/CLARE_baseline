#!/usr/bin/env python
"""L2 결과 표 — results/L2/sr_table.{csv,md} + results/L2_SR_matrix.txt.

B1 이 칸마다 sr_matrix.csv 를 갱신하므로 미완이어도 돌아간다.

    python l2_report.py                  # results/L2
    python l2_report.py results/L2_xxx   # 다른 디렉터리
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
REF = {"ER": 86.0, "R13": 79.5, "K1": 77.0, "R12": 73.5, "L0(ρ=1.0)": 60.0}
PEERS = [("R13", RES / "R13_10task"), ("K1", RES / "K1_spatial_10task"),
         ("L0", RES / "L0")]


def read_sr(d: Path):
    p = d / "sr_matrix.csv"
    if not p.exists():
        return {}, 0
    cells, K = {}, 0
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if not f[0].strip().isdigit():
            continue
        k = int(f[0]); K = max(K, len(f) - 1)
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    return cells, K


def rowavg(c, k):
    v = [c[(k, t)] for t in range(k + 1) if (k, t) in c]
    return sum(v) / len(v) if len(v) == k + 1 else None


def summarize(cells, K):
    last = [cells.get((K - 1, t)) for t in range(K)]
    diag = [cells.get((t, t)) for t in range(K)]
    done = all(v is not None for v in last)
    acq = [v for v in diag if v is not None]
    return {"avg": (sum(last) / K) if done else None,
            "bwt": (sum(last[t] - diag[t] for t in range(K - 1)) / (K - 1)
                    if done and all(v is not None for v in diag) else None),
            "acq": (sum(acq) / len(acq)) if acq else None,
            "filled": len(cells), "total": K * (K + 1) // 2}


def diag_rows(d: Path):
    p = d / "xt_diag.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def build(d: Path) -> list[str]:
    cells, K = read_sr(d)
    if not cells:
        return [f"# L2 — 아직 SR 칸이 없다 ({d}/sr_matrix.csv 없음)"]
    s = summarize(cells, K)
    cfg = {}
    p = d / "l2_config.json"
    if p.exists():
        try:
            cfg = json.loads(p.read_text())
        except Exception:
            pass
    peers = {n: read_sr(pd)[0] for n, pd in PEERS}

    L = ["# L2 — teacher-부트스트랩 x_t (앵커 보간의 행동 성분을 task-j 것으로)",
         f"LIBERO-spatial {K} task · 5000 step/task · seed 42 · 칸당 20 rollout", "",
         f"xt_mode {cfg.get('xt_mode','?')}   λ_level {cfg.get('lambda_level','?')}   "
         f"진단주기 {cfg.get('diag_every','?')} step   진단 t {cfg.get('diag_t','?')}", ""]
    head = (f"**AvgSR (마지막 행 평균) = {s['avg']:.1f}**" if s["avg"] is not None
            else f"**진행 중 — {s['filled']}/{s['total']} 칸**")
    if s["bwt"] is not None:
        head += f"   BWT {s['bwt']:+.1f}"
    if s["acq"] is not None:
        head += f"   습득(대각) {s['acq']:.1f}"
    L += [head, "", "참고값 (같은 프로토콜)  " + "   ".join(f"{k} = {v}" for k, v in REF.items()),
          "", "칸당 20 롤아웃이라 개별 칸의 이항 표준오차는 ±11%p 다. 행/열 평균으로 읽어야 한다.",
          "시드 하나(42)이므로 R13 대비 우위 주장에는 시드 반복이 더 필요하다.", "",
          "## SR matrix", "",
          "| after task | " + " | ".join(f"task{t}" for t in range(K)) + " |",
          "|---" * (K + 1) + "|"]
    for k in range(K):
        L.append(f"| {k} | " + " | ".join(
            f"{cells[(k,t)]:.0f}" if (k, t) in cells else "" for t in range(K)) + " |")

    # 행평균 대조
    L += ["", "## 행평균 대조 (스테이지 k 시점의 task 0..k 평균)", "",
          "| stage | " + " | ".join(str(k) for k in range(K)) + " |",
          "|---" * (K + 1) + "|"]
    for name, c in [("**L2**", cells)] + [(n, peers[n]) for n, _ in PEERS]:
        row = [rowavg(c, k) for k in range(K)]
        L.append(f"| {name} | " + " | ".join(
            f"{v:.1f}" if v is not None else "" for v in row) + " |")
    dl = [rowavg(cells, k) for k in range(K)]
    dr = [rowavg(peers["R13"], k) for k in range(K)]
    L.append("| L2−R13 | " + " | ".join(
        f"{a-b:+.1f}" if (a is not None and b is not None) else ""
        for a, b in zip(dl, dr)) + " |")

    # task1 궤적
    L += ["", "## task1 궤적 — 10 태스크에서 매번 무너지던 열", "",
          "| stage | " + " | ".join(str(k) for k in range(K)) + " |",
          "|---" * (K + 1) + "|"]
    for name, c in [("**L2**", cells)] + [(n, peers[n]) for n, _ in PEERS]:
        L.append(f"| {name} | " + " | ".join(
            f"{c[(k,1)]:.0f}" if (k, 1) in c else "" for k in range(K)) + " |")

    # 습득
    L += ["", "## 습득 (대각)", "",
          "| stage | " + " | ".join(str(k) for k in range(K)) + " |",
          "|---" * (K + 1) + "|"]
    for name, c in [("**L2**", cells), ("R13", peers["R13"])]:
        L.append(f"| {name} | " + " | ".join(
            f"{c[(k,k)]:.0f}" if (k, k) in c else "" for k in range(K)) + " |")

    # 내장 진단
    dg = diag_rows(d)
    if dg:
        L += ["", "## 내장 진단 — x축 어긋남 (mechanism)", "",
              "같은 (b_j, ℓ_j)·같은 ε 에서 x_t 만 바꿔 ‖v_S − v_Tj‖ 를 잰다.",
              "  r_A : x_t = (1−t)ε + t·a_cur   (R13 방식, 현재 태스크 행동)",
              "  r_B : x_t = (1−t)ε + t·â_j     (L2 방식, teacher 부트스트랩 행동)",
              "gap = r_B − r_A.  policy.eval() 로 dropout 을 끄고 잰다.", "",
              "★ 해석 주의: L2 는 r_B 좌표에서 학습하므로 gap 이 음수인 것 자체는",
              "  자기충족적이다. 읽어야 할 것은 **크기의 t-의존성**이다 —",
              "  t=0.1 에서 ≈0 이고 t=0.9 에서 커지면 두 좌표가 행동 성분에서만",
              "  갈린다는 뜻이고, 그것이 x축 어긋남의 실측 증거다.", "",
              "| stage | step | ‖â−a_cur‖ | " +
              " | ".join(f"gap(t={t})" for t in (0.1, 0.5, 0.9)) + " | " +
              " | ".join(f"r_A(t={t})" for t in (0.1, 0.5, 0.9)) + " |",
              "|---" * 9 + "|"]
        # stage 별 마지막 진단만
        last = {}
        for r in dg:
            last[r["task"]] = r
        for kk in sorted(last):
            r = last[kk]
            g = {x["t"]: x for x in r["rows"]}
            L.append(f"| {kk} | {r['step']} | {r['d_action']:.2f} | " +
                     " | ".join(f"{g[t]['gap']:+.4f}" if t in g else "" for t in (0.1, 0.5, 0.9))
                     + " | " +
                     " | ".join(f"{g[t]['r_A']:.4f}" if t in g else "" for t in (0.1, 0.5, 0.9))
                     + " |")
        L += ["", "(각 stage 의 **마지막** 진단 시점. stage 첫 스텝은 학생==teacher 라",
              " r_A=r_B=0 이 나오는데, 스냅샷 동일성과 dropout off 를 확인해 주는 값이다.)"]
    if cfg.get("base_diff"):
        L += ["", "## R13 대비 diff", ""] + [f"{i+1}. {x.split('. ',1)[-1]}"
                                            for i, x in enumerate(cfg["base_diff"])]
    return L


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "L2"
    cells, K = read_sr(d)
    L = build(d)
    if cells:
        with (d / "sr_table.csv").open("w") as f:
            f.write("after_task," + ",".join(f"task{t}" for t in range(K)) + "\n")
            for k in range(K):
                f.write(f"{k}," + ",".join(
                    f"{cells[(k,t)]:.1f}" if (k, t) in cells else "" for t in range(K)) + "\n")
    (d / "sr_table.md").write_text("\n".join(L) + "\n")
    txt = ["=" * 78, "L2 — teacher-부트스트랩 x_t", "=" * 78, ""] + L
    (RES / "L2_SR_matrix.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {d/'sr_table.md'}, {d/'sr_table.csv'}, {RES/'L2_SR_matrix.txt'}")


if __name__ == "__main__":
    main()
