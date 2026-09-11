#!/usr/bin/env python
"""L0 결과 표 — results/L0/sr_table.{csv,md} + results/L_SR_matrix.txt.

B1 이 칸마다 sr_matrix.csv 를 갱신하므로 **미완이어도** 돌아간다.
도는 중에 실행해도 되고, 완료 후 한 번 더 돌리면 최종본이 된다.

    python l0_report.py                     # results/L0
    python l0_report.py results/L0_smoke    # 다른 디렉터리
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
REF = {"ER": 86.0, "R13": 79.5, "K1": 77.0, "R12": 73.5}
R13_DIR = RES / "R13_10task"


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


def resp_rows(d: Path):
    p = d / "resp_by_stage.jsonl"
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
        return [f"# L0 — 아직 SR 칸이 없다 ({d}/sr_matrix.csv 없음 또는 빈 파일)"]
    s = summarize(cells, K)
    cfg = {}
    p = d / "l0_config.json"
    if p.exists():
        try:
            cfg = json.loads(p.read_text())
        except Exception:
            pass

    L = [f"# L0 — Implicit CARA (R13 + 명령어-앙상블 조건응답 앵커)",
         f"LIBERO-spatial {K} task, 5000 step/task, 칸당 20 rollout", "",
         f"δℓ공간 {cfg.get('delta_space','?')}   ρ_ic {cfg.get('rho_ic','?')}   "
         f"현재명령어포함 {cfg.get('include_current','?')}   λ_level {cfg.get('lambda_level','?')}", ""]
    head = (f"**AvgSR (마지막 행 평균) = {s['avg']:.1f}**" if s["avg"] is not None
            else f"**진행 중 — {s['filled']}/{s['total']} 칸**")
    if s["bwt"] is not None:
        head += f"   BWT {s['bwt']:+.1f}"
    if s["acq"] is not None:
        head += f"   습득(대각) {s['acq']:.1f}"
    L += [head, "", "참고값 (같은 프로토콜)  " + "   ".join(f"{k} = {v}" for k, v in REF.items()), "",
          "| after task | " + " | ".join(f"task{t}" for t in range(K)) + " |",
          "|---" * (K + 1) + "|"]
    for k in range(K):
        L.append(f"| {k} | " + " | ".join(
            f"{cells[(k,t)]:.0f}" if (k, t) in cells else "" for t in range(K)) + " |")

    # task1 궤적 — 10 태스크에서 매번 무너지던 열
    r13, _ = read_sr(R13_DIR)
    L += ["", "## task1 궤적 (stage 별) — L0 의 표적", "",
          "| stage | " + " | ".join(str(k) for k in range(K)) + " |",
          "|---" * (K + 1) + "|",
          "| L0 | " + " | ".join(
              f"{cells[(k,1)]:.0f}" if (k, 1) in cells else "" for k in range(K)) + " |"]
    if r13:
        L.append("| R13 | " + " | ".join(
            f"{r13[(k,1)]:.0f}" if (k, 1) in r13 else "" for k in range(K)) + " |")

    # mechanism 지표
    rr = resp_rows(d)
    if rr:
        L += ["", "## 조건응답 크기의 stage 추이 (핵심 mechanism 지표)", "",
              "resp_T = teacher 가 명령어 섭동에 보이는 응답 크기 = '보유한 routing 양'.",
              "이 값이 stage 를 따라 줄면 teacher 자체가 조건을 잃고 있다는 뜻이고,",
              "그러면 L_icara 가 지킬 대상이 사라진다(앵커의 상한).", "",
              "| stage | steps | resp_T | resp_S | ratio S/T | L_icara | λ_ic |",
              "|---|---|---|---|---|---|---|"]
        for r in rr:
            lam = r.get("lambda_ic")
            L.append(f"| {r['stage']} | {r['steps']} | {r['resp_T']:.4f} | "
                     f"{r['resp_S']:.4f} | {r['ratio']:.3f} | {r['L_icara']:.4f} | "
                     + (f"{lam:.4g} |" if lam is not None else "— |"))
    else:
        L += ["", "## 조건응답 stage 추이", "", "(아직 스테이지가 끝나지 않아 resp_by_stage.jsonl 이 없다)"]
    return L


def write_csv(d: Path, cells, K):
    with (d / "sr_table.csv").open("w") as f:
        f.write("after_task," + ",".join(f"task{t}" for t in range(K)) + "\n")
        for k in range(K):
            f.write(f"{k}," + ",".join(
                f"{cells[(k,t)]:.1f}" if (k, t) in cells else "" for t in range(K)) + "\n")


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "L0"
    cells, K = read_sr(d)
    L = build(d)
    if cells:
        write_csv(d, cells, K)
    (d / "sr_table.md").write_text("\n".join(L) + "\n")

    # 요청받은 파일: results/L_SR_matrix.txt
    txt = ["=" * 78,
           "L0 — Implicit CARA (R13 + 명령어-앙상블 조건응답 앵커)",
           "LIBERO-spatial 10 task · 5000 step/task · seed 42 · 칸당 20 rollout",
           "=" * 78, ""] + L
    (RES / "L_SR_matrix.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {d/'sr_table.md'}, {d/'sr_table.csv'}, {RES/'L_SR_matrix.txt'}")


if __name__ == "__main__":
    main()
