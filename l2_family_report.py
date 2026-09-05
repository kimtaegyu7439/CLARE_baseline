#!/usr/bin/env python
"""L2 계열 전 팔 SR 모음 — results/L2_SR_matrix.txt.

도는 중에 실행해도 된다. 미완 팔은 "진행 중 (x/55 칸)" 으로 표시하고
집계는 마지막 **완성된** 행 기준으로 낸다(최종 아님을 명시).

    python l2_family_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "L2_SR_matrix.txt"
K = 10

# (표시명, 디렉터리, 한 줄 설명)
ARMS = [
    ("L2",            "L2",                        "teacher 1-step 부트스트랩 x_t (시간-bin 관측 + 현재 state)"),
    ("L2_codebook",   "L2_codebook",               "v1 — (s,o) 코드북, 셀 대각 가우시안, 커널 가중"),
    ("…+grad",        "l2_codebook_k96_grad",      "v3 — v1 + 셀별 선형 기울기 A_k (서브셀 결합)"),
    ("…+bayes",       "l2_codebook_k96_bayes",     "런 A — v1 + GMM 사후확률 가중 (커널 대체)"),
    ("…+grad+bayes",  "l2_codebook_k96_grad_bayes","런 B — v3 + GMM 사후확률 가중"),
    ("…+fullcovS",    "l2cb_fullcov",              "v1 + p(s|j) 완전 공분산 (대각 아님)"),
    ("…+fullcovS+grad","l2cb_fullcov_grad",        "v3 + p(s|j) 완전 공분산"),
]
REF = [
    ("ER",  "ER_10task",           "과거 원시 데이터 사용 (상한 참조)", "tab"),
    ("R13", "R13_10task",          "가우시안 샘플 좌표 + level 앵커",   "csv"),
    ("K1",  "K1_spatial_10task",   "공유기저 분위수 수송",              "csv"),
    ("R12", "R12_10task",          "수송 좌표 + level 앵커",            "csv"),
    ("L0",  "L0",                  "명령어-앙상블 조건응답 앵커 (기각)", "csv"),
]


def read_csv(d: Path):
    p = d / "sr_matrix.csv"
    if not p.exists():
        return {}
    c = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if not f[0].strip().isdigit():
            continue
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip():
                c[(k, t)] = float(v)
    return c


def read_tab(d: Path):
    for p in list(d.glob("*_SR.txt")) + list(d.glob("SR.txt")):
        c = {}
        for line in p.read_text().splitlines():
            f = line.split("\t")
            if not f or not f[0].strip().isdigit():
                continue
            k = int(f[0])
            for t, v in enumerate(f[1:]):
                if v.strip():
                    c[(k, t)] = float(v)
        if c:
            return c
    return {}


def last_full_row(c):
    """가장 최근의 **완성된** 행 k (0..k 가 다 찬 행). 없으면 None."""
    best = None
    for k in range(K):
        if all((k, t) in c for t in range(k + 1)):
            best = k
    return best


def stats(c):
    k = last_full_row(c)
    if k is None:
        return None
    last = [c[(k, t)] for t in range(k + 1)]
    diag = [c.get((t, t)) for t in range(k + 1)]
    avg = sum(last) / (k + 1)
    acq = ([v for v in diag if v is not None] or [None])
    acq = sum(acq) / len(acq) if acq[0] is not None else None
    bwt = (sum(last[t] - diag[t] for t in range(k))
           / max(k, 1)) if k and all(v is not None for v in diag) else None
    return {"row": k, "avg": avg, "bwt": bwt, "acq": acq,
            "filled": len(c), "done": k == K - 1}


def main() -> None:
    L = ["=" * 86,
         "L2 계열 — LIBERO-spatial 10 task SR 모음",
         "=" * 86, "",
         "공통 프로토콜  5000 step/task · seed 42 · 45 에피소드(뒤 5개 hold-out) · p_drop=0 · w=1",
         "               칸당 20 rollout · 하삼각 55칸 · 과거 원시 데이터 미사용(ER 제외)",
         "",
         "칸당 20 롤아웃이라 개별 칸의 이항 표준오차는 ±11%p, 10칸 평균도 ±3.5%p 수준이다.",
         "시드 하나(42)이므로 팔 사이 수%p 차이는 시드 반복 없이는 확정할 수 없다.",
         "",
         "AvgSR = 마지막 완성 행의 평균 · 습득 = 대각 평균 · BWT = 마지막행 − 대각",
         ""]

    # ── 요약 ─────────────────────────────────────────────────────────────
    L += ["-" * 86, "요약", "-" * 86,
          f"{'팔':<16}{'AvgSR':>8}{'BWT':>8}{'습득':>8}  {'상태':<22}설명", ""]
    rows = []
    for name, d, desc in ARMS:
        c = read_csv(RES / d)
        s = stats(c)
        if s is None:
            L.append(f"{name:<16}{'-':>8}{'-':>8}{'-':>8}  {'미시작':<22}{desc}")
            continue
        st = "완료" if s["done"] else f"진행 중 {s['filled']}/55칸 (행{s['row']})"
        L.append(f"{name:<16}{s['avg']:8.1f}"
                 + (f"{s['bwt']:+8.1f}" if s["bwt"] is not None else f"{'-':>8}")
                 + (f"{s['acq']:8.1f}" if s["acq"] is not None else f"{'-':>8}")
                 + f"  {st:<22}{desc}")
        rows.append((name, c, s))
    L += ["", "참고 (같은 프로토콜)"]
    refs = []
    for name, d, desc, kind in REF:
        c = read_tab(RES / d) if kind == "tab" else read_csv(RES / d)
        s = stats(c)
        if s is None:
            continue
        L.append(f"{name:<16}{s['avg']:8.1f}"
                 + (f"{s['bwt']:+8.1f}" if s["bwt"] is not None else f"{'-':>8}")
                 + (f"{s['acq']:8.1f}" if s["acq"] is not None else f"{'-':>8}")
                 + f"  {'완료' if s['done'] else '진행중':<22}{desc}")
        refs.append((name, c, s))
    L.append("")

    # ── 행평균 대조 ──────────────────────────────────────────────────────
    L += ["-" * 86, "행평균 (스테이지 k 시점의 task 0..k 평균)", "-" * 86,
          f"{'stage':<16}" + "".join(f"{k:>7}" for k in range(K))]
    def rowavg(c, k):
        v = [c[(k, t)] for t in range(k + 1) if (k, t) in c]
        return sum(v) / len(v) if len(v) == k + 1 else None
    for name, c, _ in rows + refs:
        L.append(f"{name:<16}" + "".join(
            (f"{rowavg(c,k):7.1f}" if rowavg(c, k) is not None else f"{'':>7}")
            for k in range(K)))
    L.append("")

    # ── task1 궤적 ───────────────────────────────────────────────────────
    L += ["-" * 86, "task1 궤적 — 10 태스크에서 매번 무너지던 열", "-" * 86,
          f"{'stage':<16}" + "".join(f"{k:>5}" for k in range(K))]
    for name, c, _ in rows + refs:
        L.append(f"{name:<16}" + "".join(
            (f"{c[(k,1)]:5.0f}" if (k, 1) in c else f"{'':>5}") for k in range(K)))
    L.append("")

    # ── 습득(대각) ───────────────────────────────────────────────────────
    L += ["-" * 86, "습득 (대각)", "-" * 86,
          f"{'stage':<16}" + "".join(f"{k:>5}" for k in range(K))]
    for name, c, _ in rows + refs:
        L.append(f"{name:<16}" + "".join(
            (f"{c[(k,k)]:5.0f}" if (k, k) in c else f"{'':>5}") for k in range(K)))
    L.append("")

    # ── 팔별 전표 ────────────────────────────────────────────────────────
    for (name, d, desc), (nm, c, s) in zip(
            [a for a in ARMS if stats(read_csv(RES / a[1])) is not None], rows):
        L += ["-" * 86,
              f"{name}   ({d})   {'완료' if s['done'] else f'진행 중 {s[chr(39)+chr(39)] if False else s['filled']}/55칸'}",
              f"  {desc}", "-" * 86,
              f"{'after':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(K))]
        for k in range(K):
            L.append(f"{k:<8}" + "".join(
                (f"{c[(k,t)]:6.0f}" if (k, t) in c else f"{'':>6}") for t in range(K)))
        cfg = RES / d / "l2_config.json"
        if cfg.exists():
            try:
                j = json.loads(cfg.read_text())
                keys = [k for k in ("xt_mode", "codebook_k", "n_pairs", "h_scale",
                                    "grad_enable", "ridge_rho", "bayes_temp",
                                    "lambda_level") if k in j]
                L.append("  설정  " + "  ".join(f"{k}={j[k]}" for k in keys))
            except Exception:
                pass
        L.append("")

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n... saved -> {OUT}")


if __name__ == "__main__":
    main()
