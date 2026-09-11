#!/usr/bin/env python
"""L2_codebook v3(--grad_enable) 스위트별 SR — results/L2_codebook_grad_SR_matrix.txt.

같은 팔(v3)을 서로 다른 벤치마크에서 돌린 결과만 모은다. L2_SR_matrix.txt 는
libero_spatial 안에서 **팔끼리** 비교하는 표이고, 이 파일은 한 팔을 **스위트끼리**
비교하는 표다. 섞으면 안 되므로 파일을 나눴다.

지표 정의는 l2_family_report.py / tools/plot_sr_matrix.py 와 동일하다
(LIBERO 벤치마크, Liu et al. 2023).

    python L2_codebook_grad_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "L2_codebook_grad_SR_matrix.txt"
K = 10

# (표시명, 결과 디렉터리, 데이터셋 설명)
RUNS = [
    ("libero_spatial", "l2_codebook_k96_grad",
     "같은 장면·같은 물체, 배치만 다른 10 태스크"),
    ("libero_10", "l2_codebook_k96_grad_libero10",
     "LIBERO-LONG. 장기 지평 조작 10 태스크 (난도 높음)"),
    ("libero_goal", "l2_codebook_k96_grad_libero_goal",
     "같은 장면·같은 물체, 목표(goal)만 다른 10 태스크"),
    ("libero_object", "l2_codebook_k96_grad_libero_object",
     "같은 장면·같은 배치, 물체만 다른 10 태스크"),
    ("libero_spatial (seed41)", "l2_codebook_k96_grad_seed41",
     "seed 42 실행의 반복. 팔이 아니라 **시드 변동**을 재기 위한 것"),
    ("libero_10 (seed41)", "l2_codebook_k96_grad_libero10_seed41",
     "seed 42 실행의 반복. 팔이 아니라 **시드 변동**을 재기 위한 것"),
]


def read_csv(d: Path) -> dict[tuple[int, int], float]:
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


def last_full_row(c) -> int | None:
    best = None
    for k in range(K):
        if all((k, t) in c for t in range(k + 1)):
            best = k
    return best


def metrics(c, kmax):
    """LIBERO 지표. l2_family_report.libero_metrics 와 같은 식이다.

        FWT_t = R[t][t]                       갓 배운 직후
        NBT_t = mean_{k>t} (R[t][t] − R[k][t])  배운 뒤 잃은 양 (+면 망각)
        AUC_t = mean_{k>=t} R[k][t]           시퀀스 내내 낸 평균
    """
    fwt, nbt, auc = [], [], []
    for t in range(kmax + 1):
        col = [c[(k, t)] for k in range(t, kmax + 1) if (k, t) in c]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    last = [c[(kmax, t)] for t in range(kmax + 1)]
    diag = [c[(t, t)] for t in range(kmax + 1)]
    mean = lambda v: (sum(v) / len(v)) if v else None          # noqa: E731
    return {"avg": mean(last), "fwt": mean(fwt), "nbt": mean(nbt),
            "auc": mean(auc),
            "bwt": (sum(last[t] - diag[t] for t in range(kmax)) / kmax
                    if kmax else None)}


def cfg_of(d: Path) -> dict:
    p = d / "l2_codebook_config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    L = ["=" * 86,
         "L2_codebook v3 (--grad_enable) — 스위트별 SR 행렬",
         "=" * 86, "",
         "같은 팔을 서로 다른 벤치마크에서 돌린 것만 모았다.",
         "팔끼리 비교하는 표는 results/L2_SR_matrix.txt 에 따로 있다 (libero_spatial 전용).", "",
         "공통 설정  --grad_enable --chunk_backward · K=96 · n_pairs=8000 · h_scale=1.0",
         "           xt_mode=teacher · λ_level=3.0 · ridge_rho=0.05 · grad_min_frames=24",
         "           5000 step/task · 45 에피소드(뒤 5개 hold-out) · 칸당 20 rollout",
         "           ★ 시드는 블록마다 다르다. 각 블록의 '설정' 줄을 보고 읽어야 한다.", "",
         "지표 (LIBERO 벤치마크 Liu et al. 2023 — l2_family_report.py 와 같은 정의)",
         "  R[k][t] = 스테이지 k 체크포인트를 태스크 t 에서 잰 SR (t<=k 에서만 정의)",
         "  AvgSR = 마지막 행 평균                          높을수록 좋음",
         "  FWT   = mean_t R[t][t]                          높을수록 좋음 (= 대각/'습득')",
         "  NBT   = mean_t mean_{k>t} (R[t][t] − R[k][t])   **낮을수록** 좋음 (+면 망각)",
         "  AUC   = mean_t mean_{k>=t} R[k][t]              높을수록 좋음",
         "  BWT   = mean_t (R[마지막][t] − R[t][t])         높을수록 좋음", ""]

    rows = []
    for name, d, desc in RUNS:
        c = read_csv(RES / d)
        kmax = last_full_row(c)
        if kmax is None:
            L += [f"[{name}] 결과 없음 ({d})", ""]
            continue
        s = metrics(c, kmax)
        done = kmax == K - 1
        filled = len(c)
        rows.append((name, s, done, kmax, desc))

        L += ["-" * 86,
              f"{name}   ({d})   "
              + ("완료 55/55칸" if done else f"진행 중 {filled}/55칸 (행{kmax})"),
              f"  {desc}", "-" * 86,
              f"{'after':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(K))]
        for k in range(K):
            L.append(f"{k:<8}" + "".join(
                (f"{c[(k,t)]:6.0f}" if (k, t) in c else f"{'':>6}") for t in range(K)))
        # 행이 하나뿐이면(스테이지 0) NBT/BWT 가 정의되지 않아 None 이다.
        f1 = lambda v: "-" if v is None else f"{v:.1f}"          # noqa: E731
        fs = lambda v: "-" if v is None else f"{v:+.1f}"         # noqa: E731
        L.append(f"  지표  AvgSR {f1(s['avg'])}   FWT↑ {f1(s['fwt'])}"
                 f"   NBT↓ {fs(s['nbt'])}   AUC↑ {f1(s['auc'])}   BWT↑ {fs(s['bwt'])}"
                 + ("" if done else f"   (행{kmax}까지 부분 집계)"))
        cf = cfg_of(RES / d)
        if cf:
            L.append(f"  설정  seed={cf.get('seed', 42)}  "
                     f"codebook_k={cf.get('codebook_k')}  n_pairs={cf.get('n_pairs')}  "
                     f"h_scale={cf.get('h_scale')}  grad_enable={cf.get('grad_enable')}  "
                     f"ridge_rho={cf.get('ridge_rho')}  lambda_level={cf.get('lambda_level')}")
        L.append("")

    # ── 나란히 ───────────────────────────────────────────────────────────
    if rows:
        L += ["-" * 86, "스위트 대조", "-" * 86,
              f"{'스위트':<18}{'AvgSR':>8}{'FWT↑':>8}{'NBT↓':>8}{'AUC↑':>8}{'BWT↑':>8}  상태"]
        cell = lambda v, sg=False: (f"{'-':>8}" if v is None            # noqa: E731
                                    else (f"{v:+8.1f}" if sg else f"{v:8.1f}"))
        for name, s, done, kmax, _ in rows:
            L.append(f"{name:<18}" + cell(s['avg']) + cell(s['fwt'])
                     + cell(s['nbt'], True) + cell(s['auc']) + cell(s['bwt'], True)
                     + "  " + ("완료" if done else f"진행 중 (행{kmax})"))
        L.append("")

    L += ["읽을 때 주의",
          "  · 두 스위트는 난도가 다르다. libero_10(LIBERO-LONG)은 장기 지평 과제라",
          "    AICP baseline 기준으로도 libero_spatial 의 60~70% 수준에서 논다.",
          "    스위트 사이의 SR 차이를 팔의 성능 차이로 읽으면 안 된다.",
          "  · 칸당 20 롤아웃이라 개별 칸의 이항 표준오차는 ±11%p, 10칸 평균도 ±3.5%p다.",
          "    시드 하나(42)뿐이라 수 %p 차이는 시드 반복 없이 확정할 수 없다.",
          "  · --chunk_backward 는 앵커 그래디언트를 과거 태스크에 대한 **합**으로 만든다",
          "    (끄면 평균). 스테이지 9 에서 실효 가중이 λ_level 의 9 배다. 두 실행 모두",
          "    켠 조건이라 서로 비교는 되지만, λ_level=3.0 만으로 재현하려 하면 안 맞는다.", ""]

    OUT.write_text("\n".join(L) + "\n")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
