#!/usr/bin/env python
"""K1 결과 정리 — results/K_mod_none_null.txt 를 만들고 B_mod_none_null.txt 에도 반영.

    python K_report.py

results/K1*/ 의 sr_matrix.csv 와 metrics.json 을 읽어 표를 만든다. 아직 끝나지
않은 팔은 채워진 칸까지만 싣고 "진행 중" 으로 표시한다.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"

# 표시명 -> (결과 디렉토리, 스위트, 태스크 수)
ARMS = [
    ("K1  spatial 4task", "K1", "libero_spatial", 4),
    ("K1  spatial 10task", "K1_spatial_10task", "libero_spatial", 10),
    ("K1  goal 10task", "K1_goal_10task", "libero_goal", 10),
    ("K1  libero_10 10task", "K1_l10_10task", "libero_10", 10),
    ("K1  object 10task", "K1_object_10task", "libero_object", 10),
]

# 같은 프로토콜의 ER 기준선. spatial 은 기존 결과, 나머지는 run_ER_suite.sh 산출물.
ER_ARMS = [
    ("ER  spatial 10task", "ER_10task", "libero_spatial", 10),
    ("ER  goal 10task", "ER_libero_goal_10task", "libero_goal", 10),
    ("ER  libero_10 10task", "ER_libero_10_10task", "libero_10", 10),
    ("ER  object 10task", "ER_libero_object_10task", "libero_object", 10),
]
MARK = "##### K1 · 스위트별 10 태스크 (자동 생성 — K_report.py) #####"


def read_matrix(d: Path):
    """sr_matrix.csv -> {(stage, task): SR}, K. 없으면 탭 구분 *_SR.txt 를 본다."""
    src = d / "sr_matrix.csv"
    if not src.exists():
        for alt in sorted(d.glob("*SR*.txt")) if d.is_dir() else []:
            cells, K = {}, 0
            for line in alt.read_text().splitlines():
                f = line.split("\t")
                if len(f) < 2 or not f[0].strip().isdigit():
                    continue
                k = int(f[0]); K = max(K, len(f) - 1)
                for t, v in enumerate(f[1:]):
                    if v.strip():
                        cells[(k, t)] = float(v)
            if cells:
                return cells, K
        # 진행 중인 ER — 표가 아직 안 쓰였으면 프로브 jsonl 에서 직접 읽는다
        jl = d / "er_results.jsonl"
        if jl.exists():
            import json as _j
            cells, K = {}, 0
            for line in jl.read_text().splitlines():
                try:
                    r = _j.loads(line)
                except Exception:
                    continue
                if r.get("run_tag") == "er" and r.get("sr") is not None:
                    cells[(r["stage"], r["probe_task"])] = float(r["sr"])
                    K = max(K, r["stage"] + 1, r["probe_task"] + 1)
            if cells:
                return cells, K
        return None, 0
    cells, K = {}, 0
    for line in src.read_text().splitlines():
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
    """최종행 평균 / BWT / 습득 / 채워진 칸 수. 최종행이 미완이면 avg 는 None."""
    total = K * (K + 1) // 2
    filled = len(cells)
    diag = [cells.get((k, k)) for k in range(K)]
    acq = [v for v in diag if v is not None]
    last = [cells.get((K - 1, t)) for t in range(K)]
    done = all(v is not None for v in last)
    avg = sum(last) / K if done else None
    bwt = None
    if done and len(acq) == K:
        bwt = sum(last[t] - diag[t] for t in range(K - 1)) / max(1, K - 1)
    return {"filled": filled, "total": total, "avg": avg, "bwt": bwt,
            "acq": (sum(acq) / len(acq)) if acq else None, "done": filled >= total}


def fmt_matrix(cells, K, width=6):
    L = ["after\\task" + "".join(f"{t:>{width}}" for t in range(K))]
    for k in range(K):
        row = "".join(
            (f"{cells[(k,t)]:{width}.0f}" if (k, t) in cells else f"{'.':>{width}}")
            for t in range(K))
        L.append(f"{k:>10}{row}")
    return L


HEAD = """\
====================================================================================================
K_mod_none_null — K1 (공유기저 분위수 수송 앵커) 성능 기록
====================================================================================================

K1 이 무엇인가
  R13(가우시안 샘플 앵커)에서 **관측 합성 블록만** 바꾼 팔이다. 손실/teacher/
  스케줄/평가는 R13 과 한 줄도 다르지 않다 (results/K1/k1_loss_diff.txt 에
  R10.loss 대비 실제 diff 를 남긴다).

    R13   z ~ N(0,I)                        등방 가우시안 표본
          b_j = mu_j[τ] + sigma_j[τ]·z      태스크별 대각 가우시안으로 되채색

    K1    w    = Wᵀ(o − c0)                 공유 PCA 기저(r=256)로 회전
          p_i  = F_new,τ,i(w_i)             현재 태스크 분위수표의 CDF (Q=16)
          w'_i = F_j,τ,i⁻¹(p_i)             과거 태스크 표의 역CDF
          b_j  = c0 + W w' + res − m⊥_new[τ] + m⊥_j[τ]

동기
  results/R10_tsne/pca_tasks_bin5.png 실측에서 태스크 간 중심거리가 태스크 내
  산포의 3.13 배였다. 지배적 변동 축이 태스크 정체성이라 location-scale 가정
  (o = mu_j + sigma_j·ε, ε 는 태스크 공통)이 성립하지 않는다. 분위수 사상은
  그 가정을 쓰지 않고 주변분포를 그대로 옮기며, 좌표별 CDF 를 통과해도 순위
  구조(copula)가 보존되므로 좌표 간 상관이 살아남는다.

상속한 설정 (R13 과 동일)
  rolling teacher 1개, level 앵커만(structure 없음), λ_level=3, λ_anchor=1,
  anchor_norm=mean, n_bins=10, h = 0.1·median‖o−mu[τ]‖, p_drop=0, guidance_w=1,
  5000 step/task, 45 에피소드, batch 32, seed 42, 칸당 20 롤아웃, chunk_backward.

저장물 (과거 원시 데이터는 학습 중 로드하지 않는다)
  공유기저 shared_basis.pt   W (3072,256) + c0 (3072,)      3.1 MB   1개
  태스크별 stats/task{k}.pt  qtab (10,256,16) + m⊥ (10,3072)  287 KB   태스크당
  rolling teacher            학습 파라미터 스냅샷 1개                  R13 과 동일

ablation 축 (플래그. 기본값이 K1)
  --marginal  quantile(기본) / zscore     주변분포 충실도
  --basis     shared_pca(기본) / identity 좌표계
  --iid_sample                            copula 를 끊는 negative control
"""


def main() -> None:
    L = [HEAD]
    rows, mats = [], []
    for name, d, suite, K in ARMS:
        cells, Kr = read_matrix(RES / d)
        if cells is None:
            rows.append((name, suite, K, None)); continue
        s = summarize(cells, max(Kr, K))
        rows.append((name, suite, K, s))
        mats.append((name, suite, cells, max(Kr, K), s))

    L.append("-" * 100)
    L.append(f"{'팔':<24}{'스위트':<16}{'태스크':>7}{'칸':>10}{'AvgSR':>9}{'BWT':>8}{'습득':>8}")
    L.append("-" * 100)
    for name, suite, K, s in rows:
        if s is None:
            L.append(f"{name:<24}{suite:<16}{K:>7}{'미시작':>10}")
            continue
        avg = f"{s['avg']:.1f}" if s["avg"] is not None else "진행중"
        bwt = f"{s['bwt']:+.1f}" if s["bwt"] is not None else "—"
        acq = f"{s['acq']:.1f}" if s["acq"] is not None else "—"
        L.append(f"{name:<24}{suite:<16}{K:>7}{s['filled']:>5}/{s['total']:<4}"
                 f"{avg:>9}{bwt:>8}{acq:>8}")
    L.append("")
    L.append("참고값 (같은 프로토콜, p_drop=0 / anchor_agg=sum — results/B_mod_none_null.txt)")
    L.append("  libero_spatial 4 task    ER 93.8   R12 93.8   R15 91.2   R11 87.5   "
             "R14 86.2   R10 85.0   R13 85.0   joint(상한) 95.5")
    L.append("  libero_spatial 10 task   ER 86.0   R13 79.5   R12 73.5   R11 68.0   R10 67.0")
    L.append("  libero_goal / libero_object / libero_10 는 이 프로토콜의 10 태스크 ER 을")
    L.append("  아직 재지 않았다. results/ER_libero_40_SR.txt 는 40 태스크 연결 시퀀스라")
    L.append("  같은 조건이 아니므로 비교값으로 쓰지 않는다.")
    L.append("")

    for name, suite, cells, K, s in mats:
        L.append("=" * 100)
        st = "완료" if s["done"] else "진행 중"
        head = f"{name}   ({suite}, {K} task, {s['filled']}/{s['total']} 칸, {st})"
        if s["avg"] is not None:
            head += f"   AvgSR {s['avg']:.1f}"
            if s["bwt"] is not None:
                head += f"   BWT {s['bwt']:+.1f}"
            if s["acq"] is not None:
                head += f"   습득 {s['acq']:.1f}"
        L.append(head)
        L.append("=" * 100)
        L += fmt_matrix(cells, K)
        L.append("")

    out = RES / "K_mod_none_null.txt"
    out.write_text("\n".join(L) + "\n")
    print(f"saved -> {out}  ({len(L)} lines)")

    # ── B_mod_none_null.txt 의 스위트 블록 갱신 ─────────────────────────────
    er = {}
    for name, d, suite, K in ER_ARMS:
        c, Kr = read_matrix(RES / d)
        er[suite] = summarize(c, max(Kr, K)) if c else None
    B = ["", MARK, "",
         "libero_spatial 외 스위트의 10 태스크. 학습 조건은 spatial 과 같다",
         "(5000 step/task, 45 에피소드, seed 42, 칸당 20 롤아웃, p_drop=0).",
         "ER 은 배치 32 = 현재 24 + 버퍼 8, 과거 태스크당 5 에피소드.", "",
         f"{'스위트':<18}{'K1 AvgSR':>10}{'K1 BWT':>9}{'K1 습득':>9}"
         f"{'ER AvgSR':>10}{'ER BWT':>9}{'ER 습득':>9}{'K1−ER':>9}",
         "-" * 84]
    for name, d, suite, K in ARMS[1:]:
        c, Kr = read_matrix(RES / d)
        k1s = summarize(c, max(Kr, K)) if c else None
        e = er.get(suite)
        def f(v, sign=False):
            if v is None: return "—"
            return f"{v:+.1f}" if sign else f"{v:.1f}"
        d1 = (f"{k1s['avg'] - e['avg']:+.1f}"
              if (k1s and e and k1s["avg"] is not None and e["avg"] is not None) else "—")
        B.append(f"{suite:<18}"
                 f"{f(k1s and k1s['avg']):>10}{f(k1s and k1s['bwt'], True):>9}"
                 f"{f(k1s and k1s['acq']):>9}"
                 f"{f(e and e['avg']):>10}{f(e and e['bwt'], True):>9}"
                 f"{f(e and e['acq']):>9}{d1:>9}")
    B += ["", "칸별 SR 행렬은 results/K_mod_none_null.txt 와 "
              "results/ER_<suite>_10task/SR.txt 에 있다.", ""]
    bp = RES / "B_mod_none_null.txt"
    if bp.exists():
        txt = bp.read_text()
        txt = txt.split(MARK)[0].rstrip("\n") if MARK in txt else txt.rstrip("\n")
        bp.write_text(txt + "\n" + "\n".join(B) + "\n")
        print(f"saved -> {bp}  (스위트 블록 갱신)")
    for name, suite, K, s in rows:
        if s is None:
            print(f"  {name:<24} 미시작")
        else:
            a = f"{s['avg']:.1f}" if s["avg"] is not None else "진행중"
            print(f"  {name:<24} {s['filled']:>3}/{s['total']:<3} 칸  AvgSR {a}")


if __name__ == "__main__":
    main()
