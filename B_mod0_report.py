#!/usr/bin/env python
"""results/mod0/*(p_drop=0) 를 모아 B_mod_none_null.txt 로.

B_mod.txt 와 같은 구성이다. 비교 축만 다르다:
    B_mod.txt            앵커 집계  추첨 -> 합        (p_drop 0.1 고정)
    B_mod_none_null.txt  condition dropout  0.1 -> 0  (anchor_agg=sum 고정)
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
K = 4

# 표시명 -> (p_drop=0 dir, p_drop=0.1 dir, 최초 실행 dir)
ARMS = [
    ("B1 λ1",   "mod0/B1_lam1",  "mod/B1_lam1",  "B1"),
    ("B1 λ3",   "mod0/B1_lam3",  "mod/B1_lam3",  "B1_lam3"),
    ("B1 λ10",  "mod0/B1_lam10", "mod/B1_lam10", "B1_lam10"),
    ("B1 λ30",  "mod0/B1_lam30", "mod/B1_lam30", "B1_lam30"),
    ("B2 λ1",   "mod0/B2_lam1",  "mod/B2_lam1",  "B2"),
    ("B2 λ3",   "mod0/B2_lam3",  "mod/B2_lam3",  "B2_lam3"),
    ("B2 λ10",  "mod0/B2_lam10", "mod/B2_lam10", "B2_lam10"),
    ("B2 λ30",  "mod0/B2_lam30", "mod/B2_lam30", "B2_lam30"),
    ("B8 λ1",   "mod0/B8_lam1",  "mod/B8_lam1",  "B8"),
    ("B8 λ3",   "mod0/B8_lam3",  "mod/B8_lam3",  "B8_lam3"),
    ("B8 λ10",  "mod0/B8_lam10", "mod/B8_lam10", "B8_lam10"),
    ("B7",      "mod0/B7",       "mod/B7",       "B7"),
    ("B9 1023", "mod0/B9_1023",  "mod/B9_1023",  "B9_1023"),
    ("B9 0321", "mod0/B9_0321",  "mod/B9_0321",  "B9_0321"),
    ("B9 2103", "mod0/B9_2103",  "mod/B9_2103",  "B9_2103"),
    ("B9 3210", "mod0/B9_3210",  "mod/B9_3210",  "B9_3210"),
    # R10 — 수송 좌표 level+structure 앵커. p_drop=0 이라 이 표에 속한다.
    # 비교 상대가 없으므로 p.1/원본 칸은 비고, 세로 참고는 B8λ3(= 같은 λ=3 계열)다.
    ("R10",     "R10",           None,           None),
    ("R11",     "R11",           None,           None),
    ("R12",     "R12",           None,           None),
    ("R13",     "R13",           None,           None),
    ("R14",     "R14",           None,           None),
    ("R15",     "R15",           None,           None),
]


def load(d):
    if d is None:
        return None
    p = REPO / "results" / d / "metrics.json"
    if not p.exists():
        return None
    try:
        m = json.load(open(p))
    except Exception:
        return None
    return m if m.get("AvgSR_final") is not None else None


def order_of(m):
    return m.get("task_order") or list(range(K))


def matrix(d, m):
    order = order_of(m)
    name = "sr_matrix_bytask.csv" if order != list(range(K)) else "sr_matrix.csv"
    p = REPO / "results" / d / name
    if not p.exists():
        p = REPO / "results" / d / "sr_matrix.csv"
    if not p.exists():
        return None
    cells = {}
    for line in p.read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split(",")
        if not f or not f[0].strip().isdigit():
            continue
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(int(f[0]), t)] = float(v)
    return cells


def final_row(m):
    fr = m.get("final_row") or {}
    order = order_of(m)
    if order == list(range(K)):
        return [fr.get(f"task{i}") for i in range(K)]
    pos = {t: i for i, t in enumerate(order)}
    return [fr.get(f"task{pos[t]}") for t in range(K)]


def acq(m):
    ls = m.get("learning_sr") or {}
    vs = list(ls.values())
    return sum(vs) / len(vs) if vs and all(v is not None for v in vs) else None


L = ["=" * 100,
     "B_mod_none_null — condition dropout 을 끈 재실행 (p_drop 0.1 -> 0)",
     "=" * 100, "",
     "무엇이 바뀌었나",
     "  p_drop=0.1 은 매 샘플 10% 확률로 명령어를 ∅(빈 문자열)로 바꾸되 target 은",
     "  현재 태스크 정답 그대로 둔다. 즉 v(o,∅) -> v*_k 를 명시적으로 학습시킨다.",
     "",
     "  results/B_default 실측: stage 3 에서 d_3 = ‖v(o,ℓ_3)−v(o,∅)‖/‖v*‖ 가",
     "    B2λ3   0.01   (d_3<0.05 인 지점이 98.4%)   <- 무조건부 필드 = 현재 태스크",
     "    ER     0.17   (0.4%)                       <- ∅ 이 누구의 것도 아님",
     "  새 태스크가 올 때마다 기본 필드를 통째로 빼앗고, 과거 태스크는 매 스테이지",
     "  0.4 크기의 오프셋을 새 기준선 위에서 다시 세워야 한다.",
     "",
     "  p_drop 은 원래 classifier-free guidance 용인데 w 스윕에서 w>1 은 항상 손해였고",
     "  (B2λ3: w=1.00 -> 80.0, w=1.25 -> 56.2), 평가는 w=1 이라 ∅ 경로를 타지도 않는다.",
     "",
     "  그래서 p_drop 을 0 으로 두고 전부 다시 쟀다. 그 외는 B_mod 와 동일하다:",
     "  anchor_agg=sum, 5000 steps/task, 45 에피소드, 배치 32, seed 42, 칸당 20 롤아웃.",
     "",
     "열 설명   p0 = p_drop 0 (이 실험)   p.1 = p_drop 0.1 (results/B_mod.txt)",
     "          원본 = 최초 실행 (p_drop 0.1 + 앵커 추첨, results/B_compare.txt)",
     "",
     "R10  수송 좌표 위의 level+structure 앵커. 앵커 좌표를 현재 관측 o 가 아니라",
     "     o 를 과거 태스크 분포로 옮긴 b_j = μ_j[τ] + σ_j[τ]·z 에 둔다. 값(level)과",
     "     방향미분(structure)을 함께 맞춘다. teacher 는 태스크별 frozen, Ĝ 가중 off,",
     "     ℓ-swap 앵커 off. 과거 데이터는 저장하지 않는다(통계 + teacher 뿐).",
     "     p_drop=0, w=1 이라 이 표의 다른 팔들과 조건이 같다. 상세는 R10.py 참조.",
     "앵커 좌표를 어디서 얻는가 — 2x2",
     "              수송 z=(o−mu_new)/sigma_new     샘플링 z~N(0,I)",
     "  level 만          R12                          R13",
     "  +structure        R10 / R11                    R14 / R15",
     "  공통: b_j = mu_j[tau] + sigma_j[tau]·z, rolling teacher, 과거 명령어 l_j,",
     "        태스크별 (mu,sigma) 통계, p_drop=0, w=1.",
     "  수송은 현재 관측의 상관구조를 그대로 옮기고, 샘플링은 등방이라 그것이 없다.",
     "  results/R10_gauss 실측: 주변분포는 전 차원 가우시안이지만 차원 간 상관이",
     "  강하다(‖z‖²/d 산포가 독립 가정의 13배). 그 차이가 이 2x2 로 드러난다.",
     "",
     "R12  R10 에서 structure 항을 뺀 것. 수송된 점 b_j 에서 level 앵커만 건다.",
     "     b_j + h·u 에서의 두 번째 forward 가 사라져 스텝당 student forward 가",
     "     1+2K -> 1+K 로 절반이 된다. 나머지(rolling teacher, 과거 명령어 ℓ_j,",
     "     태스크별 mu/sigma, p_drop=0, w=1)는 R10 과 완전히 같다.",
     "",
     "R11  R10 에서 두 가지만 바꿨다. (1) structure 항을 제곱 -> L1 (원소 평균).",
     "     (2) 방향 u 를 상위 256 주성분 저계수 백색화. 원소별 표준화는 백색화가",
     "     아니다 — results/R10_gauss 실측에서 ‖z‖²/d 산포가 χ² 예측의 13배였고,",
     "     실제 λ_max 가 200 을 넘었다. 백색화 후 cos(u_R10,u_R11)=0.75 로 방향이",
     "     실제로 바뀌었다. 저장량은 태스크당 3.1MB 늘어난다.",
     "",
     "-" * 100,
     f"{'팔':>9}{'p0 AvgSR':>10}{'p.1 AvgSR':>11}{'Δ':>8}{'원본':>8}"
     f"{'p0 BWT':>9}{'p.1 BWT':>9}{'p0 습득':>9}{'p.1 습득':>10}",
     "-" * 100]

rows = []
for name, d0, d1, dref in ARMS:
    m0, m1, mr = load(d0), load(d1), load(dref)
    if m0 is None and m1 is None:
        continue
    f = lambda v, w=10, p=1: f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}}"
    a0 = None if m0 is None else m0["AvgSR_final"]
    a1 = None if m1 is None else m1["AvgSR_final"]
    ar = None if mr is None else mr["AvgSR_final"]
    d = (a0 - a1) if (a0 is not None and a1 is not None) else None
    L.append(f"{name:>9}{f(a0)}{f(a1, 11)}"
             + (f"{d:>+8.1f}" if d is not None else f"{'—':>8}")
             + f(ar, 8)
             + f(None if m0 is None else m0['BWT'], 9)
             + f(None if m1 is None else m1['BWT'], 9)
             + f(None if m0 is None else acq(m0), 9)
             + f(None if m1 is None else acq(m1), 10))
    rows.append((name, d0, m0, m1))

L += ["", "참고  seq-FT 35.0   ER 93.8   joint(상한) 95.5", ""]

# ── 순위 ────────────────────────────────────────────────────────────────────
rk = sorted([(m0["AvgSR_final"], m0["BWT"], n) for n, _, m0, _ in rows if m0],
            reverse=True)
if rk:
    L += ["=" * 100, "p_drop=0 순위", "=" * 100, "",
          f"{'순위':>4}{'팔':>11}{'AvgSR':>9}{'BWT':>9}", "-" * 40]
    for i, (a, b, n) in enumerate(rk, 1):
        L.append(f"{i:>4}{n:>11}{a:>9.1f}{b:>+9.1f}")
    L.append("")

# ── 최종 행 ─────────────────────────────────────────────────────────────────
L += ["=" * 100, "최종 행 (stage 3) — 태스크별", "=" * 100, "",
      f"{'팔':>9}{'':>5}" + "".join(f"{'task'+str(t):>9}" for t in range(K)),
      "-" * 100]
for name, d0, m0, m1 in rows:
    for tag, m in (("p0", m0), ("p.1", m1)):
        if m is None:
            continue
        L.append(f"{name:>9}{tag:>5}" + "".join(
            f"{v:>9.0f}" if v is not None else f"{'—':>9}" for v in final_row(m)))

# ── SR 행렬 ─────────────────────────────────────────────────────────────────
L += ["", "=" * 100, "SR 행렬 (p_drop=0) — 행 = 스테이지, 열 = 실제 task", "=" * 100]
for name, d0, m0, m1 in rows:
    if m0 is None:
        continue
    c = matrix(d0, m0)
    if not c:
        continue
    L += ["", "-" * 60,
          f"{name}   task_order: {','.join(str(t) for t in order_of(m0))}",
          "-" * 60,
          "after\\task " + "".join(f"{t:>7d}" for t in range(K))]
    for k in range(K):
        L.append(f"{k:>10d} " + "".join(
            f"{c[(k,t)]:7.0f}" if (k, t) in c else "      ." for t in range(K)))

missing = [n for n, _, m0, _ in rows if m0 is None]
if missing:
    L += ["", f"※ 아직 안 끝난 팔: {', '.join(missing)}"]

# ── 10 태스크 (별도 섹션. 위 표는 전부 4 태스크다) ─────────────────────────
K10 = 10


def cells_from_jsonl(path, tag="er"):
    c = {}
    if not path.exists():
        return c
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("run_tag") == tag and r.get("sr") is not None:
            c[(r["stage"], r["probe_task"])] = float(r["sr"])
    return c


def cells_from_csv(path):
    c = {}
    if not path.exists():
        return c
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split(",")
        if not f or not f[0].strip().isdigit():
            continue
        for t, v in enumerate(f[1:]):
            if v.strip():
                c[(int(f[0]), t)] = float(v)
    return c


TEN = [("ER",  cells_from_jsonl(REPO / "results/ER_10task/er_results.jsonl"))] + [
    (a, cells_from_csv(REPO / f"results/{a}_10task/sr_matrix.csv"))
    for a in ("R10", "R11", "R12", "R13", "R14", "R15")]
TEN = [(n, c) for n, c in TEN if c]

if TEN:
    L += ["", "=" * 100,
          "10 태스크 (libero_spatial task 0..9) — 위 표와 태스크 수가 다르다. 섞어 읽지 말 것.",
          "=" * 100, "",
          "설정은 4 태스크와 같다: 5000 steps/task, 45 에피소드, seed 42, 칸당 20 롤아웃,",
          "p_drop=0, anchor_agg=sum. R10/R11 은 --chunk_backward 를 켰다.", ""]
    for name, c in TEN:
        last = [c.get((K10 - 1, t)) for t in range(K10)]
        diag = [c.get((t, t)) for t in range(K10)]
        done = sum(1 for k in range(K10) for t in range(K10) if (k, t) in c)
        head = f"{name}   ({done}/55 칸"
        if all(v is not None for v in last + diag):
            avg = sum(last) / K10
            bwt = sum(last[i] - diag[i] for i in range(K10 - 1)) / (K10 - 1)
            head += f", 완료)   AvgSR {avg:.1f}   BWT {bwt:+.1f}   습득 {sum(diag)/K10:.1f}"
        else:
            head += ", 진행 중)"
        L += ["-" * 100, head, "-" * 100,
              "after\\task " + "".join(f"{t:>6d}" for t in range(K10))]
        for k in range(K10):
            row = "".join(f"{c[(k,t)]:6.0f}" if (k, t) in c else "     ." for t in range(K10))
            if row.strip(" ."):
                L.append(f"{k:>9d} " + row)
        L.append("")

rep = "\n".join(L)
(REPO / "results" / "B_mod_none_null.txt").write_text(rep)
print(rep)
print(f"\nsaved -> {REPO/'results'/'B_mod_none_null.txt'}")
