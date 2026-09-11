#!/usr/bin/env python
"""results/mod/* 를 모아 B_mod.txt 로. 구버전(results/*) 과 나란히 놓는다."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
K = 4

# 표시명 -> (신버전 dir, 구버전 dir)
ARMS = [
    ("B1 λ1",   "mod/B1_lam1",  "B1"),
    ("B1 λ3",   "mod/B1_lam3",  "B1_lam3"),
    ("B1 λ10",  "mod/B1_lam10", "B1_lam10"),
    ("B1 λ30",  "mod/B1_lam30", "B1_lam30"),
    ("B2 λ1",   "mod/B2_lam1",  "B2"),
    ("B2 λ3",   "mod/B2_lam3",  "B2_lam3"),
    ("B2 λ10",  "mod/B2_lam10", "B2_lam10"),
    ("B2 λ30",  "mod/B2_lam30", "B2_lam30"),
    ("B8 λ1",   "mod/B8_lam1",  "B8"),
    ("B8 λ3",   "mod/B8_lam3",  "B8_lam3"),
    ("B8 λ10",  "mod/B8_lam10", "B8_lam10"),
    ("B7",      "mod/B7",       "B7"),
    ("B9 1023", "mod/B9_1023",  "B9_1023"),
    ("B9 0321", "mod/B9_0321",  "B9_0321"),
    ("B9 2103", "mod/B9_2103",  "B9_2103"),
    ("B9 3210", "mod/B9_3210",  "B9_3210"),
]
REF = {"seq-FT": 35.0, "ER": 93.8}     # 비교 기준 (results/B_compare.txt)


def load(d):
    p = REPO / "results" / d / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def order_of(m):
    return m.get("task_order") or list(range(K))


def matrix(d, m):
    """행 = 스테이지, 열 = 실제 task. 순서를 바꾼 팔은 bytask 표를 쓴다."""
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
    fr = m.get("final_row")
    if not fr:
        return [None] * K
    order = order_of(m)
    if order == list(range(K)):
        return [fr.get(f"task{i}") for i in range(K)]
    # final_row 는 스테이지 기준이므로 실제 task 기준으로 되돌린다
    pos = {t: i for i, t in enumerate(order)}
    return [fr.get(f"task{pos[t]}") for t in range(K)]


L = ["=" * 96,
     "B_mod — 앵커 집계를 '스텝당 과거 하나 추첨' -> '과거 전부의 합' 으로 바꾼 재실행",
     "=" * 96, "",
     "무엇이 바뀌었나",
     "  구버전  j = rng.randrange(k)  한 스텝에서 배치 32개 **전부**가 같은 ℓ_j 를 받는다.",
     "          추첨이 몰리면 수백 스텝 동안 특정 과거만 앵커되고 나머지는 방치된다.",
     "          실측: 같은 배치에서 j 만 바꾼 앵커 손실이 0.0044 ~ 0.3609 (80배).",
     "  신버전  j = 0..k-1 전부의 합.  매 스텝 모든 과거가 한 번씩 들어간다.",
     "",
     "  ★ 합이므로 스테이지 k 의 실효 가중치가 k·λ 다(구버전은 기댓값 λ).",
     "    구버전 λ=3 이 신버전 λ=1 과 비슷한 세기다. 그래서 λ 를 다시 훑었다.",
     "",
     "그 외 하이퍼파라미터는 전부 동일하다. 5000 steps/task, 45 에피소드, 배치 32,",
     "seed 42, 칸당 20 롤아웃, p_drop 0.1.",
     "",
     "-" * 96,
     f"{'팔':>9}{'신 AvgSR':>10}{'구 AvgSR':>10}{'Δ':>8}{'신 BWT':>9}{'구 BWT':>9}"
     f"{'신 습득':>9}{'구 습득':>9}",
     "-" * 96]

rows = []
for name, new_d, old_d in ARMS:
    n, o = load(new_d), load(old_d)
    if n is None and o is None:
        continue
    def g(m, k_):
        return None if m is None else m.get(k_)
    na, oa = g(n, "AvgSR_final"), g(o, "AvgSR_final")
    nb, ob = g(n, "BWT"), g(o, "BWT")
    nl = None if n is None else (sum(n["learning_sr"].values()) / K
                                 if n.get("learning_sr") and all(v is not None for v in n["learning_sr"].values()) else None)
    ol = None if o is None else (sum(o["learning_sr"].values()) / K
                                 if o.get("learning_sr") and all(v is not None for v in o["learning_sr"].values()) else None)
    f = lambda v, w=10, p=1, s="": f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}}"
    d = (na - oa) if (na is not None and oa is not None) else None
    L.append(f"{name:>9}{f(na)}{f(oa)}" + (f"{d:>+8.1f}" if d is not None else f"{'—':>8}")
             + f(nb, 9) + f(ob, 9) + f(nl, 9) + f(ol, 9))
    rows.append((name, new_d, n, o))

L += ["", f"참고  seq-FT {REF['seq-FT']:.1f}   ER {REF['ER']:.1f}  (구버전 기준, results/B_compare.txt)",
      "", "=" * 96, "최종 행 (stage 3) — 태스크별", "=" * 96, "",
      f"{'팔':>9}{'':>4}" + "".join(f"{'task'+str(t):>9}" for t in range(K)),
      "-" * 96]
for name, new_d, n, o in rows:
    for tag, m in (("신", n), ("구", o)):
        if m is None:
            continue
        fr = final_row(m)
        L.append(f"{name:>9}{tag:>4}" + "".join(
            f"{v:>9.0f}" if v is not None else f"{'—':>9}" for v in fr))

L += ["", "=" * 96, "SR 행렬 (신버전) — 행 = 스테이지, 열 = 실제 task", "=" * 96]
for name, new_d, n, o in rows:
    if n is None:
        continue
    c = matrix(new_d, n)
    if not c:
        continue
    od = order_of(n)
    L += ["", "-" * 60,
          f"{name}   task_order: {','.join(str(t) for t in od)}",
          "-" * 60,
          "after\\task " + "".join(f"{t:>7d}" for t in range(K))]
    for k in range(K):
        L.append(f"{k:>10d} " + "".join(
            f"{c[(k,t)]:7.0f}" if (k, t) in c else "      ." for t in range(K)))

missing = [name for name, d, n, o in rows if n is None]
if missing:
    L += ["", f"※ 아직 안 끝난 팔: {', '.join(missing)}"]

rep = "\n".join(L)
(REPO / "results" / "B_mod.txt").write_text(rep)
print(rep)
print(f"\nsaved -> {REPO/'results'/'B_mod.txt'}")
