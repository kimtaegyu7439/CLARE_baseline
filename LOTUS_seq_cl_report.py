#!/usr/bin/env python
"""LOTUS 셋팅 순차 CL(seq-FT) 결과를 한 파일로 모은다 -> results/LOTUS_Seq_CL_SR.txt

    python LOTUS_seq_cl_report.py            결과를 모아 파일로 쓰고 요약을 출력
    python LOTUS_seq_cl_report.py --suites object,goal,50

읽는 것  results/lotus_seq_<suite>_seed<seed>/metrics.json  (L2_seq_cl.py 가 쓴다)
    R            (n_stage+1, n_task) SR 행렬. 아직 안 잰 칸은 null
    learned_at   태스크 t 를 배운 스테이지. base 태스크는 0
    metrics      AvgSR / FWT / NBT / AUC
    wall_hours   소요 시간

★ 완주하지 않은 런도 **빼지 않고** 표시한다. 채운 칸 수를 함께 적어 두면 중간 결과를
  완성본으로 오독하는 일이 없다. 시드 요약(평균)은 완주한 런만으로 낸다 — 반쯤 찬
  행렬의 AvgSR 은 다른 런과 같은 것을 재고 있지 않기 때문이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
KEYS = ("AvgSR", "FWT", "NBT", "AUC")


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", default="object,goal,50")
    p.add_argument("--seeds", default="7,42,1000")
    p.add_argument("--out", default=str(RES / "LOTUS_Seq_CL_SR.txt"))
    # 방법을 바꿔 끼울 수 있게 — 기본값은 seq-FT 라 기존 동작이 그대로다.
    p.add_argument("--prefix", default="lotus_seq_", help="결과 폴더 접두사")
    p.add_argument("--title", default="LOTUS 셋팅 순차 CL — seq-FT 기준선 (앵커·리플레이 없음)")
    p.add_argument("--loss_note", default="  손실      L = L_FM(현재). 과거 보존 장치가 없다.")
    p.add_argument("--tail_note", default="  · seq-FT 는 기준선이다. 이 표만으로 방법 간 우열을 말하지 않는다.")
    return p.parse_args()


def recompute(R, la, lifelong_only: bool) -> dict:
    """SR 행렬에서 지표를 다시 낸다. L2_V4.metrics 와 같은 정의.

        FWT_t = R[learned_at(t)][t]                   배운 직후
        NBT_t = mean_{k>learned} (FWT_t − R[k][t])     이후 잃은 양 (+면 망각)
        AUC_t = mean_{k>=learned} R[k][t]              시퀀스 내내 낸 평균
        AvgSR = 마지막 행 평균

    ★ lifelong_only 면 learned_at == 0 인 **base 태스크를 뺀다.**
      LOTUS 셋팅의 base 는 여러 태스크를 한꺼번에 joint 로 배운 구간이라,
      그 SR 은 순차 학습의 성능이 아니라 base 사전학습 성능이다. 섞어서 평균하면
      (libero_50 은 50 태스크 중 25 개가 base) 지표가 base 쪽으로 끌려간다.
    """
    idx = [t for t in range(len(la)) if (la[t] > 0 or not lifelong_only)]
    fwt, nbt, auc = [], [], []
    for t in idx:
        col = [R[k][t] for k in range(la[t], len(R))
               if t < len(R[k]) and R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    last = [R[-1][t] for t in idx if t < len(R[-1]) and R[-1][t] is not None]
    mean = lambda v: (sum(v) / len(v)) if v else None          # noqa: E731
    return {"AvgSR": mean(last), "FWT": mean(fwt), "NBT": mean(nbt), "AUC": mean(auc),
            "n_task": len(idx)}


def load(suite: str, seed: int, prefix: str = "lotus_seq_"):
    d = RES / f"{prefix}{suite}_seed{seed}"
    p = d / "metrics.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
    except Exception:
        return None
    R, la = m.get("R") or [], m.get("learned_at") or []
    if not R or not la:
        return None
    need = sum(1 for k in range(len(R)) for t in range(len(la)) if la[t] <= k)
    have = sum(1 for k in range(len(R)) for t in range(len(R[k])) if R[k][t] is not None)
    m["_have"], m["_need"] = have, need
    m["_done"] = have >= need > 0
    # lifelong 구간(base 제외)과 전체, 둘 다 다시 계산한다. metrics.json 에 저장된
    # metrics 는 전체 기준이라 metrics_all 과 일치해야 한다 — 어긋나면 정의가 바뀐 것이다.
    m["metrics_ll"] = recompute(m["R"], m["learned_at"], lifelong_only=True)
    m["metrics_all"] = recompute(m["R"], m["learned_at"], lifelong_only=False)
    return m


def stat(done: list, k: str, field: str = "metrics_ll"):
    """완주한 런들의 지표 k 에 대한 (평균, 표본표준편차, 최소, 최대).

    ★ 표본표준편차(ddof=1)다. 시드가 3 개뿐이라 이 편차 자체도 꽤 흔들린다 —
      "시드 간 재현성이 이 정도 폭"이라는 눈금이지, 신뢰구간이 아니다.
      시드가 1 개면 편차가 정의되지 않으므로 None 을 돌려준다.
    """
    v = [m[field][k] for _, m in done
         if isinstance((m.get(field) or {}).get(k), (int, float))]
    if not v:
        return None
    mean = sum(v) / len(v)
    sd = None
    if len(v) >= 2:
        sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    return mean, sd, min(v), max(v)


def matrix_block(m: dict) -> list[str]:
    R, la = m["R"], m["learned_at"]
    n_task = len(la)
    w = 4 if n_task > 20 else 6
    head = "stage" + "".join(f"{('t'+str(t)):>{w}}" for t in range(n_task))
    out = [head]
    for k, row in enumerate(R):
        cells = ""
        for t in range(n_task):
            v = row[t] if t < len(row) else None
            cells += f"{'':>{w}}" if v is None else f"{v:>{w}.0f}"
        out.append(f"{k:<5d}" + cells)
    return out


def main():
    a = parse()
    suites = [s.strip() for s in a.suites.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    L: list[str] = []
    add = L.append

    add("=" * 94)
    add(a.title)
    add("=" * 94)
    add("")
    add(a.loss_note)
    add("  프로토콜  base 체크포인트에서 출발, lifelong stage 당 10,000 step")
    add("  평가      SR 칸당 20 rollout, eval seed 7 고정 (train seed 만 바꾼다)")
    add("  행 = 스테이지 k 까지 배운 뒤, 열 = 평가 태스크. 단위 %. 빈 칸 = 아직 안 배움/안 잼")
    add("")
    add("  지표  FWT↑ 배운 직후 성능 / NBT↓ 이후 떨어진 양 / AUC↑ 전 구간 평균 / AvgSR 마지막 행 평균")
    add("")
    add("  ★ 아래 지표는 모두 **lifelong 구간(stage 1 이후에 배운 태스크)만**으로 낸다.")
    add("    base 태스크는 여러 개를 한꺼번에 joint 로 배운 구간이라 그 SR 은 순차 학습의")
    add("    성능이 아니라 base 사전학습 성능이다. 참고로 base 를 포함한 값도 괄호로 병기한다.")
    add("")

    summary: dict[str, list] = {}
    for suite in suites:
        runs = [(s, load(suite, s, a.prefix)) for s in seeds]
        runs = [(s, m) for s, m in runs if m is not None]
        add("-" * 94)
        add(f"libero_{suite}")
        add("-" * 94)
        if not runs:
            add("  결과 없음"); add(""); continue
        for seed, m in runs:
            tag = "완주" if m["_done"] else f"진행 중 {m['_have']}/{m['_need']}칸"
            wall = m.get("wall_hours")
            add(f"  train seed {seed}   ({tag}"
                + (f", {wall:.2f}h)" if isinstance(wall, (int, float)) else ")"))
            for line in matrix_block(m):
                add("    " + line)
            ll, al = m["metrics_ll"], m["metrics_all"]
            if any(isinstance(ll.get(k), (int, float)) for k in KEYS):
                add(f"    지표 (lifelong {ll['n_task']}태스크)  " + "   ".join(
                    f"{k} {ll[k]:.1f}" for k in KEYS if isinstance(ll.get(k), (int, float))))
                add(f"      base 포함 {al['n_task']}태스크    " + "   ".join(
                    f"{k} {al[k]:.1f}" for k in KEYS if isinstance(al.get(k), (int, float))))
            add("")
        done = [(s, m) for s, m in runs if m["_done"]]
        if len(done) >= 2:
            add(f"  시드 요약 (완주 {len(done)}개: {', '.join(str(s) for s, _ in done)})")
            for k in KEYS:
                st = stat(done, k)
                if st:
                    mean, sd, lo, hi = st
                    pm = f" ± {sd:4.1f}" if sd is not None else "        "
                    add(f"    {k:<6s} {mean:6.1f}{pm}   "
                        f"최소 {lo:5.1f}   최대 {hi:5.1f}")
            add("")
        summary[suite] = done

    add("=" * 94)
    add("스위트 요약 (완주한 시드의 평균 ± 시드 간 표본표준편차)")
    add("=" * 94)
    # ★ suite 칸은 15 — 'libero_object' 가 13 자라 10 으로 두면 뒤 칸이 밀린다.
    # ★ 'lifelong' 칸은 이 표가 **stage 1 이후에 배운 태스크만**으로 낸 값임을
    #   표 안에서 바로 알 수 있게 한다. 숫자가 그 태스크 수다 (base 는 빠졌다).
    add(f"  {'suite':<15}{'시드':>4}{'lifelong':>9}" + "".join(f"{k:>14}" for k in KEYS))
    for suite in suites:
        done = summary.get(suite) or []
        if not done:
            add(f"  {('libero_'+suite):<15}{0:>4}{'-':>9}"
                + "".join(f"{'-':>14}" for _ in KEYS)); continue
        n_ll = (done[0][1].get("metrics_ll") or {}).get("n_task", "-")
        row = f"  {('libero_'+suite):<15}{len(done):>4}{n_ll:>9}"
        for k in KEYS:
            st = stat(done, k)
            if not st:
                row += f"{'-':>14}"; continue
            mean, sd, _lo, _hi = st
            row += f"{mean:>8.1f} ±{sd:4.1f}" if sd is not None else f"{mean:>8.1f}      "
        add(row)
    add("")
    add("주의")
    add("  · SR 은 시뮬 롤아웃 성공률이다. 칸당 20 회라 ±5%p 단위로만 움직인다.")
    add("  · ± 는 완주한 train seed 간 표본표준편차(ddof=1)다. 시드가 3 개뿐이라 이 값 자체도")
    add("    흔들린다 — 재현성의 눈금으로 읽어라. 신뢰구간이 아니고, 롤아웃 분산도 섞여 있다.")
    add(a.tail_note)
    add("")

    txt = "\n".join(L)
    Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
