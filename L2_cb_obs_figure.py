#!/usr/bin/env python
"""코드북의 state -> observation 재현도를 태스크별 cos 분포로 그린다.

    python L2_cb_obs_figure.py --npz results/clare_v6_sqerr_goal_seed7/obs_fidelity_cos.npz

무엇을 보이려는 그림인가
    앵커는 과거 태스크 j 의 **상태** s̃_j 를 코드북에서 뽑은 다음, 그 상태에
    맞는 **관측** õ_j 를 Ridge 선형 사상으로 만들어 낸다.

        z  = (s − mean_s)/std_s ,  k* = argmin_k ‖z − c_k‖
        ô₁ = m_k* + A_k*(z − z̄_k*)          <- 셀 안에서의 **선형 근사**

    이 근사가 성립해야 "상태로 관측을 샘플링한다" 는 전제가 선다. 그래서 데모의
    (s, o) 짝을 정답으로 두고 cos(ô, o) 를 전 프레임에 대해 재어 분포로 본다.

왜 중앙값만으로는 부족한가
    cos 평균이 높아도 **꼬리가 길면** 일부 상태에서 전혀 다른 관측이 나온다는
    뜻이고, 앵커는 그 프레임에서 엉뚱한 조건으로 teacher 를 질의하게 된다.
    그래서 전 프레임 분포(왼쪽)와 기준선 대비 위치(오른쪽)를 같이 그린다.

기준선이 그림의 뼈대다
    shuffle   같은 태스크의 다른 프레임 o      <- 우연 수준 (하한)
    global    전체 프레임의 o 평균             <- 아무 정보도 안 쓴 경우
    NN(실제)  z 공간 최근접 **진짜** 프레임 o  <- 사실상 상한
    1차 ô₁ 가 상한에 가까우면 선형 근사가 된다는 진술이 선다. 하한 근처면
    코드북이 상태를 관측으로 옮기지 못한다는 뜻이다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent

# 표시 이름 -> (색, 그림에 그릴지)
# ★ 우리 런은 --sampler_mode hard_linear 이고, 그 샘플러는 관측에 σ_o 잡음을
#   **더하지 않는다**(L2_V4_seq.py:1262). 즉 앵커가 실제로 보는 관측이 곧 ô₁ 다.
#   σ_o 는 **선형 잔차** 표준편차다(L2_V4_seq.py:1173 에서 셀 내 원분산 -> 잔차로
#   교체된다). 그러니 ô₁+σ_o·ε 는 "셀 평균에 셀 분산을 얹은 것" 이 아니라
#   Ridge 조건부 평균에 그 적합의 잔차를 얹은 **조건부 가우시안**이다. 참고로만 남긴다.
SERIES = [
    ("1차 ô₁ (v4)", "Ridge 1st-order  $\\hat o_1$   (what the anchor uses)", "#E45756", True),
    ("0차 ô₀ (v1)", "0th-order  $\\hat o_0$   (cell mean, no $A_k$)",   "#4C78A8", True),
    ("셀평균+셀분산", "$m_k + \\sigma_{k}\\epsilon$   (cell mean + cell diag. cov.)", "#F58518", True),
    ("샘플 õ",      "$\\hat o_1 + \\sigma_o\\epsilon$   (+ residual noise)", "#F58518", False),
    ("shuffle",     "shuffle   (chance: a random real frame of the task)", "#BAB0AC", True),
    ("NN(실제)",    "NN (real frame)",               "#54A24B", False),
    ("global 평균", "global mean",                   "#9D9D9D", False),
]


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--main", default="1차 ô₁ (v4)", help="(a) 에서 태스크별로 펼칠 계열")
    return p.parse_args()


def main() -> int:
    a = parse()
    d = np.load(a.npz, allow_pickle=True)
    suite = str(d["suite"]); run = str(d["run"])
    keys = [k for k in d.files if "|" in k]
    tasks = sorted({int(k.split("|")[0][1:]) for k in keys})
    names = [n for n, _, _, _ in [(s[0], 0, 0, 0) for s in SERIES]]

    def get(t: int, name: str):
        k = f"t{t}|{name}"
        return d[k] if k in d.files else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.2, 5.2))
    gs = GridSpec(1, 1, left=0.085, right=0.985, top=0.845, bottom=0.135)

    # ── 태스크마다 전 계열을 **나란한 분포**로 ────────────────────────────
    #   중심 통계 하나로는 꼬리를 담을 수 없다. 본체는 0.95~0.99 에 좁게 몰려
    #   있는데 왼쪽 꼬리가 0.3 까지 내려가므로, 평균·중앙값 어느 쪽을 써도 그
    #   사실이 사라진다. 그래서 계열마다 전 프레임 분포를 그대로 그린다.
    ax = fig.add_subplot(gs[0])
    shown = [(n, lab, col) for n, lab, col, sh in SERIES if sh]
    nS = len(shown)
    off = (np.arange(nS) - (nS - 1) / 2) * (0.84 / nS)      # 슬롯 안 가로 배치
    wv = 0.84 / nS * 0.92
    means = {n: [] for n, _l, _c in shown}
    for i, t in enumerate(tasks):
        for (name, _lab, col), dx in zip(shown, off):
            x = get(t, name)
            if x is None:
                means[name].append(np.nan); continue
            vp = ax.violinplot([x], positions=[i + dx], widths=wv, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.60); b.set_edgecolor("none")
            # 5~95% 구간선과 평균점 — 꼬리의 위치를 눈으로 읽을 수 있게
            p5, p95 = np.percentile(x, 5), np.percentile(x, 95)
            ax.plot([i + dx, i + dx], [p5, p95], "-", color=col, lw=0.9, alpha=0.95)
            ax.plot([i + dx], [x.mean()], "o", color=col, ms=3.0, mec="white", mew=0.6,
                    zorder=5)
            means[name].append(x.mean())
    # 태스크 간 추세는 가는 선으로만 (분포를 가리지 않게)
    for (name, lab, col), dx in zip(shown, off):
        ax.plot(np.arange(len(tasks)) + dx, means[name], "-", color=col, lw=1.0,
                alpha=0.55, zorder=4, label=lab)

    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([str(t) for t in tasks])
    for i in range(len(tasks) - 1):
        ax.axvline(i + 0.5, color="0.90", lw=0.8, zorder=0)
    ax.set_xlabel(f"task ({suite})")
    ax.set_ylabel(r"$\cos(\hat o,\ o)$")
    ax.set_title("violin = all 3000 frames;  vertical line = 5-95% range;  dot = mean",
                 fontsize=10.5, loc="left", color="0.35")
    ax.legend(frameon=False, fontsize=9.0, ncol=2, loc="lower left",
              columnspacing=1.6, handlelength=1.8, borderaxespad=0.4)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ylo = min(np.percentile(np.concatenate(
        [get(t, n) for t in tasks if get(t, n) is not None]), 1)
        for n, _l, _c in shown)
    ax.set_ylim(max(ylo, 0.30), 1.012)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(f"Cosine-similarity analysis by sampling scheme   ({suite})",
                 fontsize=12.5, x=0.085, ha="left")
    out = Path(a.out or (REPO / "results" / f"L2_cb_obs_cos_{suite}.png"))
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))

    # ── 표 ───────────────────────────────────────────────────────────────
    W = 86
    L = ["=" * W, f"코드북 state->observation 재현도 — cos 분포  ({run}, {suite})", "=" * W, "",
         "  그림은 계열마다 **전 프레임 분포**를 그린다. 중심 통계 하나로는 꼬리를",
         "  담을 수 없기 때문이다 — 본체는 0.95~0.99 에 몰려 있고 왼쪽 꼬리만 길다.", "",
         f"  {'계열':<22}{'중앙값':>9}{'평균':>9}{'5%':>9}{'1% 꼬리':>9}{'<0.5 비율':>11}"]
    for name, lab, _c, show in SERIES:
        xs = [get(t, name) for t in tasks]
        xs = [x for x in xs if x is not None]
        if not xs:
            continue
        v = np.concatenate(xs)
        L.append(f"  {name:<22}{np.median(v):>9.4f}{v.mean():>9.4f}"
                 f"{np.percentile(v,5):>9.4f}{np.percentile(v,1):>9.4f}"
                 f"{100*(v<0.5).mean():>10.2f}%")
    L += ["",
          "  읽을 때",
          "    · 1차 ô₁ 가 NN(실제) 에 가까우면 **셀 안에서 선형 근사가 성립**한다는",
          "      진술이 선다. shuffle/global 근처면 상태가 관측을 설명하지 못한다.",
          "    · 1차 > 0차 여야 A_k 가 일을 한 것이다. 같으면 A 는 군더더기다.",
          "    · 중앙값이 높아도 '<0.5 비율' 이 크면 **일부 상태에서 엉뚱한 관측**이",
          "      나온다는 뜻이고, 앵커는 그 프레임에서 잘못된 조건으로 질의한다.", ""]
    txt = "\n".join(L)
    out.with_suffix(".txt").write_text(txt)
    print(txt)
    print(f"\nsaved -> {out}\n         {out.with_suffix('.pdf')}\n"
          f"         {out.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
