#!/usr/bin/env python
"""그룹 안에서의 PCA + K-sweep — 시간 bin vs state 셀.

  왼쪽   그룹마다 따로 PCA 한 곡선을 전부 겹쳐 그린다 (rank 맞춤).
         빨강 = 시간 bin,  파랑(투명) = state 셀.  굵은 선 = 중앙값.
  오른쪽 K 를 10→96 으로 늘릴 때 **전체 잔차 분산**이 어떻게 줄어드는가.

★ rank 함정: 그룹 크기 n 이면 공분산 rank ≤ n−1 이라 작은 그룹이 적은 성분으로
  100% 에 도달한다. 그래서 왼쪽은 **그룹당 표본 수를 맞춰서** 그린다.

★ 시간 bin 복원: 캐시에 (episode, position) 이 없어서 τ 가 감소하는 지점으로
  에피소드 경계를 찾아 복원했다. drop_n_last_frames 로 꼬리가 잘려 있어 원래 τ 와
  66% 일치하지만, K=10 잔차가 1253.6 vs 원래 1254.0 로 사실상 같다.

    python pca_within.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "pca_probe"
TOP, NS = 40, 30
KS_DEFAULT = "10,20,50,96"


def episodes(tau):
    """τ 감소 지점으로 에피소드 경계 복원 -> (pos, eplen)."""
    t = tau.cpu(); N = len(t)
    bnd = (t[1:] < t[:-1]).nonzero().squeeze(1) + 1
    st = torch.cat([torch.tensor([0]), bnd]); en = torch.cat([bnd, torch.tensor([N])])
    pos = torch.zeros(N, dtype=torch.long); ln = torch.zeros(N, dtype=torch.long)
    for a, b in zip(st.tolist(), en.tolist()):
        pos[a:b] = torch.arange(b - a); ln[a:b] = b - a
    return pos.to(tau.device), ln.to(tau.device), len(st)


def group_curves(o, lab, K, n_sub):
    g = torch.Generator().manual_seed(0); out = []
    for k in range(K):
        idx = (lab == k).nonzero().squeeze(1)
        if idx.numel() < n_sub:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=g).to(idx.device)[:n_sub]]
        x = o[idx].double(); x = x - x.mean(0)
        lam = torch.linalg.svdvals(x).pow(2) / (x.shape[0] - 1)
        c = (torch.cumsum(lam, 0) / lam.sum()).cpu().numpy() * 100
        y = np.full(TOP, 100.0); y[:min(TOP, len(c))] = c[:TOP]; out.append(y)
    return np.array(out)


def resid_var(o, lab, K):
    m = torch.zeros(K, o.shape[1], device=o.device, dtype=torch.float64)
    m.index_add_(0, lab, o.double())
    c = torch.bincount(lab, minlength=K).clamp_min(1).double()[:, None]
    return float((o.double() - (m / c)[lab]).var(0).sum())


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--right_only", action="store_true",
                    help="K sweep(오른쪽 패널)만 단독 그림으로 낸다. 그룹별 PCA 곡선을 "
                         "계산하지 않으므로 훨씬 빠르다.")
    ap.add_argument("--pdf", action="store_true", help="PNG 옆에 벡터 PDF 도 낸다")
    ap.add_argument("--title", default="", help="제목 첫 줄 (비우면 기본 문구)")
    ap.add_argument("--subtitle", default="", help="제목 둘째 줄 (비우면 기본 문구)")
    ap.add_argument("--device", default="cuda", help="cuda 또는 cpu")
    ap.add_argument("--name", default="pca_within_ksweep",
                    help="--right_only 로 낼 파일 이름 (확장자 없이)")
    ap.add_argument("--ks", default=KS_DEFAULT,
                    help="K sweep 값들, 쉼표 구분. 왼쪽 패널은 첫 K 와 마지막 K 만 그린다.")
    ap.add_argument("--annotate", choices=("all", "last", "none"), default="all",
                    help="점 위 수치 라벨. 논문 그림은 last 가 깔끔하다.")
    return ap.parse_args()


def main() -> None:
    a = parse()
    dev = a.device
    KS = tuple(int(x) for x in a.ks.replace(" ", "").split(","))
    K_LO, K_HI = KS[0], KS[-1]
    import l2_codebook as CB
    d = torch.load(OUT / "frames_task0.pt", weights_only=False)
    o, s, tau = d["o"].to(dev), d["s"].to(dev), d["tau"].to(dev)
    pos, ln, n_ep = episodes(tau)
    base = float(o.double().var(0).sum())
    print(f"프레임 {o.shape[0]}  에피소드 {n_ep}  전역 잔차 분산 {base:.1f}")

    tlab = {K: ((K * pos) // ln.clamp_min(1)).clamp(0, K - 1) for K in KS}
    slab, keff = {}, {}
    for K in KS:
        cb = CB.build_codebook(s, o, K, seed=42)
        slab[K] = torch.cdist((s - cb["mean_s"].to(dev)) / cb["std_s"].to(dev),
                              cb["c"].to(dev)).argmin(1)
        keff[K] = cb["K_eff"]

    # 왼쪽 패널(그룹별 PCA)은 --right_only 면 계산조차 하지 않는다 — 여기가 비싸다.
    C = {} if a.right_only else {
        **{("time", K): group_curves(o, tlab[K], K, NS) for K in (K_LO, K_HI)},
        **{("state", K): group_curves(o, slab[K], keff[K], NS) for K in (K_LO, K_HI)}}
    V = {("time", K): resid_var(o, tlab[K], K) for K in KS}
    V.update({("state", K): resid_var(o, slab[K], keff[K]) for K in KS})

    import matplotlib
    matplotlib.use("Agg")
    # PDF 에 Type3 대신 TrueType 을 박는다 (학회 제출 검사기 요건).
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    xs = np.arange(1, TOP + 1)

    def draw_ksweep(axr, annotate="all", ref100=True, fs=9):
        """K sweep 패널. 두 그림이 같은 코드를 쓰도록 함수로 뺐다.

        annotate: all=모든 점에 수치, last=마지막 K 만, none=없음.
        ref100:   '분할 없음 = 100%' 기준선. 논문 그림에서는 축 라벨이 그 뜻을
                  이미 담고 있어 지우고 y 범위를 데이터에 맞춘다.
        """
        ys = {}
        for nm, col, mk, lb in (("time", "tab:red", "o", "time bin"),
                                ("state", "tab:blue", "s", "state cluster")):
            ys[nm] = [100 * V[(nm, K)] / base for K in KS]
            axr.plot(KS, ys[nm], color=col, marker=mk, lw=2.4, ms=7, label=lb)
            show = KS if annotate == "all" else (KS[-1:] if annotate == "last" else ())
            for K in show:
                # 마지막 점은 오른쪽 축에 붙어 있어 가운데 정렬하면 라벨이 밖으로 샌다.
                last = K == KS[-1]
                axr.annotate(f"{100*V[(nm,K)]/base:.1f}", (K, 100 * V[(nm, K)] / base),
                             textcoords="offset points",
                             xytext=(-6 if last else 0, 8 if nm == "time" else -14),
                             ha="right" if last else "center",
                             fontsize=fs - 1, color=col)
        axr.set_xlabel("K  (number of groups)", fontsize=fs + 1)
        axr.set_ylabel("residual variance, % of global", fontsize=fs + 1)
        axr.set_xscale("log"); axr.set_xticks(KS)
        axr.set_xticklabels([str(k) for k in KS])
        axr.tick_params(labelsize=fs)
        if ref100:
            axr.axhline(100, color="0.5", ls=":", lw=1)
            axr.set_ylim(30, 105)
        else:
            lo = min(min(v) for v in ys.values()); hi = max(max(v) for v in ys.values())
            pad = 0.16 * (hi - lo)
            axr.set_ylim(lo - pad, hi + pad)
        axr.grid(alpha=.3)
        axr.legend(fontsize=fs + 1, loc="upper right" if ref100 else "lower left")

    if a.right_only:
        # 논문 삽입용 판형. 나머지 설명은 캡션이 지므로 그림 안 글자를 줄인다.
        f1, a1 = plt.subplots(figsize=(5.6, 3.9))
        draw_ksweep(a1, annotate=a.annotate, ref100=False, fs=10)
        # 제목은 한 줄만. --subtitle 을 주면 둘째 줄로 붙는다.
        t1 = a.title or "Residual variance vs K"
        a1.set_title(t1 + ("\n" + a.subtitle if a.subtitle else ""), fontsize=11.5)
        f1.tight_layout()
        fp1 = OUT / f"{a.name}.png"
        f1.savefig(fp1, dpi=140)
        if a.pdf:
            f1.savefig(fp1.with_suffix(".pdf"))
        plt.close(f1)
        print(f"saved -> {fp1}" + (f"  (+ .pdf)" if a.pdf else ""))
        for K in KS:
            t_, s_ = 100 * V[("time", K)] / base, 100 * V[("state", K)] / base
            print(f"  K={K:<4} time {t_:5.1f}%   state {s_:5.1f}%   차이 {t_-s_:+5.1f}")
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

    STY = {("time", K_LO): ("tab:red", "-"), ("time", K_HI): ("darkred", "--"),
           ("state", K_LO): ("tab:blue", "-"), ("state", K_HI): ("navy", "--")}
    for key in (("state", K_HI), ("state", K_LO), ("time", K_HI), ("time", K_LO)):
        col = "tab:blue" if key[0] == "state" else "tab:red"
        for y in C[key]:
            ax[0].plot(xs, y, color=col, alpha=0.13, lw=0.9)
    for key in (("time", K_LO), ("time", K_HI), ("state", K_LO), ("state", K_HI)):
        c, lsty = STY[key]
        ax[0].plot(xs, np.median(C[key], 0), color=c, ls=lsty, lw=2.6,
                   label=f"{key[0]} K={key[1]}  ({len(C[key])} groups)")
    ax[0].set_xlabel("number of eigenvalues summed")
    ax[0].set_ylabel("cumulative variance ratio (%)")
    ax[0].set_title(f"within-group PCA  —  {NS} frames sampled per group\n"
                    "(rank-matched; thin = individual groups, thick = median)")
    ax[0].set_xlim(1, TOP); ax[0].set_ylim(0, 100)

    draw_ksweep(ax[1])
    ax[1].set_title("pooled residual variance vs K\n"
                    "(lower = partition removed more structure)")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Time bin vs state cell — libero_spatial task0, "
                 f"{o.shape[0]} frames, DINOv2 CLS 3072-d (frozen)")
    plt.tight_layout(); plt.savefig(OUT / "pca_within.png", dpi=140)
    if a.pdf:
        fig.savefig(OUT / "pca_within.pdf")

    n_ok96 = len(C[("state", K_HI)])
    L = [f"시간 bin vs state 셀 — {o.shape[0]} 프레임 (에피소드 {n_ep}), o 3072-d", "",
         f"[1] 그룹 안 PCA — 누적 분산 비율(%) 중앙값, 그룹당 {NS} 프레임 균등 표본",
         f"{'분할':<16}{'그룹수':>7}" + "".join(f"{('k='+str(k)):>9}" for k in (1, 3, 5, 10, 20))]
    for key in (("time", K_LO), ("state", K_LO), ("time", K_HI), ("state", K_HI)):
        m = np.median(C[key], 0)
        L.append(f"{key[0]+' K='+str(key[1]):<16}{len(C[key]):>7}"
                 + "".join(f"{m[k-1]:9.2f}" for k in (1, 3, 5, 10, 20)))
    L += ["", "[2] 전체 잔차 분산 (전역 대비 %) — K sweep",
          f"{'K':>6}{'시간 bin':>14}{'state 셀':>14}{'차이':>10}"]
    for K in KS:
        t_, s_ = 100 * V[("time", K)] / base, 100 * V[("state", K)] / base
        L.append(f"{K:>6}{t_:13.1f}%{s_:13.1f}%{t_-s_:+10.1f}")
    L += ["", "시간은 K 를 10배 늘려도 포화한다(1차원이라 정보 천장이 있다).",
          "state 는 계속 줄어든다 — 16차원이라 쪼갤수록 실제 구조를 따라간다.",
          "",
          f"주의  state K={K_HI} 은 {n_ok96}/{K_HI} 클러스터만 {NS} 프레임을 넘겼다.",
          f"      고유 프레임 {o.shape[0]} 개를 {K_HI} 로 나누면 평균 "
          f"{o.shape[0]//K_HI} 개라 이미 얇다.",
          "      K 를 더 키우면 통계가 무너진다 — 이 데이터셋의 실질 상한 근처다."]
    (OUT / "pca_within.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {OUT/'pca_within.png'}")


if __name__ == "__main__":
    main()
