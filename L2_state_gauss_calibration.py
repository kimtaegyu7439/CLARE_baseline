#!/usr/bin/env python
"""코드북 셀의 가우시안 근사가 **실제 퍼짐을 얼마나 잘 설명하는가** — 두 그래프.

왜 예시 셀 그림을 대체하는가
    기존 state_gauss_cells_*.png 는 셀 2개를 뽑아 PCA 평면에 타원을 겹쳐 보여줬다.
    셀을 어떻게 고르든 "그 셀만 그런 것 아니냐"를 막을 수 없고, 2D 사영은 16차원의
    퍼짐을 다 보여주지도 못한다. 여기서는 셀 하나하나를 점으로 만들어 **전 셀의 분포**로
    답한다.

    ① 예상 비율 vs 실제 비율   가우시안이 예상하는 포함 비율(명목)을 쓸어가며,
                               그 영역 안에 실제로 들어온 held-out 데이터 비율(실제)을 잰다.
                               대각선 y=x 에 가까울수록 근사가 퍼짐을 잘 설명한다.
    ② 클러스터별 근사 편차     명목 90% 영역 하나에 대해 셀마다 실제 비율을 찍는다.
                               점 하나가 셀 하나. 90% 수평선에서 멀수록, 넓게 퍼질수록
                               셀마다 근사 품질이 다르다는 뜻이다.

정직하게 재기 위한 세 가지
    1. **held-out 에피소드**로 잰다. 표준화·k-means·셀별 (μ, Σ) 적합은 train 에피소드만
       쓰고, 평가는 한 번도 안 본 에피소드의 프레임으로 한다. 같은 점으로 적합하고
       재면 포함 비율이 낙관적으로 부풀 수밖에 없다.
    2. **원래 state 차원(16 = 2 obs step x 8)** 에서 잰다. 2D 사영은 어느 평면을
       고르느냐에 따라 답이 달라진다. 16차원에서 마할라노비스 d² 를 χ²_16 분위와 비교한다.
    3. **두 모델을 함께** 그린다.
           대각 (diag)   Σ = diag(σ²)            ← sample_codebook 이 실제로 쓰는 모델
           완전공분산     Σ = C + ridge·diag(C)   ← --full_cov_s 옵션
       대각만 그리면 "가우시안이 문제"인지 "대각 가정이 문제"인지 구분되지 않는다.

주장의 범위
    이 그림이 말하는 것은 "가우시안 근사의 **포함 비율**이 명목값과 얼마나 맞는가"
    까지다. 분포 모양 전체가 가우시안인지(KS·PIT)는 L2_state_base_gaussian_check.py 가
    따로 답한다.

    python L2_state_gauss_calibration.py --suite libero_object --tasks 0,1,2,3,4,5,6,7,8,9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import B1                                                # noqa: E402
import l2_codebook                                       # noqa: E402
from L2_state_base_gaussian_check import collect_state    # noqa: E402
from lerobot.datasets.factory import make_dataset         # noqa: E402

CK_DEFAULT = "outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model"
C_FULL, C_DIAG, C_REF = "#0072B2", "#D55E00", "0.45"


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--codebook_k", type=int, default=100, help="l2_codebook 과 같은 K")
    p.add_argument("--min_n", type=int, default=5, help="셀 병합 임계 (build_codebook 과 동일)")
    p.add_argument("--holdout_episodes", type=int, default=15,
                   help="뒤쪽 이만큼의 에피소드를 평가에만 쓴다 (기본 50개 중 15개)")
    p.add_argument("--split", choices=["episode", "frame", "all"], default="episode",
                   help="held-out 을 어떻게 떼는가. episode = 통째로 다른 에피소드(기본), "
                        "frame = 같은 에피소드 안에서 프레임을 무작위로. 두 값을 비교하면 "
                        "'셀 안이 가우시안이 아닌 것'과 '에피소드 간 변동'이 분리된다.")
    p.add_argument("--frame_holdout", type=float, default=0.30,
                   help="--split frame 에서 평가로 뺄 프레임 비율")
    p.add_argument("--min_train", type=int, default=21,
                   help="셀 적합에 필요한 train 표본 수. 16차원 공분산이 서려면 d+5")
    p.add_argument("--min_test", type=int, default=15,
                   help="셀 평가에 필요한 held-out 표본 수. 작으면 비율이 거칠어진다")
    p.add_argument("--cov_ridge", type=float, default=0.05,
                   help="완전공분산에만 붙는 ridge (l2_codebook.py 와 동일)")
    p.add_argument("--nominal", default="50,90", help="② 에 쓸 명목 비율(%). 첫 값이 주 패널")
    p.add_argument("--basis", choices=["raw", "pair"], default="raw",
                   help="pair = 쌍 기저로 바꿔 적합한다. observation.state 는 8 차원인데 "
                        "n_obs_steps=2 라 16-D 는 [t-1(8), t(8)] 이고 두 블록이 r~0.99 다. "
                        "m=(a+b)/2, d=a-b 로 바꾸면 그 상관이 기저에 흡수돼 대각 "
                        "가우시안으로도 정확히 표현된다. 되돌리기: a=m+d/2, b=m-d/2.")
    p.add_argument("--dump_whitened", action="store_true",
                   help="셀별로 표준화한 held-out state 를 whitened.npz 로 덤프한다. "
                        "z = (x - mu_hat)/sigma_hat 이라 가우시안 가정이 맞으면 "
                        "**모든 셀의 z 가 N(0,I) 하나로 모인다** — 클러스터를 고르지 "
                        "않고 전부 겹쳐 그릴 수 있다.")
    p.add_argument("--threshold", choices=["f", "chi2"], default="f",
                   help="포함 영역의 임계값. f = Hotelling T2 의 F(d,n-d) 로 (mu,Sigma) 추정 "
                        "오차까지 감안한 예측영역(기본). chi2 = 적합된 가우시안 자신의 분위 — "
                        "그 가우시안에서 **샘플링**할 때 표본이 들어가는 영역이라, 샘플링 "
                        "주장에는 이쪽이 맞는 기준일 수 있다. n=52 에서 f 가 chi2 의 1.6 배다.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=CK_DEFAULT, help="delta_timestamps 를 읽을 정책 설정")
    p.add_argument("--out", default="results/L2_state_gauss_calib")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def fit_cells(s_tr: torch.Tensor, K: int, seed: int, min_n: int):
    """train 으로만 표준화 + k-means. build_cells 와 같은 절차지만 mean/std 도 돌려준다.

    held-out 을 같은 좌표계로 옮겨 배정해야 하므로 mean/std 와 center 가 필요하다.
    """
    mean_s, std_s = s_tr.mean(0), s_tr.std(0).clamp_min(1e-8)
    zs = (s_tr - mean_s) / std_s
    c, lab, _ = l2_codebook.kmeans(zs, K, n_init=10, seed=seed)

    cnt = torch.bincount(lab, minlength=K)
    keep = (cnt >= min_n).nonzero().squeeze(1)
    drop = (cnt < min_n).nonzero().squeeze(1)
    if drop.numel() and keep.numel():
        remap = torch.arange(K, device=s_tr.device)
        remap[drop] = keep[torch.cdist(c[drop], c[keep]).argmin(1)]
        lab = remap[lab]
        uniq, lab = torch.unique(lab, return_inverse=True)
        c = c[uniq]
    return lab, c, mean_s, std_s


def cell_sigmas(x: np.ndarray, ridge: float):
    """한 셀의 (μ̂, Σ̂_full, Σ̂_diag).

    ★ 분모는 n-1(불편추정)을 쓴다. build_codebook 은 ML(n)을 쓰지만, 아래 F 기준이
      불편추정 S 를 전제로 유도돼 있다. 둘의 차이는 n/(n-1) 배(n≈50 이면 2%)이고
      기준과 통계량을 **짝이 맞게** 두는 편이 낫다.
    ridge 는 완전공분산 쪽에만 붙는다 — l2_codebook 코드가 그렇게 돼 있다.
    """
    mu = x.mean(0)
    xc = x - mu
    C = xc.T @ xc / max(1, x.shape[0] - 1)
    var = np.clip(np.diag(C), 1e-12, None)
    return (mu,
            C + ridge * np.diag(var) + 1e-8 * np.eye(x.shape[1]),
            np.diag(var))


def t2_threshold(q: float, d: int, n: int) -> float:
    """명목 비율 q 에 해당하는 T² 임계값 — **추정된** (μ̂, Σ̂) 로 잴 때.

    새 관측치 x 에 대해  T² = (x-μ̂)ᵀ Σ̂⁻¹ (x-μ̂) 는 카이제곱이 아니다. 정확히는
        T² · n(n-d) / (d(n²-1)) ~ F(d, n-d)
    이다. χ² 를 쓰면 표본이 적을수록 임계값이 작아 "가우시안이 안 맞는다"는 결론이
    저절로 나온다 — 실제로는 공분산 추정 오차를 보고 있는 것이다.
    ★ 완전공분산에는 ridge 가 붙어 이 유도가 근사가 된다(ridge 가 Σ̂ 를 부풀리므로
      포함 비율이 약간 낙관적으로 나온다). 대각 모델에는 ridge 가 없다.
    """
    return d * (n ** 2 - 1) / (n * (n - d)) * float(f_dist.ppf(q, d, n - d))


def to_pair(x: np.ndarray) -> np.ndarray:
    """[t-1(8), t(8)] -> [m(8), d(8)].  m=(a+b)/2 공통성분, d=a-b 프레임 간 변화.

    선형·가역이라 정보 손실이 없다. 대각 가우시안을 이 기저에서 적합하면
    a 와 b 사이의 r~0.99 상관을 그대로 담는다 — 원좌표에서 대각으로 뽑으면
    t-1 과 t 가 따로 놀아 실제보다 8~10 배 큰 프레임 간 변화가 생긴다.
    """
    h = x.shape[1] // 2
    a, b = x[:, :h], x[:, h:]
    return np.concatenate([(a + b) / 2.0, a - b], axis=1)


def d2(x: np.ndarray, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """마할라노비스 제곱거리. 원래 state 차원에서 그대로 잰다."""
    xc = x - mu
    return np.einsum("ij,jk,ik->i", xc, np.linalg.inv(S), xc)


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = [int(t) for t in a.tasks.split(",")]
    nominal = np.array([float(v) / 100 for v in a.nominal.split(",")])
    grid = np.concatenate([np.arange(.05, .96, .05), [.975, .99]])   # ① 의 명목 비율 축

    rows = []
    W_z, W_n, W_r, W_c = [], [], [], []   # 실제 / n / 부트스트랩 기준 / 셀 고유번호
    cell_uid = 0
    rng_boot = np.random.default_rng(a.seed + 12345)          # 셀 하나 = 한 줄
    for t in tasks:
        ns = argparse.Namespace(suite=a.suite, device="cpu", seed=a.seed, num_workers=0,
                                batch_size=32, steps_per_task=1, log_every=100,
                                eval_episodes=1, eval_batch_size=1, mode="gausscalib",
                                p_drop=0.0, lambda_anchor=0.0)
        cfg = B1.build_cfg(ns, t, a.ckpt, Path("/tmp/l2sgauss_calib"))
        ds = make_dataset(cfg)
        drop_n = int(getattr(cfg.policy, "drop_n_last_frames", 0))
        s_np, ep, _frm, _ph = collect_state(ds, 0, drop_n, a.seed)     # 0 = 전 프레임
        D = s_np.shape[1]

        eps = np.unique(ep)
        if a.split == "all":
            # ★ 코드북이 실제로 하는 것과 같다: 그 태스크의 **전 프레임**으로 셀을
            #   만들고, 그 셀의 가우시안에서 앵커를 뽑는다. 안 본 프레임으로의
            #   일반화가 아니라 "이 셀이 가우시안 모양인가"를 묻는 것이므로
            #   적합과 평가가 같은 프레임인 게 맞다.
            te = tr = np.ones(len(ep), dtype=bool)
            split_note = "all frames (fit and evaluate on the same frames)"
        elif a.split == "frame":
            # 같은 에피소드 안에서 프레임만 뺀다 — 에피소드 간 변동이 섞이지 않는다.
            te = np.random.default_rng(a.seed + t).random(len(ep)) < a.frame_holdout
            split_note = f"frame-level ({a.frame_holdout:.0%} of frames)"
        else:
            n_ho = min(a.holdout_episodes, max(1, len(eps) // 3))
            ho_eps = set(eps[-n_ho:].tolist())
            te = np.array([e in ho_eps for e in ep])
            split_note = f"episode-level (last {a.holdout_episodes} episodes)"
        if a.split != "all":
            tr = ~te
        s_tr, s_te = s_np[tr], s_np[te]
        if a.basis == "pair":
            # 셀 나누기(k-means)는 원좌표에서 그대로 한다 — 기저는 셀 **안의**
            # 분포를 어떻게 모형화하느냐의 문제지 셀을 어떻게 나누느냐가 아니다.
            s_tr_fit, s_te_fit = to_pair(s_tr), to_pair(s_te)
        else:
            s_tr_fit, s_te_fit = s_tr, s_te

        lab_tr, cen, mean_s, std_s = fit_cells(
            torch.as_tensor(s_tr, dtype=torch.float32, device=a.device),
            a.codebook_k, a.seed, a.min_n)
        lab_tr = lab_tr.cpu().numpy()
        K_eff = int(cen.shape[0])
        # held-out 을 같은 좌표계로 옮겨 가장 가까운 center 에 배정 (k-means 규칙 그대로)
        z_te = (torch.as_tensor(s_te, dtype=torch.float32, device=a.device)
                - mean_s) / std_s
        lab_te = torch.cdist(z_te, cen).argmin(1).cpu().numpy()

        used = 0
        for k in range(K_eff):
            xtr, xte = s_tr_fit[lab_tr == k], s_te_fit[lab_te == k]
            if len(xtr) < a.min_train or len(xte) < a.min_test:
                continue
            mu, S_full, S_diag = cell_sigmas(xtr, a.cov_ridge)
            n_tr = len(xtr)
            qs = set(grid.tolist()) | set(nominal.tolist())
            thr = ({q: float(chi2_dist.ppf(q, D)) for q in qs} if a.threshold == "chi2"
                   else {q: t2_threshold(q, D, n_tr) for q in qs})
            r = {"task": t, "cell": k, "n_train": n_tr, "n_test": len(xte)}
            for name, S in (("full", S_full), ("diag", S_diag)):
                dd = d2(xte, mu, S)
                ds = d2(xtr, mu, S)              # in-sample — 추정 오차만큼의 낙관을 보여준다
                r[f"cov_{name}"] = [float(np.mean(dd <= thr[q])) for q in grid]
                r[f"in_{name}"] = [float(np.mean(ds <= thr[q])) for q in grid]
                r[f"at_{name}"] = [float(np.mean(dd <= thr[q])) for q in nominal]
            if a.dump_whitened:
                # 대각 모델 기준 표준화. 샘플러(sample_codebook_hard)도 대각이라
                # 이 좌표계가 곧 "우리가 실제로 뽑는 분포"의 좌표계다.
                sd_v = np.sqrt(np.diag(S_diag))
                W_z.append(((xte - mu) / sd_v).astype(np.float32))
                W_n.append(np.full(len(xte), n_tr, dtype=np.int32))
                # ★ 모수 부트스트랩 기준. "이 셀이 **정말** 가우시안이었다면
                #   같은 셀 크기·같은 추정 절차로 무엇이 나오는가"를 직접 뽑는다.
                #   추정 오차, 대각 가정, 표본 크기, 적합=평가(=all 모드)까지
                #   전부 자동 반영되므로 분포 이론을 끌어올 필요가 없다.
                g = rng_boot.normal(size=(n_tr, D))
                gm, gs = g.mean(0), g.std(0, ddof=1)
                gt = g if a.split == "all" else rng_boot.normal(size=(len(xte), D))
                W_r.append(((gt - gm) / gs).astype(np.float32))
                # ★ 셀 고유번호. 이게 있어야 "셀마다 따로 재고 셀들 사이 분포를
                #   본다"가 가능하다. 합산해 버리면 클러스터별 보장을 말할 수 없다.
                W_c.append(np.full(len(xte), cell_uid, dtype=np.int32))
                cell_uid += 1
            rows.append(r); used += 1
        print(f"[calib] task {t}  train {len(s_tr)} / held-out {len(s_te)} 프레임 "
              f"[{split_note}]  K_eff={K_eff}  쓸 수 있는 셀 {used}", flush=True)

    if not rows:
        raise SystemExit("조건을 만족하는 셀이 없다. --min_test 를 낮추거나 --codebook_k 를 줄여라.")

    cov = {m: np.array([r[f"cov_{m}"] for r in rows]) * 100 for m in ("full", "diag")}
    ins = {m: np.array([r[f"in_{m}"] for r in rows]) * 100 for m in ("full", "diag")}
    at = {m: np.array([r[f"at_{m}"] for r in rows]) * 100 for m in ("full", "diag")}
    task_of = np.array([r["task"] for r in rows])
    D = 16

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
                                "axes.edgecolor": "0.72", "axes.linewidth": .8,
                                "xtick.color": "0.4", "ytick.color": "0.4"})
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    fig.subplots_adjust(left=.075, right=.985, top=.865, bottom=.30, wspace=.24)

    # ① 예상 비율 vs 실제 비율 ------------------------------------------------
    ax = axes[0]
    ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), lw=1.1, color=C_REF, zorder=2)
    ax.text(62, 55, "perfect", color=C_REF, fontsize=9, ha="center", va="center",
            rotation=45, rotation_mode="anchor",
            bbox=dict(fc="white", ec="none", pad=1.5))
    for m, col, lbl in (("full", C_FULL, "full covariance"),
                        ("diag", C_DIAG, "diagonal (sampler in use)")):
        med = np.median(cov[m], 0)
        lo, hi = np.percentile(cov[m], [25, 75], axis=0)
        ax.fill_between(grid * 100, lo, hi, color=col, alpha=.16, lw=0, zorder=3)
        ax.plot(grid * 100, med, color=col, lw=2.0, zorder=4, label=lbl)
        # in-sample 은 두 모델 모두 얇은 점선으로. 범례는 아래에서 한 줄만 단다.
        ax.plot(grid * 100, np.median(ins[m], 0), color=col, lw=1.1, ls=(0, (2, 2)),
                alpha=.85, zorder=4)
    ax.plot([], [], color="0.35", lw=1.1, ls=(0, (2, 2)),
            label="same clusters, in-sample")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_xlabel("Nominal coverage of the Gaussian region (%)")
    ax.set_ylabel("Actual held-out data inside (%)")
    ax.set_title("① Does the Gaussian explain the spread?", fontsize=11.5, pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    # ② 클러스터별 근사 편차 (명목 90%) ---------------------------------------
    ax = axes[1]
    q0 = nominal[-1]
    ax.axhline(q0 * 100, ls=(0, (4, 3)), lw=1.1, color=C_REF, zorder=2)
    ax.text(len(tasks) - .35, q0 * 100 + 1.5, f"nominal {q0*100:.0f}%",
            color=C_REF, fontsize=9, ha="right", va="bottom")
    rng = np.random.default_rng(a.seed)
    for j, (m, col, lbl) in enumerate((("full", C_FULL, "full covariance"),
                                       ("diag", C_DIAG, "diagonal (sampler in use)"))):
        for i, t in enumerate(tasks):
            v = at[m][task_of == t, -1]
            if not len(v):
                continue
            # 셀이 겹쳐 안 보이므로 열 안에서만 좌우로 흩는다. y(측정값)는 건드리지 않는다.
            x = i + (j - .5) * .42 + rng.uniform(-.09, .09, len(v))
            ax.scatter(x, v, s=13, color=col, alpha=.55, edgecolors="none", zorder=4,
                       label=lbl if i == 0 else None)
            ax.plot([i + (j - .5) * .42 - .13, i + (j - .5) * .42 + .13],
                    [np.median(v)] * 2, color=col, lw=2.2, zorder=5)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([str(t) for t in tasks])
    ax.set_xlim(-.6, len(tasks) - .4); ax.set_ylim(-3, 103)
    ax.set_xlabel("Task"); ax.set_ylabel(f"Inside the nominal {q0*100:.0f}% region (%)")
    ax.set_title("② How far off is each cluster?", fontsize=11.5, pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")

    n_cell = len(rows)
    med_tr = int(np.median([r["n_train"] for r in rows]))
    med_te = int(np.median([r["n_test"] for r in rows]))
    fig.suptitle("Gaussian coverage of codebook cells — measured on held-out episodes",
                 fontsize=13, y=.965)
    fig.text(.5, .175,
             f"{a.suite}, {len(tasks)} tasks, K={a.codebook_k}. Fit (standardisation, "
             f"k-means, per-cluster μ/Σ) uses train episodes only; the last "
             f"{a.holdout_episodes} episodes of each task are held out.\n"
             f"{n_cell} clusters had ≥{a.min_train} train and ≥{a.min_test} "
             f"held-out frames (median {med_tr} / {med_te}). Coverage is measured in the "
             f"original {D}-D state space, using the finite-sample threshold\n"
             f"T·n(n−d)/(d(n²−1)) ~ F(d, n−d) rather than χ², "
             f"so that estimating (μ, Σ) from n frames is already accounted for. "
             f"Dotted = the same clusters scored in-sample.\n"
             "① thick line = median over clusters, band = 25–75%.    "
             "② one dot = one cluster, bar = median; dots are spread sideways within "
             "a task only so they do not overlap.",
             ha="center", va="top", fontsize=8.8, color="0.3", linespacing=1.6)

    for ext in ("png", "pdf"):
        p = out / f"state_gauss_calibration.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None)
        print("saved ->", p)

    summary = {"suite": a.suite, "tasks": tasks, "K": a.codebook_k, "dim": D,
               "holdout_episodes": a.holdout_episodes, "n_cells": n_cell,
               "nominal_grid": grid.tolist(),
               "median_curve": {m: np.median(cov[m], 0).tolist() for m in cov},
               "iqr_curve": {m: np.percentile(cov[m], [25, 75], axis=0).tolist() for m in cov},
               "at_nominal": {f"{q*100:.0f}%": {m: {"median": float(np.median(at[m][:, i])),
                                                    "p25": float(np.percentile(at[m][:, i], 25)),
                                                    "p75": float(np.percentile(at[m][:, i], 75))}
                                                for m in at}
                              for i, q in enumerate(nominal)},
               "per_cell": rows}
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    if a.dump_whitened and W_z:
        z = np.concatenate(W_z); n = np.concatenate(W_n)
        np.savez_compressed(out / "whitened.npz", z=z, n_train=n,
                            z_ref=np.concatenate(W_r),
                            cell=np.concatenate(W_c),
                            suite=a.suite, dim=D, n_cells=n_cell, split=a.split,
                            basis=a.basis)
        print(f"saved -> {out / 'whitened.npz'}  (표준화 held-out {len(z)}점, "
              f"셀 {n_cell}개, {D}차원)")
    for i, q in enumerate(nominal):
        s = summary["at_nominal"][f"{q*100:.0f}%"]
        print(f"명목 {q*100:.0f}%  ->  full {s['full']['median']:.1f}% "
              f"(IQR {s['full']['p25']:.1f}~{s['full']['p75']:.1f})   "
              f"diag {s['diag']['median']:.1f}% "
              f"(IQR {s['diag']['p25']:.1f}~{s['diag']['p75']:.1f})")


if __name__ == "__main__":
    main()
