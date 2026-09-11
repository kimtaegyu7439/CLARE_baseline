"""D1 — 앵커 주소에서 잔차의 각도·크기를 teacher 속도 십분위별로 본다 (forward only).

가설: MSE 앵커는 ‖v‖ 가 큰 구간에 손실이 몰리므로 느린 구간의 **방향**을 덜 보호한다.
그렇다면 teacher 속도가 느린 십분위에서 학생-teacher 각도 θ 가 더 크고, 상대 크기
오차 |m| 에는 그런 격차가 없어야 한다.

de-noised 프레임: ṽ := v + ε₁.  모델이 앵커의 잡음과 일관되면 ṽ 는 예측 행동 청크와
같다. 각도·크기는 전부 ṽ 에서 잰다 (속도 프레임 v 에서 재지 않는다).

출력: d1_cells.csv, d1_taxis.csv, d1_gripper.csv, d1_sanity.csv, PNG 4장.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import (OUT, N_STAGES, H, A_DIM, POSE, GRIP, PolicyCache,
                    anchor_addresses, anchor_cond, bootstrap_xt, instructions,
                    load_codebook, vel)

EPS = 1e-8
THETA_THRESH = 30.0        # frac(θ > 30°) 의 기준
N_DECILE = 10


@torch.no_grad()
def eval_pair(pc, cb, instr, i_student, i_tp, i_to, n, bs, device, seed):
    """한 (j,i) 쌍에서 앵커 주소 n 개를 만들고 ṽ_S / ṽ_Tp / ṽ_To 를 모은다.

    반환 dict: 각 값은 (n*H,) 또는 (n,) 로 평탄화된 numpy.
    """
    S, Tp, To = pc.get(i_student), pc.get(i_tp), pc.get(i_to)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {k: [] for k in ("t", "rho_tp", "rho_to", "th_tp", "th_to",
                           "m_tp", "m_to", "e_tp", "e_to",
                           "g_tp", "g_to", "e_tp_all", "e_to_all")}
    sane_num, sane_den = 0.0, 0.0

    done = 0
    while done < n:
        b = min(bs, n - done)
        s_t, o_t = anchor_addresses(cb, b, gen, device)
        cond_S = anchor_cond(S, s_t, o_t, instr)
        cond_Tp = anchor_cond(Tp, s_t, o_t, instr)
        cond_To = anchor_cond(To, s_t, o_t, instr)

        t = torch.rand(b, device=device, generator=gen)
        eps0 = torch.randn(b, H, A_DIM, device=device, generator=gen)
        eps1 = torch.randn(b, H, A_DIM, device=device, generator=gen)
        x_t, a_hat = bootstrap_xt(Tp, cond_Tp, t, eps0, eps1)

        vS = vel(S, x_t, t, cond_S) + eps1          # ṽ_S
        vTp = vel(Tp, x_t, t, cond_Tp) + eps1       # ṽ_Tp
        vTo = vel(To, x_t, t, cond_To) + eps1       # ṽ_To

        # sanity: 잡음 규약이 맞으면 ṽ_Tp ≈ â
        sane_num += float((vTp - a_hat).flatten(1).norm(dim=1).sum())
        sane_den += float(a_hat.flatten(1).norm(dim=1).sum())

        pS = vS[..., POSE]                                             # (b,H,6)
        for tag, vT in (("tp", vTp), ("to", vTo)):
            pT = vT[..., POSE]                                         # (b,H,6)
            nS, nT = pS.norm(dim=-1), pT.norm(dim=-1)                  # (b,H)
            cos = (pS * pT).sum(-1) / (nS * nT + EPS)
            th = torch.rad2deg(torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))
            out[f"rho_{tag}"].append(nT.flatten().cpu().numpy())
            out[f"th_{tag}"].append(th.flatten().cpu().numpy())
            out[f"m_{tag}"].append(((nS - nT) / (nT + EPS)).flatten().cpu().numpy())
            out[f"e_{tag}"].append((pS - pT).pow(2).sum(-1).flatten().cpu().numpy())
            # gripper 는 스칼라라 각도가 없다 — 절대오차로 따로 본다
            out[f"g_{tag}"].append(
                (vS[..., GRIP] - vT[..., GRIP]).abs().flatten().cpu().numpy())
        # 전 dim MSE (t축 집계용, pose+gripper 전체)
        out["e_tp_all"].append((vS - vTp).pow(2).mean(dim=(1, 2)).cpu().numpy())
        out["e_to_all"].append((vS - vTo).pow(2).mean(dim=(1, 2)).cpu().numpy())
        out["t"].append(t.cpu().numpy())
        done += b

    res = {k: np.concatenate(v) for k, v in out.items() if v}
    res["sanity"] = sane_num / max(sane_den, EPS)
    # t 를 스텝 축으로 펼친 판본 (스텝별 양과 길이를 맞춘다)
    res["t_step"] = np.repeat(res["t"], H)
    return res


# ṽ = v + ε₁ 프레임이 성립하는 t 구간. 실측(sanity_probe): ‖ṽ−â‖/‖â‖ 가
# t=0 에서 0.09, t=0.3 에서 0.13, t=0.5 에서 0.22, t=1 에서 3.03 이다.
# 큰 t 에서는 x_t≈a 라 모델이 ε₁ 을 복원할 수 없고, v+ε₁ 이 잡음을 주입한다
# (flow matching 의 속도장은 주변 기대값 E[a−ε|x_t] 이다). 그 구간에서는
# ρ=‖ṽ_T‖ 가 teacher 속도가 아니라 ‖ε₁‖ 을 재므로 십분위 층화가 깨진다.
T_SELS = {"all": (0.0, 1.0), "t_le_0.3": (0.0, 0.3)}


def _tmask(t_step, sel):
    lo, hi = T_SELS[sel]
    return (t_step >= lo) & (t_step <= hi)


@torch.no_grad()
def decile_edges(pc, cb, instr, j, n, bs, device, seed):
    """task j 의 십분위 경계. (teacher_orig = stage j, 자기 task) 의 ρ 분포로 한 번 정한다.
    t 선택마다 ρ 분포가 다르므로 선택별로 따로 잡는다."""
    r = eval_pair(pc, cb, instr, j, j, j, n, bs, device, seed)
    out = {}
    for sel in T_SELS:
        rho = r["rho_to"][_tmask(r["t_step"], sel)]
        q = np.quantile(rho, np.linspace(0, 1, N_DECILE + 1))
        q[0], q[-1] = -np.inf, np.inf
        out[sel] = q
    return out, r["sanity"], r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="쌍마다 앵커 주소 수")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    instr = instructions()
    pc = PolicyCache(device)

    cells, taxis, grip, sanity, sanity_t = [], [], [], [], []
    t0 = time.perf_counter()

    # 십분위 경계 — j 마다 한 번
    edges, cbs = {}, {}
    for j in range(N_STAGES - 1):            # j < i <= N_STAGES-1
        cbs[j] = load_codebook(j, device)
        edges[j], sn, ref = decile_edges(pc, cbs[j], instr[j], j, args.n, args.batch,
                                         device, args.seed + 1000 + j)
        sanity.append({"j": j, "i": j, "kind": "decile_ref", "sanity_ratio": sn})
        # sanity 를 t 구간별로도 남긴다 — ṽ 프레임의 유효 범위 근거
        nb = np.clip((ref["t_step"] * 5).astype(int), 0, 4)
        for b in range(5):
            m = nb == b
            if m.sum():
                sanity_t.append({"j": j, "t_lo": b / 5, "t_hi": (b + 1) / 5,
                                 "rho_med": float(np.median(ref["rho_to"][m])),
                                 "n": int(m.sum())})
        print(f"[D1] j={j} 십분위 경계 확정  ρ(all) "
              f"{edges[j]['all'][1]:.4f}..{edges[j]['all'][-2]:.4f}  sanity={sn:.4f}",
              flush=True)

    for i in range(1, N_STAGES):
        for j in range(i):
            r = eval_pair(pc, cbs[j], instr[j], i, i - 1, j,
                          args.n, args.batch, device, args.seed + 100 * i + j)
            sanity.append({"j": j, "i": i, "kind": "pair", "sanity_ratio": r["sanity"]})
            for sel in T_SELS:
                tm = _tmask(r["t_step"], sel)
                for T in ("tp", "to"):
                    d = np.digitize(r[f"rho_{T}"], edges[j][sel][1:-1], right=False)
                    for k in range(N_DECILE):
                        m = (d == k) & tm
                        if m.sum() == 0:
                            continue
                        cells.append({
                            "j": j, "i": i, "teacher": T, "t_sel": sel, "decile": k + 1,
                            "n": int(m.sum()),
                            "rho_med": float(np.median(r[f"rho_{T}"][m])),
                            "theta_med": float(np.median(r[f"th_{T}"][m])),
                            "theta_p90": float(np.percentile(r[f"th_{T}"][m], 90)),
                            "theta_frac_gt30": float((r[f"th_{T}"][m] > THETA_THRESH).mean()),
                            "absm_med": float(np.median(np.abs(r[f"m_{T}"][m]))),
                            "e_mean": float(r[f"e_{T}"][m].mean()),
                            "grip_abs_mean": float(r[f"g_{T}"][m].mean()),
                        })
            for T in ("tp", "to"):
                grip.append({"j": j, "i": i, "teacher": T,
                             "grip_abs_mean": float(r[f"g_{T}"].mean()),
                             "grip_abs_p90": float(np.percentile(r[f"g_{T}"], 90))})
            # t 축 집계 — 표본(청크) 단위 MSE
            tt, ee = r["t"], r["e_to_all"]
            bins = np.clip((tt * 5).astype(int), 0, 4)
            tot = ee.sum()
            for b in range(5):
                m = bins == b
                if m.sum() == 0:
                    continue
                taxis.append({"j": j, "i": i, "t_bin": b, "t_lo": b / 5, "t_hi": (b + 1) / 5,
                              "n": int(m.sum()),
                              "e_mean": float(ee[m].mean()),
                              "e_weighted_mean": float(((1 - tt[m]) ** 2 * ee[m]).mean()),
                              "e_share": float(ee[m].sum() / max(tot, EPS))})
            hi = tt > 0.8
            w = (1 - tt) ** 2 * ee
            taxis.append({"j": j, "i": i, "t_bin": -1, "t_lo": 0.8, "t_hi": 1.0,
                          "n": int(hi.sum()),
                          "e_mean": float(ee[hi].mean()) if hi.any() else np.nan,
                          "e_weighted_mean": float(w[hi].mean()) if hi.any() else np.nan,
                          "e_share": float(ee[hi].sum() / max(tot, EPS)),
                          "e_share_weighted": float(w[hi].sum() / max(w.sum(), EPS))})
        print(f"[D1] i={i} 완료 ({time.perf_counter()-t0:.0f}s)", flush=True)

    pd.DataFrame(cells).to_csv(OUT / "d1_cells.csv", index=False)
    pd.DataFrame(taxis).to_csv(OUT / "d1_taxis.csv", index=False)
    pd.DataFrame(grip).to_csv(OUT / "d1_gripper.csv", index=False)
    pd.DataFrame(sanity).to_csv(OUT / "d1_sanity.csv", index=False)
    pd.DataFrame(sanity_t).to_csv(OUT / "d1_sanity_by_t.csv", index=False)
    print(f"[D1] CSV 저장 완료 ({time.perf_counter()-t0:.0f}s)", flush=True)
    make_plots()
    verdicts()


# ═════════════════════════════════════════════════════════════════════════════
def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = pd.read_csv(OUT / "d1_cells.csv")
    tax = pd.read_csv(OUT / "d1_taxis.csv")

    for name, col, ylab in (("d1_theta_by_decile.png", "theta_med", "median theta (deg)"),
                            ("d1_absm_by_decile.png", "absm_med", "median |m|")):
        js = [j for j in (0, 1, 4) if j in set(cells.j)]
        fig, axes = plt.subplots(1, len(js), figsize=(4.2 * len(js), 3.6), squeeze=False)
        for ax, j in zip(axes[0], js):
            sub = cells[(cells.j == j) & (cells.teacher == "tp")
                        & (cells.t_sel == "all")]
            for i in sorted(sub.i.unique()):
                s = sub[sub.i == i].sort_values("decile")
                ax.plot(s.decile, s[col], marker="o", ms=3, label=f"i={i}")
            ax.set_title(f"j={j} (T=Tp)"); ax.set_xlabel("teacher rho decile (1=slowest)")
            ax.set_ylabel(ylab); ax.grid(alpha=.3)
        axes[0][-1].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(OUT / name, dpi=140); plt.close(fig)

    # 히트맵: 느림(1-3) − 빠름(8-10) 의 median θ 차이, T=To
    sub = cells[(cells.teacher == "to") & (cells.t_sel == "all")]
    js, iss = sorted(sub.j.unique()), sorted(sub.i.unique())
    M = np.full((len(js), len(iss)), np.nan)
    for a, j in enumerate(js):
        for b, i in enumerate(iss):
            s = sub[(sub.j == j) & (sub.i == i)]
            if s.empty:
                continue
            slow = s[s.decile <= 3].theta_med.mean()
            fast = s[s.decile >= 8].theta_med.mean()
            M[a, b] = slow - fast
    fig, ax = plt.subplots(figsize=(1 + .6 * len(iss), 1 + .5 * len(js)))
    im = ax.imshow(M, cmap="RdBu_r", aspect="auto",
                   vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)))
    ax.set_xticks(range(len(iss)), iss); ax.set_yticks(range(len(js)), js)
    ax.set_xlabel("i (student stage)"); ax.set_ylabel("j (past task)")
    ax.set_title("median theta:  decile 1-3  minus  decile 8-10   (T=To)")
    fig.colorbar(im, label="delta theta (deg)")
    fig.tight_layout(); fig.savefig(OUT / "d1_theta_gap_heatmap.png", dpi=140); plt.close(fig)

    # t 구간별 MSE 기여
    b = tax[tax.t_bin >= 0].groupby("t_bin")[["e_mean", "e_weighted_mean"]].mean()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    w = 0.38; x = np.arange(len(b))
    ax.bar(x - w / 2, b.e_mean / b.e_mean.sum(), w, label="raw MSE")
    ax.bar(x + w / 2, b.e_weighted_mean / b.e_weighted_mean.sum(), w, label="(1-t)^2 weighted")
    ax.set_xticks(x, [f"{i/5:.1f}-{(i+1)/5:.1f}" for i in range(len(b))])
    ax.set_xlabel("t bin"); ax.set_ylabel("normalized contribution")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "d1_t_contribution.png", dpi=140); plt.close(fig)

    # 보조: v-tilde 프레임이 유효한 t<=0.3 만으로 본 theta 십분위 (근거는 REPORT 5절)
    js = [j for j in (0, 1, 4) if j in set(cells.j)]
    fig, axes = plt.subplots(1, len(js), figsize=(4.2 * len(js), 3.6), squeeze=False)
    for ax, j in zip(axes[0], js):
        sub = cells[(cells.j == j) & (cells.teacher == "tp") & (cells.t_sel == "t_le_0.3")]
        for i in sorted(sub.i.unique()):
            g = sub[sub.i == i].sort_values("decile")
            ax.plot(g.decile, g.theta_med, marker="o", ms=3, label=f"i={i}")
        ax.set_title(f"j={j} (T=Tp, t<=0.3)"); ax.set_xlabel("teacher rho decile (1=slow)")
        ax.set_ylabel("median theta (deg)"); ax.grid(alpha=.3)
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "d1_theta_by_decile_tle03.png", dpi=140); plt.close(fig)
    print("[D1] 그림 5장 저장", flush=True)


def verdicts():
    """P1–P3 판정을 JSON 으로 남긴다. REPORT.md 2절이 이걸 읽는다."""
    cells = pd.read_csv(OUT / "d1_cells.csv")
    tax = pd.read_csv(OUT / "d1_taxis.csv")
    sane = pd.read_csv(OUT / "d1_sanity.csv")
    v = {}

    def gap(df, col):
        s = df[df.decile <= 3][col].mean()
        f = df[df.decile >= 8][col].mean()
        return s - f, s, f

    for sel in sorted(cells.t_sel.unique()):
        C = cells[(cells.teacher == "to") & (cells.t_sel == sel)]
        # P1: 느린 십분위 θ 가 더 크고, 그 격차가 i−j 와 함께 커진다
        rows = []
        for (j, i), g in C.groupby(["j", "i"]):
            d, sl, f = gap(g, "theta_med")
            rows.append({"j": j, "i": i, "lag": i - j, "gap": d, "slow": sl, "fast": f})
        R = pd.DataFrame(rows)
        corr = float(np.corrcoef(R.lag, R.gap)[0, 1]) if len(R) > 2 else float("nan")
        v[f"P1[{sel}]"] = {
            "gap_mean": float(R.gap.mean()), "gap_frac_positive": float((R.gap > 0).mean()),
            "corr_gap_lag": corr, "slow_mean": float(R.slow.mean()),
            "fast_mean": float(R.fast.mean()), "n_pairs": len(R)}
        # P2: |m| 에는 십분위 격차가 없다
        rows = []
        for (j, i), g in C.groupby(["j", "i"]):
            d, sl, f = gap(g, "absm_med")
            rows.append({"gap": d, "slow": sl, "fast": f})
        Rm = pd.DataFrame(rows)
        v[f"P2[{sel}]"] = {
            "gap_mean": float(Rm.gap.mean()), "gap_frac_positive": float((Rm.gap > 0).mean()),
            "slow_mean": float(Rm.slow.mean()), "fast_mean": float(Rm.fast.mean()),
            "note": "|m| 은 분모가 ‖ṽ_T‖ 라 느린 십분위에서 구조적으로 부풀 수 있다"}

    # P3: raw MSE 의 과반이 t>0.8, 가중 후에는 작다
    hi = tax[tax.t_bin == -1]
    v["P3"] = {"share_raw_mean": float(hi.e_share.mean()),
               "share_weighted_mean": float(hi.e_share_weighted.mean())}
    v["sanity"] = {"max_ratio": float(sane.sanity_ratio.max()),
                   "mean_ratio": float(sane.sanity_ratio.mean()),
                   "threshold": 0.2,
                   "ok": bool(sane.sanity_ratio.max() <= 0.2)}
    (OUT / "d1_verdicts.json").write_text(json.dumps(v, indent=2, ensure_ascii=False))
    print(json.dumps(v, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
