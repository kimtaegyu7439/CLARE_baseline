#!/usr/bin/env python
"""K6 — DiT(teacher) 출력 기하 관찰. U-신호가 살아 있는가. 분석 전용.

"teacher-일관성 score" 정련의 전제를 설계 전에 관찰로 판정한다.
    U(b) = Var_{ε,t}[ g(x_t, t, b, ℓ_0) ]  가 manifold 거리의 대리이고,
    그 경사가 조향 가능한가?

정의
    프로브 집합 S   ε 4개 × t ∈ {0.1, 0.5} = 8 조합. 시작 시 1회 생성해 고정.
    x_t = (1−t)·ε                     ★ 행동 데이터를 쓰지 않는다(a 항 제외).
    v   = velocity_net(x_t, t, b, ℓ_0)
    g   = v + ε                       flow matching 목표가 a−ε 이므로 g ≈ a.
                                      ε 의존을 걷어낸 "함의된 행동" 이다.
    U(b)     = mean_S ‖g − ḡ(b)‖²     ḡ 는 S 위 평균
    U_raw(b) = 같은 식을 v 로

state 는 0 으로 고정한다. b 이외의 것으로 U 가 움직이면 관찰이 성립하지 않기 때문이다
(K5 witness 규약과 같은 이유).

α 이동의 정의 (Q3)
    방향 d 를 σ 단위로 정규화해 좌표별 RMS 변위가 정확히 α·σ 가 되게 한다.
        s = d/σ,  û = s / rms(s),  b(α) = b + α · σ ⊙ û
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sst
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B_merge import _ns

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata   # noqa: E402
from lerobot.policies.factory import make_policy                      # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging   # noqa: E402

FWD = {"n": 0}          # Q6 — teacher forward 호출 수


# ═════════════════════════════════════════════════════════════════════════════


def forward_v(policy, b, eps, tval, instr, n_obs=2, state_dim=8):
    """x_t = (1−t)·ε 로 구성한 velocity. grad 는 바깥 컨텍스트가 정한다."""
    n = b.shape[0]
    dev = b.device
    dt = next(policy.parameters()).dtype
    state = torch.zeros(n, n_obs, state_dim, device=dev, dtype=dt)
    tail = B1.cond_tail(policy, {"observation.state": state}, b.reshape(-1, b.shape[-1]).to(dt))
    lang = B1.encode_lang(policy, [instr] * n)
    cond = B1.make_cond(lang.to(dt), tail.to(dt))
    e = eps.to(dev, dt).expand(n, -1, -1)
    x_t = (1.0 - tval) * e
    t = torch.full((n,), float(tval), device=dev, dtype=dt)
    FWD["n"] += 1
    return policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond)


def probe_outputs(policy, b, probes, instr):
    """S 위의 (v, g). 반환 (|S|, n, H, A) 두 개."""
    vs, gs = [], []
    for eps, tv in probes:
        v = forward_v(policy, b, eps, tv, instr)
        vs.append(v)
        gs.append(v + eps.to(v.device, v.dtype))
    return torch.stack(vs), torch.stack(gs)


def spread(x):
    """U = mean_S ‖x − x̄‖². per-sample. x (S, n, H, A) -> (n,)."""
    m = x.mean(0, keepdim=True)
    return ((x - m) ** 2).mean(0).flatten(1).mean(1)


@torch.no_grad()
def U_of(policy, b, probes, instr, chunk=128):
    """U 와 U_raw. 큰 배치는 잘라서."""
    Us, Ur = [], []
    for s in range(0, b.shape[0], chunk):
        v, g = probe_outputs(policy, b[s:s + chunk], probes, instr)
        Us.append(spread(g).cpu()); Ur.append(spread(v).cpu())
    return torch.cat(Us), torch.cat(Ur)


def grad_U(policy, b, probes, instr):
    """∇_b U. b 는 leaf 여야 한다."""
    b = b.detach().requires_grad_(True)
    with torch.enable_grad():
        v, g = probe_outputs(policy, b, probes, instr)
        U = spread(g).sum()
        gr, = torch.autograd.grad(U, b)
    return gr.detach()


def sigma_step(base, direction, sigma, alpha):
    """좌표별 RMS 변위가 α·σ 가 되도록 이동."""
    s = direction / sigma.clamp_min(1e-6)
    rms = s.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-12)
    u = s / rms[:, None, None]
    return base + alpha * sigma * u


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--smoke_dir", default="results/K5_spatial_10task_smoke_M8")
    ap.add_argument("--teacher", default=None,
                    help="기본: <outputs>/<smoke>/task_0/checkpoints/last/pretrained_model")
    ap.add_argument("--cache", default="results/K0/emb_cache")
    ap.add_argument("--bins", default="2,5,8")
    ap.add_argument("--n_per_bin", type=int, default=256)
    ap.add_argument("--n_path", type=int, default=64, help="Q3 경로용 프레임 수")
    ap.add_argument("--n_grad", type=int, default=32, help="Q5 경사용 프레임 수")
    ap.add_argument("--n_eps", type=int, default=4, help="프로브 ε 개수")
    ap.add_argument("--n_multi", type=int, default=16, help="Q4 다봉성용 ε 개수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/K6")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    a = ap.parse_args()

    init_logging()
    torch.manual_seed(a.seed)
    g_cpu = torch.Generator().manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    sm = Path(a.smoke_dir)
    bins = [int(x) for x in a.bins.split(",")]
    TS = [0.1, 0.5]

    tpath = a.teacher or str(REPO / "outputs" / sm.name /
                             f"{a.suite}_seed42_ours/task_0/checkpoints/last/pretrained_model")
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = json.loads((sm / "instructions.json").read_text())["task0"]
    cfg = B1.build_cfg(_ns(a), 0, tpath, Path("/tmp/k6"))
    teacher = make_policy(cfg=cfg.policy, ds_meta=meta)
    teacher.eval(); teacher.requires_grad_(False)
    H = teacher.config.horizon
    A = teacher.config.action_feature.shape[0]

    st0 = torch.load(sm / "stats" / "task0.pt")
    mu0, sg0 = st0["mu"].to(device).float(), st0["sigma"].to(device).float()
    d = torch.load(Path(a.cache) / f"{a.suite}_task0.pt")
    X = d["X"].float().view(-1, 4, 768)
    T = d["T"].long()

    EPS = [torch.randn(1, H, A, generator=g_cpu) for _ in range(a.n_multi + a.n_eps)]
    probes = [(EPS[i], t) for t in TS for i in range(a.n_eps)]
    probes16 = [(EPS[i], t) for t in TS for i in range(8)]
    probes16b = [(EPS[i + 8], t) for t in TS for i in range(8)]
    print(f"[K6] teacher={tpath}")
    print(f"[K6] ℓ_0={instr!r}  |S|={len(probes)} (ε {a.n_eps} × t {TS})  seed={a.seed}  "
          f"bins={bins}  n/bin={a.n_per_bin}  state=0  x_t=(1−t)·ε (행동 미사용)", flush=True)

    R = {"config": vars(a), "probe_seed": a.seed, "t_values": TS, "bins": {}}
    ALPHA = [0, 0.5, 1, 2, 3, 4]

    for tb in bins:
        idx = torch.nonzero(T == tb, as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g_cpu)[:a.n_per_bin]]
        real = X[idx].to(device)
        n = real.shape[0]
        z = torch.randn(n, 4, 768, generator=g_cpu).clamp_(-3, 3).to(device)
        gau = (mu0[tb] + sg0[tb] * z).detach()
        sig = sg0[tb].expand_as(real)
        e = {}

        Ur_real, Uraw_real = U_of(teacher, real, probes, instr)
        Ur_gau, _ = U_of(teacher, gau, probes, instr)
        e["Q1"] = {"med_U_real": float(Ur_real.median()),
                   "med_Uraw_real": float(Uraw_real.median()),
                   "ratio_U_over_Uraw": float(Ur_real.median() / Uraw_real.median())}
        for tv in TS:
            pr = [(EPS[i], tv) for i in range(a.n_eps)]
            u1, u0 = U_of(teacher, real, pr, instr)
            e["Q1"][f"t{tv}"] = {"med_U": float(u1.median()), "med_Uraw": float(u0.median()),
                                 "ratio": float(u1.median() / u0.median())}
        lab = np.r_[np.zeros(n), np.ones(n)]
        sc = np.r_[Ur_real.numpy(), Ur_gau.numpy()]
        e["Q2"] = {"med_U_real": float(Ur_real.median()), "med_U_gauss": float(Ur_gau.median()),
                   "separation": float(Ur_gau.median() / Ur_real.median().clamp_min(1e-12)),
                   "auroc": float(roc_auc_score(lab, sc))}

        m = a.n_path
        base = real[:m]
        perm = torch.randperm(n, generator=g_cpu)[:m].to(device)
        dirs = {"normal": gau[:m] - base,
                "random": torch.randn(m, 4, 768, generator=g_cpu).to(device),
                "tangent": real[perm] - base}
        q3 = {}
        for kd, dd in dirs.items():
            curves = []
            for al in ALPHA:
                bb = base if al == 0 else sigma_step(base, dd, sig[:m], al)
                u, _ = U_of(teacher, bb, probes, instr)
                curves.append(u.numpy())
            C = np.stack(curves)
            rel = C / np.maximum(C[0], 1e-12)
            med = np.median(rel, 1)
            sp = sst.spearmanr(ALPHA[:5], med[:5]).statistic
            q3[kd] = {"alpha": ALPHA, "med": med.tolist(),
                      "q25": np.percentile(rel, 25, axis=1).tolist(),
                      "q75": np.percentile(rel, 75, axis=1).tolist(),
                      "spearman_alpha_le3": float(sp), "U3_over_U0": float(med[4])}
        e["Q3"] = q3

        pr16 = [(EPS[i], 0.1) for i in range(a.n_multi)]
        with torch.no_grad():
            _, G = probe_outputs(teacher, real[:m], pr16, instr)
        Gn = G.permute(1, 0, 2, 3).reshape(G.shape[1], a.n_multi, -1).cpu().numpy()
        sil = []
        for i in range(Gn.shape[0]):
            km = KMeans(n_clusters=2, n_init=4, random_state=a.seed).fit(Gn[i])
            if len(set(km.labels_)) > 1:
                sil.append(silhouette_score(Gn[i], km.labels_))
        sil = np.array(sil)
        e["Q4"] = {"silhouette_median": float(np.median(sil)),
                   "frac_gt_0.4": float(np.mean(sil > 0.4)), "n": int(len(sil))}

        gq = {}
        S1 = [(EPS[i], t) for t in TS for i in range(a.n_eps)]
        S2 = [(EPS[i + a.n_eps], t) for t in TS for i in range(a.n_eps)]
        for kd, bb in (("real", real[:a.n_grad]), ("gauss", gau[:a.n_grad])):
            g1, g2 = grad_U(teacher, bb, S1, instr), grad_U(teacher, bb, S2, instr)
            cos = torch.nn.functional.cosine_similarity(g1.flatten(1), g2.flatten(1), dim=1)
            h1 = grad_U(teacher, bb, probes16, instr)
            h2 = grad_U(teacher, bb, probes16b, instr)
            cos16 = torch.nn.functional.cosine_similarity(h1.flatten(1), h2.flatten(1), dim=1)
            gq[kd] = {"cos_S8": cos.cpu().numpy().tolist(),
                      "cos_S8_median": float(cos.median()),
                      "cos_S16_median": float(cos16.median()),
                      "delta": float(cos16.median() - cos.median())}
        e["Q5"] = gq

        R["bins"][str(tb)] = e
        print(f"[K6] bin{tb}  U/U_raw {e['Q1']['ratio_U_over_Uraw']:.3f}  "
              f"sep {e['Q2']['separation']:.2f}  AUROC {e['Q2']['auroc']:.3f}  "
              f"normal ρ {q3['normal']['spearman_alpha_le3']:+.2f}  "
              f"tan U3/U0 {q3['tangent']['U3_over_U0']:.2f}  "
              f"cos {gq['real']['cos_S8_median']:.3f}  "
              f"silh {e['Q4']['silhouette_median']:.3f}", flush=True)
        del real, gau, z, G, Gn
        torch.cuda.empty_cache()

    b1 = X[:64].to(device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    U_of(teacher, b1, probes, instr); torch.cuda.synchronize()
    t_U = time.perf_counter() - t0
    t0 = time.perf_counter(); grad_U(teacher, b1, probes, instr); torch.cuda.synchronize()
    t_g = time.perf_counter() - t0
    R["Q6"] = {"forwards_per_U": len(probes), "forwards_per_gradU": len(probes),
               "t_U_s_batch64": t_U, "t_gradU_s_batch64": t_g, "total_forwards": FWD["n"]}

    B = R["bins"]
    sep = float(np.mean([B[str(t)]["Q2"]["separation"] for t in bins]))
    rho_n = float(np.mean([B[str(t)]["Q3"]["normal"]["spearman_alpha_le3"] for t in bins]))
    tan = float(np.mean([B[str(t)]["Q3"]["tangent"]["U3_over_U0"] for t in bins]))
    cos_r = float(np.mean([B[str(t)]["Q5"]["real"]["cos_S8_median"] for t in bins]))
    u_ratio = float(np.mean([B[str(t)]["Q1"]["ratio_U_over_Uraw"] for t in bins]))
    if sep < 2 or rho_n < 0.9:
        verdict = ("트랙 폐쇄 — behavior-level 신호도 임베딩 조향에 불충분. "
                   "negative result 로 기록하고 동역학 축에 전념.")
    elif sep >= 3 and rho_n >= 0.9 and tan < 2 and cos_r >= 0.7:
        verdict = "정련 진행 — 네 조건 모두 충족."
    elif sep >= 3 and rho_n >= 0.9 and tan < 2:
        verdict = "재배분 강등 — U 는 ranking 용도로만(가중 재배분), 조향 불가."
    else:
        verdict = (f"조건부 미달 — separation {sep:.2f}, normal ρ {rho_n:+.2f}, "
                   f"tangent U3/U0 {tan:.2f}, cos {cos_r:.3f}.")
    flag = (f"⚠ g-보정 무효 — U 정의 재설계 필요 (med U / med U_raw = {u_ratio:.3f} > 0.7)"
            if u_ratio > 0.7 else f"g-보정 유효 (med U / med U_raw = {u_ratio:.3f})")
    R["verdict"] = verdict; R["g_correction_flag"] = flag
    R["aggregate"] = {"separation": sep, "normal_spearman": rho_n, "tangent_U3_U0": tan,
                      "cos_median_real": cos_r, "U_over_Uraw": u_ratio}

    # ── 그림 ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(len(bins), 2, figsize=(11.5, 3.4 * len(bins)), squeeze=False)
    for r, tb in enumerate(bins):
        idx = torch.nonzero(T == tb, as_tuple=True)[0][:a.n_per_bin]
        real = X[idx].to(device)
        zz = torch.randn(real.shape[0], 4, 768,
                         generator=torch.Generator().manual_seed(a.seed + tb)).clamp_(-3, 3)
        gau = mu0[tb] + sg0[tb] * zz.to(device)
        u1r, u0r = U_of(teacher, real, probes, instr)
        u1g, u0g = U_of(teacher, gau, probes, instr)
        for c, (rr, gg, nm) in enumerate(((u0r, u0g, "$U_{raw}$  (v)"),
                                          (u1r, u1g, "$U$  (g = v + ε)"))):
            axx = ax[r][c]
            lo = float(min(rr.min(), gg.min())); hi = float(max(rr.max(), gg.max()))
            bb = np.linspace(lo, hi, 50)
            axx.hist(rr.numpy(), bins=bb, color="0.45", alpha=.7, label="real")
            axx.hist(gg.numpy(), bins=bb, color="crimson", alpha=.55, label="gaussian")
            axx.set_title(f"{nm}   bin {tb}", fontsize=10.5)
            axx.set_xlabel("spread"); axx.set_ylabel("count" if c == 0 else "")
            if r == 0 and c == 0:
                axx.legend(fontsize=9, frameon=False)
        ax[r][1].text(.98, .95, f"separation {B[str(tb)]['Q2']['separation']:.2f}\n"
                                f"AUROC {B[str(tb)]['Q2']['auroc']:.3f}",
                      transform=ax[r][1].transAxes, va="top", ha="right", fontsize=9)
        del real, gau
    fig.suptitle("K6 · A — output spread of the teacher: raw v vs ε-corrected g", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "figA_spread.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(1, len(bins), figsize=(4.6 * len(bins), 4.4), squeeze=False)
    CC = {"normal": "crimson", "random": "darkorange", "tangent": "#2f6db5"}
    for i, tb in enumerate(bins):
        axx = ax[0][i]; q3 = B[str(tb)]["Q3"]
        for kd, col in CC.items():
            al = q3[kd]["alpha"]
            axx.plot(al, q3[kd]["med"], "-o", color=col, lw=2, ms=5, label=kd)
            axx.fill_between(al, q3[kd]["q25"], q3[kd]["q75"], color=col, alpha=.15)
        axx.axhline(1, color="0.6", lw=.8, ls=":")
        axx.set_xlabel(r"$\alpha$  (RMS displacement in $\sigma$ units)")
        if i == 0:
            axx.set_ylabel(r"$U(\alpha)\,/\,U(0)$"); axx.legend(fontsize=9, frameon=False)
        axx.set_title(f"bin {tb}   normal $\\rho$={q3['normal']['spearman_alpha_le3']:+.2f}",
                      fontsize=10.5)
    fig.suptitle("K6 · B — does U rise when leaving the sheet, and stay flat along it?",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "figB_path.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    allr = np.concatenate([B[str(t)]["Q5"]["real"]["cos_S8"] for t in bins])
    allg = np.concatenate([B[str(t)]["Q5"]["gauss"]["cos_S8"] for t in bins])
    bb = np.linspace(-1, 1, 41)
    ax[0].hist(allr, bins=bb, color="0.45", alpha=.7, label="real")
    ax[0].hist(allg, bins=bb, color="crimson", alpha=.55, label="gaussian")
    ax[0].axvline(0.7, color="k", ls="--", lw=1.2, label="threshold 0.7")
    ax[0].set_xlabel(r"$\cos(\nabla_b U|_S,\ \nabla_b U|_{S'})$"); ax[0].set_ylabel("count")
    ax[0].set_title(f"gradient reproducibility  |S|=8   median real {np.median(allr):.3f}",
                    fontsize=10.5)
    ax[0].legend(fontsize=9, frameon=False)
    w = 0.35; xs = np.arange(len(bins))
    ax[1].bar(xs - w / 2, [B[str(t)]["Q5"]["real"]["cos_S8_median"] for t in bins], w,
              color="0.45", label="|S|=8")
    ax[1].bar(xs + w / 2, [B[str(t)]["Q5"]["real"]["cos_S16_median"] for t in bins], w,
              color="seagreen", label="|S|=16")
    ax[1].axhline(0.7, color="k", ls="--", lw=1.2)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"bin {t}" for t in bins])
    ax[1].set_ylabel("median cosine (real)"); ax[1].set_ylim(-0.1, 1)
    ax[1].set_title("more probes → better gradient?", fontsize=10.5)
    ax[1].legend(fontsize=9, frameon=False)
    fig.suptitle("K6 · C — is the U gradient steerable?", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "figC_grad.png", dpi=300); plt.close(fig)

    json.dump(R, (out / "probe.json").open("w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f"{'bin':>4}{'U/U_raw':>10}{'separation':>12}{'AUROC':>8}{'normal ρ':>10}"
          f"{'random ρ':>10}{'tan U3/U0':>11}{'cos|S|8':>9}{'cos|S|16':>10}"
          f"{'silh med':>10}{'silh>0.4':>10}")
    print("-" * 100)
    for tb in bins:
        e = B[str(tb)]
        print(f"{tb:>4}{e['Q1']['ratio_U_over_Uraw']:10.3f}{e['Q2']['separation']:12.2f}"
              f"{e['Q2']['auroc']:8.3f}{e['Q3']['normal']['spearman_alpha_le3']:+10.2f}"
              f"{e['Q3']['random']['spearman_alpha_le3']:+10.2f}"
              f"{e['Q3']['tangent']['U3_over_U0']:11.2f}"
              f"{e['Q5']['real']['cos_S8_median']:9.3f}"
              f"{e['Q5']['real']['cos_S16_median']:10.3f}"
              f"{e['Q4']['silhouette_median']:10.3f}{e['Q4']['frac_gt_0.4']*100:9.0f}%")
    print("-" * 100)
    ag = R["aggregate"]
    print(f"{'평균':>4}{ag['U_over_Uraw']:10.3f}{ag['separation']:12.2f}{'':>8}"
          f"{ag['normal_spearman']:+10.2f}{'':>10}{ag['tangent_U3_U0']:11.2f}"
          f"{ag['cos_median_real']:9.3f}")
    print("\nQ3 곡선 (U(α)/U(0) 중앙값)")
    for tb in bins:
        for kd in ("normal", "random", "tangent"):
            q = B[str(tb)]["Q3"][kd]
            print(f"  bin{tb} {kd:<8} " + "  ".join(
                f"α={al}: {v:.2f}" for al, v in zip(q["alpha"], q["med"])))
    print("\nQ1 t별 분해 (실제 프레임)")
    for tb in bins:
        e = B[str(tb)]["Q1"]
        print(f"  bin{tb}  " + "   ".join(
            f"t={tv}: U {e[f't{tv}']['med_U']:.4g} / U_raw {e[f't{tv}']['med_Uraw']:.4g} "
            f"= {e[f't{tv}']['ratio']:.3f}" for tv in TS))
    q6 = R["Q6"]
    print(f"\nQ6 비용  U 1회 = forward {q6['forwards_per_U']}회, batch64 "
          f"{q6['t_U_s_batch64']*1000:.0f} ms   |   ∇U 1회 = forward "
          f"{q6['forwards_per_gradU']}회 + backward, batch64 "
          f"{q6['t_gradU_s_batch64']*1000:.0f} ms   |   총 forward {q6['total_forwards']}회")
    print(f"\n{flag}")
    print("=" * 100)
    print(f"판정: {verdict}")
    print("=" * 100)
    print(f"\n그림  {out/'figA_spread.png'}\n      {out/'figB_path.png'}"
          f"\n      {out/'figC_grad.png'}\n요약  {out/'probe.json'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
