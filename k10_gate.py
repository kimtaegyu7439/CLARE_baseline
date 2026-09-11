#!/usr/bin/env python
"""K10 게이트 — Langevin 표본이 manifold 에 분포로서 접근하는가. 무학습, 판정 전용.

배경
    K5b  E_wit (활성 통계)   판별은 하는데(ratio 65~130) d̂ 1.87 -> 1.87, 안 끌어올림  -> 기각
    K6   E_U   (출력 퍼짐)   신호는 잘 정의되나 대상이 manifold 거리가 아님          -> 폐쇄
  둘 다 **결정론 하강**으로 실패했다. 여기서는 두 에너지의 **곱**(= 합산 에너지)에서
  경사 + 온도 노이즈 + annealing 으로 **표본**을 뽑아, 교집합이 manifold 에 분포로서
  가까워지는지 d̂ 로 본다. T0(노이즈 0)이 같은 격자 안의 대조군이다.

샘플러 (Phase 2 의 K10-L 도 이 함수를 그대로 쓴다)
    coords=collective   ζ∈R^32,  b(ζ) = μ_0[τ] + V_0[τ]·ζ + σ⊥⊙ε_res   (ε_res 표본별 고정)
    coords=full         b 3072-d 직접,  prior 는 ‖(b−μ)/σ‖²/2
    초기화              ζ₀ ~ N(0, Λ_0[τ])        (full: b₀ = μ + σ⊙z)
    에너지              E = w_wit·Ê_wit + w_U·Ê_U + 0.5‖Λ^{-1/2}ζ‖²
                        Ê 는 각 에너지를 "실제 프레임 128개의 중앙값"으로 나눈 무차원량
    갱신                ζ ← ζ − η∇_ζE + noise·√(2ηT_m)·ξ,  ξ~N(0,I),  M=40
    η                   첫 스텝 b-공간 RMS 이동 = 0.1·mean(σ_0[τ]) 이 되도록 1회 자동 설정
    T-mode              T0(noise=0) / const(0.3) / anneal(1.0 -> 0.05 기하)

표본 b 는 저장하지 않는다. 실제 프레임은 이 스크립트의 측정에만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import k5_wstats as WS
import k5b_bench as K5B
import k6_probe as K6
from B_merge import _ns

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata   # noqa: E402
from lerobot.policies.factory import make_policy                      # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging   # noqa: E402

ARMW = {"wit": (1.0, 0.0), "U": (0.0, 1.0), "prod": (1.0, 1.0)}
TMODES = ["T0", "const", "anneal"]


# ═════════════════════════════════════════════════════════════════════════════
def bin_pca(X, T, n_bins, r, device):
    """bin 별 상위 r 주성분 V (3072,r), 고유값 Λ (r,), 잔여 좌표별 std σ⊥ (3072,)."""
    V, L, S = {}, {}, {}
    for t in range(n_bins):
        x = X[T == t]
        if x.shape[0] < r + 2:
            continue
        x = x.to(device).flatten(1)
        c = x.mean(0, keepdim=True)
        xc = x - c
        G = (xc @ xc.T).double()
        ev, U = torch.linalg.eigh(G)
        ev = ev.flip(0).clamp_min(0.0); U = U.flip(1)
        k = min(r, xc.shape[0] - 1)
        s = ev[:k].sqrt().clamp_min(1e-8)
        Vt = ((U[:, :k].T.float() @ xc) / s[:, None].float())        # (r, 3072) 정규직교
        lam = (ev[:k] / max(1, xc.shape[0] - 1)).float()
        res = xc - (xc @ Vt.T) @ Vt
        V[t] = Vt.T.contiguous().cpu()                               # (3072, r)
        L[t] = lam.cpu()
        S[t] = res.std(0).cpu()
        del x, xc, G, U, res
        torch.cuda.empty_cache()
    return V, L, S


def E_wit_fn(net, b, tau, wstats, blocks, instr, eps_probe, n_bins, floor):
    """K5b maha 에너지. 배치 통계라 스칼라 하나가 나온다."""
    with WS.BlockTap(net, blocks) as tap:
        tap.acts.clear()
        WS.witness_forward(net, b.reshape(-1, b.shape[-1]), instr, eps_probe, tap)
        return K5B.energy_var(tap.acts, tau, wstats, n_bins, "maha", floor).float()


def E_U_fn(net, b, probes, instr):
    """K6 의 U. per-sample."""
    _, g = K6.probe_outputs(net, b, probes, instr)
    return K6.spread(g)


def make_b(coords, zeta, mu, V, sperp, eps_res, sigma):
    if coords == "full":
        return zeta
    return mu + zeta @ V.T + sperp * eps_res


def prior_of(coords, zeta, lam, mu, sigma):
    if coords == "full":
        return 0.5 * (((zeta - mu) / sigma.clamp_min(1e-6)) ** 2).flatten(1).sum(1)
    return 0.5 * ((zeta ** 2) / lam.clamp_min(1e-8)).sum(1)


def temperature(mode, m, M):
    if mode == "T0":
        return 0.0, 0.0
    if mode == "const":
        return 0.3, 1.0
    return float(1.0 * (0.05 / 1.0) ** (m / max(1, M - 1))), 1.0     # anneal


def langevin(net, tb, coords, w_wit, w_U, tmode, M, mu, sig, V, lam, sperp,
             wstats, blocks, instr, eps_probe, probes, n_bins, floor,
             norm_wit, norm_U, n, gen, device, log=None,
             eta_target=0.02, step_clip=3.0):
    """표본 n 개를 M 스텝 굴린다. 반환 (b_final, 진단).

    ★ v1 실측에서 30조합 중 19개가 발산했다(노이즈 조합 전멸, T0 도 3~4스텝 뒤 상승).
      원인은 두 가지였고 둘 다 여기서 고친다.

      (a) η 과대   첫 스텝 b-공간 RMS 이동 목표를 0.1·σ -> **eta_target(기본 0.02)·σ** 로 낮춘다.
      (b) 노이즈 단위 불일치
          갱신식 ζ ← ζ − η∇E + √(2ηT)ξ 에서 η 는 **경사 크기로 보정된 값**이라
          √(2ηT) 를 ζ 에 그대로 더하면 ζ 의 자연 척도(√Λ)와 무관한 크기가 된다.
          prior 가 0.5‖Λ^{-1/2}ζ‖² 이므로 이 좌표의 자연 계량은 Λ 다. 따라서 노이즈를
          **√Λ 로 전처리**한다(full 좌표에서는 σ). 이것이 이 prior 와 정합적인 형태다.

      추가 안전장치: 스텝당 b-공간 RMS 이동을 step_clip·eta_target·σ 로 상한 (기본 3배).
      경사가 튀어도 한 스텝이 폭주하지 않는다. 발동 비율을 로그에 남긴다.
    """
    r = lam.shape[0] if coords != "full" else None
    eps_res = torch.randn(n, 3072, generator=gen).to(device) if coords != "full" else None
    if coords == "full":
        z0 = torch.randn(n, 4, 768, generator=gen).clamp_(-3, 3).to(device)
        zeta = (mu + sig * z0).reshape(n, -1)
    else:
        zeta = (torch.randn(n, r, generator=gen).to(device) * lam.sqrt())
    mu_f, sig_f = mu.reshape(1, -1), sig.reshape(1, -1)
    tau = torch.full((n,), tb, dtype=torch.long, device=device)
    eta, traj, clip_hits = None, [], 0.0
    target_rms = eta_target * float(sig.mean())
    cap = step_clip * target_rms
    pre = lam.sqrt() if coords != "full" else sig_f          # 노이즈 전처리 행렬(대각)

    def energy(zt):
        b = make_b(coords, zt, mu_f, V, sperp, eps_res, sig_f).reshape(n, 4, 768)
        tot = prior_of(coords, zt, lam, mu_f, sig_f).mean()
        if w_wit:
            tot = tot + w_wit * E_wit_fn(net, b, tau, wstats, blocks, instr,
                                         eps_probe, n_bins, floor) / norm_wit
        if w_U:
            tot = tot + w_U * E_U_fn(net, b, probes, instr).mean() / norm_U
        return tot

    for m in range(M):
        zeta = zeta.detach().requires_grad_(True)
        with torch.enable_grad():
            E = energy(zeta)
            g, = torch.autograd.grad(E, zeta)
        if eta is None:
            db = (-g if coords == "full" else -(g @ V.T))
            rms = db.pow(2).mean(1).sqrt().median().clamp_min(1e-12)
            eta = float(target_rms / rms)
            if log is not None:
                log.append(f"η={eta:.4g} (target RMS {target_rms:.4g}, cap {cap:.4g})")
        T, ns = temperature(tmode, m, M)
        step = -eta * g
        if ns > 0 and T > 0:
            step = step + ns * float(np.sqrt(2 * eta * T)) * pre * torch.randn(
                zeta.shape, generator=gen).to(device)
        # b-공간 RMS 이동 상한
        db = step if coords == "full" else step @ V.T
        rms = db.pow(2).mean(1).sqrt().clamp_min(1e-12)
        fac = (cap / rms).clamp(max=1.0)
        clip_hits += float((fac < 1).float().mean()) / M
        zeta = (zeta + step * fac[:, None]).detach()
        traj.append(float(E.detach()))
        if not np.isfinite(traj[-1]):
            break
    b = make_b(coords, zeta, mu_f, V, sperp, eps_res, sig_f).reshape(n, 4, 768).detach()
    return b, {"eta": eta, "E_traj": traj, "clip": clip_hits}


def dhat(b, real, d_real):
    return float(torch.cdist(b.flatten(1), real.flatten(1)).min(1).values.median() / d_real)


def diversity(b0, b1):
    d0 = torch.pdist(b0.flatten(1)).median().clamp_min(1e-12)
    return float(torch.pdist(b1.flatten(1)).median() / d0)


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=list(ARMW), required=True)
    ap.add_argument("--gpu_tag", default=None, help="results/K10/gpu{n} 의 n. 기본은 arm 순서")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--smoke_dir", default="results/K5_spatial_10task_smoke_M8")
    ap.add_argument("--net", default=None, help="기본: smoke 의 task0 종료 스냅샷")
    ap.add_argument("--cache", default="results/K0/emb_cache")
    ap.add_argument("--bins", default="2,5,8")
    ap.add_argument("--n", type=int, default=256, help="조합당 표본")
    ap.add_argument("--n_norm", type=int, default=128, help="Ê 무차원화용 실제 프레임")
    ap.add_argument("--M", type=int, default=40)
    ap.add_argument("--eta_target", type=float, default=0.02,
                    help="첫 스텝 b-공간 RMS 이동 목표 (σ 배수). v1 의 0.1 은 발산했다.")
    ap.add_argument("--step_clip", type=float, default=3.0,
                    help="스텝당 b-공간 RMS 이동 상한 = step_clip × eta_target × σ")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/K10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    a = ap.parse_args()

    init_logging()
    tag = a.gpu_tag or {"wit": "1", "U": "2", "prod": "3"}[a.arm]
    out = Path(a.out); gdir = out / f"gpu{tag}"; gdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    device = get_safe_torch_device(a.device, log=True)
    sm = Path(a.smoke_dir)
    bins = [int(x) for x in a.bins.split(",")]
    T0 = time.perf_counter()

    net_path = a.net or str(REPO / "outputs" / sm.name /
                            f"{a.suite}_seed42_ours/task_0/checkpoints/last/pretrained_model")
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = json.loads((sm / "instructions.json").read_text())["task0"]
    eps_probe = torch.load(sm / "eps_probe.pt").to(device)
    st0 = torch.load(sm / "stats" / "task0.pt")
    MU = st0["mu"].to(device).float(); SG = st0["sigma"].to(device).float()

    cfg = B1.build_cfg(_ns(a), 0, net_path, Path("/tmp/k10g"))
    net = make_policy(cfg=cfg.policy, ds_meta=meta)
    net.eval(); net.requires_grad_(False)
    blocks = WS.select_blocks(net, 1)                      # 6블록 전체
    H = net.config.horizon; A = net.config.action_feature.shape[0]

    d = torch.load(Path(a.cache) / f"{a.suite}_task0.pt")
    X = d["X"].float().view(-1, 4, 768); T = d["T"].long()

    # ── 자체점검 로그 (중단 없음) ────────────────────────────────────────────
    chk = [f"[K10gate] arm={a.arm} w={ARMW[a.arm]}  M={a.M}  r={a.rank}  bins={bins}  n={a.n}  "
           f"eta_target={a.eta_target}σ  step_clip={a.step_clip}×",
           f"[K10gate] net={net_path}", f"[K10gate] ℓ_0={instr!r}",
           f"[K10gate] emb_cache={a.cache}/{a.suite}_task0.pt  {tuple(X.shape)}",
           f"[K10gate] blocks={blocks}  eps_probe{tuple(eps_probe.shape)}"]
    for s in chk:
        print(s, flush=True)

    # wstats — K5b 후보(c): 스냅샷 기준, 6블록. 없으면 재계산.
    wsp = out / "wstats_snapshot_task0.pt"
    t0 = time.perf_counter()
    if wsp.exists():
        wstats = {int(k): {kk: vv for kk, vv in v.items()}
                  for k, v in torch.load(wsp)["stats"].items()}
        print(f"[K10gate] wstats 로드 {wsp}", flush=True)
    else:
        wstats = K5B.collect_from_cls(net, X, T, instr, eps_probe, blocks, a.n_bins, device)
        torch.save({"stats": {str(k): v for k, v in wstats.items()}, "blocks": blocks}, wsp)
        print(f"[K10gate] wstats 수집 {time.perf_counter()-t0:.1f}s -> {wsp}", flush=True)
    floor = K5B.maha_floor(wstats, blocks)

    EPS = [torch.randn(1, H, A, generator=gen) for _ in range(4)]
    probes = [(EPS[i], t) for t in (0.1, 0.5) for i in range(4)]

    # ── bin 별 PCA ───────────────────────────────────────────────────────────
    pcap = out / "pca_task0.pt"
    if pcap.exists():
        P = torch.load(pcap); Vb, Lb, Sb = P["V"], P["lam"], P["sperp"]
        print(f"[K10gate] PCA 로드 {pcap}", flush=True)
    else:
        Vb, Lb, Sb = bin_pca(X, T, a.n_bins, a.rank, device)
        torch.save({"V": Vb, "lam": Lb, "sperp": Sb, "r": a.rank}, pcap)
        print(f"[K10gate] PCA 계산 -> {pcap}  (r={a.rank})", flush=True)

    # ── Ê 무차원화 상수 + 기준선 ─────────────────────────────────────────────
    norm, base = {}, {}
    for tb in bins:
        idx = torch.nonzero(T == tb, as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=gen)]
        rn = X[idx[:a.n_norm]].to(device)
        tn = torch.full((rn.shape[0],), tb, dtype=torch.long, device=device)
        with torch.no_grad():
            nw = float(E_wit_fn(net, rn, tn, wstats, blocks, instr, eps_probe, a.n_bins, floor))
            nu = float(E_U_fn(net, rn, probes, instr).median())
        real = X[idx[:a.n]].to(device)
        D = torch.cdist(real.flatten(1), real.flatten(1)); D.fill_diagonal_(float("inf"))
        d_real = float(D.min(1).values.median())
        z = torch.randn(real.shape[0], 4, 768, generator=gen).clamp_(-3, 3).to(device)
        g0 = MU[tb] + SG[tb] * z
        with torch.no_grad():
            gw = float(E_wit_fn(net, g0, torch.full((g0.shape[0],), tb, dtype=torch.long,
                                                    device=device), wstats, blocks, instr,
                                eps_probe, a.n_bins, floor))
            gu = float(E_U_fn(net, g0, probes, instr).median())
        norm[tb] = {"wit": max(nw, 1e-12), "U": max(nu, 1e-12)}
        base[tb] = {"d_real": d_real, "d_hat_gauss": dhat(g0, real, d_real),
                    "Ehat_wit_gauss": gw / max(nw, 1e-12), "Ehat_U_gauss": gu / max(nu, 1e-12),
                    "n_real": int(real.shape[0])}
        print(f"[K10gate] bin{tb}  Ê 상수 wit={nw:.5g} U={nu:.5g}  d_real={d_real:.2f}  "
              f"가우시안 d̂={base[tb]['d_hat_gauss']:.3f}", flush=True)
        del rn, real, g0, D
        torch.cuda.empty_cache()

    # ── 격자 ────────────────────────────────────────────────────────────────
    combos = [(tm, "collective") for tm in TMODES]
    if a.arm == "prod":
        combos.append(("anneal", "full"))
    w_wit, w_U = ARMW[a.arm]

    R = {"arm": a.arm, "w": [w_wit, w_U], "M": a.M, "rank": a.rank, "bins": bins,
         "n": a.n, "seed": a.seed, "net": net_path, "selfcheck": chk,
         "baseline": {str(k): v for k, v in base.items()},
         "norm": {str(k): v for k, v in norm.items()}, "combos": {}}

    for tmode, coords in combos:
        for tb in bins:
            key = f"{a.arm}|{tmode}|{coords}|bin{tb}"
            try:
                if tb not in Vb:
                    R["combos"][key] = {"status": "no_pca"}; continue
                idx = torch.nonzero(T == tb, as_tuple=True)[0]
                idx = idx[torch.randperm(len(idx), generator=gen)[:a.n]]
                real = X[idx].to(device)
                n = real.shape[0]
                V = Vb[tb].to(device); lam = Lb[tb].to(device); sp = Sb[tb].to(device)
                lg = []
                t0 = time.perf_counter()
                # before: 같은 초기화의 표본
                gsave = torch.Generator().manual_seed(a.seed + tb)
                b_before, _ = langevin(net, tb, coords, 0.0, 0.0, "T0", 1, MU[tb], SG[tb],
                                       V, lam, sp, wstats, blocks, instr, eps_probe, probes,
                                       a.n_bins, floor, norm[tb]["wit"], norm[tb]["U"],
                                       n, torch.Generator().manual_seed(a.seed + tb), device,
                                       eta_target=a.eta_target, step_clip=a.step_clip)
                b_after, dg = langevin(net, tb, coords, w_wit, w_U, tmode, a.M, MU[tb], SG[tb],
                                       V, lam, sp, wstats, blocks, instr, eps_probe, probes,
                                       a.n_bins, floor, norm[tb]["wit"], norm[tb]["U"],
                                       n, torch.Generator().manual_seed(a.seed + tb), device, lg,
                                       eta_target=a.eta_target, step_clip=a.step_clip)
                if not torch.isfinite(b_after).all():
                    R["combos"][key] = {"status": "diverged"}
                    print(f"[K10gate] {key}  diverged", flush=True)
                    continue
                dr = base[tb]["d_real"]
                tt = torch.full((n,), tb, dtype=torch.long, device=device)
                with torch.no_grad():
                    ew_a = float(E_wit_fn(net, b_after, tt, wstats, blocks, instr, eps_probe,
                                          a.n_bins, floor)) / norm[tb]["wit"]
                    eu_a = float(E_U_fn(net, b_after, probes, instr).median()) / norm[tb]["U"]
                    ew_b = float(E_wit_fn(net, b_before, tt, wstats, blocks, instr, eps_probe,
                                          a.n_bins, floor)) / norm[tb]["wit"]
                    eu_b = float(E_U_fn(net, b_before, probes, instr).median()) / norm[tb]["U"]
                R["combos"][key] = {
                    "status": "ok", "tmode": tmode, "coords": coords, "bin": tb,
                    "d_hat_before": dhat(b_before, real, dr),
                    "d_hat_after": dhat(b_after, real, dr),
                    "diversity": diversity(b_before, b_after),
                    "Ehat_wit_before": ew_b, "Ehat_wit_after": ew_a,
                    "Ehat_U_before": eu_b, "Ehat_U_after": eu_a,
                    "eta": dg["eta"], "E_traj": dg["E_traj"], "clip": dg.get("clip"),
                    "sec": time.perf_counter() - t0, "log": lg}
                c = R["combos"][key]
                print(f"[K10gate] {key}  d̂ {c['d_hat_before']:.3f}->{c['d_hat_after']:.3f}  "
                      f"div {c['diversity']:.3f}  Ê_wit {ew_b:.3f}->{ew_a:.3f}  "
                      f"Ê_U {eu_b:.3f}->{eu_a:.3f}  ({c['sec']:.0f}s)", flush=True)
                if len(lg):
                    print(f"           {'; '.join(lg)}  E 첫10 "
                          + " ".join(f"{v:.3g}" for v in dg["E_traj"][:10]), flush=True)
                del real, b_before, b_after, V, lam, sp
                torch.cuda.empty_cache()
            except Exception as ex:                       # fail-soft
                R["combos"][key] = {"status": "error", "err": repr(ex),
                                    "tb": traceback.format_exc()[-800:]}
                print(f"[K10gate] {key}  ERROR {ex!r}", flush=True)
                torch.cuda.empty_cache()
            json.dump(R, (gdir / "partial.json").open("w"), indent=1, ensure_ascii=False)

    R["wall_s"] = time.perf_counter() - T0
    json.dump(R, (gdir / "partial.json").open("w"), indent=1, ensure_ascii=False)
    print(f"\n[K10gate] {a.arm} 완료 {R['wall_s']/60:.1f}분 -> {gdir/'partial.json'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
