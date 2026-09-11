#!/usr/bin/env python
"""K5b — witness 판별력 벤치. 학습 없음, 판정 전용.

K5 정련의 witness 가 "실제 manifold 위 점"과 "가우시안 합성점"을 실제로 구별하는지
후보별로 재고, 사전 등록된 기준으로 본 실행 witness 를 정한다.

배경 (K5 smoke 실측)
    사전학습 witness, blocks [0,4]:  ratio = E(gauss)/E(real) = 1.43  (< 2)
    정련 후 E = 0.0156 < E(real) = 0.0288  ->  통계 과최적화(overshoot) 의심.

후보
    (a) 사전학습망, blocks [0,4]        현재 설정 (대조)
    (b) 사전학습망, blocks 6개 전부
    (c) task0 스냅샷(rolling teacher 후보), blocks 6개 전부
    (d) DINOv2 자신 — ★ 구조적으로 불가능. 파이프라인의 CLS 는
        outputs.pooler_output, 즉 **마지막 레이어** 산물이다
        (modeling_dit_flow_mt.py:303). b 를 주입해 읽을 이후 레이어가 없다.
        infeasible 로 보고하고 건너뛴다.

E 변형
    plain  Σ_l [‖μ̂_l−μ_l‖² + ‖σ̂_l−σ_l‖²] / d_h
    maha   Σ_l [‖(μ̂_l−μ_l)/σ_l‖² + ‖(σ̂_l−σ_l)/σ_l‖²] / d_h,  σ_l 하한 = 채널 σ 중앙값 × 0.1

판정 (사전 등록)
    (i) ratio ≥ 3
    (ii) overshoot = E(real)/E(refined) ≤ 1.2  (조기종료 하)
    (iii) d̂_after 가 d̂_before 대비 30% 이상 감소 & d̂_after ≤ 1.5
    셋 다 만족하는 후보 중 ratio 최대를 권고. 전부 미달이면
    "활성-통계 정련 기각 — K5 는 M=0(=R13) 으로 후퇴".

주의
    통계 수집과 정련의 forward 규약은 K5 와 문자 그대로 같다
    (t=0, x_t=eps_probe, ℓ_0, state=0). witness 와 통계는 반드시 같은 망·같은 규약이며
    후보끼리 섞지 않는다. 실제 프레임은 이 벤치의 측정에만 쓰고 학습 코드로 넘기지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import k5_wstats as WS
from B_merge import _ns

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata   # noqa: E402
from lerobot.policies.factory import make_policy                      # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging   # noqa: E402

PRETRAIN = "/home/sa090180/Models/dit_flow_mt_libero_90_pretrain"


# ═════════════════════════════════════════════════════════════════════════════
def energy_var(acts, tau, stats, n_bins, mode: str, floor: dict | None = None):
    """K5 의 배치 통계를 그대로 쓰되 plain / maha 를 고른다."""
    bs = WS.batch_stats(acts, tau, n_bins)
    tot = None
    for bi, (mu, sd, cnt) in bs.items():
        ref = stats[bi]
        rm = ref["mu"].to(mu.device, mu.dtype)
        rs = ref["sigma"].to(mu.device, mu.dtype)
        H = mu.shape[-1]
        w = (cnt / cnt.sum().clamp_min(1)).to(mu.dtype)
        if mode == "maha":
            s = rs.clamp_min(floor[bi])
            e_mu = (((mu - rm) / s) ** 2).sum(-1) / H
            e_sd = (((sd - rs) / s) ** 2).sum(-1) / H
        else:
            e_mu = ((mu - rm) ** 2).sum(-1) / H
            e_sd = ((sd - rs) ** 2).sum(-1) / H
        e_sd = torch.where(cnt > 1, e_sd, torch.zeros_like(e_sd))
        e = (w * (e_mu + e_sd)).sum()
        tot = e if tot is None else tot + e
    return tot


def maha_floor(stats, blocks):
    """채널 σ 중앙값 × 0.1. 블록별 스칼라."""
    return {bi: float(0.1 * stats[bi]["sigma"][stats[bi]["sigma"] > 0].median())
            for bi in blocks}


def load_policy(ckpt, meta, device, args):
    cfg = B1.build_cfg(_ns(args), 0, str(ckpt), Path("/tmp/k5b"))
    p = make_policy(cfg=cfg.policy, ds_meta=meta)
    p.eval(); p.requires_grad_(False)
    return p


@torch.no_grad()
def collect_from_cls(witness, X, T, instr, eps_probe, blocks, n_bins, device, bs=64):
    """캐시된 CLS 로 블록별 채널 통계. K5.collect_wstats 와 같은 규약, 같은 누적식."""
    acc = {}
    with WS.BlockTap(witness, blocks) as tap:
        for s in range(0, X.shape[0], bs):
            x = X[s:s + bs].to(device)
            tau = T[s:s + bs].to(device)
            tap.acts.clear()
            WS.witness_forward(witness, x.reshape(-1, 768), instr, eps_probe, tap)
            for bi, a in tap.acts.items():
                y = a.detach().permute(1, 0, 2).double()
                B_, Tn, H = y.shape
                if bi not in acc:
                    acc[bi] = [torch.zeros(n_bins, H, device=device, dtype=torch.float64),
                               torch.zeros(n_bins, H, device=device, dtype=torch.float64),
                               torch.zeros(n_bins, device=device, dtype=torch.float64)]
                sm, sq, c = acc[bi]
                sm.index_add_(0, tau, y.sum(1))
                sq.index_add_(0, tau, (y * y).sum(1))
                c.index_add_(0, tau, torch.full((B_,), float(Tn), device=device,
                                                dtype=torch.float64))
    out = {}
    for bi, (sm, sq, c) in acc.items():
        cc = c.clamp_min(1.0)[:, None]
        mu = sm / cc
        out[bi] = {"mu": mu.float().cpu(),
                   "sigma": (sq / cc - mu ** 2).clamp_min(0).sqrt().float().cpu(),
                   "count": c.float().cpu(), "d_h": int(mu.shape[-1])}
    return out


def E_of(witness, x, tau, stats, blocks, instr, eps_probe, n_bins, mode, floor,
         grad=False):
    with WS.BlockTap(witness, blocks) as tap:
        tap.acts.clear()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            WS.witness_forward(witness, x.reshape(-1, x.shape[-1]), instr, eps_probe, tap)
            return energy_var(tap.acts, tau, stats, n_bins, mode, floor).float()


def nn_stats(A, B_):
    """A 각 행에서 B_ 로의 최근접 L2 거리 중앙값."""
    return float(torch.cdist(A.flatten(1), B_.flatten(1)).min(1).values.median())


def loo_scale(X):
    D = torch.cdist(X.flatten(1), X.flatten(1))
    D.fill_diagonal_(float("inf"))
    return float(D.min(1).values.median())


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--smoke_dir", default="results/K5_spatial_10task_smoke_M8")
    ap.add_argument("--snapshot", default=None,
                    help="기본: <smoke 의 outputs>/task_0/checkpoints/last/pretrained_model")
    ap.add_argument("--cache", default="results/K0/emb_cache")
    ap.add_argument("--bins", default="2,5,8")
    ap.add_argument("--n_per_bin", type=int, default=256)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--M_max", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/K5b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    a = ap.parse_args()

    init_logging()
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    sm = Path(a.smoke_dir)
    bins = [int(x) for x in a.bins.split(",")]

    snap = a.snapshot or str(REPO / "outputs" / sm.name /
                             f"{a.suite}_seed42_ours/task_0/checkpoints/last/pretrained_model")
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = json.loads((sm / "instructions.json").read_text())["task0"]
    eps_probe = torch.load(sm / "eps_probe.pt").to(device)
    st0 = torch.load(sm / "stats" / "task0.pt")
    mu0 = st0["mu"].to(device).float()
    sg0 = st0["sigma"].to(device).float()

    print(f"[K5b] smoke={sm}")
    print(f"[K5b] snapshot={snap}")
    print(f"[K5b] ℓ_0={instr!r}  eps_probe{tuple(eps_probe.shape)}  bins={bins}  "
          f"n/bin={a.n_per_bin}  M_max={a.M_max}  ρ={a.rho}")

    # 실제 task0 CLS (진단 전용)
    d = torch.load(Path(a.cache) / f"{a.suite}_task0.pt")
    X, T = d["X"].float(), d["T"].long()
    X = X.view(X.shape[0], -1, 768)
    print(f"[K5b] 실제 프레임 {tuple(X.shape)}  (results/K0/emb_cache — 측정 전용)")

    # ── 후보 witness ────────────────────────────────────────────────────────
    cands = []
    t0 = time.perf_counter()
    w_pre = load_policy(PRETRAIN, meta, device, a)
    blocks_all = WS.select_blocks(w_pre, 1)
    st_pre = collect_from_cls(w_pre, X, T, instr, eps_probe, blocks_all, a.n_bins, device)
    t_pre = time.perf_counter() - t0
    cands.append(("a_pretrained_blocks04", w_pre, [0, 4], st_pre, t_pre))
    cands.append(("b_pretrained_blocksAll", w_pre, blocks_all, st_pre, t_pre))

    t0 = time.perf_counter()
    if Path(snap).is_dir():
        w_snap = load_policy(snap, meta, device, a)
        st_snap = collect_from_cls(w_snap, X, T, instr, eps_probe, blocks_all, a.n_bins, device)
        cands.append(("c_task0snapshot_blocksAll", w_snap, blocks_all, st_snap,
                      time.perf_counter() - t0))
    else:
        print(f"[K5b] ⚠ 스냅샷 없음 -> (c) 건너뜀: {snap}")

    infeasible = {"d_dinov2": "CLS = outputs.pooler_output (DINOv2 마지막 레이어). "
                              "주입 후 읽을 이후 레이어가 없어 구조적으로 불가."}
    print(f"[K5b] (d) DINOv2 witness: infeasible — {infeasible['d_dinov2']}")
    print(f"[K5b] 블록 전체 {blocks_all}  통계 수집 사전학습 {t_pre:.1f}s"
          + (f"  스냅샷 {cands[-1][4]:.1f}s" if len(cands) > 2 else ""))

    # ── 측정 ────────────────────────────────────────────────────────────────
    rows, raw = [], {}
    for name, w, blocks, stats, t_col in cands:
        fl = maha_floor(stats, blocks)
        for mode in ("plain", "maha"):
            key = f"{name}|{mode}"
            per_bin, t_ref0 = {}, time.perf_counter()
            for tb in bins:
                idx = torch.nonzero(T == tb, as_tuple=True)[0]
                if len(idx) > a.n_per_bin:
                    idx = idx[torch.randperm(len(idx), generator=g)[:a.n_per_bin]]
                real = X[idx].to(device)
                n = real.shape[0]
                tau = torch.full((n,), tb, dtype=torch.long, device=device)
                eps = torch.randn(n, real.shape[1], 768, generator=g).clamp_(-3, 3).to(device)
                b0 = (mu0[tb] + sg0[tb] * eps).detach()
                sig = sg0[tb].expand_as(b0)

                kw = dict(stats=stats, blocks=blocks, instr=instr, eps_probe=eps_probe,
                          n_bins=a.n_bins, mode=mode, floor=fl)
                e_real = float(E_of(w, real, tau, **kw))
                e_b0 = float(E_of(w, b0, tau, **kw))
                # E(real) 배치 간 분산 — 64 x 4
                sub = [float(E_of(w, real[i * 64:(i + 1) * 64],
                                  tau[i * 64:(i + 1) * 64], **kw))
                       for i in range(max(1, n // 64))]

                # 정련 (조기종료: E <= mean E(real))
                b, eta, used_M, clip = b0.clone(), None, 0, 0.0
                for m in range(a.M_max):
                    b = b.detach().requires_grad_(True)
                    E = E_of(w, b, tau, grad=True, **kw)
                    if float(E) <= e_real:
                        b = b.detach(); break
                    gr, = torch.autograd.grad(E, b)
                    if eta is None:
                        r = (gr.abs() / sig.clamp_min(1e-6)).flatten(1).max(1).values
                        eta = float(a.rho / r.median().clamp_min(1e-12))
                    dstep = -eta * gr
                    sc = (dstep.abs() / sig.clamp_min(1e-6)).flatten(1).max(1).values
                    fac = (a.rho / sc.clamp_min(1e-12)).clamp(max=1.0)
                    clip += float((fac < 1).float().mean())
                    b = (b + dstep * fac[:, None, None]).detach()
                    used_M = m + 1
                e_ref = float(E_of(w, b, tau, **kw))

                d_real = loo_scale(real)
                per_bin[tb] = {
                    "E_real": e_real, "E_real_batches_mean": float(np.mean(sub)),
                    "E_real_batches_std": float(np.std(sub)),
                    "E_b0": e_b0, "E_refined": e_ref,
                    "ratio": e_b0 / max(e_real, 1e-12),
                    "overshoot": e_real / max(e_ref, 1e-12),
                    "M_used": used_M, "eta": eta,
                    "clip_frac": clip / max(1, used_M),
                    "d_hat_before": nn_stats(b0, real) / max(d_real, 1e-12),
                    "d_hat_after": nn_stats(b, real) / max(d_real, 1e-12),
                    "diversity": float(torch.pdist(b.flatten(1)).median()
                                       / torch.pdist(b0.flatten(1)).median().clamp_min(1e-12)),
                    "d_real_loo": d_real, "n": n}
                del real, b0, b
                torch.cuda.empty_cache()

            agg = {k: float(np.mean([per_bin[t][k] for t in bins]))
                   for k in ("ratio", "overshoot", "M_used", "clip_frac",
                             "d_hat_before", "d_hat_after", "diversity")}
            agg["t_refine_s"] = time.perf_counter() - t_ref0
            agg["t_collect_s"] = t_col
            raw[key] = {"per_bin": per_bin, "agg": agg,
                        "blocks": blocks, "witness": name, "E": mode}
            rows.append((name, mode, agg))
            print(f"[K5b] {name:<26} {mode:<5}  ratio {agg['ratio']:5.2f}  "
                  f"overshoot {agg['overshoot']:5.2f}  M {agg['M_used']:.1f}  "
                  f"d̂ {agg['d_hat_before']:.2f}->{agg['d_hat_after']:.2f}  "
                  f"div {agg['diversity']:.2f}  ({agg['t_refine_s']:.0f}s)", flush=True)

    # ── 블록별 기여 ((b),(c) 만) ────────────────────────────────────────────
    blockwise = {}
    for name, w, blocks, stats, _ in cands:
        if not name.startswith(("b_", "c_")):
            continue
        fl = maha_floor(stats, blocks)
        per = {}
        for bi in blocks:
            rr = []
            for tb in bins:
                idx = torch.nonzero(T == tb, as_tuple=True)[0][:a.n_per_bin]
                real = X[idx].to(device)
                n = real.shape[0]
                tau = torch.full((n,), tb, dtype=torch.long, device=device)
                eps = torch.randn(n, real.shape[1], 768, generator=g).clamp_(-3, 3).to(device)
                b0 = mu0[tb] + sg0[tb] * eps
                kw = dict(stats=stats, blocks=[bi], instr=instr, eps_probe=eps_probe,
                          n_bins=a.n_bins, mode="maha", floor=fl)
                rr.append(float(E_of(w, b0, tau, **kw))
                          / max(float(E_of(w, real, tau, **kw)), 1e-12))
                del real, b0
            per[str(bi)] = float(np.mean(rr))
        blockwise[name] = per
        print(f"[K5b] {name} 블록별 ratio(maha): "
              + "  ".join(f"b{k}={v:.2f}" for k, v in per.items()))

    # ── 판정 ────────────────────────────────────────────────────────────────
    passed = []
    for name, mode, ag in rows:
        i = ag["ratio"] >= 3
        ii = ag["overshoot"] <= 1.2
        iii = (ag["d_hat_after"] <= 0.7 * ag["d_hat_before"]) and ag["d_hat_after"] <= 1.5
        if i and ii and iii:
            passed.append((ag["ratio"], name, mode))
    if passed:
        passed.sort(reverse=True)
        r, name, mode = passed[0]
        verdict = f"채택 — {name} / E={mode} (ratio {r:.2f}). 세 조건 모두 충족."
        if name.startswith("c_"):
            verdict += (" ★ rolling teacher 를 witness 로 쓰는 구성이다. 태스크마다 "
                        "witness 통계를 새로 수집해야 하며 그 비용은 표의 t_collect 이다.")
    else:
        verdict = ("활성-통계 정련 기각 — K5 는 M=0(=R13) 으로 후퇴, negative result 로 기록. "
                   "어느 후보도 (i) ratio≥3, (ii) overshoot≤1.2, (iii) d̂ 30%↓ & ≤1.5 를 "
                   "동시에 만족하지 못했다.")

    # ── 표 ──────────────────────────────────────────────────────────────────
    L = ["# K5b — witness 판별력 벤치", "",
         f"suite={a.suite}  bins={bins}  n/bin={a.n_per_bin}  M_max={a.M_max}  ρ={a.rho}",
         f"witness 규약: t=0, x_t=eps_probe, ℓ_0, state=0 (K5 와 동일)", "",
         "| 후보 | E | ratio | overshoot | 평균 M | clip | d̂_before → d̂_after | diversity | 수집 s | 정련 s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, mode, ag in rows:
        L.append(f"| {name} | {mode} | {ag['ratio']:.2f} | {ag['overshoot']:.2f} | "
                 f"{ag['M_used']:.1f} | {ag['clip_frac']*100:.0f}% | "
                 f"{ag['d_hat_before']:.2f} → {ag['d_hat_after']:.2f} | "
                 f"{ag['diversity']:.2f} | {ag['t_collect_s']:.0f} | {ag['t_refine_s']:.0f} |")
    L += ["", "판정 기준: (i) ratio ≥ 3   (ii) overshoot ≤ 1.2   "
              "(iii) d̂_after ≤ 0.7·d̂_before 이고 d̂_after ≤ 1.5", "",
          f"**{verdict}**", "",
          "(d) DINOv2 witness: **infeasible** — " + infeasible["d_dinov2"]]
    if blockwise:
        L += ["", "블록별 ratio (maha)"]
        for k, v in blockwise.items():
            L.append(f"  {k}: " + "  ".join(f"b{i}={x:.2f}" for i, x in v.items()))
    md = "\n".join(L)
    (out / "bench_table.md").write_text(md + "\n")
    json.dump({"config": vars(a), "rows": raw, "blockwise": blockwise,
               "infeasible": infeasible, "verdict": verdict},
              (out / "bench.json").open("w"), indent=2, ensure_ascii=False)
    print("\n" + md)
    print(f"\nsaved -> {out/'bench_table.md'}, {out/'bench.json'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
