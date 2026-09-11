#!/usr/bin/env python
"""K10 Phase 2 — 10-task 학습 3팔. R13 프로토콜 그대로, 앵커 부분만 교체.

    K10L    R13 + Langevin 합성 (게이트가 고른 설정)
    K7b     R13 + 잔차-EMA task 배분
    K10LB   둘 결합 (★ EMA 는 **정련 전 b0 잔차**로만 갱신 — 자기 증폭 차단)

R13 앵커의 j 집계는 **전체 합 (1/K)Σ_j** 다 (R10.loss: sum(lvl)/len(lvl)). 따라서 K7b 는
"샘플링" 분기가 아니라 "가중 재배분" 분기를 탄다:
    loss = λ·Σ_j (K·p_j)·(1/K)·L_j          총예산은 R13 과 같고 가중만 옮긴다.

Langevin 은 게이트(k10_gate.py)의 에너지·온도 스케줄을 그대로 import 한다. 다만 학습 배치는
**bin 이 섞여 있어** 게이트의 단일 bin 드라이버를 쓸 수 없다. 표본별 bin 인덱싱 드라이버만
여기서 새로 쓰고, 에너지(E_wit 배치통계·E_U per-sample)·prior·T 스케줄은 게이트 함수를 쓴다.

★ 주기 정련 P (비용 통제, 명시적 완화)
  매 스텝이 아니라 P 스텝마다 앵커 배치 전체를 **fresh z 에서 완전히 재생성·정련**하고,
  그 사이 스텝들은 같은 b 를 쓰되 (x_t, t, ε) 은 매 스텝 신선하다. P 는 시작 20 스텝 실측으로
  "스텝당 총 시간 ≤ R13 기준 2.5배"가 되는 최소값(상한 16)으로 자동 설정한다.
  b 의 P 주기 초과 캐시는 없다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10
import R13
import k5_wstats as WS
import k5b_bench as K5B
import k6_probe as K6
import k10_gate as GATE

ARMS = ("K10L", "K7b", "K10LB")


# ═════════════════════════════════════════════════════════════════════════════
#  K7b — 잔차-EMA 배분
# ═════════════════════════════════════════════════════════════════════════════
class EMAAlloc:
    """R̄_j ← (1−β)R̄_j + β·mean‖r0_j‖²,  p_j = max((R̄_j+c)/Σ(R̄+c), p_min) 후 재정규화."""

    def __init__(self, beta=0.05, p_min_scale=0.5):
        self.beta = beta
        self.p_min_scale = p_min_scale
        self.R: dict[int, float] = {}

    def reset_stage(self):
        self.R = {}                       # stage 시작 시 균등 출발

    def update(self, j: int, r2: float):
        self.R[j] = r2 if j not in self.R else (1 - self.beta) * self.R[j] + self.beta * r2

    def probs(self, js: list[int]) -> dict[int, float]:
        K = len(js)
        if not js:
            return {}
        if any(j not in self.R for j in js):          # 첫 관측 전엔 균등
            return {j: 1.0 / K for j in js}
        m = float(np.mean([self.R[j] for j in js]))
        c = 0.1 * m
        raw = {j: self.R[j] + c for j in js}
        s = sum(raw.values())
        p = {j: raw[j] / s for j in js}
        pm = self.p_min_scale / K
        p = {j: max(v, pm) for j, v in p.items()}
        s = sum(p.values())
        return {j: v / s for j, v in p.items()}


# ═════════════════════════════════════════════════════════════════════════════
#  Langevin — 표본별 bin 인덱싱 드라이버 (에너지/스케줄은 게이트 함수)
# ═════════════════════════════════════════════════════════════════════════════
def langevin_batch(net, o_shape, tau, coords, w_wit, w_U, tmode, M,
                   MU, SG, V, LAM, SP, wstats, blocks, instr, eps_probe, probes,
                   n_bins, floor, norm_wit, norm_U, device, eta_box, eu_every=3,
                   eta_target=0.02, step_clip=3.0):
    """과거 태스크 j 의 좌표를 M 스텝 Langevin 으로 생성. bin 이 섞인 배치를 그대로 받는다."""
    n = tau.shape[0]
    mu = MU[tau]                                        # (n,4,768)
    sg = SG[tau]
    mu_f, sg_f = mu.reshape(n, -1), sg.reshape(n, -1)
    if coords == "full":
        z0 = torch.randn(n, 4, 768, device=device).clamp_(-3, 3)
        zeta = (mu + sg * z0).reshape(n, -1)
        Vt = lam = sp = eres = None
    else:
        Vt = V[tau]                                     # (n,3072,r)
        lam = LAM[tau]                                  # (n,r)
        sp = SP[tau]                                    # (n,3072)
        eres = torch.randn(n, 3072, device=device)
        zeta = torch.randn(n, lam.shape[1], device=device) * lam.sqrt()

    def to_b(zt):
        if coords == "full":
            return zt.reshape(n, 4, 768)
        return (mu_f + torch.bmm(Vt, zt.unsqueeze(-1)).squeeze(-1) + sp * eres).reshape(n, 4, 768)

    def prior(zt):
        if coords == "full":
            return 0.5 * (((zt - mu_f) / sg_f.clamp_min(1e-6)) ** 2).sum(1)
        return 0.5 * ((zt ** 2) / lam.clamp_min(1e-8)).sum(1)

    traj, e_first, clip_hits = [], None, 0.0
    target_rms = eta_target * float(sg.mean())
    cap = step_clip * target_rms
    pre = lam.sqrt() if coords != "full" else sg_f      # 노이즈 전처리 — 게이트와 동일
    for m in range(M):
        zeta = zeta.detach().requires_grad_(True)
        with torch.enable_grad():
            b = to_b(zeta)
            tot = prior(zeta).mean()
            if w_wit:
                tot = tot + w_wit * GATE.E_wit_fn(net, b, tau, wstats, blocks, instr,
                                                  eps_probe, n_bins, floor) / norm_wit
            if w_U and (m % eu_every == 0):
                tot = tot + w_U * GATE.E_U_fn(net, b, probes, instr).mean() / norm_U
            g, = torch.autograd.grad(tot, zeta)
        if e_first is None:
            e_first = float(tot.detach())
        if eta_box["eta"] is None:                      # 첫 스텝에서 1회 자동 설정
            db = -g if coords == "full" else -torch.bmm(Vt, g.unsqueeze(-1)).squeeze(-1)
            rms = db.pow(2).mean(1).sqrt().median().clamp_min(1e-12)
            eta_box["eta"] = float(target_rms / rms)
            logging.info(f"[K10] Langevin η 자동 설정 = {eta_box['eta']:.4g} "
                         f"(target RMS {target_rms:.4g}, cap {cap:.4g})")
        eta = eta_box["eta"]
        T, ns = GATE.temperature(tmode, m, M)
        step = -eta * g
        if ns > 0 and T > 0:
            step = step + ns * float(np.sqrt(2 * eta * T)) * pre * torch.randn_like(zeta)
        db = step if coords == "full" else torch.bmm(Vt, step.unsqueeze(-1)).squeeze(-1)
        rms = db.pow(2).mean(1).sqrt().clamp_min(1e-12)
        fac = (cap / rms).clamp(max=1.0)
        clip_hits += float((fac < 1).float().mean()) / M
        zeta = (zeta + step * fac[:, None]).detach()
        traj.append(float(tot.detach()))
        if not np.isfinite(traj[-1]):
            break
    b = to_b(zeta).detach()
    return b, {"E0": e_first, "E1": traj[-1] if traj else None, "clip": clip_hits}


# ═════════════════════════════════════════════════════════════════════════════
class K10Anchor(R13.R13Anchor):
    """세 팔 공용. args.arm 으로 분기한다."""

    name = "K10"

    def __init__(self, args):
        super().__init__(args)
        self.arm = args.arm
        self.use_lang = args.arm in ("K10L", "K10LB")
        self.use_ema = args.arm in ("K7b", "K10LB")
        self.alloc = EMAAlloc(args.beta) if self.use_ema else None
        self.wstats: dict[int, dict] = {}
        self.pca: dict[int, dict] = {}          # j -> {V, lam, sperp}
        self.blocks = None
        self.eps_probe = None
        self.probes = None
        self.norm = {"wit": None, "U": None}
        self.eta_box = {"eta": None}
        self.P = None
        self._b_cache = None                    # (step_mod, {j: b}) — P 주기에서만 유효
        self._b_step = -10 ** 9
        self.fallback = 0
        self.n_synth = 0
        self._t_hist = []
        (self.out / "wstats").mkdir(exist_ok=True)
        (self.out / "pca").mkdir(exist_ok=True)
        self.klog = (self.out / "k10.jsonl").open("a")

    def describe(self):
        s = f"{self.arm} — R13"
        if self.use_lang:
            s += f" + Langevin({self.a.sel['arm']}/{self.a.sel['T_mode']}/{self.a.sel['coords']}, M={self.a.M})"
        if self.use_ema:
            s += f" + 잔차-EMA 배분(β={self.a.beta})"
        return s + f", rolling teacher 1개 + 통계 {len(self.stats)}개, λ_lvl={self.lam_lvl}"

    # ── 태스크 시작 ─────────────────────────────────────────────────────────
    def on_task_start(self, policy, k, args, instructions, device, **kw):
        super().on_task_start(policy, k, args, instructions, device, **kw)
        self._kw, self._k = kw, k
        self._instr_cur = instructions[f"task{k}"]
        self._logged = False
        # ★ 스테이지가 바뀌면 과거 j 집합이 늘어난다. 이전 스테이지의 b 캐시를 그대로
        #   쓰면 새로 생긴 j 키가 없어 KeyError 가 난다(실측: stage 2 에서 KeyError: 1).
        self._b_cache, self._b_step = None, -10 ** 9
        if self.use_ema:
            self.alloc.reset_stage()
        if self.use_lang and self.eps_probe is None:
            H, A = policy.config.horizon, policy.config.action_feature.shape[0]
            g = torch.Generator().manual_seed(20260829)
            self.eps_probe = torch.randn(1, H, A, generator=g).to(device)
            torch.save(self.eps_probe.cpu(), self.out / "eps_probe.pt")
            E = [torch.randn(1, H, A, generator=g) for _ in range(2)]
            self.probes = [(E[i], 0.1) for i in range(2)]     # 경제판: ε2개 × t=0.1
            self.blocks = WS.select_blocks(policy, 1)
            logging.info(f"[K10] Langevin 준비  blocks={self.blocks}  "
                         f"probes={len(self.probes)}  M={self.a.M}")

    # ── 태스크 종료: teacher + μ/σ + (K10L) PCA·wstats ──────────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        cur_mu = self.cur["mu"].clone()
        super().on_task_end(policy, k, args, instructions, device, **kw)
        if not self.use_lang:
            return
        t0 = time.perf_counter()
        ds, cfg, eps = self._kw["dataset"], self._kw["cfg"], self._kw["train_eps"]
        prep = self._kw["prep"]
        was = policy.training; policy.eval()
        # 현재 태스크 프레임의 CLS 를 한 번 모아 PCA 와 wstats 에 함께 쓴다.
        X, T = K5B_collect_cls(policy, ds, eps, cfg, device, self.n_bins,
                               args.batch_size, prep, getattr(args, "stats_workers", 4))
        Vb, Lb, Sb = GATE.bin_pca(X.view(-1, 4, 768), T, self.n_bins, self.a.rank, device)
        # wstats 는 **직전 스냅샷(= 방금 갱신된 rolling teacher)** 기준
        ws = K5B.collect_from_cls(self.teacher, X.view(-1, 4, 768), T, self._instr_cur,
                                  self.eps_probe, self.blocks, self.n_bins, device)
        if was:
            policy.train()
        self.pca[k] = {"V": {t: v.cpu() for t, v in Vb.items()},
                       "lam": {t: v.cpu() for t, v in Lb.items()},
                       "sperp": {t: v.cpu() for t, v in Sb.items()}}
        self.wstats[k] = ws
        torch.save({"V": self.pca[k]["V"], "lam": self.pca[k]["lam"],
                    "sperp": self.pca[k]["sperp"], "r": self.a.rank},
                   self.out / "pca" / f"task{k}.pt")
        torch.save({"stats": {str(b): v for b, v in ws.items()}, "blocks": self.blocks},
                   self.out / "wstats" / f"task{k}.pt")
        # Ê 무차원화 상수 — 첫 태스크에서 1회
        if self.norm["wit"] is None:
            fl = K5B.maha_floor(ws, self.blocks)
            sel = X.view(-1, 4, 768)[:128].to(device)
            tt = T[:128].to(device)
            with torch.no_grad():
                self.norm["wit"] = max(float(GATE.E_wit_fn(
                    self.teacher, sel, tt, ws, self.blocks, self._instr_cur,
                    self.eps_probe, self.n_bins, fl)), 1e-12)
                self.norm["U"] = max(float(GATE.E_U_fn(
                    self.teacher, sel, self.probes, self._instr_cur).median()), 1e-12)
            logging.info(f"[K10] Ê 상수  wit={self.norm['wit']:.5g}  U={self.norm['U']:.5g}")
        del X, T
        torch.cuda.empty_cache()
        logging.info(f"[K10] task {k} PCA(r={self.a.rank}) + wstats "
                     f"{time.perf_counter()-t0:.1f}s")

    # ── 손실 ────────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        t_start = time.perf_counter()
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        o = cls.view(n, -1, cls.shape[-1]).float()
        tau = R10.phase_bins(batch, self.ep_len, self.n_bins).to(device)
        js = sorted(self.stats)
        K = len(js)

        # ── 합성 ─────────────────────────────────────────────────────────
        z = torch.randn_like(o).clamp_(-3.0, 3.0).detach()
        b0 = {j: (self.stats[j]["mu"].to(device)[tau]
                  + self.stats[j]["sigma"].to(device)[tau] * z).detach() for j in js}
        bmap, diag = b0, {}
        if self.use_lang:
            need = ((self.P is None) or (self._b_cache is None)
                    or (self.step - self._b_step >= self.P)
                    or any(j not in self._b_cache for j in js))
            if need:
                bmap, diag = self._synth(policy, tau, js, device, instructions)
                self._b_cache, self._b_step = bmap, self.step
            else:
                bmap = self._b_cache
        chunk = self.a.chunk_backward
        p = self.alloc.probs(js) if self.use_ema else {j: 1.0 / K for j in js}

        lvl = []
        for j in js:
            past = [instructions[f"task{j}"]] * n

            def fwd(pol, c):
                flat = c.reshape(-1, c.shape[-1]).to(x_t.dtype)
                tl = B1.cond_tail(pol, batch, flat)
                return pol.dit_flow.velocity_net(
                    noisy_actions=x_t, time=t,
                    global_cond=B1.make_cond(B1.encode_lang(pol, past), tl))

            b_j = bmap[j]
            with torch.no_grad():
                vt0 = fwd(self.teacher, b_j)
            vs0 = fwd(policy, b_j)
            r0 = vs0 - vt0.to(vs0.dtype)
            L_j = self.reduce_level(r0)

            # ★ EMA 는 **정련 전 b0** 잔차로만. 정련 좌표로 갱신하면 배분이 자기 증폭한다.
            if self.use_ema:
                if self.use_lang:
                    with torch.no_grad():
                        r_ema = fwd(policy, b0[j]) - fwd(self.teacher, b0[j])
                    self.alloc.update(j, float(self.reduce_level(r_ema)))
                else:
                    self.alloc.update(j, float(L_j.detach()))

            wgt = (K * p[j]) / K                 # = p_j. 총예산 Σ = 1 로 R13 과 같다.
            term = self.lam_lvl * wgt * L_j
            if chunk:
                (args.lambda_anchor * term).backward()
                lvl.append(term.detach())
            else:
                lvl.append(term)

        out = sum(lvl)
        if chunk:
            out = out.detach()
        self.lam_str = 0.0
        self.step += 1
        self._t_hist.append(time.perf_counter() - t_start)
        self._autoP()

        if self.step % self.a.log_every_anchor == 0 or not self._logged:
            rec = {"task": k, "step": self.step, "L": float(out), "K": K, "P": self.P,
                   "p": {str(j): round(p[j], 4) for j in js}}
            if self.use_ema:
                rec["Rbar"] = {str(j): round(v, 6) for j, v in self.alloc.R.items()}
            if self.use_lang:
                rec.update({k2: v for k2, v in diag.items()})
                rec["fallback_rate"] = self.fallback / max(1, self.n_synth)
            self.klog.write(json.dumps(rec) + "\n"); self.klog.flush()
            if not self._logged:
                logging.info(f"[K10][sanity] task{k} arm={self.arm} K={K} P={self.P}  "
                             f"Σp={sum(p.values()):.3f} p_min={min(p.values()):.3f}  "
                             + (f"Ê {diag.get('E0')} -> {diag.get('E1')}  "
                                f"div {diag.get('diversity')}  " if self.use_lang else "")
                             + f"L={float(out):.5f}")
                self._logged = True
        return out

    # ── 정련 ────────────────────────────────────────────────────────────────
    def _synth(self, policy, tau, js, device, instructions):
        sel = self.a.sel
        w_wit, w_U = GATE.ARMW[sel["arm"]]
        out, diag = {}, {}
        for j in js:
            self.n_synth += 1
            try:
                P = self.pca[j]
                nb = self.n_bins
                V = torch.stack([P["V"].get(t, torch.zeros(3072, self.a.rank))
                                 for t in range(nb)]).to(device)
                LAM = torch.stack([P["lam"].get(t, torch.ones(self.a.rank))
                                   for t in range(nb)]).to(device)
                SP = torch.stack([P["sperp"].get(t, torch.zeros(3072))
                                  for t in range(nb)]).to(device)
                ws = self.wstats[j]
                fl = K5B.maha_floor(ws, self.blocks)
                b, d = langevin_batch(
                    self.teacher, None, tau, sel["coords"], w_wit, w_U, sel["T_mode"],
                    self.a.M, self.stats[j]["mu"].to(device),
                    self.stats[j]["sigma"].to(device), V, LAM, SP, ws, self.blocks,
                    instructions[f"task{j}"], self.eps_probe, self.probes, nb, fl,
                    self.norm["wit"] or 1.0, self.norm["U"] or 1.0, device, self.eta_box,
                    eta_target=self.a.eta_target, step_clip=self.a.step_clip)
                if not torch.isfinite(b).all():
                    raise FloatingPointError("Langevin NaN")
                out[j] = b
                if j == js[0]:
                    b0 = (self.stats[j]["mu"].to(device)[tau]
                          + self.stats[j]["sigma"].to(device)[tau]
                          * torch.randn_like(b).clamp_(-3, 3))
                    diag = {"E0": round(d["E0"], 5), "E1": round(d["E1"], 5),
                            "clip": round(d.get("clip", 0.0), 3),
                            "diversity": round(float(
                                torch.pdist(b.flatten(1)).median()
                                / torch.pdist(b0.flatten(1)).median().clamp_min(1e-12)), 3)}
            except Exception as ex:                        # fail-soft
                self.fallback += 1
                out[j] = (self.stats[j]["mu"].to(device)[tau]
                          + self.stats[j]["sigma"].to(device)[tau]
                          * torch.randn(tau.shape[0], 4, 768, device=device).clamp_(-3, 3))
                if self.fallback <= 3:
                    logging.warning(f"[K10] Langevin 폴백 j={j}: {ex!r}")
        return out, diag

    def _autoP(self):
        """시작 20 스텝 실측으로 '스텝당 시간 ≤ R13 의 2.5배' 최소 P (상한 16)."""
        if not self.use_lang or self.P is not None:
            return
        if len(self._t_hist) < 20:
            return
        t_syn = float(np.median(self._t_hist[:5]))          # 정련 포함 스텝
        t_pln = float(np.median(sorted(self._t_hist)[:5]))  # 정련 없는 스텝
        budget = 2.5 * max(t_pln, 1e-6)
        extra = max(t_syn - t_pln, 0.0)
        need = extra / max(budget - t_pln, 1e-6)
        self.P = int(min(16, max(1, np.ceil(need))))
        logging.info(f"[K10] P 자동 설정 = {self.P}  (정련 스텝 {t_syn*1000:.0f}ms, "
                     f"일반 {t_pln*1000:.0f}ms, 예산 {budget*1000:.0f}ms)")


def K5B_collect_cls(policy, dataset, train_eps, cfg, device, n_bins, batch_size, prep,
                    workers=4):
    """현재 태스크 학습 프레임의 CLS 와 bin. k1.collect_cls 를 그대로 쓴다."""
    import k1 as K1MOD
    return K1MOD.collect_cls(policy, dataset, train_eps, cfg, device, n_bins,
                             batch_size, prep, workers=workers)


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=10)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--stats_workers", type=int, default=4)
    ap.add_argument("--chunk_backward", action="store_true", default=True)
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--M", type=int, default=12, help="Langevin 스텝 (경제판)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--eta_target", type=float, default=0.02)
    ap.add_argument("--step_clip", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=0.05, help="EMA 계수")
    ap.add_argument("--selected", default="results/K10/selected.json")
    ap.add_argument("--num_workers", type=int, default=6)
    args = ap.parse_args()

    args.rho = 0.0; args.warmup_steps = 0; args.n_white = 0
    args.use_ghat_weight = False; args.lambda_swap = 0.0
    args.batch_size = 32; args.p_drop = 0.0

    sp = Path(args.selected)
    args.sel = (json.loads(sp.read_text()) if sp.exists()
                else {"arm": "prod", "T_mode": "anneal", "coords": "collective",
                      "gate_failed": True, "note": "selected.json 없음 — default"})
    out_dir = Path(args.out) if args.out else REPO / "results" / args.arm
    args.out_dir = str(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = K10Anchor(args)
    argv = ["B1.py", "--p_drop", "0", "--guidance_w", "1.0", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / out_dir.name),
            "--num_tasks", str(args.num_tasks), "--suite", args.suite,
            "--num_workers", str(args.num_workers)]
    if args.teacher_bf16:
        argv.append("--teacher_bf16")

    json.dump({"arm": args.arm, "base": "R13", "selected": args.sel, "M": args.M,
               "rank": args.rank, "beta": args.beta,
               "j_aggregation": "weighted sum over all j (R13 은 (1/K)Σ)",
               "ema_source": "pre-refinement b0 residual" if args.arm == "K10LB" else
                             ("L_j" if args.arm == "K7b" else None),
               "lambda_level": args.lambda_level, "n_bins": args.n_bins,
               "teacher": "rolling (1 snapshot)", "p_drop": 0.0, "guidance_w": 1.0,
               "batch_size": 32, "suite": args.suite, "argv": argv},
              (out_dir / "k10_config.json").open("w"), indent=2, ensure_ascii=False)
    print(f"[K10] arm={args.arm}  selected={args.sel}  out={out_dir}", flush=True)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    R10.write_table(out_dir, arm=args.arm, subtitle=f"{args.arm} (selected={args.sel})")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
