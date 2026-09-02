#!/usr/bin/env python
"""L2_codebook_bayes — 코드북 앵커의 가중치를 커널(c-거리) → 베이즈 사후확률로 교체.

무엇을 바꾸나
    기존(v1/v3)   w = softmax(−‖zs(s̃) − c‖² / h²)
                  c 는 k-means 중심, h 는 전역 대역폭 — **알고리즘 부산물** 이다.
                  생성 스토리(셀 k 의 가우시안에서 s̃ 를 뽑음)와 조회 스토리(c 까지의
                  유클리드 거리)가 서로 다르고, 셀별 모양(σ_s)과 빈도(π)를 무시한다.

    이번(bayes)   이미 저장 중인 {π, μ_s, σ_s} 가 **그 자체로 GMM 을 정의한다.**
                  그 GMM 의 사후확률(responsibility)로 가중한다.

                      r_j(s̃) = π_j N(s̃; μ_s,j, diag σ²_s,j) / Σ_k π_k N(s̃; μ_s,k, ...)

                  õ = Σ_j r_j (셀 j 통계) 는 전확률 법칙 그대로다 — 구조는 그대로 두고
                  **가중치의 출처만** 생성 모형과 일치시킨다.

    소멸: 가중 경로의 c, h, 표준화 환승 zs(). 새 하이퍼파라미터 없음
          (τ = bayes_temp 는 폴백용 config, 기본 1.0 = 순수 베이즈).

원시공간에서 계산한다
    r 은 π·N(s̃; μ_s, σ_s) 이므로 **정규화 안 된 원시 s 공간**에서 바로 계산된다.
    커널 가중이 zs 공간으로 환승해야 했던 것과 달리 좌표계 변환이 없다.
    (부수 효과: "왜 z-정규화 뒤에 거리를 재나" 라는 문제 자체가 사라진다 —
     σ_s,j 로 나누는 마할라노비스가 셀별로 자동 적용된다.)

런 B(grad+bayes)의 δ
    스펙은 δ_j = (s̃ − μ_s,j)/std_s 로 재정의하라고 하는데, v3 구현이 이미 그것이다:
        zbar_k = (μ_s,k − mean_s)/std_s,  zs_i = (s_i − mean_s)/std_s
        δ = zs_i − zbar_k = (s_i − μ_s,k)/std_s      ← 동일
    그리고 zbar 은 k-means 중심 c 가 아니라 **셀 평균 μ_s** 에서 유도된다.
    따라서 A_k 회귀식 변경은 필요 없고, 바뀌는 것은 블렌드 가중(w → r) 뿐이다.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import l2_codebook as CB

OUT_DIR = REPO / "results" / "L2_codebook_bayes"


# ═════════════════════════════════════════════════════════════════════════════
def bayes_logits(cb: dict, s: torch.Tensor) -> torch.Tensor:
    """log π_j + log N(s; μ_s,j, diag σ²_s,j)  (상수항 제외). (B, K)."""
    mu, sig = cb["mu_s"], cb["sig_s"]                      # (K,16)
    logC = cb["_logC"]                                     # (K,) 로드 시 1회
    d2 = ((s[:, None, :] - mu[None]) / sig[None]).pow(2).sum(-1)   # (B,K)
    return logC[None] - 0.5 * d2


def bayes_r(cb: dict, s: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    lg = bayes_logits(cb, s)
    assert torch.isfinite(lg).all(), "bayes logit 에 non-finite 가 있다"
    return torch.softmax(lg / temp, dim=1)


def sample_codebook_bayes(cb: dict, n: int, gen=None, temp: float = 1.0):
    """v1/v3 의 sample_codebook 과 동일하되 가중치만 r 로 교체."""
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]
    m, sig_o = cb["m"], cb["sig_o"]
    k = torch.multinomial(pi, n, replacement=True, generator=gen)
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)
    s = mu_s[k] + sig_s[k] * eps                            # 생성: 변화 없음
    r = bayes_r(cb, s, temp)                                # (n,K)
    o_mu = r @ m
    if cb.get("A") is not None:
        # pred_j = m_j + A_j δ_j,  δ_j = (s − μ_s,j)/std_s   (v3 와 같은 δ)
        A = cb["A"].float()                                 # (K,D,16)
        d = (s[:, None, :] - mu_s[None]) / cb["std_s"][None]
        u = (r[:, :, None] * d).reshape(n, -1)
        o_mu = o_mu + u @ A.permute(0, 2, 1).reshape(-1, A.shape[1])
    o_sd = (r @ sig_o.pow(2)).clamp_min(0.0).sqrt()
    o = o_mu + o_sd * torch.randn(n, m.shape[1], device=m.device, generator=gen)
    assert torch.isfinite(s).all() and torch.isfinite(o).all(), "bayes 샘플에 NaN"
    return s, o


def add_logC(cb: dict) -> dict:
    """셀별 상수 logC = log π − ½ Σ_d log σ_s². 코드북 로드 시 1회."""
    cb["_logC"] = (cb["pi"].clamp_min(1e-12).log()
                   - 0.5 * cb["sig_s"].pow(2).log().sum(1))
    return cb


# ═════════════════════════════════════════════════════════════════════════════
class BayesAnchor(CB.L2CodebookAnchor):
    """가중치만 GMM 사후확률로 바꾼 코드북 앵커."""

    name = "L2_codebook_bayes"

    def __init__(self, args):
        super().__init__(args)
        self.bayes_temp = args.bayes_temp

    def describe(self):
        return (f"L2_codebook_bayes{'+grad' if self.grad_enable else ''} — "
                f"GMM 사후확률 가중 (τ={self.bayes_temp}), K={self.K}, "
                f"코드북 {len(self.books)}개")

    def _sample(self, cb, n, gen=None):
        return sample_codebook_bayes(cb, n, gen, self.bayes_temp)

    def on_task_end(self, policy, k, args, instructions, device, **kw):
        super().on_task_end(policy, k, args, instructions, device, **kw)
        add_logC(self.books[k])                    # GPU 상주 코드북에 상수 부착

    # ── §4 sanity ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _extra_sanity(self, cb, cbd, s_real, o_real, device, W, X, r_real):
        add_logC(cbd)
        L = ["", "── §4 베이즈 가중 진단 ──"]
        pi, mu_s, sig_s = cbd["pi"], cbd["mu_s"], cbd["sig_s"]
        K = pi.shape[0]
        g = torch.Generator(device=device).manual_seed(777)
        kk = torch.multinomial(pi, 1000, replacement=True, generator=g)
        eps = torch.randn(1000, mu_s.shape[1], device=device, generator=g)
        s_t = mu_s[kk] + sig_s[kk] * eps

        # 1. r 유효성
        lg = bayes_logits(cbd, s_t)
        r = torch.softmax(lg / self.bayes_temp, 1)
        L.append(f"1 r 유효성  합 min/max={float(r.sum(1).min()):.6f}/{float(r.sum(1).max()):.6f}  "
                 f"logit finite={bool(torch.isfinite(lg).all())}  NaN={int(torch.isnan(r).sum())}")

        # 2. 자기일관 일치율
        agree = float((r.argmax(1) == kk).float().mean())
        L.append(f"2 자기일관  argmax_j r_j == 생성 셀  일치율={agree:.3f}")
        L.append("   (100%가 아닌 것은 정상 — 저-π 셀 표본을 고-π 이웃이 가져가는 것은")
        L.append("    베이즈의 올바른 동작이다. 보고만 하고 수정하지 않는다.)")

        # 3. ESS
        ess = 1.0 / r.pow(2).sum(1)
        q = lambda v, p: float(v.quantile(p))
        L.append(f"3 ESS = 1/Σr²  중앙값={float(ess.median()):.2f}  "
                 f"IQR={q(ess,0.25):.2f}~{q(ess,0.75):.2f}  (K={K})")
        if float(ess.median()) < 1.05:
            L.append("   ⚠ 소프트 경고: ESS≈1 — 블렌딩이 사실상 사라졌다(하드 배정에 수렴)."
                     + ("  단 grad 팔은 A_jδ_j 가 서브셀 결합을 유지하므로 양성."
                        if self.grad_enable else
                        "  런 A 에서는 셀 해상도 결합으로 퇴화 — 유효하되 조악."))
            L.append("   τ 조정은 이 수치를 보고 별도 결정한다. 선제 조정하지 않는다.")

        # 4. 구가중(커널) 대비 — 같은 s̃ 배치
        z = (s_t - cbd["mean_s"]) / cbd["std_s"]
        w = torch.softmax(-torch.cdist(z, cbd["c"]).pow(2) / cbd["h"] ** 2, 1)
        top1 = float((w.argmax(1) == r.argmax(1)).float().mean())
        tv = float((0.5 * (w - r).abs().sum(1)).mean())
        L += [f"4 구가중 대비  top-1 일치율={top1:.3f}  평균 total variation ½Σ|w−r|={tv:.4f}",
              f"   ESS  커널={float((1/w.pow(2).sum(1)).median()):.2f}  베이즈={float(ess.median()):.2f}"]
        if tv < 0.10:
            L.append("   → 두 가중이 거의 같다. SR 변화 기대치를 낮게 볼 것 "
                     "(교체는 원리화였고 성능 레버는 아니었다는 판정 정보).")

        # 5. 합성 품질 재검
        def deff(x):
            x = x - x.mean(0)
            lam = torch.linalg.svdvals(x.double()).pow(2) / (x.shape[0] - 1)
            return float(lam.sum() ** 2 / lam.pow(2).sum()), float(lam.sum())
        g = torch.Generator(device=device).manual_seed(1234)
        s_b, o_b = sample_codebook_bayes(cbd, 1000, g, self.bayes_temp)
        sel = torch.randperm(o_real.shape[0], device=device)[:1000]
        dr, vr = deff(o_real[sel]); db, vb = deff(o_b)
        okd = abs(db / dr - 1) <= 0.20
        okv = 0.8 <= vb / vr <= 1.2
        L += [f"5 합성 품질  d_eff 실측={dr:.2f} bayes={db:.2f} (비 {db/dr:.2f}) "
              f"{'OK' if okd else '★±20% 이탈★'}   var 비={vb/vr:.2f} "
              f"{'OK' if okv else '★이탈★'}"]
        rmse = float((o_b.double()
                      - torch.cat([s_b, torch.ones(1000, 1, device=device)], 1).double() @ W)
                     .pow(2).mean().sqrt())
        L.append(f"   서브셀 정합 RMSE(õ−f(s̃))={rmse:.4f}  실측 잔차={r_real:.4f}  "
                 f"비={rmse/max(r_real,1e-12):.2f}")
        return L


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--codebook_k", type=int, default=96)
    ap.add_argument("--n_pairs", type=int, default=8000)
    ap.add_argument("--h_scale", type=float, default=1.0)
    ap.add_argument("--grad_enable", action="store_true")
    ap.add_argument("--ridge_rho", type=float, default=0.05)
    ap.add_argument("--grad_min_frames", type=int, default=24)
    ap.add_argument("--bayes_temp", type=float, default=1.0,
                    help="사후확률 온도. 1.0 = 순수 베이즈(폴백용 노브)")
    ap.add_argument("--xt_mode", choices=["teacher", "current"], default="teacher")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    CB._forbid_time_bin()

    out_dir = Path(args.out) if args.out else OUT_DIR
    args.out_dir = str(out_dir)
    args.batch_size = 32
    args.p_drop = 0.0
    args.seed = 42
    out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = BayesAnchor(args)

    argv = ["B1.py", "--p_drop", "0", "--guidance_w", "1.0", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / out_dir.name)]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    cfg = {
        "arm": "L2_codebook_bayes" + ("+grad" if args.grad_enable else ""),
        "base": "L2_codebook" + ("(grad)" if args.grad_enable else "(diag)"),
        "base_diff": [
            "1. 앵커 가중치를 커널 softmax(−‖zs(s̃)−c‖²/h²) 에서 GMM 사후확률 r 로 교체.",
            "2. r_j ∝ π_j N(s̃; μ_s,j, diag σ²_s,j) — 이미 저장 중인 통계만 쓴다. 새 파라미터 0.",
            "3. 원시 s 공간에서 계산 — zs() 환승·c·h 가 가중 경로에서 사라진다.",
            "4. 빌드는 무변경. c·h 는 파일에 남기되 이 팔에서는 deprecated(sanity 구가중 대비용).",
            "5. grad 팔의 δ=(s̃−μ_s,j)/std_s 는 v3 구현과 이미 동일 — A_k 회귀식 변경 없음.",
        ],
        "bayes_temp": args.bayes_temp, "codebook_k": args.codebook_k,
        "n_pairs": args.n_pairs, "grad_enable": args.grad_enable,
        "ridge_rho": args.ridge_rho, "grad_min_frames": args.grad_min_frames,
        "lambda_level": args.lambda_level, "chunk_backward": args.chunk_backward,
        "teacher": "rolling (1 snapshot)", "p_drop": 0.0, "guidance_w": 1.0, "argv": argv,
    }
    json.dump(cfg, (out_dir / "bayes_config.json").open("w"), indent=2, ensure_ascii=False)
    json.dump(cfg, (out_dir / "l2_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
