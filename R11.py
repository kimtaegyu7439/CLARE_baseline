#!/usr/bin/env python
"""R11 — R10 에서 structure 항을 L1 으로, 방향 u 를 실제 백색화로 바꾼 팔.

R10 과 다른 점은 정확히 둘이다. 나머지(수송 b_j, rolling teacher, 과거 명령어 ℓ_j,
level 항, λ_struct 자동 설정, p_drop=0, w=1)는 전부 같다.

  (1) structure 항의 축약을 제곱 -> **L1**
      L_struct = mean |(r1 − r0)/h|          (원소 평균)
      제곱은 큰 성분 하나에 지배되지만 L1 은 성분들에 고르게 가중을 준다.
      자코비안 차이가 소수의 방향에 몰려 있을 때 그 방향만 맞추고 끝나는 것을
      막는다. λ_struct 는 자동 설정이라 스케일 차이는 알아서 흡수된다.

  (2) 방향 u 를 **백색화**
      R10 은 u = z/‖z‖ 를 썼는데 z 는 원소별 표준화일 뿐 백색화가 아니다.
      results/R10_gauss 실측: ‖z‖²/d 의 표준편차가 0.33 으로, 차원이 독립이라면
      나왔어야 할 χ² 예측(0.026)의 **13배**다. 즉 소수의 공통 인자가 z 를
      지배하고 있어서 u 가 등방이 아니라 그 인자 방향으로 치우쳐 있었다.

      전 차원 공분산은 3072² = 37MB/bin 이라 저장할 수 없으므로, 상위 k개
      주성분만 단위분산으로 눌러 주는 **저계수 백색화**를 쓴다.

          z_w = z + Σ_{i<k} (λ_i^{-1/2} − 1) ⟨z, e_i⟩ e_i
          u   = z_w / ‖z_w‖

      k = --n_white (기본 256). 저장량은 태스크당 k×3072 float32 = 3.1 MB 로,
      teacher 하나(740 MiB)에 비하면 무시할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10

OUT_DIR = REPO / "results" / "R11"


class R11Anchor(R10.R10Anchor):
    """R10 + (L1 structure, 백색화된 방향)."""

    name = "R11"

    def __init__(self, args):
        super().__init__(args)
        self.n_white = args.n_white          # >0 이면 compute_stats 가 기저를 만든다

    def describe(self):
        return (f"R11 — 수송 앵커 + L1 structure + 백색화(k={self.n_white}), "
                f"rolling teacher 1개 + 통계 {len(self.stats)}개, λ_lvl={self.lam_lvl}")

    # ── (1) structure 를 L1 으로 ────────────────────────────────────────────
    def reduce_struct(self, x):
        if self.a.anchor_norm == "sum":
            return x.flatten(1).abs().sum(1).mean()
        return x.flatten(1).abs().mean(1).mean()

    # ── (2) 방향을 백색화 ───────────────────────────────────────────────────
    def direction(self, z):
        """상위 k 주성분을 단위분산으로 누른 뒤 단위화.

        기저는 **현재 태스크**의 것을 쓴다. u 는 "지금 관측이 자기 분포 안에서
        어느 쪽으로 벗어나 있는가"를 나타내는 양이기 때문이다.
        """
        cur = self.cur
        if cur is None or "white_V" not in cur:
            return super().direction(z)
        V = cur["white_V"].to(z.device)                      # (k, d)
        lam = cur["white_lam"].to(z.device)                  # (k,)
        flat = z.flatten(1)                                  # (B, d)
        proj = flat @ V.t()                                  # (B, k)
        scale = lam.rsqrt() - 1.0                            # λ^{-1/2} − 1
        zw = flat + (proj * scale) @ V
        zw = zw / zw.norm(dim=1).clamp_min(1e-8)[:, None]
        return zw.view_as(z)

    def on_task_start(self, policy, k, args, instructions, device, **kw):
        super().on_task_start(policy, k, args, instructions, device, **kw)
        cur = self.cur
        if "white_lam" in cur:
            lam = cur["white_lam"]
            logging.info(
                f"[R11] task {k} 백색화 k={lam.numel()}  "
                f"설명분산 {100*cur['white_frac']:.1f}%  "
                f"λ_max={float(lam[0]):.3f}  λ_min={float(lam[-1]):.4f}  "
                f"(원소별 표준화 후이므로 평균 고유값이 1 이면 이미 등방)")
            # 백색화가 실제로 등방으로 만들었는지 한 배치에서 확인한다
            self._check_white = True

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        out = super().loss(policy, batch, tail, x_t, t, k, instructions, rng, args, device)
        if getattr(self, "_check_white", False) and k > 0:
            self._check_white = False
            with torch.no_grad():
                cls = getattr(self, "cls", None)
                if cls is None:
                    cls = B1.rgb_cls(policy, batch)
                n = batch["observation.state"].shape[0]
                o = cls.view(n, -1, cls.shape[-1]).float()
                tau = R10.phase_bins(batch, self.ep_len, self.n_bins).to(device)
                mu_n = self.cur["mu"].to(device); sg_n = self.cur["sigma"].to(device)
                z = ((o - mu_n[tau]) / sg_n[tau].clamp_min(self.cur["sigma_floor"])
                     ).clamp_(-3.0, 3.0)
                u0 = super().direction(z)                    # R10 방식
                u1 = self.direction(z)                       # R11 방식
                d = z.shape[1] * z.shape[2]
                logging.info(
                    f"[R11][check] ‖u‖={float(u1.flatten(1).norm(dim=1).mean()):.4f}  "
                    f"cos(u_R10,u_R11)={float((u0*u1).flatten(1).sum(1).mean()):.4f}  "
                    f"‖z‖²/d={float(z.flatten(1).pow(2).sum(1).mean()/d):.4f}  "
                    f"백색화 후 ‖z_w‖²/d 산포 감소를 여기서 확인")
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--n_white", type=int, default=256,
                    help="백색화할 상위 주성분 개수. 0 이면 R10 과 같아진다.")
    ap.add_argument("--stats_batches", type=int, default=0,
                    help="통계를 앞 N 배치로만 낸다(0=전수). R10 과 동일.")
    ap.add_argument("--chunk_backward", action="store_true",
                    help="과거 태스크마다 즉시 backward. R10 과 동일한 동작.")
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--use_ghat_weight", action="store_true")
    ap.add_argument("--lambda_swap", type=float, default=0.0)
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else OUT_DIR
    args.out_dir = str(out_dir)
    args.batch_size = 32
    args.p_drop = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = R11Anchor(args)

    argv = ["B1.py",
            "--p_drop", "0",
            "--guidance_w", "1.0",
            "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "R11")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "R11", "base": "R10", "lambda_level": args.lambda_level,
               "rho": args.rho, "n_bins": args.n_bins, "n_white": args.n_white,
               "anchor_norm": args.anchor_norm, "struct_norm": "L1",
               "chunk_backward": args.chunk_backward, "stats_batches": args.stats_batches,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0, "argv": argv},
              (out_dir / "r11_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    R10.write_table(out_dir, arm="R11",
                    subtitle="R10 + L1 structure + 백색화된 방향 u")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
