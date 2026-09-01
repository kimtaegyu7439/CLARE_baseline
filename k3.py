#!/usr/bin/env python
"""K3 — K1 에서 **잔여 성분을 뺀** 팔. 공유기저 사영 + 분위수 사상까지만.

K1 의 수송은 기저 안(w)과 기저 밖(res) 두 조각으로 나뉜다.

    K1   b_j = c0 + W w'  +  res − m⊥_new[τ] + m⊥_j[τ]
                            └───────── 잔여 성분 ─────────┘
    K3   b_j = c0 + W w'

즉 K3 는 관측을 공유기저가 펼치는 r 차원 아핀 부분공간 (c0 + span(W)) 위로
완전히 눌러 버린다. 기저 밖 성분은 모양도 평균도 전달되지 않는다.

왜 재는가
  results/K0_10task 실측: 태스크 0 에서 만든 기저가 다른 태스크의 변동을
  상한의 66~92% 밖에 담지 못한다(r=256, task4 66% / task9 69%). 즉 res 는
  무시할 만한 잔재가 아니라 실제 변동의 상당 부분이다. K1 은 그것을 현재
  프레임에서 그대로 가져와(모양 유지) 평균만 과거 쪽으로 옮긴다.
  K3 는 그 조각을 통째로 버린다. 두 팔의 차이가 곧 "기저 밖 성분이 앵커
  좌표로서 값을 하는가"에 대한 답이다.

K1 과 다른 곳은 transport 의 마지막 한 줄뿐이다. 손실/teacher/스케줄/평가/
통계 계산은 전부 그대로다. m⊥ 는 계산·저장까지 K1 과 똑같이 하되 쓰지 않는다
(저장 형식을 맞춰 두면 K1 과 파일 단위로 비교할 수 있다).
"""
from __future__ import annotations

import argparse
import difflib
import inspect
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import k1 as K1MOD

OUT_DIR = REPO / "results" / "K3"


class K3Anchor(K1MOD.K1Anchor):
    """K1 과 동일하되 b_j 를 공유기저 부분공간 위로만 만든다."""

    name = "K3"
    use_residual = False

    def describe(self):
        return (f"K3 — 공유기저 사영 + 분위수 사상만(잔여 성분 없음), rolling teacher "
                f"1개 + 표 {len(self.stats)}개, basis={self.basis}, "
                f"marginal={self.marginal}, iid={self.iid_sample}, Q={self.Q}, "
                f"r={self.r}, λ_lvl={self.lam_lvl}, bins={self.n_bins}")

    # ── K1.transport 의 복사본. 마지막 복원 한 줄만 다르다. ──────────────────
    def transport(self, w, res, tau, j):
        """w -> 과거 태스크 j 좌표로. b_j = c0 + W w'  (잔여 성분 없음)."""
        dev = w.device
        B, r = w.shape
        qtab_j = self.stats[j]["qtab"]
        qj = qtab_j[tau].reshape(-1, self.Q)                             # (B*r, Q)

        if self.marginal == "zscore":
            # 표에서 med/s 만 써서 표준화 -> 과거 med/s 로 재채색. 주변분포 형태는 버린다.
            hh = self._hhat(w, tau)
            mj = K1MOD.quant_at(qtab_j, 0.5)[tau]
            sj = ((K1MOD.quant_at(qtab_j, 0.841) - K1MOD.quant_at(qtab_j, 0.159)) / 2)[tau]
            wp = mj + sj.clamp_min(self.cur["s_floor"]) * hh
            self.clamp_frac = 0.0
        else:
            probs = K1MOD.make_probs(self.Q, dev)
            if self.iid_sample:
                # copula 를 끊는 negative control — 좌표마다 독립 균등 p.
                p = probs[0] + (probs[-1] - probs[0]) * torch.rand(B * r, device=dev)
                self.clamp_frac = 0.0
            else:
                qn = self.cur["qtab"][tau].reshape(-1, self.Q)           # (B*r, Q)
                p, out = K1MOD.cdf_forward(qn, probs, w.reshape(-1))
                self.clamp_frac = float(out.float().mean())
            wp = K1MOD.cdf_inverse(qj, p).view(B, r)

        # ★★★ K1 과 다른 유일한 줄 — res 와 m⊥ 항을 넣지 않는다 ★★★
        b = self.c0 + (wp if self.W is None else wp @ self.W.T)
        return b.reshape(B, -1, 768)


def transport_diff() -> str:
    """K1.transport 대비 실제로 달라진 줄. 완료 보고용이자 회귀 감시용."""
    a = inspect.getsource(K1MOD.K1Anchor.transport).splitlines()
    b = inspect.getsource(K3Anchor.transport).splitlines()
    return "\n".join(l for l in difflib.unified_diff(a, b, "K1.transport", "K3.transport", n=0)
                     if l.startswith(("+", "-", "@")) and not l.startswith(("+++", "---")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── K1 에서 그대로 상속 (기본값 동일) ───────────────────────────────────
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--stats_workers", type=int, default=4)
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quantiles", type=int, default=16, help="분위수 개수 Q")
    ap.add_argument("--rank", type=int, default=256, help="공유 PCA 기저 차원 r")
    ap.add_argument("--marginal", choices=["quantile", "zscore"], default="quantile")
    ap.add_argument("--basis", choices=["shared_pca", "identity"], default="shared_pca")
    ap.add_argument("--iid_sample", action="store_true")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--r13_ref", default="K1(4task) = 97.5, K1(10task) = 77.0, "
                                        "R13(10task) = 79.5")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    # K1 이 고정하는 값들 — 그대로 상속한다
    args.rho = 0.0
    args.warmup_steps = 0
    args.n_white = 0
    args.use_ghat_weight = False
    args.lambda_swap = 0.0

    out_dir = Path(args.out) if args.out else OUT_DIR
    args.out_dir = str(out_dir)
    args.batch_size = 32
    args.p_drop = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = K3Anchor(args)

    argv = ["B1.py",
            "--p_drop", "0",
            "--guidance_w", "1.0",
            "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / out_dir.name)]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru
    if "--suite" in argv:
        args.suite = argv[argv.index("--suite") + 1]

    json.dump({"arm": "K3", "base": "K1",
               "anchor_coord": "shared-basis quantile transport, NO residual",
               "residual": False,
               "marginal": args.marginal, "basis": args.basis,
               "iid_sample": args.iid_sample, "Q": args.quantiles, "r": args.rank,
               "structure_term": False, "sample_z": False,
               "lambda_level": args.lambda_level, "n_bins": args.n_bins,
               "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
               "stats_batches": args.stats_batches, "stats_workers": args.stats_workers,
               "rho": args.rho, "warmup_steps": args.warmup_steps,
               "use_ghat_weight": args.use_ghat_weight, "lambda_swap": args.lambda_swap,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0, "batch_size": 32,
               "student_forward_per_step": "1+K", "suite": args.suite, "argv": argv},
              (out_dir / "k3_config.json").open("w"), indent=2, ensure_ascii=False)

    (out_dir / "k3_transport_diff.txt").write_text(transport_diff())
    (out_dir / "k1_loss_diff.txt").write_text(K1MOD.loss_diff())
    print(f"[K3] K1.transport 대비 차이 -> {out_dir/'k3_transport_diff.txt'}")

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    # write_table 은 k1 것을 그대로 쓰되 머리글만 K3 로 고친다 (k1.py 는 건드리지 않는다)
    K1MOD.write_table(out_dir, args.suite, args.r13_ref)
    md = out_dir / "sr_table.md"
    if md.exists():
        L = md.read_text().splitlines()
        L[0] = L[0].replace("# K1 — 공유기저 분위수 수송 level 앵커",
                            "# K3 — 공유기저 사영 + 분위수만 (잔여 성분 없음)")
        md.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
