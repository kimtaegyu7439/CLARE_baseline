#!/usr/bin/env python
"""R14 — R10 에서 앵커 좌표만 가우시안 샘플링으로 바꾼 것.

  R10   z = (o − mu_new[τ]) / sigma_new[τ]   현재 관측의 실제 편차 (수송)
  R14   z ~ N(0, I)                           매 스텝 새 난수 (샘플링)
  공통    b_j = mu_j[τ] + sigma_j[τ] · z

R12/R13 이 level 앵커만으로 이 축을 봤다면, R14 은 structure 항이 있는
R10 위에서 같은 축을 본다. 네 팔이 2x2 를 이룬다.

           수송            샘플링
  level    R12             R13
  +struct  R10 / R11       R14 / R15

그 외(rolling teacher, 과거 명령어 ℓ_j, 태스크별 mu/sigma, p_drop=0, w=1)는
R10 과 완전히 같다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10
import R10

OUT_DIR = REPO / "results" / "R14"


class R14Anchor(R10.R10Anchor):
    """R10 + 샘플링."""

    name = "R14"
    sample_z = True          # R10 과의 유일한 차이

    def describe(self):
        return "R14 — " + super().describe().split("—", 1)[1].strip() + " + 가우시안 샘플 좌표"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--n_white", type=int, default=0)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--chunk_backward", action="store_true")
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

    B1.ANCHOR = R14Anchor(args)

    argv = ["B1.py", "--p_drop", "0", "--guidance_w", "1.0", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "R14")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "R14", "base": "R10", "anchor_coord": "gaussian sample",
               "sample_z": True, "lambda_level": args.lambda_level,
               "n_bins": args.n_bins, "n_white": args.n_white,
               "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0, "argv": argv},
              (out_dir / "r14_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    R10.write_table(out_dir, arm="R14", subtitle="R10 + 샘플링 (가우시안 샘플 좌표)")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
