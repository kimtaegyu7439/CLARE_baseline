#!/usr/bin/env python
"""R12 — R10 에서 structure 항을 뺀 것. 수송된 점 위의 level 앵커만.

R10 은 수송된 좌표 b_j 에서 값(level)과 방향미분(structure) 둘을 맞췄다.
R12 는 앞의 하나만 남긴다. 즉 "과거 태스크의 관측 분포에서 뽑은 점에 과거
명령어를 물려 teacher 와 값을 맞춘다"가 전부다.

  z   = clip((o − mu_new[τ]) / max(sigma_new[τ], floor), −3, 3)
  b_j = mu_j[τ] + sigma_j[τ] · z                          detach
  L   = L_FM + λ_level · mean_j ‖v_S(x_t,t,b_j,ℓ_j) − v_T(x_t,t,b_j,ℓ_j)‖²

R10 과 다른 점은 그것 하나뿐이다. teacher 는 rolling(스냅샷 1개), 명령어는
저장해 둔 과거 것, 통계는 태스크별 (mu, sigma), p_drop=0, w=1 로 전부 같다.

부수 효과: 방향미분을 위해 b_j + h·u 에서 한 번 더 돌 필요가 없어져 스텝당
student forward 가 1+2K -> **1+K** 로 절반이 된다. 10 태스크 마지막 스테이지
기준 19회 -> 10회다.

왜 재는가
  R10/R11 의 이득이 (a) 수송 자체에서 오는지 (b) structure 항에서 오는지
  분리되지 않는다. R12 는 (a) 만 남긴 팔이라 그 분해를 준다.
    R12 > B8λ3   -> 수송이 효과의 원천
    R10 > R12    -> structure 항이 추가로 기여
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

OUT_DIR = REPO / "results" / "R12"


class R12Anchor(R10.R10Anchor):
    """수송 좌표 위의 level 앵커만. structure 항 없음."""

    name = "R12"
    use_struct = False           # b_j + h·u forward 를 아예 하지 않는다

    def describe(self):
        return (f"R12 — 수송 앵커(level 만), rolling teacher 1개 + "
                f"통계 {len(self.stats)}개, λ_lvl={self.lam_lvl}, bins={self.n_bins}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0, help="B8λ3 의 λ 와 동일")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    # R10Anchor 가 참조하지만 R12 에서는 의미가 없는 항목들. 값만 채워 둔다.
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

    B1.ANCHOR = R12Anchor(args)

    argv = ["B1.py",
            "--p_drop", "0",
            "--guidance_w", "1.0",
            "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "R12")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "R12", "base": "R10", "structure_term": False,
               "lambda_level": args.lambda_level, "n_bins": args.n_bins,
               "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
               "stats_batches": args.stats_batches,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0,
               "student_forward_per_step": "1+K", "argv": argv},
              (out_dir / "r12_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    R10.write_table(out_dir, arm="R12", subtitle="수송 좌표 level 앵커만 (structure 없음)")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
