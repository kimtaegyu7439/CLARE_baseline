#!/usr/bin/env python
"""R13 — R12 와 같되 앵커 좌표를 과거 태스크의 가우시안에서 **샘플링**한다.

R12(수송)  z = (o − mu_new[τ]) / sigma_new[τ]        현재 관측의 실제 편차
R13(샘플)  z ~ N(0, I)                                매 스텝 새 난수
공통       b_j = mu_j[τ] + sigma_j[τ] · z             앵커 좌표

즉 R13 은 "과거 태스크 j 의 관측 분포를 가우시안으로 근사하고 거기서 뽑은 점"에
과거 명령어 ℓ_j 를 물려 rolling teacher 와 값을 맞춘다. 과거 데이터를 하나도
보관하지 않고 **분포 파라미터만으로 좌표를 생성**하는 형태다.

무엇이 달라지는가
  좌표의 출처     R12 현재 배치가 결정 / R13 난수가 결정
  커버리지        R12 현재 관측이 닿는 범위 / R13 과거 분포 전체
  상관구조        R12 z 가 담고 있어 보존 / R13 없음(등방)

  ★ 마지막 줄이 핵심 위험이다. results/R10_gauss 실측에서 임베딩은 주변분포는
    가우시안이지만 차원 간 상관이 강했다(‖z‖²/d 산포가 독립 가정의 13배).
    등방 난수로 뽑으면 그 상관이 사라져 b_j 가 실제 관측 매니폴드를 벗어난
    "허공의 점"이 될 수 있다. R12 vs R13 이 그것을 실험으로 답한다.

structure 항은 R12 와 마찬가지로 없다(level 앵커만). forward 는 1+K.
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

OUT_DIR = REPO / "results" / "R13"


class R13Anchor(R10.R10Anchor):
    """가우시안 샘플링 좌표 위의 level 앵커."""

    name = "R13"
    use_struct = False       # structure 항 없음 (R12 와 동일)
    sample_z = True          # z ~ N(0,I) — 이것이 R12 와의 유일한 차이

    def describe(self):
        return (f"R13 — 가우시안 샘플 앵커(level 만), rolling teacher 1개 + "
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

    B1.ANCHOR = R13Anchor(args)

    argv = ["B1.py",
            "--p_drop", "0",
            "--guidance_w", "1.0",
            "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "R13")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "R13", "base": "R12", "anchor_coord": "gaussian sample",
               "structure_term": False, "sample_z": True,
               "lambda_level": args.lambda_level, "n_bins": args.n_bins,
               "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
               "stats_batches": args.stats_batches,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0,
               "student_forward_per_step": "1+K", "argv": argv},
              (out_dir / "r13_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    R10.write_table(out_dir, arm="R13",
                    subtitle="과거 태스크 가우시안에서 샘플링한 좌표 + level 앵커")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
