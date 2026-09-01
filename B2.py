#!/usr/bin/env python
"""B2 — teacher 고정 (frozen per-task teachers).

B1 의 rolling teacher 는 세대마다 앵커 목표가 오염된다. 측정으로 확인된 사실이다
(results/B1_drift/report.txt): 세대 거리 k−j 가 커질수록 상대 드리프트가
0.28 → 0.43 → 0.50 → 0.54 로 커지고 SR 은 71 → 31 → 18.5 → 0 으로 붕괴한다.
앵커 자체는 자기 목표에 충실한데(anchor_loss 0.004~0.008) 그 목표가 표류한다.

B2 는 그 지점 하나만 바꾼다. 태스크 j 를 막 배운 시점의 모델 teacher_j 를 **영구 보관**
하고, ℓ_j 앵커는 언제나 teacher_j 를 목표로 삼는다. 세대 누적이 끊긴다.

  대가:  모델을 태스크 수만큼 GPU 에 들고 있어야 한다 (ditflow_mt 195M x 4byte ≈ 780MB,
         4태스크면 약 3GB). --teacher_bf16 로 절반이 된다.
  유지:  과거 관측/액션은 여전히 하나도 저장하지 않는다. rehearsal-free 다.

그 외 모든 것은 B1 과 같다. 학습 루프·FM·평가·산출물은 B1.py 를 그대로 쓴다.

사용법
    python B2.py --smoke
    python B2.py                 # B1 기본 세팅 (libero_spatial 4태스크, 5000 steps)
"""
from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

OUT_DIR = REPO / "results" / "B2"


class FrozenTeachers:
    """태스크마다 그때의 스냅샷을 따로 보관한다. 목표가 절대 표류하지 않는다."""

    name = "frozen"

    def __init__(self, bf16: bool = False):
        self.teachers: dict[int, object] = {}
        self.bf16 = bf16

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or not self.teachers or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        # 과거 스테이지마다 **자기 시점의** teacher 를 쓴다. 이게 B1 과의 차이다.
        # 집계 방식(sum/mean/sample)은 B1 이 정한다.
        return B1.anchor_over_tasks(policy, self.teachers.get, batch, tail,
                                    x_t, t, k, instructions, rng, args, device,
                                    cls=getattr(self, "cls", None))

    def on_task_end(self, policy, k, args, instructions, device, **kw):
        # 직전 것을 버리지 않는다 — 이게 B1 과의 유일한 차이다.
        self.teachers[k] = B1.snapshot(policy, self.bf16 or args.teacher_bf16)
        torch.cuda.empty_cache()

    def describe(self):
        return f"frozen teachers — 태스크별 스냅샷 {len(self.teachers)}개 보관"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true",
                    help="보관 모델을 bf16 으로. 태스크가 많을 때 메모리 절반.")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[],
                    help="이 뒤의 인자는 B1.py 로 그대로 넘긴다")
    args = ap.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = FrozenTeachers(bf16=args.teacher_bf16)

    argv = ["B1.py",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "B2")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "B2", "anchor": "frozen_per_task_teachers",
               "teacher_bf16": args.teacher_bf16, "passthru": args.passthru},
              (out_dir / "arm.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
