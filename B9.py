#!/usr/bin/env python
"""B9 — 태스크 순서를 뒤집어 task 1 취약성의 원인을 가른다.

관측
  모든 앵커 팔에서 task 1 의 최종 SR 이 0~40 이다(ER 은 100). task 1 이 어려운 것도
  아니다 — 습득 SR 은 75~100 이고 ER 이 100 으로 지킨다.
  CLIP 명령어 유사도에서 최대 공선 쌍이 (0,1)=0.952 이므로 "명령어가 겹쳐 분리가
  안 된다"는 설명이 그럴듯하다. 그런데 **그 관계는 대칭인데 결과는 비대칭**이다:
      task 0  최종 85~100  (살아남음)
      task 1  최종  0~40   (죽음)
  공선성만으로는 어느 쪽이 죽는지 예측할 수 없다.

개입
  학습 순서를 0,1,2,3 -> **1,0,2,3** 으로 바꾼다. 나머지는 전부 동일
  (B2 frozen teachers + λ=3, 현재 최고 팔인 B2λ3 와 같은 설정).

판정
  task 0 이 죽고 task 1 이 살면   -> 공선 쌍에서 **나중에 배운 쪽이 죽는다**.
                                    순서가 원인이고 task 1 고유 성질이 아니다.
  여전히 task 1 이 죽으면        -> task 1 자체의 성질(궤적 난이도 등).
                                    공선성+순서 가설 기각.

구현 주의
  B1.py 의 --task_order 는 데이터셋·환경 선택만 바꾸고, SR 행렬·teacher 인덱스·
  instructions 는 **스테이지 기준**으로 유지한다. 그래야 AvgSR/BWT 정의가 다른 팔과
  같아진다. 태스크 기준으로 보려면 metrics.json 의 task_order 로 되매핑한다.

사용법
    python B9.py --smoke
    python B9.py                        # 순서 1,0,2,3 / frozen teachers / λ=3
    python B9.py --order 2,1,0,3
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B2 import FrozenTeachers          # 앵커는 B2 것을 그대로 쓴다

OUT_DIR = REPO / "results" / "B9"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--order", default="1,0,2,3")
    ap.add_argument("--lambda_anchor", type=float, default=3.0)
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    tag = args.out or ("B9_" + args.order.replace(",", ""))
    out_dir = REPO / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = FrozenTeachers(bf16=args.teacher_bf16)

    argv = ["B1.py",
            "--lambda_anchor", str(args.lambda_anchor),
            "--task_order", args.order,
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / tag)]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "B9", "anchor": "frozen_teachers", "task_order": args.order,
               "lambda_anchor": args.lambda_anchor,
               "note": "B2λ3 와 순서만 다름", "passthru": args.passthru},
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
