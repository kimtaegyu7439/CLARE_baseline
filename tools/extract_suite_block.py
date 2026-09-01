#!/usr/bin/env python
"""libero_40 SR 행렬에서 한 스위트 블록(10×10)을 잘라 별도 SR txt로 쓴다.

libero_40은 네 스위트를 이어 붙인 40스테이지 시퀀스다:

    스테이지  0..9  libero_10 / 10..19 libero_goal / 20..29 libero_spatial / 30..39 libero_object

스위트 s의 블록은 행=스테이지 10s..10s+9, 열=태스크 10s..10s+9 이고, 그 자체로
완전한 하삼각 10×10이다 -- 그 스위트를 배우는 동안의 학습/망각이 그대로 담긴다.

★ 주의: 이건 **libero_40 시퀀스 안에서 잰 값**이지, 사전학습 모델에서 바로 시작하는
  단독 libero_object 벤치마크(bash/*/eval_libero_object*.sh)와 같은 수가 아니다.
  블록 시작 시점에 모델은 이미 앞 스위트 30개 태스크를 거쳤다. 논문에 단독 벤치마크
  숫자로 실으면 안 된다.

    python tools/extract_suite_block.py \
        outputs/ER_eval/libero_40/seed42/libero_40_SR.txt --suite object
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SUITES = {"10": 0, "goal": 1, "spatial": 2, "object": 3}


def load(path: Path, n: int = 40) -> np.ndarray:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    m = np.full((n, n), np.nan)
    for ln in lines[1:]:
        parts = ln.split("\t")
        k = int(parts[0])
        for t, v in enumerate(parts[1:]):
            if v.strip() not in ("", "-", "nan"):
                m[k, t] = float(v)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="libero_40_SR.txt")
    ap.add_argument("--suite", choices=SUITES, default="object")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    off = SUITES[args.suite] * 10
    m = load(args.src)
    blk = m[off:off + 10, off:off + 10]

    out = args.out or args.src.with_name(f"libero_40_{args.suite}_block_SR.txt")
    tag = f"LIBERO_{args.suite.upper()}_BLOCK"
    with out.open("w") as f:
        # 첫 줄 열 라벨은 블록 안에서의 0..9. 원래 libero_40 인덱스는 +off.
        f.write(f"# {args.src}  스테이지/태스크 {off}..{off + 9} 블록 "
                f"(libero_40 시퀀스 내부 측정 -- 단독 벤치마크 아님)\n")
        f.write(f"{tag}\t" + "\t".join(str(t) for t in range(10)) + "\n")
        for k in range(10):
            row = blk[k, :k + 1]
            f.write(f"{k}\t" + "\t".join("" if np.isnan(v) else f"{v:.0f}" for v in row) + "\n")

    filled = int(np.isfinite(blk).sum())
    diag = blk[np.diag_indices(10)]
    last = blk[9, :]
    print(f"[{args.suite:8s}] {out}")
    print(f"           칸 {filled}/55   "
          f"학습직후 평균(diag) {np.nanmean(diag):.1f}   "
          f"블록끝 평균(stage{off + 9}) {np.nanmean(last):.1f}   "
          f"망각 {np.nanmean(diag - last):+.1f}")


if __name__ == "__main__":
    main()
