#!/usr/bin/env python
"""B6 — B5 의 질의점 수 N 만 키운다. 다른 것은 전부 같다.

물음: 고정 앵커점을 충분히 많이 두면, 매 스텝 새 점을 뽑는 B1 수준으로 회복되는가.

지금까지의 관측
  매 스텝 새 앵커점   B1 62.5 / B2 76.2   (한 태스크에 5000x32 = 160,000 개, 전부 다름)
  고정 앵커점         B3 32.5 / B4 37.5 / B5 (stage2 31.7)
                      B5 는 1,024 점을 156 회씩 반복해서 본다.
  cond_move(o_0)      B1 0.198  vs  B5 0.357   — 고정점은 자기 영역조차 덜 지킨다

가설 두 개가 정반대를 예측한다.
  (i)  앵커 범위가 좁아서다      -> N 을 키우면 SR 이 오른다
  (ii) '고정'이라는 성질 자체가 문제다 -> N 을 키워도 안 오른다

B5 와의 차분은 **N 뿐이다**. K(시드), W(waypoint), ODE 스텝, λ, 학습 세팅 모두 동일.
앵커 클래스도 B5.SelfRolloutAnchor 를 그대로 import 해서 쓴다(복제하지 않는다).

  B5: N=32   -> 32 x 4 x 8 =  1,024 점,  0.60 MB/task
  B6: N=512  -> 512 x 4 x 8 = 16,384 점,  9.4 MB/task   (16 배)

사용법
    python B6.py --smoke
    python B6.py                 # N=512
    python B6.py --n_query 128   # 다른 N 로 sweep
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B5 import SelfRolloutAnchor          # 앵커 본체는 B5 것을 그대로 쓴다

OUT_DIR = REPO / "results" / "B6"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--n_query", type=int, default=512, help="B5 는 32. 여기만 다르다.")
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--n_waypoints", type=int, default=8)
    ap.add_argument("--ode_steps", type=int, default=100)
    ap.add_argument("--pool", type=int, default=None,
                    help="선별 후보 풀. 기본은 max(4N, 2048).")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = OUT_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    n_q = min(args.n_query, 16) if args.smoke else args.n_query
    pool = args.pool if args.pool else max(4 * n_q, 2048)
    if args.smoke:
        pool = min(pool, 128)
    ode = min(args.ode_steps, 20) if args.smoke else args.ode_steps

    B1.ANCHOR = SelfRolloutAnchor(
        lambda_a=args.lambda_a, n_query=n_q, n_seeds=args.n_seeds,
        n_waypoints=args.n_waypoints, ode_steps=ode, pool=pool, out_dir=out_dir)

    argv = ["B1.py", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir), "--ckpt_root", str(REPO / "outputs" / "B6")]
    if args.smoke:
        argv.append("--smoke")
    argv += args.passthru

    json.dump({"arm": "B6", "anchor": "self_rollout_large_N", "lambda_a": args.lambda_a,
               "n_query": n_q, "n_seeds": args.n_seeds, "n_waypoints": args.n_waypoints,
               "ode_steps": ode, "pool": pool, "note": "B5 와 N 만 다름",
               "passthru": args.passthru},
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
