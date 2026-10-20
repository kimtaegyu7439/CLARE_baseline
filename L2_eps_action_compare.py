#!/usr/bin/env python
"""앵커가 쓰는 노이즈 ε 와 실제 액션 a 의 크기를 비교한다.

    python L2_eps_action_compare.py
    python L2_eps_action_compare.py --suites libero_goal,libero_object --out ...

왜 재는가
    flow matching 경로는 x_t = (1−t)ε + t·a 이고 속도는 v = a − ε 다. 앵커는
    student 와 teacher 를 **같은 x_t** 에서 평가하므로 두 속도가 ε 를 공유한다.
    ‖ε‖ 이 ‖a‖ 보다 크면 v 의 방향이 ε 에 끌려가, 두 정책이 전혀 다른 액션을
    원해도 속도 각도는 작게 나온다. 그 비 ‖ε‖/‖a‖ 가 압축의 세기를 정한다.

    ε 는 torch.randn_like(x_t) 라 액션과 **같은 shape (B,H,A)** 의 표준정규다.
    앵커의 방향 항은 pose 6 축만 쓰므로 그 부분 노름을 기준으로 본다.

측정 대상
    a : 데모 액션을 정책의 MIN_MAX 통계로 [-1,1] 정규화한 것. 모델이 보는 값이다.
    ε : N(0, I) 에서 뽑은 같은 shape 의 표본.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent
DATA = Path("/home/sa090180/Datasets/lerobot/continuallearning")
POSE = 6
ADIM = 7


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", default="libero_goal,libero_object,libero_spatial")
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--files_per_task", type=int, default=0,
                   help="0 이면 **전 에피소드**. parquet 1 개 = 에피소드 1 개다.")
    p.add_argument("--norm_ckpt",
                   default="/home/sa090180/Models/dit_flow_mt_libero_90_pretrain/model.safetensors",
                   help="정규화 통계 출처. CLARE 셋팅은 전 스위트가 이 체크포인트를 쓴다.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO / "results" / "libero_epsilon_action_compare.txt"))
    return p.parse_args()


def load_norm(path: str):
    d = load_file(path)
    return (d["normalize_targets.buffer_action.min"].numpy(),
            d["normalize_targets.buffer_action.max"].numpy())


def task_stats(suite: str, k: int, mn, mx, nfiles: int):
    fs = sorted(glob.glob(str(DATA / f"{suite}_image_task_{k}" / "data" / "**" / "*.parquet"),
                          recursive=True))
    if nfiles:
        fs = fs[:nfiles]
    if not fs:
        return None
    A = np.concatenate([np.stack(pd.read_parquet(f)["action"].values) for f in fs])
    An = (A - mn) / (mx - mn + 1e-8) * 2 - 1            # 모델이 보는 [-1,1] 값
    rho6 = np.linalg.norm(An[:, :POSE], axis=1)
    rho7 = np.linalg.norm(An, axis=1)
    g = A[:, 6]
    sw = np.where(np.sign(g[1:]) != np.sign(g[:-1]))[0]
    if len(sw):
        d = np.min(np.abs(np.arange(len(A))[:, None] - (sw[None, :] + 0.5)), axis=1)
        near = rho6[d < 8]
    else:
        near = np.array([])
    return {"n": len(A), "rho": rho6, "p50": np.median(rho6), "mean": rho6.mean(),
            "p05": np.percentile(rho6, 5), "p95": np.percentile(rho6, 95),
            "p50_7": np.median(rho7),
            "near": (np.median(near) if len(near) else np.nan),
            "n_sw": len(sw)}


def main() -> None:
    a = parse()
    torch.manual_seed(a.seed)
    mn, mx = load_norm(a.norm_ckpt)
    suites = [s.strip() for s in a.suites.split(",")]

    # ε 는 액션과 같은 shape 의 표준정규. 해석적 기대값도 같이 적는다.
    N = 2_000_000
    e6 = torch.randn(N, POSE).norm(dim=1).numpy()
    e7 = torch.randn(N, ADIM).norm(dim=1).numpy()

    W = 92
    L: list[str] = ["=" * W,
                    "앵커 노이즈 ε 와 액션 a 의 크기 비교 — LIBERO (정규화된 [-1,1] 좌표)",
                    "=" * W, "",
                    "  flow matching 경로   x_t = (1−t)·ε + t·a ,   속도  v = a − ε",
                    "",
                    "  앵커는 student 와 teacher 를 **같은 x_t** 에서 평가하므로 두 속도가 ε 를",
                    "  공유한다. ‖ε‖ 이 ‖a‖ 보다 크면 v 의 방향이 ε 에 끌려가, 두 정책이 서로",
                    "  다른 액션을 원해도 **속도 각도는 작게** 나온다. 그 비가 압축의 세기다.",
                    "",
                    f"  ε ~ N(0, I), 액션과 같은 shape (B,H,{ADIM}). 방향 항은 pose {POSE} 축만 쓴다.",
                    f"  정규화 통계: {Path(a.norm_ckpt).parent.name}",
                    ""]

    L += ["-" * W, "① 노이즈 ε 의 크기 (해석적 — 데이터와 무관)", "-" * W,
          f"  {'차원':<22}{'중앙값':>10}{'평균':>10}{'5%':>9}{'95%':>9}",
          f"  {'pose 6 축  ‖ε[0:6]‖':<22}{np.median(e6):>10.3f}{e6.mean():>10.3f}"
          f"{np.percentile(e6,5):>9.3f}{np.percentile(e6,95):>9.3f}",
          f"  {'전체 7 축  ‖ε‖':<22}{np.median(e7):>10.3f}{e7.mean():>10.3f}"
          f"{np.percentile(e7,5):>9.3f}{np.percentile(e7,95):>9.3f}",
          f"  (참고) √6 = {np.sqrt(6):.3f},  √7 = {np.sqrt(7):.3f}", ""]

    allp50 = []
    for suite in suites:
        L += ["-" * W, f"② 액션 크기 ‖a[0:6]‖ — {suite}", "-" * W,
              f"  {'task':<6}{'프레임':>9}{'중앙값':>10}{'평균':>9}{'5%':>8}{'95%':>8}"
              f"{'파지근처':>10}{'전환수':>8}"]
        rows = []
        for k in range(a.n_tasks):
            st = task_stats(suite, k, mn, mx, a.files_per_task)
            if st is None:
                continue
            rows.append(st)
            near = "-" if np.isnan(st["near"]) else f"{st['near']:.3f}"
            L.append(f"  t{k:<5}{st['n']:>9}{st['p50']:>10.3f}{st['mean']:>9.3f}"
                     f"{st['p05']:>8.3f}{st['p95']:>8.3f}{near:>10}{st['n_sw']:>8}")
        if rows:
            # ★ 스위트 요약은 **전 프레임을 모아** 낸 통계다. 태스크별 중앙값을
            #   다시 평균하면 프레임 수가 다른 태스크가 같은 무게를 갖게 되고,
            #   ε 쪽(표본 중앙값)과 집계 방식도 어긋난다.
            pool = np.concatenate([r["rho"] for r in rows])
            allp50.append((suite, float(np.median(pool)), float(pool.mean()), len(pool)))
            L.append(f"  {'전체':<6}{len(pool):>9}{np.median(pool):>10.3f}{pool.mean():>9.3f}"
                     f"{np.percentile(pool,5):>8.3f}{np.percentile(pool,95):>8.3f}"
                     f"   (태스크 풀드)")
        L.append("")

    L += ["=" * W, "③ 비교 — ε 가 액션보다 몇 배 큰가", "=" * W,
          f"  {'스위트':<16}{'‖a‖ 중앙':>11}{'‖ε‖ 중앙':>11}{'비 ε/a':>10}"
          f"{'‖v‖ 추정':>11}{'액션 비중':>11}"]
    me, mee = float(np.median(e6)), float(e6.mean())
    L[-1] = (f"  {'스위트':<16}{'프레임':>9}{'‖a‖ 중앙':>10}{'‖a‖ 평균':>10}"
             f"{'‖ε‖ 중앙':>10}{'비(중앙)':>9}{'‖v‖ 추정':>10}{'액션 비중':>10}")
    for suite, p50, pmean, npool in allp50:
        # v = a − ε, 두 벡터가 독립이면 E‖v‖² ≈ E‖a‖² + E‖ε‖²
        vn = float(np.sqrt(p50 ** 2 + me ** 2))
        L.append(f"  {suite:<16}{npool:>9}{p50:>10.3f}{pmean:>10.3f}{me:>10.3f}"
                 f"{me/p50:>9.2f}{vn:>10.3f}{100*p50**2/(p50**2+me**2):>9.1f}%")
    L += ["",
          "읽을 때",
          "  · 'ε/a' 가 클수록 속도 v 가 노이즈에 끌려가 각도가 압축된다.",
          "    액션이 직각(90°)으로 틀어져도 속도 각도는 27° 로 보이는 식이다.",
          "  · '액션 비중' 은 ‖v‖² 중 액션이 차지하는 몫이다. 이 값이 작을수록",
          "    속도 각도가 정책 불일치를 적게 반영한다.",
          "  · 압축은 t 에도 의존한다. t 가 1 에 가까우면 x_t 가 액션에 붙어",
          "    압축이 사라진다 (측정: t=0.1 에서 6.4배, t=0.9 에서 1.0배).",
          "  · '파지근처' 는 그리퍼 전환 ±8 스텝의 ‖a[0:6]‖ 중앙값이다. 그 구간에서",
          "    액션이 작아져 ε 대비 더 묻힌다 — 보호가 가장 필요한 곳에서 신호가 가장 약하다.",
          "  · 이 표는 **액션 크기**만 본다. teacher 가 내는 flow velocity 의 실측",
          "    노름은 별도이며, 앵커 로그와 L2_V6_gripper_probe.py 에서 ‖v̄‖ ≈ 2.5 로 쟀다.",
          ""]

    txt = "\n".join(L)
    Path(a.out).write_text(txt)
    print(txt)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
