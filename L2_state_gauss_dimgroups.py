#!/usr/bin/env python
"""셀 내부 가우시안 적합도를 **차원 묶음별로** 잰다 — 그리퍼가 범인인지 가른다.

    python L2_state_gauss_dimgroups.py --suite libero_goal --tasks 0,1,2,3,4,5,6,7,8,9

묻는 것  "셀 안에서 가우시안이 맞나"를 pose(x,y,z,r,p,y) 와 gripper 로 나눠 본다.
    L2_state_gauss_calibration.py 의 Figure (a) 와 같은 검정이되, 적합·검정에
    쓰는 차원만 바꾼다.

★ 공정한 비교를 위해 **셀 나누기는 세 설정 모두 동일**하다 — 원좌표 16 차원
  전체로 k-means 를 한 번만 돌리고, 그 라벨을 세 설정이 공유한다. 차원을 빼고
  클러스터링까지 다시 하면 "다른 셀의 적합도"를 비교하는 것이 되어 답이 안 된다.

★ 임계값은 설정마다 d 가 달라 다시 계산한다 — T²·n(n−d)/(d(n²−1)) ~ F(d,n−d).
  d 를 고정해 두면 차원 수가 적은 설정이 저절로 유리해진다.

★ 기저는 pair 로 둔다([m,d] = [(a+b)/2, a−b]). 원좌표에서 대각 가우시안을 적합하면
  t−1 과 t 의 r≈0.99 상관을 못 담아 프레임 간 변화가 실제보다 8~10 배 커진다.
  --basis raw 로 원좌표도 볼 수 있다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import B1                                                          # noqa: E402
from L2_state_gauss_calibration import (CK_DEFAULT, cell_sigmas,   # noqa: E402
                                        collect_state, d2, fit_cells,
                                        t2_threshold, to_pair)
from lerobot.datasets.factory import make_dataset                  # noqa: E402

# observation.state(8) = x,y,z,roll,pitch,yaw,gripper,gripper, 2 시점이라 16.
# pair 기저도 모터 순서가 그대로여서 같은 인덱스를 쓴다.
POSE = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
GRIP = [6, 7, 14, 15]
# ★ d 가 다르면 비교가 공정하지 않다. 포함 비율의 과잉은 차원 수에도 의존하므로
#   (중심 집중 효과가 d 에 따라 다르게 나타난다) **d=4 로 맞춘 묶음**을 함께 둔다.
#   아래 세 개는 전부 (모터 2개) x (m, d) = 4 차원이다.
GROUPS = (("전체 16차원", list(range(16))),
          ("pose 12차원 (그리퍼 제외)", POSE),
          ("[d=4] gripper", GRIP),
          ("[d=4] 위치 x,y", [0, 1, 8, 9]),
          ("[d=4] 위치 y,z", [1, 2, 9, 10]),
          ("[d=4] 회전 roll,pitch", [3, 4, 11, 12]),
          ("[d=4] 회전 pitch,yaw", [4, 5, 12, 13]))


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--codebook_k", type=int, default=100)
    p.add_argument("--min_n", type=int, default=5)
    p.add_argument("--cov_ridge", type=float, default=0.05)
    p.add_argument("--nominal", default="50,90")
    p.add_argument("--basis", choices=["raw", "pair"], default="pair")
    p.add_argument("--min_train", type=int, default=21)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=CK_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/L2_state_gauss_dimgroups.txt")
    return p.parse_args()


def main():
    a = parse()
    tasks = [int(x) for x in a.tasks.split(",") if x.strip()]
    nominal = [float(x) / 100 for x in a.nominal.split(",")]
    # 설정별로 셀마다의 실제 포함 비율을 모은다. acc[설정][명목] = [셀별 비율]
    acc = {g: {q: [] for q in nominal} for g, _ in GROUPS}
    n_cells = 0

    for t in tasks:
        # ★ 원본 캘리브레이션과 같은 Namespace 를 쓴다 — 같은 프레임을 모아야
        #   Figure (a) 와 비교 가능한 수치가 나온다.
        ns = argparse.Namespace(suite=a.suite, device="cpu", seed=a.seed, num_workers=0,
                                batch_size=32, steps_per_task=1, log_every=100,
                                eval_episodes=1, eval_batch_size=1, mode="gausscalib",
                                p_drop=0.0, lambda_anchor=0.0)
        cfg = B1.build_cfg(ns, t, a.ckpt, Path("/tmp/l2sgauss_dimgrp"))
        ds = make_dataset(cfg)
        drop_n = int(getattr(cfg.policy, "drop_n_last_frames", 0))
        s_np, _ep, _frm, _ph = collect_state(ds, 0, drop_n, a.seed)

        # ★ 셀 나누기는 한 번만, 16 차원 원좌표로. 세 설정이 이 라벨을 공유한다.
        lab, _cen, _m, _s = fit_cells(
            torch.as_tensor(s_np, dtype=torch.float32, device=a.device),
            a.codebook_k, a.seed, a.min_n)
        lab = lab.cpu().numpy()
        K_eff = int(lab.max()) + 1

        # 적합 기저. 코드북이 실제로 쓰는 좌표계와 같게 둔다.
        fit = to_pair(s_np) if a.basis == "pair" else s_np

        used = 0
        for k in range(K_eff):
            sel = lab == k
            n_tr = int(sel.sum())
            if n_tr < a.min_train:
                continue
            ok = False
            for gname, dims in GROUPS:
                D = len(dims)
                if n_tr <= D + 1:          # F(d, n−d) 가 정의되지 않는다
                    continue
                x = fit[sel][:, dims]
                mu, S_full, S_diag = cell_sigmas(x, a.cov_ridge)
                dd = d2(x, mu, S_diag)     # 대각 = 샘플러가 실제로 쓰는 모델
                for q in nominal:
                    thr = t2_threshold(q, D, n_tr)
                    acc[gname][q].append(float(np.mean(dd <= thr)))
                ok = True
            used += ok
        n_cells += used
        print(f"[dimgrp] task {t}  프레임 {len(s_np)}  K_eff={K_eff}  쓸 수 있는 셀 {used}",
              flush=True)

    L = []
    add = L.append
    add("=" * 88)
    add(f"셀 내부 가우시안 적합도 — 차원 묶음별  ({a.suite}, task {a.tasks})")
    add("=" * 88)
    add("")
    add(f"  셀 {n_cells}개   K={a.codebook_k}   기저={a.basis}   대각 공분산(샘플러와 동일)")
    add("  셀 나누기는 16 차원 원좌표로 한 번만 — 세 줄이 **같은 셀**을 본다.")
    add("  임계값은 설정마다 d 를 반영해 다시 계산한다 (F(d, n−d)).")
    add("")
    add("  명목 비율(가우시안이라면 이만큼 들어와야 함) 대비 실제 포함 비율의 중앙값.")
    add("  오차가 0 에 가까울수록 그 차원 묶음이 셀 안에서 가우시안에 가깝다.")
    add("")
    hdr = f"  {'차원 묶음':<28}{'d':>3}"
    for q in nominal:
        hdr += f"{'명목'+format(q*100,'.0f')+'%':>12}{'오차':>8}"
    add(hdr)
    add("  " + "-" * 84)
    for gname, dims in GROUPS:
        row = f"  {gname:<28}{len(dims):>3}"
        for q in nominal:
            v = acc[gname][q]
            if not v:
                row += f"{'-':>12}{'-':>8}"; continue
            med = float(np.median(v)) * 100
            row += f"{med:>11.1f}%{med - q*100:>+8.1f}"
        add(row)
    add("")
    txt = "\n".join(L)
    Path(a.out).write_text(txt + "\n")
    print()
    print(txt)


if __name__ == "__main__":
    main()
