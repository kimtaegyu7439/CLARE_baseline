#!/usr/bin/env python
"""속도장이 현재 태스크로 채워지는 과정을 스텝 축으로 잰다.

가설(사용자)
  flow matching 은 연속 공간 위의 속도장을 맞춘다. 학습이 길어질수록 그 공간이
  현재 태스크에 맞는 방향으로 점점 채워진다. 다음 태스크가 들어올 '빈 자리'가
  사라진다. 페널티(앵커)가 없으면 그냥 덮어쓰면 되지만, 있으면 못 덮는다.

재는 것 — task 0 만 학습한 체크포인트들(1000..20000 스텝)에서

  ① 평면 지도
     한 샘플마다 ε 에서 출발하는 두 방향  g_0 = a_0 − ε (task0 동작),
     g_1 = a_1 − ε (task1 동작) 을 잡아 2D 평면을 만든다. flow 경로
     x_t = ε + t·g 가 이 평면 안의 직선이므로 두 태스크 경로가 함께 보인다.
     격자점마다 v 를 재서  cos(v,g_0) − cos(v,g_1) 로 색칠 = 누구를 가리키는가.

  ② 점유율
     task 1 경로 근방에서 cos(v,g_0) > cos(v,g_1) 인 지점 비율.
     '이미 task 0 이 차지한 자리'.

  ③ 학습 비용
     task 1 경로 위에서 ‖v − g_1‖/‖g_1‖. 새 태스크를 맞추려면 이만큼 움직여야 한다.

  ④ 페널티의 저항
     Ĝ = ‖v(o_1, ℓ_0) − g_1‖/‖g_1‖. 앵커가 지키려는 값과 task1 정답의 충돌량.
     이게 커질수록 페널티가 새 학습을 더 세게 막는다.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B_merge import _ns

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/B_fill/task0_s20000")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--n_planes", type=int, default=16, help="평면 지도에 쓸 샘플 수")
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--n_path", type=int, default=256, help="경로 통계용 샘플 수")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_fill")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = [B1.task_instruction(f"{ds_prefix}{i}") for i in range(2)]

    ckpts = sorted((REPO / a.root / "checkpoints").glob("0*"),
                   key=lambda p: int(p.name))
    ckpts = [(int(p.name), p / "pretrained_model") for p in ckpts
             if (p / "pretrained_model").is_dir()]
    if not ckpts:
        raise SystemExit(f"체크포인트 없음: {REPO/a.root}/checkpoints")
    print("체크포인트:", [s for s, _ in ckpts])

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_fill"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    seed_pol = load(ckpts[-1][1])

    def batches(j, n):
        cfg = B1.build_cfg(_ns(a), j, str(ckpts[-1][1]), Path("/tmp/b_fill"))
        ds = make_dataset(cfg)
        sp = EpisodeAwareSampler(ds.episode_data_index,
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sp,
                                         num_workers=0, drop_last=True)
        torch.manual_seed(a.seed + j); it = iter(dl)
        return [B1.prep_batch(seed_pol, B1.to_device(next(it), device))
                for _ in range((n + a.batch_size - 1) // a.batch_size)]

    B1_ = batches(1, max(a.n_path, a.n_planes))      # task 1 관측 + 정답
    B0_ = batches(0, max(a.n_path, a.n_planes))      # task 0 정답 (동작 방향만 쓴다)
    A1 = torch.cat([b["action"] for b in B1_])       # (N,16,7)
    A0 = torch.cat([b["action"] for b in B0_])
    N = min(A1.shape[0], A0.shape[0], a.n_path)
    torch.manual_seed(a.seed * 3)
    EPS = torch.randn(N, *A1.shape[1:], device=device)
    T = torch.rand(N, device=device)
    G1 = A1[:N] - EPS                                 # task1 목표 속도
    G0 = A0[:N] - EPS                                 # task0 목표 속도
    XT = EPS + T[:, None, None] * G1                  # task1 경로 위의 점

    # 평면 기저: u = ĝ1,  w = ĝ0 에서 u 성분을 뺀 것
    def unit(x):
        return x / x.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
    U = unit(G1)
    W = G0 - (G0 * U).flatten(1).sum(1)[:, None, None] * U
    W = unit(W)
    L1 = G1.flatten(1).norm(dim=1)                    # ‖g1‖
    del seed_pol; torch.cuda.empty_cache()

    def cos(x, y):
        return (x * y).flatten(1).sum(1) / (x.flatten(1).norm(dim=1).clamp_min(1e-8)
                                            * y.flatten(1).norm(dim=1).clamp_min(1e-8))

    # 평면 격자 좌표 (‖g1‖ 단위)
    P = torch.linspace(0.0, 1.15, a.grid)             # u 축 = task1 진행 방향
    Q = torch.linspace(-0.55, 1.15, a.grid)           # w 축 = task0 쪽 성분
    nP = min(a.n_planes, N)

    rows, maps = [], {}
    for step, path in ckpts:
        pol = load(path); net = pol.dit_flow.velocity_net
        with torch.no_grad():
            # ── 경로 통계 ───────────────────────────────────────────────
            claimed = cost = conflict = 0.0
            nb = 0
            for s in range(0, N, a.batch_size):
                e = min(s + a.batch_size, N)
                idx = slice(s, e)
                bi = s // a.batch_size
                b = B1_[bi]
                n = e - s
                tail = B1.cond_tail(pol, {k: (v[:n] if torch.is_tensor(v) else v[:n])
                                          for k, v in b.items()})
                c1 = B1.make_cond(B1.encode_lang(pol, [instr[1]] * n), tail)
                c0 = B1.make_cond(B1.encode_lang(pol, [instr[0]] * n), tail)
                v1 = net(noisy_actions=XT[idx], time=T[idx], global_cond=c1)
                v0 = net(noisy_actions=XT[idx], time=T[idx], global_cond=c0)
                claimed += float((cos(v1, G0[idx]) > cos(v1, G1[idx])).float().mean())
                cost += float(((v1 - G1[idx]).flatten(1).norm(dim=1) / L1[idx]).mean())
                conflict += float(((v0 - G1[idx]).flatten(1).norm(dim=1) / L1[idx]).mean())
                nb += 1
            rows.append({"step": step, "claimed": claimed / nb,
                         "cost": cost / nb, "conflict": conflict / nb})

            # ── 평면 지도 ───────────────────────────────────────────────
            b = B1_[0]
            tail = B1.cond_tail(pol, {k: (v[:nP] if torch.is_tensor(v) else v[:nP])
                                      for k, v in b.items()})
            cond = B1.make_cond(B1.encode_lang(pol, [instr[1]] * nP), tail)
            M = torch.zeros(a.grid, a.grid)
            for ip, p in enumerate(P):
                for iq, q in enumerate(Q):
                    x = (EPS[:nP] + p * L1[:nP, None, None] * U[:nP]
                         + q * L1[:nP, None, None] * W[:nP])
                    t = torch.full((nP,), float(p.clamp(0, 1)), device=device)
                    v = net(noisy_actions=x, time=t, global_cond=cond)
                    M[ip, iq] = float((cos(v, G0[:nP]) - cos(v, G1[:nP])).mean())
            maps[step] = M.tolist()
        del pol; torch.cuda.empty_cache()
        r = rows[-1]
        print(f"[fill] step {step:6d}  점유율 {r['claimed']*100:5.1f}%  "
              f"학습비용 {r['cost']:.3f}  충돌 Ĝ {r['conflict']:.3f}", flush=True)

    json.dump({"rows": rows, "maps": maps,
               "P": P.tolist(), "Q": Q.tolist()},
              (out / "probe.json").open("w"))
    print(f"saved -> {out/'probe.json'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
