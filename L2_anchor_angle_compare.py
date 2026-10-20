#!/usr/bin/env python
"""앵커 손실의 방향 항 e 를 속도 공간과 액션 공간에서 각각 재어 비교한다.

    python L2_anchor_angle_compare.py --run results/clare_v6_sqerr_goal_seed7 \
        --ckpt outputs/CLARE_V6/sqerr_seed7 --stages 3,6,9

무엇을 비교하나
    앵커는 과거 태스크 j 마다 student 와 teacher 의 속도를 **같은 질의점 x_j** 에서
    평가한다. 방향 항은

        e = 1 − u_s · u_t ,   u = (벡터)/‖벡터‖

    인데 그 "벡터" 를 무엇으로 두느냐가 V6 와 V7 의 차이다.

        V6 (속도)  v_s, v_t                         <- 노트 그대로
        V7 (액션)  a = x_j + (1−t)·v   의 a_s, a_t   <- 공통 노이즈를 뺀 끝점

    선형 경로에서 v = a − ε 이고 두 속도가 **같은 ε 를 공유**하므로, 차이 벡터는
    상쇄되지만(v_s−v_t = a_s−a_t) **각도는 상쇄되지 않는다**. ‖ε‖ 이 ‖a‖ 보다
    2.5~2.9 배 커서 두 속도가 둘 다 −ε 쪽을 가리키기 때문이다.

★ 같은 질의 (x_j, t, s̃, õ) 에서 두 값을 동시에 뽑는다. 샘플링 차이가 섞이지
  않으므로 비는 순수하게 파라미터화 차이다.
★ 압축 정도는 t 에 의존한다(t→1 이면 x_j 가 액션에 붙어 압축이 사라진다).
  그래서 t 구간별로 나눠 적는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import L2_V4_seq as B1
from L2_V6 import DEAAnchor, split_mag_dir, to_action


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stages", default="3,6,9", help="이 스테이지들의 정책으로 잰다")
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval_seed", type=int, default=7)
    p.add_argument("--eval_episodes", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--steps_per_task", type=int, default=10000)
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--log_every_anchor", type=int, default=10 ** 9)
    # 앵커 생성자용 (코드북은 저장본을 쓰므로 실제로는 안 쓰인다)
    p.add_argument("--codebook_k", type=int, default=96)
    p.add_argument("--n_pairs", type=int, default=8000)
    p.add_argument("--h_scale", type=float, default=1.0)
    p.add_argument("--ridge_rho", type=float, default=0.05)
    p.add_argument("--grad_min_frames", type=int, default=64)
    p.add_argument("--lambda_level", type=float, default=3.0)
    p.add_argument("--sampler_mode", default="hard_linear")
    p.add_argument("--anchor_weight", default="speedinv")
    p.add_argument("--teacher_bf16", action="store_true", default=False)
    p.add_argument("--pose_dims", type=int, default=None)
    p.add_argument("--anchor_form", default="dea")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--eps_w", type=float, default=1e-3)
    p.add_argument("--anchor_reduce", default="mean")
    p.add_argument("--dir_space", default="vel")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    a.out_dir = a.run
    return a


def main() -> int:
    a = parse()
    B1.init_logging()
    run, ck = Path(a.run), Path(a.ckpt)
    out = Path(a.out) if a.out else run / "angle_compare.txt"
    device = B1.get_safe_torch_device(a.device, log=True)
    ckdir = sorted(ck.glob("*_ours"))[0] if list(ck.glob("*_ours")) else ck
    P = B1.POSE_DIMS
    instr = json.loads((run / "instructions.json").read_text())

    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy

    W = 88
    L = ["=" * W, f"앵커 방향 항 e — 속도 공간 vs 액션 공간  ({run.name}, {a.suite})",
         "=" * W, "",
         "  같은 질의 (x_j, t, s̃, õ) 에서 두 값을 동시에 뽑는다.",
         "    e_vel = 1 − cos(v_s, v_t)                  <- V6 (노트 그대로)",
         "    e_act = 1 − cos(a_s, a_t),  a = x_j+(1−t)v  <- V7 (공통 ε 제거)",
         "", "  v = a − ε 이고 두 속도가 같은 ε 를 공유한다. 차이 벡터는 상쇄되지만",
         "  각도는 상쇄되지 않는다 — ‖ε‖/‖a‖ ≈ 2.6 이라 둘 다 −ε 쪽을 가리킨다.", ""]

    for k in [int(x) for x in a.stages.split(",")]:
        sp = ckdir / f"task_{k}" / "checkpoints" / "last" / "pretrained_model"
        tp = ckdir / f"task_{k-1}" / "checkpoints" / "last" / "pretrained_model"
        if not (sp / "model.safetensors").exists() or not (tp / "model.safetensors").exists():
            L.append(f"stage {k}: 체크포인트 없음 — 건너뜀"); continue

        cfg = B1.build_cfg(a, k, str(sp), run)
        ds = make_dataset(cfg)
        policy = make_policy(cfg=cfg.policy, ds_meta=ds.meta); policy.to(device); policy.eval()
        anchor = DEAAnchor(a)
        anchor.out = run
        for j in range(k):
            f = run / "codebooks" / f"codebook_task{j}.pt"
            if f.exists():
                anchor.books[j] = B1.to_device_cb(
                    torch.load(f, map_location="cpu", weights_only=False), device)
        tcfg = B1.build_cfg(a, k - 1, str(tp), run)
        anchor.teacher = make_policy(cfg=tcfg.policy, ds_meta=ds.meta)
        anchor.teacher.to(device); anchor.teacher.eval()
        for q_ in anchor.teacher.parameters():
            q_.requires_grad_(False)

        loader = torch.utils.data.DataLoader(
            ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=True,
            multiprocessing_context="spawn" if a.num_workers > 0 else None)
        js = sorted(anchor.books)
        EV, EA, TT = [], [], []
        with torch.no_grad():
            for bi, raw in enumerate(loader):
                if bi >= a.batches:
                    break
                batch = B1.prep_batch(policy, B1.to_device(raw, device))
                anchor.cls = B1.rgb_cls(policy, batch)
                n = batch["observation.state"].shape[0]
                st_shape = batch["observation.state"].shape
                x_t, t, _ = B1.sample_fm(policy, batch)
                tcol = t[:, None, None]
                for j in js:
                    q = anchor._query(policy, batch, x_t, t, k, instr, j, n,
                                      st_shape, tcol, device)
                    v_s = anchor._student_v(policy, q, t)
                    _, ev, _, _, _ = split_mag_dir(v_s, q["v_t"], P)
                    a_s = to_action(v_s, q["x_j"], t)
                    a_t = to_action(q["v_t"], q["x_j"], t)
                    _, ea, _, _, _ = split_mag_dir(a_s, a_t, P)
                    EV.append(ev.cpu()); EA.append(ea.cpu())
                    TT.append(t.cpu().unsqueeze(1).expand_as(ev))
        del policy, anchor, ds
        torch.cuda.empty_cache()
        if not EV:
            continue
        ev = torch.cat([x.flatten() for x in EV]); ea = torch.cat([x.flatten() for x in EA])
        tt = torch.cat([x.flatten() for x in TT])

        L += ["-" * W, f"stage {k}   (student = task {k}, teacher = task {k-1}, 과거 {len(js)}개)",
              "-" * W,
              f"  {'t 구간':<12}{'표본':>9}{'e_vel':>11}{'e_act':>11}{'배율':>8}"
              f"{'각도_vel':>10}{'각도_act':>10}"]
        for lo, hi in ((0.0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)):
            m = (tt >= lo) & (tt < hi)
            if not m.any():
                continue
            v, c = float(ev[m].mean()), float(ea[m].mean())
            dv = float(torch.rad2deg(torch.acos((1 - ev[m]).clamp(-1, 1))).mean())
            da = float(torch.rad2deg(torch.acos((1 - ea[m]).clamp(-1, 1))).mean())
            L.append(f"  {f'{lo:.1f}~{hi:.1f}':<12}{int(m.sum()):>9}{v:>11.5f}{c:>11.5f}"
                     f"{c/max(v,1e-12):>8.1f}{dv:>10.2f}°{da:>9.2f}°")
        v, c = float(ev.mean()), float(ea.mean())
        L.append(f"  {'전체':<12}{len(ev):>9}{v:>11.5f}{c:>11.5f}{c/max(v,1e-12):>8.1f}"
                 f"{float(torch.rad2deg(torch.acos((1-ev).clamp(-1,1))).mean()):>10.2f}°"
                 f"{float(torch.rad2deg(torch.acos((1-ea).clamp(-1,1))).mean()):>9.2f}°")
        L.append("")

    L += ["읽을 때",
          "  · '배율' = e_act / e_vel. 1 보다 크면 속도 각도가 실제 불일치를 과소평가한다.",
          "  · t 가 작을수록 x_j 가 노이즈에 가까워 압축이 세다. t→1 이면 1 에 수렴한다.",
          "  · t 는 균등 샘플링되므로, 속도 공간을 쓰면 **같은 정책 불일치가 t 에 따라",
          "    다른 e 로 들어온다**. β 가 상수인 unifmd/dea 에서만 생기는 왜곡이다",
          "    (sqerr 은 계수 2·r_s·r̄ 가 t 와 함께 커져 자동 보정된다).",
          ""]
    txt = "\n".join(L)
    out.write_text(txt); print(txt); print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
