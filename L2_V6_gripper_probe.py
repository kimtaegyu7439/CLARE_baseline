#!/usr/bin/env python
"""그리퍼 개폐 시점 근처에서 방향오차 e 와 DEA 가중 w 가 어떻게 되는가.

    python L2_V6_gripper_probe.py --run results/clare_v6_dea_goal_seed7 \
        --ckpt outputs/CLARE_V6/dea_seed7 --task 9 --batches 8

왜 보는가
    pose 속도가 작아지는 구간(정지·파지 전환)에서는 방향이 불안정해 각도차가
    커질 수 있다. 그렇다면 DEA 가 그 구간을 더 보호해 줘야 한다. 그런데 DEA 의
    w_j 는 **태스크 축(j)** 에서만 정규화된다. 그래서 두 가지를 따로 봐야 한다.

      (A) e 가 파지 전환 근처에서 실제로 커지는가      <- 전제 확인
      (B) 그 자리에서 **태스크 간 e 의 편차**가 커지는가 <- DEA 가 쓸 수 있는 것

    (A) 가 참이어도 (B) 가 거짓이면 — 즉 모든 과거 태스크가 똑같이 커지면 —
    w_j 는 균일하게 남고 DEA 는 아무것도 재배분하지 않는다. 그 경우 필요한 것은
    태스크 간 재배분이 아니라 **(t,h) 축 가중**이다.

전환 시점 정의
    teacher 부트스트랩 â (B,H,A) 의 gripper 축(index 6) 부호가 바뀌는 h.
    LIBERO 그리퍼는 ±1 이진이라 부호 변화가 곧 개폐 명령 전환이다.
    각 (b,h) 에 대해 **가장 가까운 전환까지의 거리** d = min|h − h_sw| 로 묶는다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import L2_V4_seq as B1
from L2_V6 import DEAAnchor, dea_weights, split_mag_dir


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="codebooks/ 가 있는 결과 폴더")
    p.add_argument("--ckpt", required=True, help="ckpt_root (task_*/ 가 있는 상위)")
    p.add_argument("--task", type=int, default=9, help="이 스테이지의 정책으로 잰다")
    p.add_argument("--batches", type=int, default=8)
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
    p.add_argument("--eps_w", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--anchor_form", default="dea")
    p.add_argument("--anchor_reduce", default="mean")
    p.add_argument("--out", default=None)
    # DEAAnchor.__init__ 가 읽는 값들. 진단에서는 코드북을 새로 만들지 않으므로
    # 실제로 쓰이지 않지만 생성자가 요구한다.
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
    a = p.parse_args()
    a.out_dir = a.run
    return a


def switch_distance(a_hat: torch.Tensor, grip_dim: int = 6) -> torch.Tensor:
    """(B,H,A) -> (B,H) 가장 가까운 그리퍼 전환까지의 거리. 전환이 없으면 큰 값."""
    g = a_hat[..., grip_dim]                                   # (B,H)
    sw = (g[:, 1:] * g[:, :-1]) < 0                            # (B,H-1) 부호 변화
    B, H = g.shape
    idx = torch.arange(H, device=g.device).view(1, H, 1).float()
    sw_pos = (torch.arange(H - 1, device=g.device).float() + 0.5).view(1, 1, H - 1)
    big = torch.full((B, H), float(H), device=g.device)
    if not sw.any():
        return big
    d = (idx - sw_pos).abs()                                   # (B,H,H-1)
    d = torch.where(sw.view(B, 1, H - 1), d, torch.full_like(d, float(H)))
    return d.min(-1).values                                    # (B,H)


def main() -> int:
    a = parse()
    B1.init_logging()
    run, ck = Path(a.run), Path(a.ckpt)
    out = Path(a.out) if a.out else run / "gripper_probe.txt"
    device = B1.get_safe_torch_device(a.device, log=True)

    ckdir = sorted(ck.glob("*_ours"))[0] if list(ck.glob("*_ours")) else ck
    pol_path = ckdir / f"task_{a.task}" / "checkpoints" / "last" / "pretrained_model"
    if not (pol_path / "model.safetensors").exists():
        raise SystemExit(f"정책 체크포인트 없음: {pol_path}")

    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy

    cfg = B1.build_cfg(a, a.task, str(pol_path), run)
    ds = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds.meta); policy.to(device); policy.eval()

    # 앵커 객체를 만들고 코드북을 붙인다(학습은 하지 않는다)
    anchor = DEAAnchor(a)
    anchor.out = run
    for j in range(a.task):
        f = run / "codebooks" / f"codebook_task{j}.pt"
        if f.exists():
            anchor.books[j] = B1.to_device_cb(
                torch.load(f, map_location="cpu", weights_only=False), device)
    # ★ teacher 는 **직전 스테이지**(task_{k-1}) 체크포인트다. 학습 중 앵커가
    #   보던 것과 같게 맞춘다. 자기 자신을 스냅샷하면 student==teacher 라 e=0 이
    #   되어 아무것도 안 보인다(처음에 그렇게 짰다가 전부 0 이 나왔다).
    tpath = ckdir / f"task_{a.task - 1}" / "checkpoints" / "last" / "pretrained_model"
    if not (tpath / "model.safetensors").exists():
        raise SystemExit(f"teacher 체크포인트 없음: {tpath}")
    tcfg = B1.build_cfg(a, a.task - 1, str(tpath), run)
    anchor.teacher = make_policy(cfg=tcfg.policy, ds_meta=ds.meta)
    anchor.teacher.to(device); anchor.teacher.eval()
    for q_ in anchor.teacher.parameters():
        q_.requires_grad_(False)
    print(f"[probe] teacher = task {a.task - 1}, student = task {a.task}")
    js = sorted(anchor.books)
    N = len(js)
    print(f"[probe] 과거 태스크 {N}개, 정책 = task {a.task}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=True,
        multiprocessing_context="spawn" if a.num_workers > 0 else None)

    import json as _json
    instr = _json.loads((run / "instructions.json").read_text())
    E, D, W, R, EA, RA = [], [], [], [], [], []
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
            es, vs, ds_, rs_, eas, ras = [], [], [], [], [], []
            for j in js:
                q = anchor._query(policy, batch, x_t, t, a.task, instr, j, n,
                                  st_shape, tcol, device)
                v_s = anchor._student_v(policy, q, t)
                _, e, r_s, r_t, va = split_mag_dir(v_s, q["v_t"], B1.POSE_DIMS)
                # ★ 액션 수준. 선형 경로에서 v = (a − x_t)/(1−t) 이므로
                #   â = x_t + (1−t)·v 로 끝점을 되살린다. 속도 각도는 두 v 가
                #   같은 x_t(노이즈 지배)를 공유해 작게 나오지만, 액션 각도는
                #   그 공통 성분이 빠져 실제 불일치를 본다.
                one_t = (1.0 - t).view(-1, 1, 1)
                a_s = q["x_j"] + one_t * v_s
                a_t = q["x_j"] + one_t * q["v_t"]
                _, e_a, ra_s, ra_t, va_a = split_mag_dir(a_s, a_t, B1.POSE_DIMS)
                eas.append(e_a); ras.append(ra_t)
                # 부트스트랩 â 는 x_j 로부터 역산하지 않고 teacher 속도로 다시 만든다
                es.append(e); vs.append(va); rs_.append(r_t)
                ds_.append(switch_distance(q["a_hat"]))    # 부트스트랩 액션의 gripper 부호
            Es, Vs = torch.stack(es), torch.stack(vs)       # (N,B,H)
            Ws = dea_weights(Es, Vs, a.eps_w)               # (N,B,H)
            E.append(Es.cpu()); W.append(Ws.cpu())
            D.append(torch.stack(ds_).cpu()); R.append(torch.stack(rs_).cpu())
            EA.append(torch.stack(eas).cpu()); RA.append(torch.stack(ras).cpu())

    E = torch.cat(E, 1); W = torch.cat(W, 1); D = torch.cat(D, 1); R = torch.cat(R, 1)
    EA = torch.cat(EA, 1); RA = torch.cat(RA, 1)
    d0 = D[0]                                               # (B,H) 태스크 0 기준 거리
    L = ["=" * 86,
         f"그리퍼 전환 근처의 방향오차와 DEA 가중  ({run.name}, task {a.task}, N={N})",
         "=" * 86, "",
         "  (A) e 가 전환 근처에서 커지는가   (B) 태스크 간 편차가 커지는가",
         "      w_max·N 이 1.0 이면 균일(재배분 없음), 클수록 한 태스크로 몰린 것",
         "", f"  {'거리 d':<8}{'칸수':>7}{'e_vel':>10}{'e_act':>10}"
         f"{'배율':>7}{'‖v̄‖':>8}{'‖â‖':>8}{'w_max·N':>9}", "  " + "-" * 68]
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 100)]
    for lo, hi in bins:
        m = (d0 >= lo) & (d0 < hi)                          # (B,H)
        if not m.any():
            continue
        e_sel = E[:, m]                                     # (N, n_sel)
        w_sel = W[:, m]
        r_sel = R[:, m]
        cv = (e_sel.std(0) / e_sel.mean(0).clamp_min(1e-9)).mean()   # 태스크 축 변동계수
        lab = f"{lo}~{hi-1}" if hi < 100 else f"{lo}+"
        ea_sel, ra_sel = EA[:, m], RA[:, m]
        ev, ea = float(e_sel.mean()), float(ea_sel.mean())
        L.append(f"  {lab:<8}{int(m.sum()):>7}{ev:>10.5f}{ea:>10.5f}"
                 f"{(ea / max(ev, 1e-12)):>7.1f}{float(r_sel.mean()):>8.3f}"
                 f"{float(ra_sel.mean()):>8.3f}{float(w_sel.max(0).values.mean()) * N:>9.3f}")
    L += ["",
          "읽을 때",
          "  · e 평균이 d 가 작을수록 크면 (A) 성립 — 전환 근처에서 방향이 흔들린다.",
          "  · 그런데 DEA 가 쓸 수 있는 건 **태스크 간** 편차뿐이다. 변동계수와",
          "    w_max·N 이 d 와 무관하게 평평하면, 전환 근처에서도 재배분은 일어나지",
          "    않는다 — 모든 과거 태스크가 똑같이 흔들린다는 뜻이다.",
          "  · 그 경우 필요한 것은 태스크 축이 아니라 **(t,h) 축 가중**이다.",
          "  · ‖v̄_pose‖ 가 작을수록 방향이 불안정해진다. e 와 같이 움직이는지 본다.",
          ""]
    txt = "\n".join(L)
    out.write_text(txt); print(txt); print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
