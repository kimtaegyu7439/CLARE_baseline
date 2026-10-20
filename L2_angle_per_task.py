#!/usr/bin/env python
"""앵커 각도를 **과거 태스크별로** 재어 npz 로 떨구고, 그림 3 패널을 그린다.

    python L2_angle_per_task.py --run results/clare_v6_sqerr_goal_seed7 \
        --ckpt outputs/CLARE_V6/sqerr_seed7 --stage 9 --batches 16
    python L2_angle_per_task.py --npz results/angle_per_task_stage9.npz   # 재측정 없이 다시 그리기

무엇을 재는가
    앵커는 과거 태스크 j 마다 **같은 질의점** (x_j, t, s̃_j, õ_j) 에서 student 와
    teacher 의 속도를 뽑는다. 그 쌍에서 두 각도를 **동시에** 낸다.

        α_vel = arccos cos(v_s, v_T)
        α_act = arccos cos(a_s, a_T),   a = x_j + (1−t)·v

    같은 질의에서 나오므로 샘플링 차이가 섞이지 않는다 — 차이는 순수하게
    파라미터화(속도 vs 끝점)뿐이다.

왜 '퍼짐' 을 보는가
    평균이 크다는 것만으로는 쓸모를 말할 수 없다. 앵커가 각도로 무언가를
    판단하려면(DEA 는 각 (b,h) 에서 **태스크를 줄 세운다**) 태스크 간 차이가
    태스크 내 요동보다 커야 한다. 그래서 η² = 태스크간분산/전체분산 을 같이 낸다.
    η² 가 작으면 순위가 잡음이고, w_j ∝ e_j + ε_w 는 잡음을 따라간다.

★ 측정은 **속도 손실로 학습한 팔**(V6 sqerr = MSE 기준선)에서 하는 것이 공정하다.
  액션 각도로 학습한 V7/V9 로 재면 액션 쪽에 유리하게 편향된다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="results/clare_v6_sqerr_goal_seed7")
    p.add_argument("--ckpt", default="outputs/CLARE_V6/sqerr_seed7")
    p.add_argument("--stage", type=int, default=9, help="이 스테이지 정책으로 잰다")
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--h_lo", type=int, default=1, help='실행되는 horizon 구간 [lo,hi). dit_flow 는 actions[:, 1:9] 만 로봇에 보낸다 (n_obs_steps-1=1 부터 n_action_steps=8 개). index 0 은 t-1 이라 이미 지난 시점이고, 9~15 는 미리보기라 폐기된다. 16 전체를 재면 **쓰이지도 않는 액션**이 2/3 를 차지한다.')
    p.add_argument("--h_hi", type=int, default=9)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--npz", default=None, help="주면 측정을 건너뛰고 이 파일로 그린다")
    p.add_argument("--out_npz", default=None)
    p.add_argument("--out_png", default=None)
    # 앵커 생성자용 (코드북은 저장본을 쓴다)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval_seed", type=int, default=7)
    p.add_argument("--eval_episodes", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--steps_per_task", type=int, default=10000)
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--log_every_anchor", type=int, default=10 ** 9)
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
    p.add_argument("--anchor_form", default="sqerr")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--eps_w", type=float, default=1e-3)
    p.add_argument("--anchor_reduce", default="mean")
    p.add_argument("--dir_space", default="vel")
    a = p.parse_args()
    a.out_dir = a.run
    return a


# ═════════════════════════════════════════════════════════════════════════════
#  측정
# ═════════════════════════════════════════════════════════════════════════════
def measure(a) -> dict:
    import torch

    import L2_V4_seq as B1
    from L2_V6 import DEAAnchor, split_mag_dir, to_action

    B1.init_logging()
    run, ck = Path(a.run), Path(a.ckpt)
    device = B1.get_safe_torch_device(a.device, log=True)
    ckdir = sorted(ck.glob("*_ours"))[0] if list(ck.glob("*_ours")) else ck
    P = B1.POSE_DIMS
    instr = json.loads((run / "instructions.json").read_text())
    k = a.stage

    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy

    sp = ckdir / f"task_{k}" / "checkpoints" / "last" / "pretrained_model"
    tp = ckdir / f"task_{k-1}" / "checkpoints" / "last" / "pretrained_model"
    for q in (sp, tp):
        if not (q / "model.safetensors").exists():
            raise SystemExit(f"체크포인트 없음: {q}")

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
    for q in anchor.teacher.parameters():
        q.requires_grad_(False)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=True,
        multiprocessing_context="spawn" if a.num_workers > 0 else None)
    js = sorted(anchor.books)

    # 태스크 라벨까지 같이 모은다. 전부 (칸,) 1차원으로 펴서 쌓는다.
    AV, AA, TT, JJ = [], [], [], []
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
                # e = 1 − cos  ->  각도(도)
                deg = lambda e: torch.rad2deg(torch.acos((1 - e).clamp(-1, 1)))  # noqa: E731
                # ★ 로봇에 실제로 나가는 horizon 구간만 남긴다 (기본 1:9).
                lo, hi = a.h_lo, a.h_hi
                ev, ea = ev[:, lo:hi], ea[:, lo:hi]
                AV.append(deg(ev).flatten().cpu())
                AA.append(deg(ea).flatten().cpu())
                TT.append(t[:, None].expand_as(ev).flatten().cpu())
                JJ.append(torch.full((ev.numel(),), j, dtype=torch.int16))
    out = {"a_vel": torch.cat(AV).numpy().astype(np.float32),
           "a_act": torch.cat(AA).numpy().astype(np.float32),
           "t": torch.cat(TT).numpy().astype(np.float32),
           "j": torch.cat(JJ).numpy().astype(np.int16),
           "stage": np.int32(k), "suite": np.str_(a.suite), "run": np.str_(run.name),
           "h_lo": np.int32(a.h_lo), "h_hi": np.int32(a.h_hi)}
    del policy, anchor, ds
    torch.cuda.empty_cache()
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  η²  — 각 (b,h) 칸에서 태스크를 줄 세울 수 있는가
# ═════════════════════════════════════════════════════════════════════════════
def eta_sq(x: np.ndarray, j: np.ndarray) -> float:
    """태스크 간 분산 / 전체 분산. 일원분산분석의 η² 다.

    DEA 의 w_j 는 각 (b,h) 에서 **태스크 축으로** 정규화된다. 그래서 '태스크가
    서로 다른 정도' 가 '같은 태스크 안의 요동' 에 묻히면 순위가 잡음이 된다.
    η² -> 0 이면 w_j 는 사실상 균일 가중에 잡음을 얹은 것이다.
    """
    gm = x.mean()
    ss_t = sum(((x[j == u].mean() - gm) ** 2) * (j == u).sum() for u in np.unique(j))
    ss_all = ((x - gm) ** 2).sum()
    return float(ss_t / ss_all) if ss_all > 0 else 0.0


def main() -> int:
    a = parse()
    if a.npz:
        d = dict(np.load(a.npz, allow_pickle=True))
        src = Path(a.npz)
    else:
        d = measure(a)
        src = Path(a.out_npz or (REPO / "results" / f"angle_per_task_stage{a.stage}.npz"))
        src.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(src, **d)
        print(f"측정 저장 -> {src}")

    av, aa, tt, jj = d["a_vel"], d["a_act"], d["t"], d["j"]
    stage = int(d["stage"])
    png = Path(a.out_png or (REPO / "results" / f"L2_angle_vel_vs_act_stage{stage}.png"))

    # ── 표를 먼저 찍는다 (그림 없이도 읽히게) ─────────────────────────────
    W = 78
    L = ["=" * W,
         f"앵커 각도 — 속도 공간 vs 액션 공간  (stage {stage}, {d['suite']}, {d['run']})",
         "=" * W, "",
         "  같은 질의점에서 두 각도를 동시에 뽑았다. 차이는 파라미터화뿐이다.",
         "",
         f"  {'태스크':<8}{'표본':>9}{'α_vel 중앙':>12}{'IQR':>8}{'α_act 중앙':>12}{'IQR':>8}{'배율':>7}"]
    for u in np.unique(jj):
        m = jj == u
        mv, ma = np.median(av[m]), np.median(aa[m])
        iv = np.percentile(av[m], 75) - np.percentile(av[m], 25)
        ia = np.percentile(aa[m], 75) - np.percentile(aa[m], 25)
        L.append(f"  t{int(u):<7}{int(m.sum()):>9}{mv:>11.2f}°{iv:>7.2f}°"
                 f"{ma:>11.2f}°{ia:>7.2f}°{ma/max(mv,1e-9):>7.2f}")
    mv, ma = np.median(av), np.median(aa)
    iv = np.percentile(av, 75) - np.percentile(av, 25)
    ia = np.percentile(aa, 75) - np.percentile(aa, 25)
    L += [f"  {'전체':<8}{len(av):>9}{mv:>11.2f}°{iv:>7.2f}°{ma:>11.2f}°{ia:>7.2f}°"
          f"{ma/max(mv,1e-9):>7.2f}", "",
          f"  퍼짐(IQR) 비  액션/속도 = {ia/max(iv,1e-9):.2f}배", "",
          "  t 구간별 비  α_act/α_vel (칸마다의 비의 중앙값) vs 소각 예측 2.9(1−t)",
          f"  {'t':<11}{'표본':>8}{'비 중앙값':>11}{'2.9(1−t)':>11}{'Spearman':>11}"]
    _r = aa / np.maximum(av, 1e-9)
    _e = np.linspace(0, 1, 11)
    for _lo, _hi in zip(_e[:-1], _e[1:]):
        _m = (tt >= _lo) & (tt < _hi)
        if not _m.any():
            continue
        _c = 0.5 * (_lo + _hi)
        _sp = np.corrcoef(np.argsort(np.argsort(av[_m])),
                          np.argsort(np.argsort(aa[_m])))[0, 1]
        L.append(f"  {f'{_lo:.1f}~{_hi:.1f}':<11}{int(_m.sum()):>8}"
                 f"{np.median(_r[_m]):>11.2f}{2.9*(1-_c):>11.2f}{_sp:>11.3f}")
    _spa = np.corrcoef(np.argsort(np.argsort(av)), np.argsort(np.argsort(aa)))[0, 1]
    L += [f"  {'전체':<11}{len(av):>8}{np.median(_r):>11.2f}{'-':>11}{_spa:>11.3f}",
          "    ★ 전체 상관은 0.17 로 낮지만 **t 를 고정하면 0.66~0.70** 이다.",
          "      (b) 의 반비례처럼 보이는 모양은 상관이 없어서가 아니라,",
          "      t 가 0->1 로 쓸고 지나가며 점을 부채살 위로 옮기기 때문이다.",
          f"  η² (태스크간분산/전체분산)   속도 {eta_sq(av, jj):.4f}   "
          f"액션 {eta_sq(aa, jj):.4f}",
          "    η² 는 '각 칸에서 태스크를 줄 세울 수 있는가' 다. DEA 의 w_j 가 쓰는",
          "    신호가 이것이고, 작으면 순위가 잡음이라 재배분이 해롭다.", ""]
    txt = "\n".join(L)
    (png.with_suffix(".txt")).write_text(txt)
    print(txt)

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    C_V, C_A = "#4C78A8", "#F58518"       # 속도, 액션
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = GridSpec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.30,
                  left=0.055, right=0.985, top=0.86, bottom=0.155)

    # (a) 태스크별 등맞댄 half-violin
    ax = fig.add_subplot(gs[0])
    us = np.unique(jj)
    FLOOR = 1e-3                      # 0.001° 미만은 이 값으로. log 가 −inf 가 되는 것만 막는다
    lv, la = np.log10(np.maximum(av, FLOOR)), np.log10(np.maximum(aa, FLOOR))
    for i, u in enumerate(us):
        m = jj == u
        for x, col, side in ((lv[m], C_V, -1), (la[m], C_A, +1)):
            vp = ax.violinplot([x], positions=[i], widths=1.0, showextrema=False)
            for b in vp["bodies"]:
                pts = b.get_paths()[0].vertices
                pts[:, 0] = np.clip(pts[:, 0], i, np.inf) if side > 0 \
                    else np.clip(pts[:, 0], -np.inf, i)
                b.set_facecolor(col); b.set_alpha(0.75); b.set_edgecolor("none")
    for x, col, lab in ((lv, C_V, "velocity"), (la, C_A, "action")):
        med = [np.median(x[jj == u]) for u in us]
        ax.plot(range(len(us)), med, "-o", color=col, ms=3.4, lw=1.5,
                mec="white", mew=0.7, label=lab, zorder=5)
    # 눈금은 실제 각도로 되돌려 단다
    ticks = np.array([0.01, 0.1, 1.0, 10.0])
    ax.set_yticks(np.log10(ticks))
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.set_yticks(np.log10(np.concatenate([k * ticks for k in range(2, 10)])), minor=True)
    ax.set_ylim(np.log10(0.012), np.log10(45))
    # ★ 분포가 꼬리가 길다(전체 최대 37°, 중앙값 0.6°). 그대로 두면 바이올린이
    #   바닥에 눌려 아무것도 안 보인다. 조밀한 구간만 보이게 자르고 그 사실을 적는다.
    ax.set_xticks(range(len(us))); ax.set_xticklabels([str(int(u)) for u in us])
    ax.set_xlabel("past task $j$  (libero_goal)"); ax.set_ylabel(r"angle $\alpha$  [deg]")
    ax.set_title("(a) per-task student-teacher angle", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) 관계 — 대각선 위/아래 + t 색
    ax = fig.add_subplot(gs[1])
    n = len(av); idx = np.random.default_rng(0).choice(n, min(n, 14000), replace=False)
    sc = ax.scatter(av[idx], aa[idx], c=tt[idx], s=2.6, alpha=0.5,
                    cmap="viridis", linewidths=0, rasterized=True)
    hi = float(max(np.percentile(av, 99.5), np.percentile(aa, 99.5)))
    # 소각 근사에서  α_act/α_vel ≈ (1−t)·‖v_T‖/‖a_T‖ ≈ 2.9(1−t)  — 기울기가
    # **t 로만** 정해지는 원점 직선족이다. 실측 비가 10 개 t 구간 전부에서
    # 이 값과 맞는다(2.78/2.47/.../0.12 vs 2.75/2.46/.../0.15).
    K = 2.9
    for tv in (0.1, 0.4, 0.655, 0.9):
        sl = K * (1 - tv)
        xe = min(hi, hi / max(sl, 1e-9))
        ax.plot([0, xe], [0, sl * xe], "-", color="0.30", lw=0.9, alpha=0.75, zorder=4)
        # 라벨은 선 **안쪽** 에 둔다 (축 밖으로 나가면 제목을 가린다).
        fr = 0.62
        ax.annotate(f"$t$={tv:g}", xy=(fr * xe, fr * sl * xe), xytext=(5, -5),
                    textcoords="offset points", fontsize=8, color="0.25", ha="left",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72))
    ax.plot([0, hi], [0, hi], "--", color="0.55", lw=1.0, zorder=3)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel(r"$\alpha_{\rm vel}$  [deg]"); ax.set_ylabel(r"$\alpha_{\rm act}$  [deg]")
    ax.set_title("(b) both angles at the same query point", fontsize=11, loc="left")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046); cb.set_label("t", fontsize=9)
    cb.outline.set_visible(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (c) t 의존성 — 중앙값 + IQR
    ax = fig.add_subplot(gs[2])
    edges = np.linspace(0, 1, 21); ctr = 0.5 * (edges[1:] + edges[:-1])
    # 계단 좌표: e0,e1,e1,e2,e2,...  -> 구간 [0,1] 을 빈틈없이 덮는다
    xs = np.repeat(edges, 2)[1:-1]
    for x, col, lab in ((av, C_V, "velocity"), (aa, C_A, "action")):
        q = np.array([[np.percentile(x[(tt >= lo) & (tt < hi2)], p) for p in (25, 50, 75)]
                      if ((tt >= lo) & (tt < hi2)).any() else [np.nan] * 3
                      for lo, hi2 in zip(edges[:-1], edges[1:])])
        ax.fill_between(xs, np.repeat(q[:, 0], 2), np.repeat(q[:, 2], 2),
                        color=col, alpha=0.20, lw=0)
        ax.plot(xs, np.repeat(q[:, 1], 2), "-", color=col, lw=1.6, label=lab)
        ax.plot(ctr, q[:, 1], "o", color=col, ms=3.4, mec="white", mew=0.7)
    # 두 중앙값 곡선이 만나는 t 를 선형보간으로 찾아 표시한다. 이 교차가
    # 이 그림의 요점이다 — 어느 파라미터화가 '민감한지' 가 t 에 따라 뒤집힌다.
    mv_ = np.array([np.median(av[(tt >= lo) & (tt < hi2)]) if ((tt >= lo) & (tt < hi2)).any()
                    else np.nan for lo, hi2 in zip(edges[:-1], edges[1:])])
    ma_ = np.array([np.median(aa[(tt >= lo) & (tt < hi2)]) if ((tt >= lo) & (tt < hi2)).any()
                    else np.nan for lo, hi2 in zip(edges[:-1], edges[1:])])
    dd = ma_ - mv_
    xs = np.where(np.sign(dd[1:]) != np.sign(dd[:-1]))[0]
    if len(xs):
        i0 = xs[0]
        w = dd[i0] / (dd[i0] - dd[i0 + 1])
        tx = ctr[i0] + w * (ctr[i0 + 1] - ctr[i0])
        ax.axvline(tx, color="0.45", ls=":", lw=1.1, zorder=1)
        ax.annotate(f"crossover\n$t\\approx${tx:.2f}", xy=(tx, ax.get_ylim()[1] * 0.92),
                    xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                    color="0.35", va="top")
    ax.set_xlabel("flow time $t$"); ax.set_ylabel(r"angle $\alpha$  [deg]")
    ax.set_title("(c) dependence on $t$  (median, IQR band)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9); ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Directional term of the anchor loss: velocity vs. action parameterization"
                 f"   (stage {stage}, {d['suite']}, MSE-anchored run)",
                 fontsize=12.5, x=0.055, ha="left")
    fig.savefig(png, dpi=200)
    fig.savefig(png.with_suffix(".pdf"))
    print(f"\nsaved -> {png}\n         {png.with_suffix('.pdf')}"
          f"\n         {png.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
