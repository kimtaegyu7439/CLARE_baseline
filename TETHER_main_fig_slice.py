#!/usr/bin/env python
"""TETHER main figure — 파지 시도 위치. 두 모델을 한 패널에 겹친 1x3 (main).

묻는 것 하나
    "task 0 조건을 받았는데, 팔이 **어디에서 파지를 시도하는가**?"

    분포는 한 번만 그리고 그 위에 두 모델을 함께 올린다. 모델마다 패널을 따로 두면
    같은 expert 분포가 반복되고, 위 패널의 점 위치를 기억한 채 아래로 시선을 옮겨야
    한다. 겹치면 그 비교가 한 패널 안에서 끝난다.

        연한 파랑 영역   task 0 시연 20개의 파지 위치 (90%)
        연한 주황 영역   최신 task 시연 20개의 파지 위치 (90%)
        짙은 회색 ×      Seq-FT 의 파지 시도      ← ×는 **모델 표시**다. 실패 표시가 아니다
        보라 ●           TETHER 의 파지 시도

파지 시도의 정의 — 모든 rollout 에 같은 규칙
    궤적의 **z 최소점**. 시연·rollout·성공·실패에 예외 없이 같은 규칙을 쓴다.
    성공한 것만 표시하면 망각이 심한 rollout 일수록 그림에서 사라져, "최신 task 쪽으로
    잘못 간다"를 보여줘야 할 패널이 도리어 비어 버린다.
    시도 자체가 없었던 rollout(테이블 높이까지 내려간 적 없는 것)은 임의 위치를 넣지
    않고 개수만 적는다. 판정 기준은 시연 파지 z 의 최댓값이고 이 데이터에서는 0 개다.

    gripper 닫힘을 쓰지 않는 이유: rollout 캐시에 EE xyz 만 있고 gripper 신호가 없다.
    두 곡선에 다른 규칙을 쓰면 비교가 성립하지 않는다.

두 배치
    overlay (기본)  1x3. 두 모델을 한 패널에. main 용.
    perrow          모델을 행으로 나누고 task 성공/실패를 ● / ○ 로 구분. appendix 용.
                    --include_control 로 control 열(조건 == 최신 task)까지 넣는다.

숫자는 그림 밖에
    패널 아래에는 task SR 표만 둔다. σ 거리와 영역 포함 개수는 panel_slice_stats.txt 에
    남긴다 — 그림이 답해야 할 질문은 "어디에서 시도하는가" 하나다.

    python TETHER_main_fig_slice.py                                  # main 1x3
    python TETHER_main_fig_slice.py --layout perrow --include_control  # appendix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

C_T0, C_LAT = "#0072B2", "#D55E00"          # expert 분포 (Okabe-Ito)
C_SEQ, C_TET = "#4D4D4D", "#6A3D9A"         # 모델: 짙은 회색 / 보라
C_CTRL = "#9A9A9A"
MASS = 0.90                                  # 그리는 분포 등수위선 (질량)


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="results/TETHER_main_fig_spatial")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--layout", choices=["overlay", "perrow"], default="overlay")
    p.add_argument("--stages", default=None,
                   help="열로 쓸 단계. 기본 1,2,3 (--include_control 이면 0,1,2,3)")
    p.add_argument("--include_control", action="store_true")
    p.add_argument("--row_names", default="Seq-FT,TETHER")
    p.add_argument("--eval_task", type=int, default=0)
    p.add_argument("--n_ref_demos", type=int, default=20)
    p.add_argument("--out", default=None)
    return p.parse_args()


def grasp_xy(tr):
    return tr[int(np.argmin(tr[:, 2])), :2]


def ellipse_of(P):
    mu, cov = P.mean(0), np.cov(P.T)
    val, vec = np.linalg.eigh(cov)
    ang = float(np.degrees(np.arctan2(*vec[:, 1][::-1])))
    sizes = {m: tuple(2 * np.sqrt(chi2.ppf(m, 2)) * np.sqrt(np.maximum(val, 1e-12)))
             for m in (0.50, MASS)}
    return mu, cov, ang, sizes


def ellipse_extent(mu, ang, wh):
    w, h = wh
    t = np.radians(ang)
    dx, dy = np.hypot(w / 2 * np.cos(t), h / 2 * np.sin(t)), np.hypot(w / 2 * np.sin(t), h / 2 * np.cos(t))
    return np.array([[mu[0] - dx, mu[1] - dy], [mu[0] + dx, mu[1] + dy]])


def maha(P, mu, cov):
    d = P - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d))


def inside(P, mu, cov, mass=MASS):
    """등수위선 안에 든 점의 불리언. maha² <= chi2 임계값."""
    return maha(P, mu, cov) ** 2 <= chi2.ppf(mass, 2)


def main():
    a = parse()
    cache = Path(a.cache)
    data = np.load(cache / "rollouts.npz", allow_pickle=True)["data"].item()
    stages_str = a.stages or ("0,1,2,3" if a.include_control else "1,2,3")
    stages = [int(s) for s in stages_str.split(",")]
    rows = [s.strip() for s in a.row_names.split(",")]
    default_name = ("TETHER_fig_slice_appendix" if a.layout == "perrow"
                    else "TETHER_main_fig_slice")
    out = Path(a.out) if a.out else cache / default_name

    from TETHER_main_fig import expert_traj, seq_spec
    spec = seq_spec(a.suite, max(stages + [a.eval_task]) + 1)

    print("expert 로딩...")
    ell, z_demo = {}, []
    for t in sorted(set(stages) | {a.eval_task}):
        E = [np.asarray(x) for x in expert_traj(spec["ds_root"][t], None, a.n_ref_demos, None)]
        ell[t] = ellipse_of(np.stack([grasp_xy(x) for x in E]))
        z_demo += [float(x[:, 2].min()) for x in E]
        print(f"   task {t}: 시연 {len(E)}개")
    z_thr = float(np.max(z_demo))
    print(f"   파지 시도 판정 z <= {z_thr:.4f} (시연 파지 z 의 최댓값)")

    panels, lines = {}, []
    for name in rows:
        for k in stages:
            d = data[f"{name}|{k}"]
            T = [np.asarray(x) for x in d["traj"]]
            ok = np.asarray(d["success"], bool)
            tried = np.array([float(x[:, 2].min()) <= z_thr for x in T])
            P = np.stack([grasp_xy(x) for x in T])
            pan = dict(sr=d["sr"], n=len(T), ok=ok, tried=tried, pts=P[tried],
                       hit=P[ok & tried], miss=P[~ok & tried], none=int((~tried).sum()))
            panels[(name, k)] = pan
            if len(pan["pts"]):
                mu0, cov0 = ell[a.eval_task][0], ell[a.eval_task][1]
                muL, covL = ell[k][0], ell[k][1]
                in0, inL = inside(pan["pts"], mu0, cov0), inside(pan["pts"], muL, covL)
                lines.append(
                    f"{name:8s} col{k}  attempts={len(pan['pts'])}/{pan['n']}  "
                    f"task_success={int(ok.sum())}  "
                    f"in_task0_{int(MASS*100)}%={int(in0.sum())}  "
                    f"in_latest_{int(MASS*100)}%={int(inL.sum())}  "
                    f"outside_both={int((~in0 & ~inL).sum())}  "
                    f"d_task0={np.median(maha(pan['pts'], mu0, cov0)):.2f}  "
                    f"d_latest={np.median(maha(pan['pts'], muL, covL)):.2f}")

    # 축 — 전 패널 공통. 분포(90%)와 그려지는 모든 점을 담고 여백은 작게.
    box = [ellipse_extent(ell[t][0], ell[t][2], ell[t][3][MASS]) for t in ell]
    box += [np.stack([p["pts"].min(0), p["pts"].max(0)]) for p in panels.values()
            if len(p["pts"])]
    B = np.concatenate(box)
    pad = 0.055 * max(np.ptp(B[:, 0]), np.ptp(B[:, 1]))
    xl = (B[:, 0].min() - pad, B[:, 0].max() + pad)
    yl = (B[:, 1].min() - pad, B[:, 1].max() + pad)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
                                "axes.edgecolor": "0.72", "axes.linewidth": .8,
                                "xtick.color": "0.4", "ytick.color": "0.4"})
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import MaxNLocator

    def draw_clouds(ax, k, ctrl):
        """expert 분포 — 한 단계(90%)만. 연한 채움 + 가는 경계선."""
        groups = ([(ell[a.eval_task], C_CTRL)] if ctrl
                  else [(ell[k], C_LAT), (ell[a.eval_task], C_T0)])
        for (mu, _cov, ang, sz), col in groups:
            ax.add_patch(Ellipse(mu, *sz[MASS], angle=ang, facecolor=col, alpha=.13,
                                 edgecolor="none", zorder=2))
            ax.add_patch(Ellipse(mu, *sz[MASS], angle=ang, facecolor="none",
                                 edgecolor=col, lw=.8, alpha=.55, zorder=3))

    def style(ax, first_col):
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=9, length=3, width=.8)
        if not first_col:
            ax.set_yticklabels([])

    ncol = len(stages)
    if a.layout == "overlay":
        # ── 1x3: 두 모델을 한 패널에 ─────────────────────────────────────────
        pw = 3.7
        ph = pw * float(np.clip((yl[1] - yl[0]) / (xl[1] - xl[0]), .5, 1.5))
        fw, fh = pw * ncol + 1.35, ph + 2.45
        fig, axes = plt.subplots(1, ncol, figsize=(fw, fh), squeeze=False)
        fig.subplots_adjust(left=1.28 / fw, right=1 - .18 / fw, top=1 - .62 / fh,
                            bottom=1.72 / fh, wspace=.10)
        for c, k in enumerate(stages):
            ax = axes[0][c]
            draw_clouds(ax, k, k == a.eval_task)
            ps, pt = panels[(rows[0], k)]["pts"], panels[(rows[1], k)]["pts"]
            if len(ps):
                ax.scatter(ps[:, 0], ps[:, 1], s=78, marker="x", c=C_SEQ,
                           linewidths=1.7, zorder=6)
            if len(pt):
                ax.scatter(pt[:, 0], pt[:, 1], s=62, marker="o", c=C_TET,
                           edgecolor="white", linewidth=.8, zorder=7)
            ax.set_title(("After task 0  (control)" if k == a.eval_task
                          else f"After tasks 0–{k}"), fontsize=12.5, pad=9)
            style(ax, c == 0)

        fig.canvas.draw()
        bb0 = axes[0][0].get_position()
        # 패널 아래: task SR 표 (문장 반복 대신 표 한 덩어리)
        y_hdr = bb0.y0 - .085
        dy = .052
        fig.text(1.20 / fw, y_hdr, "Task SR", ha="right", va="center", fontsize=10.5,
                 color="0.25")
        for i, name in enumerate(rows):
            fig.text(1.20 / fw, y_hdr - (i + 1) * dy, name, ha="right", va="center",
                     fontsize=10.5, color=(C_SEQ if i == 0 else C_TET))
        for c, k in enumerate(stages):
            bb = axes[0][c].get_position()
            xc = bb.x0 + bb.width / 2
            # 열 제목은 쓰지 않는다 — 값이 각 패널 바로 아래 정렬돼 있고,
            # 패널 제목(After tasks 0-k)이 바로 위에 있어 중복이다.
            for i, name in enumerate(rows):
                pan = panels[(name, k)]
                fig.text(xc, y_hdr - (i + 1) * dy, f"{pan['sr']:.0f}%", ha="center",
                         va="center", fontsize=11,
                         color=(C_SEQ if i == 0 else C_TET))
        miss = sum(p["none"] for p in panels.values())
        if miss:
            fig.text(.5, y_hdr - 3 * dy, f"no grasp attempt detected: {miss} rollouts",
                     ha="center", va="center", fontsize=9.5, color="0.45")

        fig.text(.5, bb0.y0 - .045, "EE x (m)", ha="center", va="center", fontsize=11)
        fig.text(.52 / fw, bb0.y0 + bb0.height / 2, "EE y (m)", rotation=90,
                 ha="center", va="center", fontsize=11)
        h = [plt.Rectangle((0, 0), 1, 1, fc=C_T0, alpha=.30, ec=C_T0, lw=.8),
             plt.Rectangle((0, 0), 1, 1, fc=C_LAT, alpha=.30, ec=C_LAT, lw=.8),
             plt.Line2D([], [], color=C_SEQ, marker="x", ls="none", ms=9, mew=1.7),
             plt.Line2D([], [], color=C_TET, marker="o", ls="none", ms=8.5, mec="white")]
        lb = [f"Task 0 expert ({int(MASS*100)}%)", f"Latest-task expert ({int(MASS*100)}%)",
              "Seq-FT grasp attempt", "TETHER grasp attempt"]
        fig.legend(h, lb, loc="lower center", ncol=4, frameon=False, fontsize=11,
                   bbox_to_anchor=(.5, .012), handlelength=1.3, columnspacing=2.0,
                   handletextpad=.7)
        fig.suptitle("All policies evaluated on Task 0", fontsize=13.5, y=1 - .14 / fh)
    else:
        # ── 2xN: 모델을 행으로, task 성공/실패를 ● / ○ 로 (appendix) ────────
        pw = 3.5
        ph = pw * float(np.clip((yl[1] - yl[0]) / (xl[1] - xl[0]), .5, 1.5))
        fw, fh = pw * ncol + 1.45, ph * 2 + 2.25
        fig, axes = plt.subplots(2, ncol, figsize=(fw, fh), squeeze=False)
        fig.subplots_adjust(left=1.40 / fw, right=1 - .18 / fw, top=1 - .62 / fh,
                            bottom=1.55 / fh, wspace=.10, hspace=.30)
        for r, name in enumerate(rows):
            col_m = C_SEQ if r == 0 else C_TET
            for c, k in enumerate(stages):
                ax = axes[r][c]
                pan = panels[(name, k)]
                draw_clouds(ax, k, k == a.eval_task)
                if len(pan["miss"]):
                    ax.scatter(pan["miss"][:, 0], pan["miss"][:, 1], s=60,
                               facecolor="none", edgecolor=col_m, linewidth=1.3, zorder=6)
                if len(pan["hit"]):
                    ax.scatter(pan["hit"][:, 0], pan["hit"][:, 1], s=60, c=col_m,
                               edgecolor="white", linewidth=.9, zorder=7)
                if r == 0:
                    ax.set_title(("After task 0  (control)" if k == a.eval_task
                                  else f"After tasks 0–{k}"), fontsize=12, pad=9)
                    ax.set_xticklabels([])
                style(ax, c == 0)
        fig.canvas.draw()
        for r, name in enumerate(rows):
            for c, k in enumerate(stages):
                pan = panels[(name, k)]
                bb = axes[r][c].get_position()
                off = .030 if r == 0 else .052
                txt = f"Task SR: {pan['sr']:.0f}%  ({int(pan['ok'].sum())}/{pan['n']})"
                if pan["none"]:
                    txt += f"    No attempt: {pan['none']}/{pan['n']}"
                fig.text(bb.x0 + bb.width / 2, bb.y0 - off, txt, ha="center", va="top",
                         fontsize=10.5, color="0.2")
            bb = axes[r][0].get_position()
            fig.text(.20 / fw, bb.y0 + bb.height / 2, name, rotation=90, va="center",
                     ha="center", fontsize=13, color=(C_SEQ if r == 0 else C_TET))
        top_bb, bot_bb = axes[0][0].get_position(), axes[1][0].get_position()
        fig.text(.5, bot_bb.y0 - .095, "EE x (m)", ha="center", va="center", fontsize=11)
        fig.text(.62 / fw, (top_bb.y1 + bot_bb.y0) / 2, "EE y (m)", rotation=90,
                 ha="center", va="center", fontsize=11)
        h = [plt.Rectangle((0, 0), 1, 1, fc=C_T0, alpha=.30, ec=C_T0, lw=.8),
             plt.Rectangle((0, 0), 1, 1, fc=C_LAT, alpha=.30, ec=C_LAT, lw=.8),
             plt.Line2D([], [], color="0.2", marker="o", ls="none", ms=8.5, mec="white"),
             plt.Line2D([], [], color="0.2", marker="o", ls="none", ms=8.5, mfc="none",
                        mew=1.3)]
        lb = [f"Task 0 expert ({int(MASS*100)}%)", f"Latest-task expert ({int(MASS*100)}%)",
              "attempt — task succeeded", "attempt — task failed"]
        fig.legend(h, lb, loc="lower center", ncol=4, frameon=False, fontsize=11,
                   bbox_to_anchor=(.5, .012), handlelength=1.2, columnspacing=2.2,
                   handletextpad=.7)
        fig.suptitle("All policies evaluated on Task 0", fontsize=13.5, y=1 - .14 / fh)

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None)
        print("saved ->", f"{out}.{ext}")
    (cache / "panel_slice_stats.txt").write_text(
        "파지 시도 위치 — 정량 (그림에는 넣지 않는다)\n"
        f"시도 판정: 궤적 z 최소값 <= {z_thr:.4f} (시연 파지 z 의 최댓값)\n"
        f"in_* = 해당 expert 분포의 {int(MASS*100)}% 등수위선 안에 든 시도 수\n"
        "d_*   = 마할라노비스 거리 중앙값(σ). 성공·실패 시도를 모두 포함한다.\n\n"
        + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
