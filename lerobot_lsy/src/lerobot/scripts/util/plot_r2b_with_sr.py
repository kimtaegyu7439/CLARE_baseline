#!/usr/bin/env python
"""R2-B(모드 센서스) 두 패널 아래에 SR 패널을 붙여 한 장으로 그린다.

왜 필요한가
    R2-B는 "중심이 움직였나 / 경계가 움직였나"만 보여준다. 그 움직임이 실제로 성공률과
    같이 가는지는 같은 x축(CL stage k) 위에 SR을 놓고 봐야 읽힌다.

어떤 키를 쓰나
    JSONL 의 center_shift / assign_change 는 200개 관측 **전체**의 median 이라 쓰면 안 된다.
    78.5%가 단봉이고 단봉 관측은 assign_change 가 정의상 0이므로 전체 median 이 구조적으로
    0이 된다(실제로 모든 행이 0.0 이다). R2.py 의 그림은
        (a) center_shift_rel      = 중심 이동 / 모드 간 거리, 다봉 관측만
        (b) assign_change_multi   = 다봉 관측만의 median
    을 쓴다. 여기서도 같은 키를 써야 두 그림이 같은 것을 말한다.

SR 출처
    R1의 r1_results.jsonl (probe_task=0 롤아웃의 에피소드별 success). 다섯 팔 × 네
    스테이지가 모두 채워져 있는 유일한 소스라 이걸 쓴다. R2의 checkpoint_sticky에도
    sr이 있지만 실험 A를 돌린 타깃에만 있어 구멍이 난다.

    ★ R1은 max_steps=300, R2는 500으로 돌았다. 각각 내부적으로는 일관되지만 두 실험의
      SR을 같은 축에서 비교할 때는 이 차이를 감안해야 한다(R2의 fresh_check가 재는 gap이
      seq@0에서 0.033이었다). 여기서는 SR을 전부 R1에서 가져오므로 패널 내부는 일관된다.

사용 예
    python plot_r2b_with_sr.py \
        --r2_dir=outputs/R2/libero_spatial_seed42_probe0 \
        --r1_dir=outputs/R1/libero_spatial_seed42_probe0
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# R2.py의 그림과 같은 색·이름을 쓴다. 다르면 두 그림을 나란히 놓고 읽을 수 없다.
STYLE = {
    "seq": ("Seq (fine-tune)", "tab:blue"),
    "ewc": ("EWC", "tab:orange"),
    "frozen": ("Frozen (lambda=inf)", "tab:purple"),
    "er": ("ER", "tab:green"),
    "packnet": ("PackNet (task ID known)", "tab:brown"),
}
ORDER = ["seq", "ewc", "frozen", "er", "packnet"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def census_table(rows: list[dict]) -> dict:
    """(method, stage) -> census 행. append-only 이므로 뒤쪽(최신)이 이긴다."""
    out = {}
    for r in rows:
        if r.get("kind") == "census":
            out[(r["method"], r["stage"])] = r
    return out


def sr_table(rows: list[dict]) -> dict:
    """(method, stage) -> (SR%, n). R1의 에피소드별 success를 집계한다.

    ★ r1_results.jsonl 은 append-only 다. 같은 팔을 다시 돌리면 옛 행이 그대로 남아
      그냥 평균 내면 같은 롤아웃을 두 번 세게 된다(실측: seq/ewc/frozen 이 60개,
      er/packnet 이 30개로 잡혔다). (method, stage, rollout) 마다 뒤쪽(최신)만 남긴다.
    """
    uniq: dict = {}
    for r in rows:
        if r.get("success") is None:
            continue
        uniq[(r["method"], r["stage"], r.get("rollout"))] = bool(r["success"])
    acc: dict = {}
    for (m, s, _), ok in uniq.items():
        acc.setdefault((m, s), []).append(ok)
    return {k: (100.0 * np.mean(v), len(v)) for k, v in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r2_dir", required=True)
    p.add_argument("--r1_dir", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--packnet_task_id_known", default="true",
                   help="PackNet 을 본래 방식(테스트 시 task ID 로 마스크 선택)으로 표시한다. "
                        "false 면 저장된 그대로(마스크 미적용) 찍는다.")
    args = p.parse_args()

    r2_dir, r1_dir = Path(args.r2_dir), Path(args.r1_dir)
    cen = census_table(load_jsonl(r2_dir / "r2_results.jsonl"))
    srs = sr_table(load_jsonl(r1_dir / "r1_results.jsonl"))
    out = Path(args.out) if args.out else r2_dir / "R2_B_census_with_SR.png"

    # ── PackNet: 본래 방식대로 task ID 로 마스크를 골라 씌운 값 ────────────────
    # 지금 파이프라인(R1/R2/make_policy)은 mask.safetensors 를 아예 읽지 않아서, 저장된
    # 값은 "task 3 까지 학습한 통짜 모델을 task 0 에 던진 것"이다. PackNet 은 테스트 시
    # task ID 를 알고 그 태스크의 마스크를 씌우는 방법이므로 그건 PackNet 이 아니다.
    #
    # 검증 두 가지로 정답을 직접 유도할 수 있어 롤아웃을 다시 돌리지 않는다.
    #   (1) task0 마스크(mask==1 만 남김)를 씌운 모델이 네 스테이지에서 비트 단위로 동일
    #       (다른 원소 0/193,623,045, max|Δ|=0). bias/LayerNorm 은 task 0 부터 동결이라 함께 보존.
    #   (2) task_0 체크포인트는 mask!=1 인 32,851,238 자리가 이미 전부 0이다
    #       (가지치기가 0으로 만들고 restore_protected 가 그대로 유지).
    #   -> 마스크를 씌운 모델 = stage 1 체크포인트 그 자체. 모든 스테이지에서 같은 함수다.
    # 상수인 곡선을 다시 측정하면 롤아웃 비결정성(실측 ~3pp)만 얹히므로 stage 1 의
    # 측정값을 그대로 쓴다. 유도이지 추정이 아니다.
    if str(args.packnet_task_id_known).lower() in ("1", "true", "yes"):
        base_c, base_s = cen.get(("packnet", 0)), srs.get(("packnet", 0))
        for s_ in sorted({k[1] for k in list(cen) + list(srs)}):
            if base_c is not None:
                cen[("packnet", s_)] = base_c
            if base_s is not None:
                srs[("packnet", s_)] = base_s

    methods = [m for m in ORDER if any(k[0] == m for k in cen) or any(k[0] == m for k in srs)]
    stages = sorted({k[1] for k in list(cen) + list(srs)})
    xs = [s + 1 for s in stages]          # CL stage k = tasks 0..k-1 learned

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, :])

    def series(ax, key, q25, q75):
        for m in methods:
            pts = [(s + 1, cen[(m, s)]) for s in stages if (m, s) in cen
                   and cen[(m, s)].get(key) is not None]
            if not pts:
                continue
            x = [a for a, _ in pts]
            y = [b[key] for _, b in pts]
            lo = [b.get(q25, np.nan) for _, b in pts]
            hi = [b.get(q75, np.nan) for _, b in pts]
            lab, col = STYLE[m]
            ax.plot(x, y, "-o", ms=5, color=col, label=lab)
            if not all(np.isnan(lo)):
                ax.fill_between(x, lo, hi, color=col, alpha=0.18)

    series(ax_a, "center_shift_rel", "center_shift_rel_q25", "center_shift_rel_q75")
    ax_a.axhline(1.0, color="crimson", ls="--", lw=1.2)
    ax_a.text(0.02, 1.02, "centres moved further than the gap between modes",
              color="crimson", fontsize=8, transform=ax_a.get_yaxis_transform())
    ax_a.set(xlabel="CL stage k (tasks 0..k-1 learned)",
             ylabel="center_shift / distance between modes",
             title="(a) did the mode centres stay put?")

    series(ax_b, "assign_change_multi", "assign_change_multi_q25", "assign_change_multi_q75")
    ax_b.axhline(0.5, color="crimson", ls="--", lw=1.2)
    ax_b.text(0.02, 0.51, "chance (2 modes: assignments fully scrambled)",
              color="crimson", fontsize=8, transform=ax_b.get_yaxis_transform())
    ax_b.set(xlabel="CL stage k (tasks 0..k-1 learned)",
             ylabel="assign_change: fraction of $a_0$ reassigned",
             title="(b) did the boundaries move?")

    for m in methods:
        pts = [(s + 1, srs[(m, s)][0]) for s in stages if (m, s) in srs]
        if not pts:
            continue
        lab, col = STYLE[m]
        ax_s.plot(*zip(*pts), "-o", ms=6, color=col, label=lab)
    n_ep = {v[1] for v in srs.values()}
    ax_s.set(xlabel="CL stage k (tasks 0..k-1 learned)", ylabel="SR on task 0 (%)",
             title=f"(c) does it still succeed?  —  R1 rollouts, "
                   f"n={sorted(n_ep)[0] if n_ep else '?'} episodes/point, max_steps=300")
    ax_s.set_ylim(-4, 104)

    for ax in (ax_a, ax_b, ax_s):
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
    ax_s.legend(fontsize=9, ncol=len(methods), loc="lower left")

    fig.suptitle("R2-B + success rate on the same CL-stage axis", fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "PackNet is shown as its method intends: the task-0 mask is applied at test time. "
             "That model is bit-identical across stages, so its curve is flat by construction.",
             ha="center", fontsize=8.5, style="italic", color="0.3")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved figure -> {out}")

    csv = out.with_suffix(".csv")
    lines = ["method,cl_stage,center_shift_rel,assign_change_multi,n_multimodal,sr_percent,n_episodes"]
    for m in methods:
        for s in stages:
            c = cen.get((m, s), {})
            sr = srs.get((m, s))
            if not c and not sr:
                continue
            lines.append(",".join([
                m, str(s + 1),
                "" if c.get("center_shift_rel") is None else f"{c['center_shift_rel']:.6g}",
                "" if c.get("assign_change_multi") is None else f"{c['assign_change_multi']:.6g}",
                "" if c.get("n_multimodal") is None else str(c["n_multimodal"]),
                "" if sr is None else f"{sr[0]:.4g}",
                "" if sr is None else str(sr[1]),
            ]))
    csv.write_text("\n".join(lines) + "\n")
    print(f"saved table  -> {csv}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
