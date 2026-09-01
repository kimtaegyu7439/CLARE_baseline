#!/usr/bin/env python
"""LIBERO-40 SR 행렬을 Fig.4 스타일 하삼각 히트맵으로 그린다.

입력은 eval_common.sh가 뱉는 `<bench>_SR.txt` (탭 구분, 1행=헤더, 1열=스테이지)
또는 같은 디렉터리의 `summary.csv`. 둘 다 행=체크포인트(스테이지), 열=태스크다.

    python tools/plot_sr_matrix.py \
        --panel CLARE outputs/CLARE_eval/libero_40/seed42/libero_40_SR.txt \
        --panel ER    outputs/ER_eval/libero_40/seed42/libero_40_SR.txt \
        -o figs/libero40_sr.pdf

--panel을 하나만 주면 한 칸짜리 그림이 나온다(ER 평가가 끝나기 전에 CLARE만 보기).

지표는 LIBERO 벤치마크(Liu et al., 2023) 정의를 그대로 쓴다. R[k][t]는 스테이지 k
체크포인트를 태스크 t에서 잰 성공률이고 t<=k에서만 정의된다.

    FWT_t = R[t][t]                                      갓 배운 직후의 성능
    NBT_t = mean_{k>t} (FWT_t - R[k][t])                 이후에 얼마나 잃었나 (+면 망각)
    AUC_t = mean_{k>=t} R[k][t]                          그 태스크가 시퀀스 내내 낸 평균

세 지표 모두 태스크에 대해 평균한다. 마지막 태스크는 뒤따르는 스테이지가 없어
NBT가 정의되지 않으므로 NBT 평균에서만 제외한다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ── 입력 읽기 ────────────────────────────────────────────────────────────────
def load_matrix(path: Path, n_tasks: int | None = None) -> np.ndarray:
    """SR 행렬을 (N,N) float 배열로 읽는다. 빈 칸/미평가는 NaN.

    path가 디렉터리면 stage*/task*/eval_info.json을 직접 읽는다. 평가가 도는 중에도
    쓸 수 있고, 워커 여섯 개 중 누가 마지막에 summary.csv를 쓰는지에 의존하지 않는다.
    """
    if path.is_dir():
        import json

        n = n_tasks or 40
        m = np.full((n, n), np.nan)
        for p in path.glob("stage*/task*/eval_info.json"):
            k = int(p.parent.parent.name.removeprefix("stage"))
            t = int(p.parent.name.removeprefix("task"))
            try:
                v = json.loads(p.read_text())["aggregated"].get("pc_success")
            except Exception as e:                   # 중간에 죽어 잘린 파일은 건너뛴다
                print(f"  WARN 읽기 실패 {p}: {e}")
                continue
            if v is not None and k < n and t < n:
                m[k, t] = float(v)
        return m

    if path.suffix == ".csv":
        rows = list(csv.reader(path.open()))
        header, body = rows[0], [r for r in rows[1:] if r and r[0].startswith("stage")]
        # summary.csv에는 맨 뒤에 avg_seen 열이 붙는다.
        n_cols = sum(1 for c in header if c.startswith("task"))
        cells = [[c.strip() for c in r[1 : 1 + n_cols]] for r in body]
    else:
        # SR.txt: 헤더 1행 + 스테이지마다 "인덱스 값 값 …" (하삼각이라 줄마다 길이가 다르다)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        body = [ln.split("\t")[1:] for ln in lines[1:]]
        n_cols = len(lines[0].split("\t")) - 1
        cells = [[c.strip() for c in r] for r in body]

    n = n_tasks or n_cols
    m = np.full((n, n), np.nan)
    for k, row in enumerate(cells[:n]):
        for t, v in enumerate(row[:n]):
            if v not in ("", "-", "nan"):
                m[k, t] = float(v)
    return m


# ── 지표 ─────────────────────────────────────────────────────────────────────
def metrics(m: np.ndarray) -> dict[str, float]:
    n = m.shape[0]
    fwt, nbt, auc = [], [], []
    for t in range(n):
        col = m[t:, t]                       # 태스크 t를 배운 시점부터 끝까지
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        fwt.append(col[0])
        auc.append(col.mean())
        if col.size > 1:                     # 마지막 태스크는 NBT가 정의되지 않는다
            nbt.append(col[0] - col[1:].mean())
    return {
        "AUC": float(np.mean(auc)),
        "FWT": float(np.mean(fwt)),
        "NBT": float(np.mean(nbt)),
    }


# ── 그리기 ───────────────────────────────────────────────────────────────────
def draw(panels: list[tuple[str, np.ndarray]], out: Path, *, cmap="RdYlGn", dpi=300):
    n = panels[0][1].shape[0]
    tick = [0] + list(range(4, n, 5))        # 1, 5, 10, … (0-based 인덱스)
    ticklab = [i + 1 for i in tick]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.4 * len(panels) + 0.4, 3.9), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    norm = Normalize(0, 100)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.82")                       # 아직 배우지 않은 상삼각 = 회색

    for ax, (name, m) in zip(axes, panels):
        ax.imshow(np.ma.masked_invalid(m), cmap=cm, norm=norm,
                  interpolation="nearest", aspect="equal")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Task", fontsize=10)
        ax.set_xticks(tick, ticklab, fontsize=8)
        ax.set_yticks(tick, ticklab, fontsize=8)
        ax.tick_params(length=2)
        if ax is axes[0]:
            ax.set_ylabel("Stage", fontsize=10)

        s = metrics(m)
        ax.text(
            0.975, 0.975,
            f"AUC = {s['AUC']:.1f}\nFWT = {s['FWT']:.1f}\nNBT = {s['NBT']:.1f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="0.3", lw=0.6),
        )

    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cm), ax=axes.tolist(),
        orientation="horizontal", fraction=0.055, pad=0.02, aspect=40,
    )
    cb.set_label("Success rate [%]", fontsize=10)
    cb.set_ticks(range(0, 101, 20))
    cb.ax.tick_params(labelsize=8, length=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    for alt in {out.with_suffix(".png"), out.with_suffix(".pdf")} - {out}:
        fig.savefig(alt, dpi=dpi, bbox_inches="tight")
    print(f"[plot] saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", nargs=2, action="append", metavar=("NAME", "PATH"),
                    required=True, help="패널 제목과 SR 파일. 여러 번 줄 수 있다.")
    ap.add_argument("-n", "--n-tasks", type=int, default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("figs/libero40_sr.pdf"))
    ap.add_argument("--cmap", default="RdYlGn")
    args = ap.parse_args()

    panels = []
    for name, path in args.panel:
        m = load_matrix(Path(path), args.n_tasks)
        s = metrics(m)
        filled = int(np.isfinite(m).sum())
        n = m.shape[0]
        print(f"[{name:6s}] {filled}/{n * (n + 1) // 2} cells  "
              f"AUC={s['AUC']:.1f}  FWT={s['FWT']:.1f}  NBT={s['NBT']:.1f}")
        panels.append((name, m))

    draw(panels, args.out, cmap=args.cmap)


if __name__ == "__main__":
    main()
