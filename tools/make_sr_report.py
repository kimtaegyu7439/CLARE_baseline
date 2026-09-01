#!/usr/bin/env python3
"""모든 CL 실험의 SR 행렬과 지표(ACC/BWT)를 하나의 txt로 모은다.

행 = 스테이지 k 체크포인트, 열 = 태스크 j, 값 = 성공률(%). 하삼각.
값의 출처는 두 가지이고 리포트에 그대로 표기한다:
  sweep    평가 스크립트가 남긴 eval_info.json (칸별 파일)
  trainlog 학습 스크립트가 스테이지 끝에서 돌린 평가를 로그에서 추출
둘 다 eval_peft.py:eval_policy_with_env_init 롤아웃이라 지표는 같은 뜻이다.
"""
import json, re, sys, glob, unicodedata
from pathlib import Path

ROOT = Path("/home/sa090180/clare")

# (방법, 스위트, 태스크수, 출처종류, 경로)
SOURCES = [
    ("ER",    "libero_10",      10, "sweep",    "outputs/ER_eval/libero_10/seed42"),
    ("ER",    "libero_goal",    10, "sweep",    "outputs/ER_eval/libero_goal/seed42"),
    ("ER",    "libero_spatial", 10, "sweep",    "outputs/ER_eval/libero_spatial/seed42"),
    ("ER",    "libero_object",  10, "sweep",    "outputs/ER_eval/libero_object/seed42"),
    ("ER",    "libero_40",      40, "sweep",    "outputs/ER_eval/libero_40/seed42"),
    ("CLARE", "libero_10",      10, "txt",      "outputs/libero_10/clare/libero_10_SR.txt"),
    ("CLARE", "libero_goal",    10, "txt",      "outputs/libero_goal/clare/goal_SR.txt"),
    ("CLARE", "libero_spatial", 10, "txt",      "outputs/libero_spatial/clare/spatial_SR.txt"),
    ("CLARE", "libero_object",  10, "sweep",    "outputs/CLARE_eval/libero_object/seed42"),
    ("CLARE", "libero_40",      40, "sweep",    "outputs/CLARE_eval/libero_40/seed42"),
]


def w(text):
    """한글은 터미널에서 두 칸을 먹는다. 표를 맞추려면 문자 수가 아니라 폭을 세야 한다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def pad(text, width, right=False):
    fill = " " * max(0, width - w(text))
    return fill + text if right else text + fill


def from_sweep(path):
    """칸별 eval_info.json에서 성공률과 에피소드 수를 읽는다."""
    cells, eps = {}, set()
    for p in glob.glob(f"{ROOT/path}/stage*/task*/eval_info.json"):
        k = int(p.split("/stage")[1].split("/")[0])
        t = int(p.split("/task")[1].split("/")[0])
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        cells[(k, t)] = d["aggregated"]["pc_success"]
        eps.add(len(d.get("per_episode", [])))
    return cells, (eps.pop() if len(eps) == 1 else None)


def from_txt(path):
    """이미 만들어 둔 SR txt를 되읽는다. 에피소드 수는 파일에 없다."""
    cells = {}
    for line in (ROOT / path).read_text().splitlines()[1:]:
        f = line.split("\t")
        if not f or not f[0].strip().isdigit():
            continue
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    return cells, 100        # 학습 스크립트의 N_EVAL=100


def metrics(cells, n):
    """ACC(최종 스테이지 평균), BWT(역방향 전이), 대각(습득 직후) 평균."""
    last = [cells.get((n - 1, t)) for t in range(n)]
    diag = [cells.get((t, t)) for t in range(n)]
    if any(v is None for v in last) or any(v is None for v in diag):
        return None
    acc = sum(last) / n
    bwt = sum(last[t] - diag[t] for t in range(n - 1)) / (n - 1)
    return acc, bwt, sum(diag) / n


def matrix_block(cells, n, tag):
    out = [f"{tag}\t" + "\t".join(str(t) for t in range(n))]
    for k in range(n):
        row = [cells.get((k, t)) for t in range(k + 1)]
        out.append(f"{k}\t" + "\t".join("" if v is None else f"{v:.0f}" for v in row))
    return "\n".join(out)


def main():
    loaded, lines = [], []
    for method, bench, n, kind, path in SOURCES:
        if not (ROOT / path).exists():
            print(f"  SKIP {method} {bench}: 없음 ({path})", file=sys.stderr)
            continue
        cells, eps = (from_sweep if kind == "sweep" else from_txt)(path)
        want = n * (n + 1) // 2
        loaded.append((method, bench, n, kind, path, cells, eps, len(cells), want))

    lines.append("=" * 78)
    lines.append("CL SR 리포트 — LIBERO / seed 42")
    lines.append("=" * 78)
    lines.append("")
    lines.append("행 = 스테이지 k 체크포인트, 열 = 태스크 j, 값 = 성공률(%). 하삼각.")
    lines.append("모든 값은 시뮬레이터 롤아웃 성공률이다 (오프라인 지표 아님).")
    lines.append("")
    lines.append("  ACC  최종 스테이지에서 전 태스크 평균 SR")
    lines.append("  BWT  최종 SR - 습득 직후 SR 의 평균 (음수 = 망각)")
    lines.append("  습득  대각 평균. 각 태스크를 막 배웠을 때의 SR")
    lines.append("")
    lines.append("출처  sweep    = 평가 스크립트 산출 eval_info.json")
    lines.append("      trainlog = 학습 스크립트가 스테이지 끝에 돌린 평가 (같은 롤아웃 함수)")
    lines.append("")

    lines.append("-" * 78)
    lines.append("요약")
    lines.append("-" * 78)
    lines.append(pad("방법", 7) + pad("스위트", 17) + pad("칸", 9, True)
                 + pad("롤아웃/칸", 11, True) + pad("ACC", 8, True)
                 + pad("BWT", 8, True) + pad("습득", 8, True) + "  출처")
    for method, bench, n, kind, path, cells, eps, got, want in loaded:
        m = metrics(cells, n)
        src = "sweep" if kind == "sweep" else "trainlog"
        cell_s = f"{got}/{want}"
        eps_s = str(eps) if eps else "?"
        if m:
            acc, bwt, dia = m
            lines.append(pad(method, 7) + pad(bench, 17) + pad(cell_s, 9, True)
                         + pad(eps_s, 11, True) + pad(f"{acc:.1f}", 8, True)
                         + pad(f"{bwt:+.1f}", 8, True) + pad(f"{dia:.1f}", 8, True)
                         + "  " + src)
        else:
            lines.append(pad(method, 7) + pad(bench, 17) + pad(cell_s, 9, True)
                         + pad(eps_s, 11, True) + pad("-", 8, True) + pad("-", 8, True)
                         + pad("-", 8, True) + "  " + src + " (미완)")
    lines.append("")
    lines.append("주의: libero_40은 칸당 20 롤아웃이라 이항 표준오차가 최대 +-11%p다.")
    lines.append("      개별 칸이 아니라 행/블록 평균으로 읽어야 한다.")
    lines.append("      단일 스위트 4종은 칸당 100 롤아웃, 표준오차 최대 +-5%p.")
    lines.append("      학습 시드는 42 하나뿐이다. 방법 간 수%p 차이는 시드 반복 없이 단정할 수 없다.")
    lines.append("")

    # libero_40 블록 분해 — 4개 스위트를 이어 붙인 시퀀스라 순서 효과가 드러난다
    for method, bench, n, kind, path, cells, eps, got, want in loaded:
        if n != 40 or metrics(cells, n) is None:
            continue
        lines.append("-" * 78)
        lines.append(f"{method} {bench} — 블록별 분해 (학습 순서: 10 -> goal -> spatial -> object)")
        lines.append("-" * 78)
        lines.append(pad("블록", 17) + pad("태스크", 9, True) + pad("최종", 8, True)
                     + pad("습득", 8, True) + pad("망각", 8, True))
        for name, off in [("libero_10", 0), ("libero_goal", 10),
                          ("libero_spatial", 20), ("libero_object", 30)]:
            fin = sum(cells[(39, t)] for t in range(off, off + 10)) / 10
            dia = sum(cells[(t, t)] for t in range(off, off + 10)) / 10
            lines.append(pad(name, 17) + pad(f"{off}-{off+9}", 9, True)
                         + pad(f"{fin:.1f}", 8, True) + pad(f"{dia:.1f}", 8, True)
                         + pad(f"{fin-dia:+.1f}", 8, True))
        lines.append("")

    for method, bench, n, kind, path, cells, eps, got, want in loaded:
        lines.append("=" * 78)
        m = metrics(cells, n)
        head = f"{method}  {bench}  ({got}/{want}칸, 칸당 {eps or '?'} 롤아웃, " \
               f"{'sweep' if kind == 'sweep' else 'trainlog'})"
        lines.append(head)
        if m:
            lines.append(f"ACC {m[0]:.1f}   BWT {m[1]:+.1f}   습득 {m[2]:.1f}")
        lines.append("=" * 78)
        lines.append(matrix_block(cells, n, f"LIBERO_{bench.removeprefix('libero_').upper()}"))
        lines.append("")

    out = ROOT / "results" / "SR_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"saved -> {out}  ({len(loaded)}개 실험)")


if __name__ == "__main__":
    main()
