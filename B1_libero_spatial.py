#!/usr/bin/env python
"""B1 을 libero_spatial 10태스크 전체로 확장하고 10x10 SR 표를 낸다.

방법론 코드는 B1.py 를 그대로 재사용한다(임포트해서 쓴다. 복제하지 않는다).
이 파일이 추가로 하는 일은 세 가지뿐이다.

  1. 기본값을 libero_spatial / 10태스크로 바꿔 B1.main() 을 호출
  2. 결과를 저장소 표준 `*_SR.txt` 형식(탭 구분 하삼각)으로 내보내
     기존 CLARE/ER 표와 같은 파서로 읽히게 한다
  3. CLARE / ER 표를 나란히 놓은 비교 리포트를 만든다

"CLARE setting = 백본만 쓴다"의 뜻:
  CLARE 는 베이스를 얼리고 어댑터만 학습하지만, B1 은 어댑터가 없는 전체 파인튜닝이다.
  공유하는 것은 **같은 사전학습 백본** dit_flow_mt_libero_90_pretrain 하나이며,
  이는 CLARE/ER 두 베이스라인이 모두 출발점으로 쓴 체크포인트다(bash/clare/env.sh:29-31).
  스테이지 연결은 ER 과 같은 방식이다 — task k>0 은 task k-1 체크포인트에서 이어받는다.
  CLARE 처럼 매 스테이지 PRETRAIN_PATH 로 되돌리면 어댑터가 없는 B1 에서는 이전 태스크
  학습이 통째로 사라져 CL 실험 자체가 성립하지 않는다.

사용법
    python B1_libero_spatial.py --smoke
    python B1_libero_spatial.py                       # 기본 = CLARE/ER 표와 같은 세팅
    python B1_libero_spatial.py --mirror_e0           # E0/B1 4태스크 실행과 같은 세팅
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import B1  # noqa: E402  방법론 본체. 이 파일은 B1.py 를 수정하지 않는다.

SUITE = "libero_spatial"
NUM_TASKS = 10
OUT_DIR = REPO / "results" / "B1_libero_spatial"

# 비교 대상. 둘 다 칸당 100 롤아웃이다(results/SR_report.txt 참조).
REF_TABLES = {
    "CLARE": REPO / "outputs" / "libero_spatial" / "clare" / "spatial_SR.txt",
    "ER": REPO / "outputs" / "ER_eval" / SUITE / "seed42" / "libero_spatial_SR.txt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mirror_e0", action="store_true",
                   help="CLARE/ER 표가 아니라 E0/B1 4태스크 실행과 세팅을 맞춘다 "
                        "(steps 5000, 에피소드 45, 롤아웃 20). 훨씬 빠르지만 기존 표와 자가 다르다.")
    p.add_argument("--steps_per_task", type=int, default=None)
    p.add_argument("--episodes_per_task", type=int, default=None)
    p.add_argument("--eval_episodes", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--mode", choices=["ours", "baseline"], default="ours")
    p.add_argument("--p_drop", type=float, default=0.1)
    p.add_argument("--lambda_anchor", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_tasks", type=int, default=NUM_TASKS)
    p.add_argument("--guidance_w", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report_only", action="store_true",
                   help="학습 없이 이미 있는 sr_matrix.csv 로 표/리포트만 다시 만든다")
    a = p.parse_args()

    # 기본은 CLARE/ER libero_spatial 표와 동일한 자다. 비교가 목적이기 때문이다.
    #   bash/clare/clare_libero_spatial.sh : STEPS=20000, N_EVAL=100, BS_EVAL=50, 전체 50 에피소드
    #   bash/er/ER_libero_spatial.sh       : 동일
    if a.mirror_e0:
        defaults = dict(steps_per_task=5000, episodes_per_task=45,
                        eval_episodes=20, eval_batch_size=20)
    else:
        defaults = dict(steps_per_task=20000, episodes_per_task=50,
                        eval_episodes=100, eval_batch_size=50)
    for k, v in defaults.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


# ═════════════════════════════════════════════════════════════════════════════
#  표 입출력
# ═════════════════════════════════════════════════════════════════════════════
def read_sr_txt(path: Path, n: int) -> dict[tuple[int, int], float]:
    """저장소 표준 `*_SR.txt`(탭 구분 하삼각)를 읽는다."""
    cells: dict[tuple[int, int], float] = {}
    if not path.exists():
        return cells
    for line in path.read_text().splitlines()[1:]:
        f = line.split("\t")
        if not f or not f[0].strip().isdigit():
            continue
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip() and t <= k < n:
                cells[(k, t)] = float(v)
    return cells


def read_sr_csv(path: Path, n: int) -> dict[tuple[int, int], float]:
    """B1 이 쓴 sr_matrix.csv 를 읽는다."""
    cells: dict[tuple[int, int], float] = {}
    if not path.exists():
        return cells
    for line in path.read_text().splitlines()[1:]:
        f = line.split(",")
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    return cells


def write_sr_txt(cells: dict, n: int, out: Path, tag: str) -> None:
    """기존 CLARE/ER 표와 같은 형식으로 내보낸다. 같은 파서가 그대로 동작한다."""
    with out.open("w") as f:
        f.write(f"{tag}\t" + "\t".join(str(t) for t in range(n)) + "\n")
        for k in range(n):
            row = [cells.get((k, t)) for t in range(k + 1)]
            f.write(f"{k}\t" + "\t".join("" if v is None else f"{v:.0f}" for v in row) + "\n")


def metrics_of(cells: dict, n: int) -> dict | None:
    last = [cells.get((n - 1, t)) for t in range(n)]
    diag = [cells.get((t, t)) for t in range(n)]
    if any(v is None for v in last) or any(v is None for v in diag):
        return None
    return {
        "ACC": sum(last) / n,
        "BWT": sum(last[t] - diag[t] for t in range(n - 1)) / (n - 1),
        "acq": sum(diag) / n,
        "final": last,
        "diag": diag,
    }


def build_report(cells: dict, n: int, args, out_dir: Path) -> str:
    """B1 / CLARE / ER 를 한 파일에 나란히 놓는다."""
    L = ["=" * 78,
         f"B1 — {SUITE} {n}태스크  SR 리포트  (seed {args.seed}, mode={args.mode})",
         "=" * 78, ""]
    setting = ("B1/E0 미러" if args.mirror_e0 else "CLARE/ER 표와 동일")
    L += [f"학습      steps/task {args.steps_per_task}, 에피소드 {args.episodes_per_task}, "
          f"batch 32  [{setting}]",
          f"평가      칸당 {args.eval_episodes} 롤아웃, start_seed {args.seed}",
          f"방법      p_drop {args.p_drop}, lambda_anchor {args.lambda_anchor}",
          "백본      dit_flow_mt_libero_90_pretrain (CLARE/ER 와 동일)",
          "스테이지   task k>0 은 task k-1 체크포인트에서 이어받음 (ER 과 같은 방식)", ""]

    if args.mirror_e0:
        L += ["주의: CLARE/ER 표는 steps 20000 / 에피소드 50 / 롤아웃 100 이다.",
              "      학습 예산과 평가 표본이 달라 대각(습득) 값의 직접 비교는 조심해야 한다.",
              "      망각 폭(BWT)과 행 모양은 예산 차이에 덜 민감하므로 그쪽을 먼저 보라.", ""]

    rows = [("B1", cells)]
    for name, path in REF_TABLES.items():
        ref = read_sr_txt(path, n)
        if ref:
            rows.append((name, ref))

    L += ["-" * 78, "요약", "-" * 78,
          f"{'방법':8s}{'ACC':>9s}{'BWT':>9s}{'습득':>9s}   비고"]
    for name, c in rows:
        m = metrics_of(c, n)
        note = "" if name == "B1" else "칸당 100 롤아웃"
        if m:
            L.append(f"{name:8s}{m['ACC']:9.1f}{m['BWT']:+9.1f}{m['acq']:9.1f}   {note}")
        else:
            got = len(c)
            L.append(f"{name:8s}{'-':>9s}{'-':>9s}{'-':>9s}   미완 ({got}/{n*(n+1)//2}칸)")
    L.append("")

    for name, c in rows:
        m = metrics_of(c, n)
        if not m:
            continue
        L += ["-" * 78, f"{name} — 최종 행(stage {n-1}) vs 습득(대각)", "-" * 78,
              "task   " + "".join(f"{t:>6d}" for t in range(n)),
              "습득   " + "".join(f"{v:6.0f}" for v in m["diag"]),
              "최종   " + "".join(f"{v:6.0f}" for v in m["final"]),
              "차이   " + "".join(f"{m['final'][t]-m['diag'][t]:+6.0f}" for t in range(n)), ""]

    for name, c in rows:
        L += ["=" * 78, f"{name}  SR matrix   행 = 태스크 k 학습 후, 열 = 평가 태스크",
              "=" * 78,
              f"LIBERO_SPATIAL\t" + "\t".join(str(t) for t in range(n))]
        for k in range(n):
            row = [c.get((k, t)) for t in range(k + 1)]
            L.append(f"{k}\t" + "\t".join("" if v is None else f"{v:.0f}" for v in row))
        L.append("")

    return "\n".join(L)


def emit(args, out_dir: Path) -> None:
    n = args.num_tasks
    cells = read_sr_csv(out_dir / "sr_matrix.csv", n)
    if not cells:
        print("[B1-spatial] sr_matrix.csv 가 비어 있어 표를 만들지 않는다")
        return
    txt = out_dir / f"{SUITE}_SR.txt"
    write_sr_txt(cells, n, txt, "LIBERO_SPATIAL")
    report = build_report(cells, n, args, out_dir)
    (out_dir / "SR_report.txt").write_text(report)
    print(report)
    print(f"\nsaved -> {txt}")
    print(f"saved -> {out_dir / 'SR_report.txt'}")


def main() -> None:
    args = parse_args()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        emit(args, out_dir)
        return

    # B1.py 의 argparse 로 그대로 넘긴다. 방법론 코드는 한 줄도 복제하지 않는다.
    argv = [
        "B1.py",
        "--suite", SUITE,
        "--num_tasks", str(args.num_tasks),
        "--steps_per_task", str(args.steps_per_task),
        "--episodes_per_task", str(args.episodes_per_task),
        "--eval_episodes", str(args.eval_episodes),
        "--eval_batch_size", str(args.eval_batch_size),
        "--mode", args.mode,
        "--p_drop", str(args.p_drop),
        "--lambda_anchor", str(args.lambda_anchor),
        "--seed", str(args.seed),
        "--guidance_w", str(args.guidance_w),
        "--device", args.device,
        "--out_dir", str(out_dir),
        "--ckpt_root", str(REPO / "outputs" / "B1_libero_spatial"),
    ]
    if args.smoke:
        argv.append("--smoke")

    json.dump(vars(args), (out_dir / "spatial_config.json").open("w"), indent=2, ensure_ascii=False)
    old_argv, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old_argv

    emit(args, out_dir)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
