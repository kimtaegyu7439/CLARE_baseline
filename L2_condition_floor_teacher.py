#!/usr/bin/env python
"""조건 붕괴 바닥(floor) 대 실측 손실 — **학습 목적함수 판, 실제 데이터**.

재는 것은 학습이 실제로 최소화하는 목적함수 그 자체다.

    L = ‖v_S(ℓ_t) − (a−ε)‖²  +  λ · Σ_{j<t} ‖v_S(ℓ_j) − v_T(ℓ_j)‖²
        └─ L_FM (현재 태스크) ─┘     └───── L_anchor (과거 태스크들) ─────┘

★ 이 파일의 핵심 — 앵커 항의 **student 와 teacher 모두 실제 task t 데이터**를 받는다.
  학습 중 앵커는 과거 코드북에서 합성한 (s̃, õ) 와 teacher 부트스트랩 x_t 를 쓰지만,
  floor 와 비교하려면 양쪽이 **같은 실제 좌표**에 있어야 한다. 합성 좌표에서 잰 손실을
  실제 데이터로 정의한 바닥과 견주면 서로 다른 분포의 수를 비교하는 셈이 된다.

  teacher 는 **stage t−1 체크포인트**다 — 학습이 쓰는 rolling teacher 와 같다.
  과거 태스크 j 의 목표 v*_j 도 이 teacher 가 ℓ_j 를 들었을 때의 출력으로 둔다
  (학습의 앵커 정의 그대로).

이론
    조건맹 해(모든 ℓ 에 같은 출력 u)의 손실은 아래로 내려갈 수 없다:
        floor = min_u [ w_t‖u − v*_t‖² + Σ_{j<t} w_j‖u − v*_j‖² ],  w_t=1, w_j=λ
              = Σ_i w_i ‖v*_i − ū‖²,   ū = Σ_i w_i v*_i / W,  W = 1 + λ·t
    N=2 에서 λ/(1+λ)·‖v*_t − v*_j‖² 로 환원된다(B_split.py 와 같은 식).
    조건별로 갈라지면 원리상 0 까지 갈 수 있으므로 이 바닥이 분화 압력이다.

    actual < floor  -> 조건 분화가 실제로 이득을 냈다
    actual ≈ floor  -> 조건맹 해 근처에 머물러 있다

효율
    DINOv2 CLS 는 동결이라 모든 체크포인트에서 값이 같다. 실제 배치를 한 번만 돌려
    (state, cls, x_t, t, target) 을 캐시하고, 모델마다 cond_tail 의 학습 대상 투영만
    다시 돌린다. 그래서 행마다 모델을 2개(student, teacher)만 올리면 된다.

    python L2_condition_floor_teacher.py \
        --ckpt_root outputs/L2_codebook/libero_spatial_seed41_ours \
        --instructions results/l2_codebook_k96_grad_seed41/instructions.json \
        --tasks 1,2,3,4,5,6,7,8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import torch                                                          # noqa: E402

from l2_codebook_id import (                                          # noqa: E402
    build_cfg, cond_tail, encode_lang, make_cond, prep_batch, rgb_cls, sample_fm,
)
from lerobot.datasets.factory import make_dataset                     # noqa: E402
from lerobot.datasets.utils import cycle                              # noqa: E402
from lerobot.policies.factory import make_policy                      # noqa: E402
from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device  # noqa: E402
from lerobot.utils.utils import init_logging                          # noqa: E402


def mse(a, b) -> float:
    return float((a - b).pow(2).mean())


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_root", default=str(REPO / "outputs" / "L2_codebook"
                                             / "libero_spatial_seed41_ours"))
    p.add_argument("--instructions",
                   default=str(REPO / "results" / "l2_codebook_k96_grad_seed41"
                               / "instructions.json"))
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", default="1,2,3,4,5,6,7,8")
    p.add_argument("--n_batches", type=int, default=40)
    p.add_argument("--n_draws", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--episodes_per_task", type=int, default=45)
    p.add_argument("--lam", type=float, default=3.0,
                   help="λ_level. 학습 실효 가중은 λ_a(1.0) x λ_lvl(3.0) = 3.0")
    p.add_argument("--split", choices=["train", "holdout"], default="train")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(REPO / "results"
                                        / "L2_condition_floor_teacher.txt"))
    return p.parse_args()


def ckpt_of(a, k: int) -> Path:
    return Path(a.ckpt_root) / f"task_{k}" / "checkpoints" / "last" / "pretrained_model"


def _cfg_args(a):
    # build_cfg(l2_codebook_id.py) 가 읽는 필드 전부. 원본 L2_condition_floor.py 와 동일.
    return SimpleNamespace(suite=a.suite, device=a.device, seed=a.seed,
                           num_workers=a.num_workers, batch_size=a.batch_size,
                           steps_per_task=1, log_every=10,
                           eval_episodes=1, eval_batch_size=1)


def build_dataset(a, t: int):
    cfg = build_cfg(_cfg_args(a), t, str(ckpt_of(a, t)),
                    Path(os.environ.get("SCRATCH", "/tmp")) / f"cfloor_t{t}")
    ds = make_dataset(cfg)
    total = ds.meta.total_episodes
    holdout = total - a.episodes_per_task
    if holdout > 0:
        tr, ho = split_episodes(cfg.dataset.repo_id, cfg.dataset.root, holdout)
        eps = ho if a.split == "holdout" else tr
    else:
        eps = list(range(total))
    return cfg, ds, eps


def load_policy(a, k: int, ds_meta, device):
    cfg = build_cfg(_cfg_args(a), k, str(ckpt_of(a, k)),
                    Path(os.environ.get("SCRATCH", "/tmp")) / f"cfloor_t{k}")
    pol = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    return pol.to(device).eval().requires_grad_(False)


@torch.no_grad()
def cache_coords(a, cfg, ds, eps, policy, device):
    """task t 의 **실제** 배치에서 (state, cls, x_t, tt, target) 을 캐시한다.

    cls 는 동결 DINOv2 출력이라 어떤 체크포인트로 뽑아도 같다 — 한 번만 뽑고
    모델마다 cond_tail 의 학습 대상 투영만 다시 돌린다.
    (ε, t) 는 배치·draw 로 시드를 고정해 행마다 같은 좌표를 쓴다.
    """
    # ★ num_workers=0 으로 고정한다. 행마다 40배치만 읽고 로더를 버리는데, spawn 워커를
    #   그때마다 띄웠다 내리면 종료 시점에 "worker killed by signal: Aborted" 로 죽는다
    #   (실측: task 2 진입에서 재현). 읽는 양이 적어 워커 없이도 충분하다.
    loader = torch.utils.data.DataLoader(
        ds, num_workers=0, batch_size=a.batch_size,
        sampler=episode_sampler(cfg, ds, eps), pin_memory=True, drop_last=True)
    it = cycle(loader)
    out = []
    for b in range(a.n_batches):
        batch = prep_batch(policy, to_device(next(it), device))
        cls = rgb_cls(policy, batch)                       # (B*4,768) 동결
        state = batch["observation.state"]
        for d in range(a.n_draws):
            torch.manual_seed(1_000_003 * b + 97 * d + 7)
            x_t, tt, target = sample_fm(policy, batch)
            out.append({"state": state, "cls": cls,
                        "x_t": x_t, "tt": tt, "target": target})
    del loader, it
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def velocities(policy, coords, instr_text: str):
    """이 모델이 instr_text 조건에서 내는 속도. 캐시된 **실제** 좌표 위에서만 돈다."""
    net = policy.dit_flow.velocity_net
    vs = []
    for c in coords:
        B = c["state"].shape[0]
        tail = cond_tail(policy, {"observation.state": c["state"]}, c["cls"])
        cond = make_cond(encode_lang(policy, [instr_text] * B), tail)
        vs.append(net(noisy_actions=c["x_t"], time=c["tt"], global_cond=cond))
    return vs


@torch.no_grad()
def measure_row(a, t: int, instr: dict, device) -> dict:
    """행 t. student=stage t, teacher=stage t−1(rolling, 학습과 동일).

    조건은 ℓ_0..ℓ_t 의 N=t+1 개. 목표는
        v*_t = a − ε              (현재 태스크, 실측)
        v*_j = v_T(ℓ_j)   j<t     (과거 태스크, teacher 가 실제 데이터에서 낸 값)
    """
    N, lam = t + 1, a.lam
    cfg, ds, eps = build_dataset(a, t)
    ds_meta = ds.meta

    student = load_policy(a, t, ds_meta, device)
    coords = cache_coords(a, cfg, ds, eps, student, device)
    V_S = [velocities(student, coords, instr[f"task{i}"]) for i in range(N)]
    del student
    torch.cuda.empty_cache()

    teacher = load_policy(a, t - 1, ds_meta, device)
    V_T = [velocities(teacher, coords, instr[f"task{j}"]) for j in range(t)]
    del teacher, ds
    torch.cuda.empty_cache()

    w = [lam] * t + [1.0]                      # 과거 각 λ, 현재 1 (인덱스 i=0..t)
    W = sum(w)
    acc = {k: 0.0 for k in ("e_fm", "e_anc", "actual", "floor", "G2", "split",
                            "tgt_pow")}
    per = {"actual": [], "floor": []}        # 표본별 값 (불확실성 추정용)
    for bi, c in enumerate(coords):
        tgt = c["target"]
        V_star = [V_T[j][bi] for j in range(t)] + [tgt]          # v*_0..v*_t
        # 실측 목적함수
        e_fm = mse(V_S[t][bi], tgt)
        e_anc = sum(mse(V_S[j][bi], V_star[j]) for j in range(t))
        acc["e_fm"] += e_fm
        acc["e_anc"] += e_anc
        acc["actual"] += e_fm + lam * e_anc
        # 조건맹 바닥 — 같은 가중의 가중분산
        ubar = sum(wi * v for wi, v in zip(w, V_star)) / W
        fl = sum(wi * mse(v, ubar) for wi, v in zip(w, V_star))
        acc["floor"] += fl
        per["actual"].append(e_fm + lam * e_anc)
        per["floor"].append(fl)
        # 참고: 과거-현재 충돌량
        acc["G2"] += sum(mse(V_star[j], tgt) for j in range(t)) / max(t, 1)
        # 분화 정도: student 가 과거 조건과 현재 조건을 얼마나 갈랐나
        s = 0.0
        for j in range(t):
            num = (V_S[j][bi] - V_S[t][bi]).flatten(1).norm(dim=1).mean()
            den = (V_star[j] - tgt).flatten(1).norm(dim=1).mean().clamp_min(1e-8)
            s += float(num / den)
        acc["split"] += s / max(t, 1)
        acc["tgt_pow"] += float(tgt.pow(2).mean())
    n = len(coords)
    out = {k: v / n for k, v in acc.items()}
    # ── 불확실성: 표본(coord set)별 값으로 표준오차와 분할 반분을 낸다 ──────────
    # 적은 배치로 재면 우연히 쉬운/어려운 구간만 볼 수 있다. per_actual/per_floor 의
    # 산포를 같이 보고해 "표본이 충분한가"를 판단할 수 있게 한다.
    import statistics as st
    pa, pf = per["actual"], per["floor"]
    half = len(pa) // 2
    out["se_actual"] = st.stdev(pa) / (len(pa) ** 0.5) if len(pa) > 1 else 0.0
    out["se_floor"] = st.stdev(pf) / (len(pf) ** 0.5) if len(pf) > 1 else 0.0
    out["ratio"] = out["actual"] / max(out["floor"], 1e-12)
    pr = [x / max(y, 1e-12) for x, y in zip(pa, pf)]
    out["se_ratio"] = st.stdev(pr) / (len(pr) ** 0.5) if len(pr) > 1 else 0.0
    out["ratio_half1"] = (sum(pa[:half]) / max(sum(pf[:half]), 1e-12)) if half else None
    out["ratio_half2"] = (sum(pa[half:]) / max(sum(pf[half:]), 1e-12)) if half else None
    out["ratio_min"], out["ratio_max"] = min(pr), max(pr)
    out["frac_below_floor"] = sum(1 for x in pr if x < 1.0) / len(pr)
    out.update(N=N, t=t, lam=lam, n_coord=n, n_samples=n * a.batch_size,
               student=str(ckpt_of(a, t)), teacher=str(ckpt_of(a, t - 1)))
    return out


def main():
    a = parse()
    init_logging()
    device = torch.device(a.device)
    instr = json.loads(Path(a.instructions).read_text())
    rows, t0 = [], time.perf_counter()
    for t in [int(x) for x in a.tasks.split(",")]:
        if not (ckpt_of(a, t).exists() and ckpt_of(a, t - 1).exists()):
            print(f"  task {t}: 체크포인트 없음 — 건너뜀", flush=True)
            continue
        try:
            r = measure_row(a, t, instr, device)
        except Exception as e:                      # 한 행이 죽어도 나머지는 살린다
            print(f"  task {t}: 실패 — {type(e).__name__}: {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        json.dump(rows, Path(a.out).with_suffix(".json").open("w"),
                  indent=2, ensure_ascii=False)      # 칸마다 저장
        g = r["floor"] - r["actual"]
        print(f"  task {t}  N={r['N']}  floor={r['floor']:.5f}  actual={r['actual']:.5f}  "
              f"gap={g:+.5f} ({100*g/max(r['floor'],1e-12):.1f}%)  "
              f"ratio={r['actual']/max(r['floor'],1e-12):.2f}  split={r['split']:.3f}"
              f"  [{time.perf_counter()-t0:.0f}s]", flush=True)

    W = 108
    L = ["=" * W,
         "조건 붕괴 바닥 대 실측 손실 — 학습 목적함수 판 (앵커 입력 = 실제 데이터)",
         "=" * W, "",
         "재는 것은 학습이 실제로 최소화하는 목적함수 그 자체다.",
         "    L = ‖v_S(ℓ_t) − (a−ε)‖²  +  λ·Σ_{j<t} ‖v_S(ℓ_j) − v_T(ℓ_j)‖²",
         "",
         "★ 앵커 항의 student 와 teacher **모두 실제 task t 데이터**를 받는다.",
         "  학습 중 앵커는 과거 코드북의 합성 (s̃,õ) 를 쓰지만, floor 와 비교하려면",
         "  양쪽이 같은 실제 좌표 위에 있어야 한다. 따라서 이 표의 actual 은",
         "  diagnostics.jsonl 의 학습 손실과 **같은 수가 아니다**.",
         "",
         "  teacher = stage t−1 체크포인트 (학습이 쓰는 rolling teacher 와 동일)",
         "  목표     v*_t = a − ε (실측),  v*_j = v_T(ℓ_j) (j<t)",
         "",
         "floor  = min_u Σ_i w_i‖u − v*_i‖² = Σ_i w_i‖v*_i − ū‖²   (w_t=1, w_j=λ)",
         "         조건을 무시하는 해가 내려갈 수 있는 최소 손실.",
         "         N=2 에서 λ/(1+λ)·‖v*_t − v*_j‖² 로 환원된다.",
         "actual = 실제 도달한 목적함수 값. actual < floor 면 조건 분화가 이득을 냈다.",
         "",
         f"체크포인트  {a.ckpt_root}",
         f"좌표        task t 의 실제 데이터({a.split} split), "
         f"{a.n_batches}배치 x {a.n_draws}draw x B{a.batch_size}",
         f"λ           {a.lam}", "",
         "-" * W,
         f"{'row':<8}{'N':>3}{'floor':>10}{'actual':>10}{'ratio':>7}{'±SE':>7}"
         f"{'반분1':>7}{'반분2':>7}{'<1 비율':>8}{'L_FM':>9}{'L_anc':>9}{'split':>7}",
         "-" * W]
    for r in rows:
        L.append(f"task {r['t']:<3}{r['N']:>3}{r['floor']:>10.5f}{r['actual']:>10.5f}"
                 f"{r['ratio']:>7.2f}{r['se_ratio']:>7.3f}"
                 f"{(r['ratio_half1'] or 0):>7.2f}{(r['ratio_half2'] or 0):>7.2f}"
                 f"{100*r['frac_below_floor']:>7.0f}%"
                 f"{r['e_fm']:>9.5f}{r['e_anc']:>9.5f}{r['split']:>7.3f}")
    L += ["-" * W, "",
          "읽는 법",
          "  · ratio = actual/floor.  <1 이면 분화가 이득을 냈다, ≈1 이면 조건맹 해 근처.",
          "  · ±SE 는 표본(coord set)별 ratio 의 표준오차. 반분1/반분2 는 표본을 앞뒤로",
          "    반씩 갈라 따로 계산한 ratio 다 — 둘이 크게 다르면 표본이 부족하거나",
          "    배치 간 편향이 있다는 뜻이다. '<1 비율' 은 개별 표본 중 floor 밑으로",
          "    내려간 비율로, 평균 하나로는 안 보이는 분포를 드러낸다.",
          "  · L_FM / L_anc 는 actual 의 두 구성요소다 (actual = L_FM + λ·L_anc).",
          "  · split = mean_j ‖v_S(ℓ_j) − v_S(ℓ_t)‖ / ‖v*_j − (a−ε)‖.",
          "    gap 이 '이득이 났나'라면 split 은 '실제로 갈라졌나'다. 함께 읽어야 한다.",
          "  · 행 t 는 과거 태스크 t 개를 전부 조건으로 넣는다(N = t+1).",
          "    학습의 chunk backward 와 같이 과거는 각각 λ 로 더해진다.", ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    json.dump(rows, Path(a.out).with_suffix(".json").open("w"),
              indent=2, ensure_ascii=False)
    print("\n".join(L))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
