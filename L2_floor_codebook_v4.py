#!/usr/bin/env python
"""조건 붕괴 바닥 — **앵커가 실제로 작동하는 좌표(코드북)** 에서 재는 판.

L2_floor_lotus_v4.py 는 현재 태스크의 실제 데이터 좌표에서 쟀다. 그런데 학습은
앵커 항을 그 좌표에서 최소화한 적이 없다 — 과거 코드북에서 뽑은 (s̃, õ) 와 teacher
부트스트랩 x_t 에서만 했다. 그래서 여기서는 **학습이 실제로 본 좌표**에서 잰다.

    좌표   과거 태스크 j0 의 코드북에서 (s̃, õ) 를 뽑고,
           â = ε₀ + v_T(ε₀, 0, cond_T(ℓ_{j0})),  x_t = (1−t)ε₁ + t·â
           (L2_V4.anchor_loss 와 같은 경로)
    목표   v_j = v_T(x_t, t, cond(ℓ_j))        j ∈ 과거 전부
    실측   Σ_j ‖v_S(ℓ_j) − v_j‖²
    바닥   Σ_j ‖v_j − v̄‖²                      v̄ = 과거 목표들의 평균

★ FM 항이 없다. 합성 좌표에는 정답 액션 a 가 없어 v_0 = a−ε 를 만들 수 없다.
  따라서 이 표는 **앵커 항만의** 진단이다: "앵커가 걸린 바로 그 좌표에서, 정책이
  과거 명령어들을 붕괴한 해보다 잘 구분하는가". 가중은 전부 같아 λ 는 약분된다.

★ 코드북은 디스크에 없어 재생성한다. 코드북 입력 (s, o) 는 **모델과 무관**하다 —
  o 는 동결 DINOv2 CLS 이고 s 는 정규화된 상태인데 정규화 버퍼가 전 체크포인트에서
  동일함을 확인했다. 그래서 정책 하나로 전부 만든다.

    python L2_floor_codebook_v4.py --run results/l2_codebook_v4_goal_seed7 --suite goal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import torch                                                          # noqa: E402

import L2_V4 as V4                                                    # noqa: E402
from L2_floor_lotus_v4 import load_policy, student_ckpt, teacher_ckpt, mse  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata   # noqa: E402
from lerobot.utils.utils import init_logging                          # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--suite", default="goal", choices=["goal", "object", "50"])
    p.add_argument("--stages", default="")
    p.add_argument("--n_draws", type=int, default=40, help="과거 태스크마다 좌표 묶음 수")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--k_per_demo", type=int, default=2)
    p.add_argument("--k_min", type=int, default=8)
    p.add_argument("--h_scale", type=float, default=1.0)
    p.add_argument("--ridge_rho", type=float, default=0.05)
    p.add_argument("--grad_min_frames", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    return p.parse_args()


def build_all_books(a, spec, policy, upto: int, device):
    """태스크 0..upto-1 의 코드북. base 는 한 스플릿에 여러 태스크가 들어 있다."""
    nb = spec["n_base"]
    books = {}
    base_ids = [t for t in range(min(nb, upto))]
    if base_ids:
        books.update(V4.make_books(policy, spec["root"] / "base", device, a,
                                   task_ids=list(range(nb)),
                                   n_demos=spec["base_demos"]))
        books = {t: v for t, v in books.items() if t < upto}
    for t in range(nb, upto):
        k = t - nb + 1                                   # 그 태스크를 배운 stage
        books.update(V4.make_books(policy, spec["root"] / f"stage{k}", device, a,
                                   task_ids=[t], n_demos=spec["stage_demos"]))
    return books


@torch.no_grad()
def measure_stage(a, spec, k: int, device) -> dict:
    run = Path(a.run)
    cur = spec["n_base"] - 1 + k
    past = list(range(cur))
    N = len(past)
    meta = LeRobotDatasetMetadata("local/lotus", root=str(spec["root"] / "base"))

    student = load_policy(student_ckpt(run, k), meta, device)
    teacher = load_policy(teacher_ckpt(run, k, a.suite), meta, device)

    # 코드북은 모델 무관 — student 하나로 전부 만든다
    books = build_all_books(a, spec, student, cur, device)
    missing = [j for j in past if j not in books]
    if missing:
        raise RuntimeError(f"코드북 없음: {missing}")

    st_shape = None
    acc = {"actual": 0.0, "floor": 0.0, "split": 0.0, "G2": 0.0}
    per = {"actual": [], "floor": []}
    n_obs, d_state = spec.get("n_obs", 2), None
    gen = torch.Generator(device=device).manual_seed(a.seed)

    for j0 in past:                       # 앵커 항마다 자기 좌표에서
        cb = books[j0]
        for d in range(a.n_draws):
            n = a.batch_size
            s_t, o_t = V4.sample_codebook_hard(cb, n, gen)
            s_t, o_t = s_t.detach(), o_t.detach()
            if st_shape is None:
                d_state = s_t.shape[1] // n_obs
                st_shape = (n, n_obs, d_state)
            bb = {"observation.state": s_t.view(st_shape)}
            cls_j = o_t.reshape(-1, 768)

            # teacher 부트스트랩 x_t — 학습(L2_V4.anchor_loss)과 같은 경로
            tlT = V4.cond_tail(teacher, bb, cls_j)
            condT0 = V4.make_cond(V4.encode_lang(teacher, [spec["instr"][j0]] * n), tlT)
            t_ = torch.rand(n, device=device, generator=gen)
            eps0 = torch.randn(n, 16, 7, device=device, generator=gen)
            eps1 = torch.randn(n, 16, 7, device=device, generator=gen)
            netT = teacher.dit_flow.velocity_net
            a_hat = eps0 + netT(noisy_actions=eps0, time=torch.zeros_like(t_),
                                global_cond=condT0)
            x_t = ((1 - t_[:, None, None]) * eps1 + t_[:, None, None] * a_hat).detach()

            # 같은 좌표에서 과거 명령어 전부에 대한 teacher 목표와 student 출력
            tlS = V4.cond_tail(student, bb, cls_j)
            netS = student.dit_flow.velocity_net
            V_T, V_S = [], []
            for j in past:
                cT = V4.make_cond(V4.encode_lang(teacher, [spec["instr"][j]] * n), tlT)
                cS = V4.make_cond(V4.encode_lang(student, [spec["instr"][j]] * n), tlS)
                V_T.append(netT(noisy_actions=x_t, time=t_, global_cond=cT))
                V_S.append(netS(noisy_actions=x_t, time=t_, global_cond=cS))

            vbar = sum(V_T) / N
            fl = sum(mse(v, vbar) for v in V_T)                    # 붕괴 해의 바닥
            act = sum(mse(s, v) for s, v in zip(V_S, V_T))         # 실측
            acc["floor"] += fl
            acc["actual"] += act
            i0 = past.index(j0)
            sp = 0.0
            for j in range(N):
                if j == i0:
                    continue
                num = (V_S[j] - V_S[i0]).flatten(1).norm(dim=1).mean()
                den = (V_T[j] - V_T[i0]).flatten(1).norm(dim=1).mean().clamp_min(1e-8)
                sp += float(num / den)
            acc["split"] += sp / max(N - 1, 1)
            acc["G2"] += sum(mse(V_T[i], V_T[j]) for i in range(N)
                             for j in range(i + 1, N)) / max(N * (N - 1) / 2, 1)
            per["actual"].append(act); per["floor"].append(fl)

    del student, teacher, books
    torch.cuda.empty_cache()
    m = len(per["actual"])
    out = {key: v / m for key, v in acc.items()}
    import statistics as st
    pr = [x / max(y, 1e-12) for x, y in zip(per["actual"], per["floor"])]
    h = len(pr) // 2
    out["ratio"] = out["actual"] / max(out["floor"], 1e-12)
    out["se_ratio"] = st.stdev(pr) / (len(pr) ** 0.5) if len(pr) > 1 else 0.0
    out["ratio_half1"] = sum(per["actual"][:h]) / max(sum(per["floor"][:h]), 1e-12)
    out["ratio_half2"] = sum(per["actual"][h:]) / max(sum(per["floor"][h:]), 1e-12)
    out["frac_below"] = sum(1 for x in pr if x < 1.0) / len(pr)
    out.update(stage=k, cur_task=cur, N=N, n_coord=m, n_samples=m * a.batch_size)
    return out


def main():
    a = parse()
    init_logging()
    device = torch.device(a.device)
    run = Path(a.run)
    spec = V4.suite_spec(a.suite)
    stages = ([int(x) for x in a.stages.split(",")] if a.stages
              else sorted(int(p.name.replace("stage", ""))
                          for p in (run / "checkpoints").glob("stage*")))
    out_path = Path(a.out) if a.out else run / "floor_codebook.txt"

    rows, t0 = [], time.perf_counter()
    for k in stages:
        try:
            r = measure_stage(a, spec, k, device)
        except Exception as e:
            print(f"  stage {k}: 실패 — {type(e).__name__}: {str(e)[:140]}", flush=True)
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        json.dump(rows, out_path.with_suffix(".json").open("w"),
                  indent=2, ensure_ascii=False)
        g = 100 * (r["floor"] - r["actual"]) / max(r["floor"], 1e-12)
        print(f"  stage {k}  t{r['cur_task']}  N={r['N']}  floor={r['floor']:.5f}  "
              f"actual={r['actual']:.5f}  바닥아래 {g:+.1f}%  ratio={r['ratio']:.3f}  "
              f"split={r['split']:.3f}  [{time.perf_counter()-t0:.0f}s]", flush=True)

    Wd = 100
    L = ["=" * Wd,
         f"조건 붕괴 바닥 — 앵커 좌표(코드북) 판  /  LOTUS {a.suite} v4(TETHER)",
         "=" * Wd, "",
         f"런   {run}", "",
         "좌표는 학습의 앵커가 실제로 본 곳이다 — 과거 코드북에서 (s̃,õ) 를 뽑고",
         "teacher 부트스트랩으로 x_t 를 만든다 (L2_V4.anchor_loss 와 같은 경로).",
         "",
         "  목표   v_j = v_T(x_t, t, ℓ_j)        j = 과거 태스크 전부",
         "  실측   Σ_j ‖v_S(ℓ_j) − v_j‖²",
         "  바닥   Σ_j ‖v_j − v̄‖²   (붕괴 해가 낼 수 있는 최소. 가중이 같아 λ 는 약분)",
         "",
         "★ FM 항이 없다 — 합성 좌표에는 정답 액션이 없다. 앵커 항만의 진단이다.", "",
         "-" * Wd,
         f"{'stage':<7}{'현재':>5}{'N':>4}{'floor':>11}{'actual':>11}{'바닥아래%':>11}"
         f"{'ratio':>8}{'±SE':>8}{'반분1':>8}{'반분2':>8}{'<1':>6}{'split':>7}",
         "-" * Wd]
    for r in rows:
        g = 100 * (r["floor"] - r["actual"]) / max(r["floor"], 1e-12)
        L.append(f"{r['stage']:<7}{'t'+str(r['cur_task']):>5}{r['N']:>4}"
                 f"{r['floor']:>11.5f}{r['actual']:>11.5f}{g:>+11.1f}"
                 f"{r['ratio']:>8.3f}{r['se_ratio']:>8.3f}"
                 f"{r['ratio_half1']:>8.3f}{r['ratio_half2']:>8.3f}"
                 f"{100*r['frac_below']:>5.0f}%{r['split']:>7.3f}")
    L += ["-" * Wd, "",
          "읽는 법",
          "  · 바닥아래% > 0 이면 정책이 과거 명령어들을 붕괴 해보다 잘 구분한다는 뜻.",
          "  · ratio = actual/floor.  <1 이 바닥 아래.",
          "  · 이 표는 앵커 항만 본다. 현재 태스크 FM 항은 합성 좌표에 정답이 없어 뺐다.",
          "    실제 데이터 좌표 판본은 floor_v4.txt 를 보라 — 두 표는 묻는 질문이 다르다.", ""]
    out_path.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
