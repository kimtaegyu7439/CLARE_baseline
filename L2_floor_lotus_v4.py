#!/usr/bin/env python
"""조건 붕괴 바닥 L* 대 실측 손실 — **LOTUS 셋팅 v4(TETHER)** 판.

논문 4.2 의 진단을 그대로 구현한다. 한 좌표에서 여러 조건이 동시에 요구될 때,
조건을 무시하는(붕괴한) 정책이 낼 수 있는 최소 손실이 바닥 L* 이고, 실측 손실이
그 아래면 정책이 그 성분을 읽고 있다는 뜻이다.

    L(u) = μ‖u − v_0‖² + λ Σ_{j=1}^N ‖u − v_j‖²                      ... (6)
    u*   = (μ v_0 + λ Σ v_j) / (μ + λN)                              ... (7)
    L*   = λ/(μ+λN) · ( μ Σ_j ‖v_0−v_j‖² + λ Σ_{i<j} ‖v_i−v_j‖² )    ... (8)

    v_0        새 태스크의 목표 = a − ε   (현재 스테이지 데이터, 실측)
    v_1..v_N   과거 태스크들의 조건에서 **teacher 가 낸 속도**
    μ = 1 (L_FM 의 암묵 가중), λ = lambda_level

스왑은 **한 번에 한 성분**이다 (논문 Scope). 기본은 instruction swap —
(x_t, t)·관측·상태를 현재 스테이지 것으로 고정하고 **명령어만** 과거 것으로 바꾼다.

    L* 계산은 가중분산 형태 Σ_i w_i‖v_i − ū‖² 로 한다. 부록 A 의 항등식으로 (8) 과
    같은 값이고(수치 확인: N=1 에서 μλ/(μ+λ)‖v_0−v_1‖² 와 상대오차 2e-8), 쌍합 형태로
    다시 계산해 `pair_form` 열에 같이 찍어 둔다.

LOTUS 셋팅의 계보 (l2_codebook_id 판과 다른 점)
    · base phase 가 있다. base 체크포인트(Models/Lotus_clare_libero_<suite>_pretrain)가
      6 태스크를 joint 로 배운 상태이고, stage1 의 teacher 는 **그 base** 다.
    · stage k 의 student = <out>/checkpoints/stage{k}/checkpoints/last/pretrained_model,
      teacher = stage k−1 (k≥2) 또는 base (k=1).
    · stage k 가 새로 배우는 태스크는 n_base−1+k, 그 이전 전부가 과거 조건이다.
      goal: stage1 -> 현재 t6, 과거 t0..t5 (N=6);  stage4 -> 현재 t9, 과거 t0..t8 (N=9).
    · 좌표는 LOTUS_setting/<DIR>/stage{k} 스플릿의 **실제** 데이터다.

    python L2_floor_lotus_v4.py --run results/l2_codebook_v4_goal_seed7 --suite goal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import torch                                                          # noqa: E402

import L2_V4 as V4                                                    # noqa: E402
from lerobot.configs.policies import PreTrainedConfig                 # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata   # noqa: E402
from lerobot.datasets.utils import cycle                              # noqa: E402
from lerobot.policies.factory import make_policy                      # noqa: E402
from lerobot.scripts.E0 import to_device                              # noqa: E402
from lerobot.utils.utils import init_logging                          # noqa: E402

MODEL_ROOT = Path("/home/sa090180/Models")


def mse(a, b) -> float:
    return float((a - b).pow(2).mean())


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="v4 런 디렉터리 (checkpoints/stage*)")
    p.add_argument("--suite", default="goal", choices=["goal", "object", "50"])
    p.add_argument("--stages", default="", help="비우면 있는 stage 전부")
    p.add_argument("--n_batches", type=int, default=40)
    p.add_argument("--n_draws", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mu", type=float, default=1.0, help="L_FM 의 암묵 가중")
    p.add_argument("--lam", type=float, default=3.0, help="λ_level")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    return p.parse_args()


def student_ckpt(run: Path, k: int) -> Path:
    return run / "checkpoints" / f"stage{k}" / "checkpoints" / "last" / "pretrained_model"


def teacher_ckpt(run: Path, k: int, suite: str) -> Path:
    """stage k 의 teacher. k=1 은 base phase 체크포인트다."""
    if k >= 2:
        return student_ckpt(run, k - 1)
    return MODEL_ROOT / f"Lotus_clare_libero_{suite}_pretrain"


def load_policy(path: Path, ds_meta, device):
    cfg = PreTrainedConfig.from_pretrained(str(path))
    cfg.pretrained_path = str(path)
    cfg.device = str(device)
    cfg.push_to_hub = False
    pol = make_policy(cfg=cfg, ds_meta=ds_meta)
    return pol.to(device).eval().requires_grad_(False)


@torch.no_grad()
def cache_coords(a, ds, policy, device):
    """현재 스테이지의 **실제** 배치에서 (state, cls, x_t, tt, target) 을 캐시한다.

    cls 는 동결 DINOv2 라 모든 체크포인트에서 값이 같다 — 한 번만 뽑고 모델마다
    cond_tail 의 학습 대상 투영만 다시 돌린다. num_workers=0 으로 둔다(행마다
    로더를 버리는데 spawn 워커는 종료에서 죽는다).
    """
    loader = torch.utils.data.DataLoader(
        ds, num_workers=0, batch_size=a.batch_size, shuffle=True,
        pin_memory=True, drop_last=True)
    it = cycle(loader)
    out = []
    for b in range(a.n_batches):
        batch = V4.prep_batch(policy, to_device(next(it), device))
        cls = V4.rgb_cls(policy, batch)
        state = batch["observation.state"]
        for d in range(a.n_draws):
            torch.manual_seed(1_000_003 * b + 97 * d + 7)
            x_t, tt, target = V4.sample_fm(policy, batch)
            out.append({"state": state, "cls": cls,
                        "x_t": x_t, "tt": tt, "target": target})
    del loader, it
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def velocities(policy, coords, text: str):
    """이 모델이 text 조건에서 내는 속도. 캐시된 실제 좌표 위에서만 돈다."""
    net = policy.dit_flow.velocity_net
    vs = []
    for c in coords:
        B = c["state"].shape[0]
        tail = V4.cond_tail(policy, {"observation.state": c["state"]}, c["cls"])
        cond = V4.make_cond(V4.encode_lang(policy, [text] * B), tail)
        vs.append(net(noisy_actions=c["x_t"], time=c["tt"], global_cond=cond))
    return vs


@torch.no_grad()
def measure_stage(a, spec, k: int, device) -> dict:
    """stage k. 현재 태스크 = n_base−1+k, 과거 = 그 앞 전부 (N = 과거 개수)."""
    run = Path(a.run)
    cur = spec["n_base"] - 1 + k
    past = list(range(cur))
    N, mu, lam = len(past), a.mu, a.lam

    sroot = spec["root"] / f"stage{k}"
    meta = LeRobotDatasetMetadata("local/lotus", root=str(sroot))

    sp = student_ckpt(run, k)
    tp = teacher_ckpt(run, k, a.suite)
    student = load_policy(sp, meta, device)
    ds = V4.load_split(sroot, student.config)
    coords = cache_coords(a, ds, student, device)
    V_S = {i: velocities(student, coords, spec["instr"][i]) for i in past + [cur]}
    del student, ds
    torch.cuda.empty_cache()

    teacher = load_policy(tp, meta, device)
    V_T = {j: velocities(teacher, coords, spec["instr"][j]) for j in past}
    del teacher
    torch.cuda.empty_cache()

    w = [mu] + [lam] * N                       # v_0 = 현재, v_1..v_N = 과거
    W = sum(w)
    acc = {key: 0.0 for key in ("e_fm", "e_anc", "actual", "floor", "pair_form",
                                "G2", "split", "floor_old", "actual_old")}
    per = {"actual": [], "floor": []}
    for bi, c in enumerate(coords):
        v0 = c["target"]
        vs = [v0] + [V_T[j][bi] for j in past]          # v_0 .. v_N
        e_fm = mse(V_S[cur][bi], v0)
        e_anc = sum(mse(V_S[j][bi], V_T[j][bi]) for j in past)
        act = mu * e_fm + lam * e_anc
        ubar = sum(wi * v for wi, v in zip(w, vs)) / W
        fl = sum(wi * mse(v, ubar) for wi, v in zip(w, vs))
        # 식 (8) 의 쌍합 형태로 다시 계산 (부록 A 항등식 확인용)
        pf = (lam / W) * (mu * sum(mse(vs[0], vs[j + 1]) for j in range(N))
                          + lam * sum(mse(vs[i + 1], vs[j + 1])
                                      for i in range(N) for j in range(i + 1, N)))
        acc["e_fm"] += e_fm; acc["e_anc"] += e_anc
        acc["actual"] += act; acc["floor"] += fl; acc["pair_form"] += pf
        acc["G2"] += sum(mse(vs[0], vs[j + 1]) for j in range(N)) / max(N, 1)
        s = 0.0
        for j in past:
            num = (V_S[j][bi] - V_S[cur][bi]).flatten(1).norm(dim=1).mean()
            den = (V_T[j][bi] - v0).flatten(1).norm(dim=1).mean().clamp_min(1e-8)
            s += float(num / den)
        acc["split"] += s / max(N, 1)
        # ── 구 방식(L2_condition_floor.py) 을 같은 좌표에서 함께 계산한다 ──────
        #   floor_old  = ‖V̄ − v_0‖²   V̄ = **student 자신의** 조건별 출력 평균
        #   actual_old = ‖v_S(ℓ_0) − v_0‖²  = L_FM 만
        #   식 (6)/(8) 과 달리 앵커 항이 양쪽 다 없고 v_j 도 teacher 가 아니다.
        #   논문 Table 2 의 27–50% 가 이 정의에서 나온 값인지 대조하기 위한 것이다.
        Vbar_old = sum(V_S[i][bi] for i in past + [cur]) / (N + 1)
        acc["floor_old"] += mse(Vbar_old, v0)
        acc["actual_old"] += e_fm
        per["actual"].append(act); per["floor"].append(fl)

    n = len(coords)
    out = {key: v / n for key, v in acc.items()}
    import statistics as st
    pr = [x / max(y, 1e-12) for x, y in zip(per["actual"], per["floor"])]
    h = len(pr) // 2
    out["ratio"] = out["actual"] / max(out["floor"], 1e-12)
    out["se_ratio"] = st.stdev(pr) / (len(pr) ** 0.5) if len(pr) > 1 else 0.0
    out["ratio_half1"] = sum(per["actual"][:h]) / max(sum(per["floor"][:h]), 1e-12)
    out["ratio_half2"] = sum(per["actual"][h:]) / max(sum(per["floor"][h:]), 1e-12)
    out["frac_below"] = sum(1 for x in pr if x < 1.0) / len(pr)
    out["gap_old_pct"] = 100 * (out["floor_old"] - out["actual_old"]) / max(out["floor_old"], 1e-12)
    out.update(stage=k, cur_task=cur, N=N, mu=mu, lam=lam,
               n_samples=n * a.batch_size, student=str(sp), teacher=str(tp))
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
    out_path = Path(a.out) if a.out else run / "floor_v4.txt"

    rows, t0 = [], time.perf_counter()
    for k in stages:
        if not student_ckpt(run, k).exists():
            print(f"  stage {k}: student 없음 — 건너뜀", flush=True)
            continue
        if not (teacher_ckpt(run, k, a.suite) / "model.safetensors").exists():
            print(f"  stage {k}: teacher 없음 — 건너뜀 ({teacher_ckpt(run,k,a.suite)})",
                  flush=True)
            continue
        try:
            r = measure_stage(a, spec, k, device)
        except Exception as e:
            print(f"  stage {k}: 실패 — {type(e).__name__}: {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        json.dump(rows, out_path.with_suffix(".json").open("w"),
                  indent=2, ensure_ascii=False)
        g = r["floor"] - r["actual"]
        print(f"  stage {k}  t{r['cur_task']}  N={r['N']}  L*={r['floor']:.5f}  "
              f"actual={r['actual']:.5f}  gap={g:+.5f} "
              f"({100*g/max(r['floor'],1e-12):+.1f}%)  ratio={r['ratio']:.2f}  "
              f"split={r['split']:.3f}  [{time.perf_counter()-t0:.0f}s]", flush=True)

    Wd = 112
    L = ["=" * Wd,
         f"조건 붕괴 바닥 L* 대 실측 손실 — LOTUS {a.suite} / v4(TETHER)",
         "=" * Wd, "",
         f"런        {run}",
         f"μ={a.mu}  λ={a.lam}   스왑 성분 = instruction (관측·상태·(x_t,t) 는 현재 스테이지 고정)",
         f"좌표      LOTUS_setting stage{{k}} 의 실제 데이터, "
         f"{a.n_batches}배치 x {a.n_draws}draw x B{a.batch_size}", "",
         "  L(u)   = μ‖u−v_0‖² + λ Σ_j ‖u−v_j‖²                     (6)",
         "  L*     = λ/(μ+λN)·( μ Σ_j‖v_0−v_j‖² + λ Σ_{i<j}‖v_i−v_j‖² )   (8)",
         "  actual = μ‖v_S(ℓ_0)−v_0‖² + λ Σ_j ‖v_S(ℓ_j)−v_j‖²",
         "  v_0 = a−ε (현재 스테이지 실측),  v_j = teacher 가 ℓ_j 조건에서 낸 속도", "",
         "-" * Wd,
         f"{'stage':<7}{'현재':>5}{'N':>4}{'L*':>10}{'actual':>10}{'gap%':>9}{'ratio':>7}"
         f"{'±SE':>7}{'반분1':>7}{'반분2':>7}{'<1':>6}{'L_FM':>9}{'L_anc':>9}{'split':>7}"
         f"{'구gap%':>8}",
         "-" * Wd]
    for r in rows:
        g = 100 * (r["floor"] - r["actual"]) / max(r["floor"], 1e-12)
        L.append(f"{r['stage']:<7}{'t'+str(r['cur_task']):>5}{r['N']:>4}"
                 f"{r['floor']:>10.5f}{r['actual']:>10.5f}{g:>+9.1f}"
                 f"{r['ratio']:>7.2f}{r['se_ratio']:>7.3f}"
                 f"{r['ratio_half1']:>7.2f}{r['ratio_half2']:>7.2f}"
                 f"{100*r['frac_below']:>5.0f}%{r['e_fm']:>9.5f}{r['e_anc']:>9.5f}"
                 f"{r['split']:>7.3f}{r['gap_old_pct']:>+8.1f}")
    if rows:
        d = max(abs(r["floor"] - r["pair_form"]) / max(r["floor"], 1e-12) for r in rows)
        L += ["-" * Wd, "",
              f"항등식 확인: 가중분산 형태와 식 (8) 쌍합 형태의 최대 상대차 {d:.2e}"]
    L += ["",
          "읽는 법",
          "  · gap% > 0 이면 실측이 바닥 아래 — 정책이 instruction 을 읽고 있다는 뜻.",
          "    논문 Table 2 의 '27–50% below the floor' 가 이 열이다.",
          "  · ratio = actual/L*.  <1 이 바닥 아래.",
          "  · ±SE / 반분1·반분2 / <1 은 표본 편향 점검이다. 반분 둘이 크게 다르면",
          "    표본 구간에 의존한다는 뜻이므로 배치를 늘려야 한다.",
          "  · split = mean_j ‖v_S(ℓ_j)−v_S(ℓ_0)‖ / ‖v_j−v_0‖.  0 이면 조건맹.",
          "  · stage1 의 teacher 는 base phase 체크포인트다(과거 6 태스크를 joint 로 배운 것).",
          "",
          "  · 구gap% 는 L2_condition_floor.py 의 옛 정의를 같은 좌표에서 계산한 것이다:",
          "      floor_old  = ‖V̄ − v_0‖²,  V̄ = **student 자신의** 조건별 출력 평균",
          "      actual_old = ‖v_S(ℓ_0) − v_0‖²  (= L_FM 만, 앵커 항 없음)",
          "    식 (6)/(8) 과는 다른 양이다 — 앵커 항이 없고 v_j 가 teacher 가 아니다.",
          "    두 열을 나란히 두면 논문 Table 2 의 수치가 어느 정의에서 나온 것인지 보인다.", ""]
    out_path.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
