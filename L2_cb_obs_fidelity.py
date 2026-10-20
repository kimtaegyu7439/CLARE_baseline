#!/usr/bin/env python
"""코드북이 뽑는 õ 가 **그 상태의 진짜 관측**과 얼마나 같은가.

    python L2_cb_obs_fidelity.py --run results/clare_v4_goal_seed7 --suite libero_goal
    python L2_cb_obs_fidelity.py --run ... --tasks 0,3,9 --n_frames 2000

────────────────────────────────────────────────────────────────────────────
무엇을 재는가
────────────────────────────────────────────────────────────────────────────
데모 프레임은 (s, o) 가 **짝**이다. s 는 proprioceptive state (16-d), o 는 그
프레임의 DINOv2 CLS 특징 (3072-d). 그러니 "상태 s 를 줬을 때 코드북이 내놓는
관측"을 그 프레임의 **정답 o** 와 직접 맞대 볼 수 있다. 이게 앵커의 전제
( s̃ 에 맞는 õ 가 나온다 )가 실제로 성립하는지에 대한 가장 직접적인 검사다.

예측 경로 (sample_codebook_hard 와 **같은 식**, 잡음만 뺀 조건부 평균)

    z   = (s − mean_s) / std_s                 표준화
    k*  = argmin_k ‖z − c_k‖                   경성 배정 (k-means 규칙)
    ô₁  = m_k* + A_k*(z − z̄_k*)                1차 (v4)
    ô₀  = m_k*                                 0차 (v1) — A 를 안 쓰는 옛 방식
    õ   = ô₁ + σ_o,k* ⊙ ε                      샘플러가 실제로 내보내는 것

비교 기준선
    global   전체 프레임의 o 평균                (아무 정보도 안 쓴 경우)
    shuffle  같은 태스크의 **다른** 프레임 o     (우연 수준)
    NN       z 공간 최근접 **실제** 프레임의 o   (사실상 상한)

지표
    cos      코사인 유사도 (높을수록 좋음)
    relL2    ‖ô − o‖ / ‖o − ō‖  (낮을수록 좋음. 1.0 = global 평균과 같은 수준)
    R²       1 − SS_res/SS_tot  (차원 전체 합산)

★ held-out 으로 잰다. 코드북은 저수지 8000 쌍으로 만들어졌으므로, 여기서는
  그 태스크의 프레임을 새로 뽑아 쓴다. 완전한 분리는 아니지만(같은 데모 풀),
  적어도 코드북 적합에 쓰인 바로 그 샘플은 아니다. --holdout_frac 로 조절한다.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

import L2_V4_seq as B1

REPO = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="codebooks/ 가 있는 결과 폴더 (예: results/clare_v4_goal_seed7)")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--tasks", default="all", help="all 또는 0,3,9")
    p.add_argument("--n_frames", type=int, default=3000, help="태스크당 평가 프레임 수")
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps_per_task", type=int, default=10000)   # cfg 조립용(안 쓰임)
    p.add_argument("--eval_episodes", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--eval_seed", type=int, default=7)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out", default=None, help="기본: <run>/obs_fidelity.txt")
    return p.parse_args()


# ── 지표 ────────────────────────────────────────────────────────────────────
def scores(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """pred/true: (N, D).  cos / relL2 / R² 를 돌려준다."""
    mu = true.mean(0, keepdim=True)
    res = pred - true
    ss_res = res.pow(2).sum()
    ss_tot = (true - mu).pow(2).sum()
    cos = torch.nn.functional.cosine_similarity(pred, true, dim=1)
    # relL2 는 "global 평균을 쓸 때 대비" 라 1.0 이 기준선이다
    rel = res.norm(dim=1) / (true - mu).norm(dim=1).clamp_min(1e-9)
    return {"cos": float(cos.mean()), "cos_med": float(cos.median()),
            "relL2": float(rel.mean()), "R2": float(1 - ss_res / ss_tot.clamp_min(1e-12)),
            # 분포 그림용 원시값. 요약만 보면 '평균은 높은데 꼬리가 길다' 를 놓친다.
            "cos_raw": cos.cpu().numpy().astype("float32")}


@torch.no_grad()
def collect(policy, cfg, n_want: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(s, o) 짝을 n_want 개 모은다. B1 의 수집 경로를 그대로 쓴다."""
    from lerobot.datasets.factory import make_dataset
    ds = make_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        shuffle=True, drop_last=False,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None)
    S, O = [], []
    got = 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        batch = B1.prep_batch(policy, batch)
        cls = B1.rgb_cls(policy, batch)                      # (B*4, 768)
        s = batch["observation.state"].detach().flatten(1).float()   # (B,16)
        o = cls.detach().reshape(s.shape[0], -1).float()             # (B,3072)
        S.append(s.cpu()); O.append(o.cpu())
        got += s.shape[0]
        if got >= n_want:
            break
    del ds
    return torch.cat(S)[:n_want], torch.cat(O)[:n_want]


def evaluate(cb: dict, s: torch.Tensor, o: torch.Tensor, gen) -> dict:
    """코드북 cb 로 s 에서 o 를 예측하고 기준선과 비교한다."""
    z = (s - cb["mean_s"]) / cb["std_s"]                     # (N,16)
    k = torch.cdist(z, cb["c"]).argmin(1)                    # (N,) 경성 배정
    zc = z - cb["zbar"][k]                                   # (N,16) 셀 중심 기준 편차

    o0 = cb["m"][k]                                          # (N,3072) 0차
    # A: (Ke,3072,16) -> 배정된 셀만 모아 배치 matmul
    o1 = o0 + torch.einsum("ndk,nk->nd", cb["A"][k], zc)     # (N,3072) 1차
    # ── 0차 생성 모델: 셀 평균 + 셀 **원** 대각 공분산 ───────────────────────
    # 코드북의 sig_o 는 Ridge 1차 항을 뺀 **잔차** 표준편차로 덮어써져 있다
    # (L2_V4_seq.py:1173). 셀 안 o 의 원래 산포는 저장돼 있지 않으므로, 평가
    # 프레임을 같은 규칙으로 셀에 배정해 셀별·차원별 표준편차를 직접 추정한다.
    # 표본이 2 개 미만인 셀은 추정이 불가능하므로 전역 표준편차로 되돌린다.
    Ke = cb["m"].shape[0]
    sig_raw = o.std(0, unbiased=False).expand(Ke, -1).clone()   # (Ke,3072) 기본값=전역
    for kk in torch.unique(k):
        sel = o[k == kk]
        if sel.shape[0] >= 2:
            sig_raw[kk] = sel.std(0, unbiased=False)
    o0g = o0 + torch.randn(o1.shape, generator=gen) * sig_raw[k]   # (N,3072)

    noise = torch.randn(o1.shape, generator=gen) * cb["sig_o"][k]
    osamp = o1 + noise                                       # (N,3072) 조건부 가우시안(참고)

    gmean = o.mean(0, keepdim=True).expand_as(o)             # 기준선: 전역 평균
    perm = torch.randperm(o.shape[0], generator=gen)
    oshuf = o[perm]                                          # 기준선: 다른 프레임
    # 기준선: z 공간 최근접 **실제** 프레임(자기 자신 제외)
    d = torch.cdist(z, z); d.fill_diagonal_(float("inf"))
    onn = o[d.argmin(1)]

    return {
        "1차 ô₁ (v4)":  scores(o1, o),
        "0차 ô₀ (v1)":  scores(o0, o),
        "셀평균+셀분산":  scores(o0g, o),
        "샘플 õ":        scores(osamp, o),
        "global 평균":   scores(gmean, o),
        "shuffle":       scores(oshuf, o),
        "NN(실제)":      scores(onn, o),
    }


def main() -> int:
    a = parse()
    B1.init_logging()
    run = Path(a.run)
    cbdir = run / "codebooks"
    if not cbdir.exists():
        raise SystemExit(f"코드북 폴더가 없다: {cbdir}")
    out = Path(a.out) if a.out else run / "obs_fidelity.txt"
    device = B1.get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)

    files = sorted(cbdir.glob("codebook_task*.pt"),
                   key=lambda p: int(p.stem.split("task")[-1]))
    want = ([int(x) for x in a.tasks.split(",")] if a.tasks != "all"
            else [int(f.stem.split("task")[-1]) for f in files])

    # 정책은 o(=DINOv2 CLS)를 뽑는 데만 쓴다. 어떤 체크포인트든 백본은 동결이라
    # 같은 특징이 나온다 — task 0 체크포인트를 쓴다.
    cfg0 = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    ck_root = Path(cfg0.get("ckpt_root", "")) if cfg0.get("ckpt_root") else None
    pol_path = None
    if ck_root:
        cand = ck_root / f"{a.suite}_seed{cfg0.get('seed', 7)}_{B1.MODE}" / "task_0" / "checkpoints" / "last" / "pretrained_model"
        if (cand / "model.safetensors").exists():
            pol_path = str(cand)
    if pol_path is None:
        import os
        pol_path = os.environ.get("PRETRAIN_PATH",
                                  "/home/sa090180/Models/dit_flow_mt_libero_90_pretrain")
    logging.info(f"[fid] 특징 추출용 정책: {pol_path}")

    L: list[str] = ["=" * 92,
                    f"코드북 관측 재현도 — {run.name}  ({a.suite})",
                    "=" * 92, "",
                    "  데모의 (s, o) 짝을 정답으로 두고, 상태 s 에서 코드북이 내놓는 관측을 맞대본다.",
                    "    ô₁ = m_k + A_k(z − z̄_k)   1차 (v4 가 실제로 쓰는 조건부 평균)",
                    "    ô₀ = m_k                   0차 (A 를 안 쓰던 옛 방식)",
                    "    õ  = ô₁ + σ_o,k ⊙ ε        샘플러가 내보내는 값(잡음 포함)",
                    "",
                    "  cos↑ 코사인 유사도   relL2↓ ‖ô−o‖/‖o−ō‖ (1.0 = 전역평균 수준)   R²↑",
                    f"  태스크당 {a.n_frames} 프레임", ""]

    from lerobot.policies.factory import make_policy
    from lerobot.datasets.factory import make_dataset

    summary = {}
    raw: dict = {}
    for t in want:
        f = cbdir / f"codebook_task{t}.pt"
        if not f.exists():
            L.append(f"[task {t}] 코드북 없음"); continue
        cb = torch.load(f, map_location="cpu", weights_only=False)
        # ★ 코드북은 half 로 저장돼 있을 수 있다(용량 때문). 지표 계산은 전부
        #   float32 로 한다 — half 로 3072 차원을 더하면 누적 오차가 지표를 흔든다.
        cb = {kk: (vv.float() if torch.is_tensor(vv) and vv.is_floating_point() else vv)
              for kk, vv in cb.items()}
        cfg = B1.build_cfg(a, t, pol_path, run)
        ds = make_dataset(cfg)
        policy = make_policy(cfg=cfg.policy, ds_meta=ds.meta); policy.to(device); policy.eval()
        del ds
        s, o = collect(policy, cfg, a.n_frames, device)
        del policy
        torch.cuda.empty_cache()

        gen = torch.Generator().manual_seed(a.seed + t)
        res = evaluate(cb, s, o, gen)
        summary[t] = res
        for name, v in res.items():
            raw[f"t{t}|{name}"] = v["cos_raw"]

        L.append("-" * 92)
        L.append(f"task {t}   K_eff={cb['K_eff']}  프레임 {len(s)}")
        L.append("-" * 92)
        L.append(f"  {'예측':<16}{'cos↑':>9}{'cos중앙':>9}{'relL2↓':>9}{'R²↑':>9}")
        for name, v in res.items():
            L.append(f"  {name:<16}{v['cos']:>9.4f}{v['cos_med']:>9.4f}"
                     f"{v['relL2']:>9.4f}{v['R2']:>9.4f}")
        L.append("")

    if summary:
        L.append("=" * 92); L.append("태스크 평균"); L.append("=" * 92)
        L.append(f"  {'예측':<16}{'cos↑':>9}{'relL2↓':>9}{'R²↑':>9}")
        for name in next(iter(summary.values())):
            cs = [summary[t][name]["cos"] for t in summary]
            rl = [summary[t][name]["relL2"] for t in summary]
            r2 = [summary[t][name]["R2"] for t in summary]
            L.append(f"  {name:<16}{sum(cs)/len(cs):>9.4f}"
                     f"{sum(rl)/len(rl):>9.4f}{sum(r2)/len(r2):>9.4f}")
        L += ["",
              "읽을 때",
              "  · 1차가 0차보다 좋아야 A_k 가 일을 한 것이다. 둘이 같으면 A 는 군더더기다.",
              "  · relL2 가 1.0 을 넘으면 **전역 평균보다 나쁘다** — 그 경우 코드북이",
              "    상태를 관측으로 옮기지 못한다는 뜻이고, 앵커의 전제가 흔들린다.",
              "  · shuffle 이 우연 수준, NN(실제) 이 사실상 상한이다. 1차가 그 사이",
              "    어디쯤인지가 핵심이다.",
              "  · 샘플 õ 는 1차에 σ_o 잡음을 더한 것이라 항상 1차보다 나쁘다.",
              "    그 격차가 크면 σ_o 가 과하다는 신호다.",
              ""]

    if raw:
        import numpy as np
        npz = out.with_name(out.stem + "_cos.npz")
        np.savez_compressed(npz, suite=np.str_(a.suite), run=np.str_(run.name), **raw)
        L.append(f"  원시 cos 분포 -> {npz.name}")
        print(f"원시 cos 저장 -> {npz}")

    txt = "\n".join(L)
    out.write_text(txt)
    print(txt)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
