#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""H5 — H4의 Fisher 충돌 측정을 **실제 CL 궤적 위에서** 다시 잰다.

왜 H5가 따로 필요한가
    H4는 모든 것을 하나의 고정된 앵커 θ*(사전학습 체크포인트)에서 잰다. 그건 의도된
    설계였다 — 파라미터가 이동한 효과를 섞지 않아야 "겹침"이 태스크 차이 때문이라고
    말할 수 있기 때문이다.
    하지만 그래서 H4는 "CL을 시작하기 전에 이미 충돌이 예정돼 있다"까지만 보인다.
    정작 EWC가 실패하는 곳은 순차 학습이 진행되며 **앵커가 계속 옮겨 다니는** 상황이고,
    그 상황에서도 충돌이 유지되는지는 H4가 답하지 못한다.

    반론이 가능하기 때문이다: "θ*에서는 겹쳤지만, 태스크 0을 배우고 나면 파라미터가
    옮겨가서 태스크 1과는 안 겹칠 수도 있지 않나?"  H5가 그 반론을 닫는다.

H5가 재는 것 — 지표는 H4와 **글자 그대로 동일**하고 앵커만 다르다
    stage k 에서
        앵커 θ*_k = 태스크 k-1까지 학습을 끝낸 파라미터
                    (= E0가 EWC 앵커로 실제 사용한 바로 그 값)
        F_j, g_j  = 태스크 j의 Fisher/그래디언트를 **그 θ*_k 에서** 새로 측정 (j=0..k)
        F_old = Σ_{j<k} F_j,   F_new = F_k
    로 두고 H4의 analyze_pair를 그대로 태운다. 그래서 두 결과가 같은 축 위에 놓인다.

    ★ 지표 함수를 복사하지 않고 H4에서 import한다. 조금이라도 다르면 H4 대 H5 비교가
      성립하지 않기 때문이다. H5가 새로 하는 일은 "어느 파라미터에서 재는가" 하나뿐이다.

    같이 남기는 CL 전용 스칼라 두 개
        anchor_drift          ‖θ*_k − θ_pretrain‖ / ‖θ_pretrain‖
                              앵커가 실제로 얼마나 움직였나. 이게 0에 가까우면
                              H5와 H4가 같아지는 게 당연하므로 반드시 같이 봐야 한다.
        stored_fresh_cosine   E0가 들고 다닌 누적 Fisher(과거 앵커에서 잰 값) 대
                              지금 앵커에서 새로 잰 F_old 의 cosine.
                              1보다 낮을수록 EWC가 **낡은 곡률**을 붙들고 있다는 뜻이다.

읽는 법 (H4와 동일)
    pareto_gain ≈ 0.5 → 좋은 λ가 존재하지 않음(퇴화)      1.0 → λ로 분리 가능
    H4(고정 앵커)와 H5(CL 앵커)가 비슷하면, 충돌은 앵커 위치가 아니라
    태스크 구조 자체에서 오는 것이라는 뜻이다 → EWC로는 못 푼다.

전제
    E0가 만든 CL 체크포인트가 있어야 한다 (--cl_root). 학습은 하지 않는다.
        <cl_root>/task_{k}/checkpoints/last/pretrained_model
        <cl_root>/task_{k}/ewc_state.pt            (있으면 stored_fresh_cosine을 낸다)
    시뮬레이터(gym_libero)는 필요 없다. SR은 E0가 잰다.

사용 예
    python H5.py --cl_root=outputs/E0/.../lam100 --policy.path=<pretrain> --output_dir=... --num_tasks=4
    python H5.py --plot_only --results=... --out=... --h4_results=...
"""

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import torch
import torch.multiprocessing as mp
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, get_safe_torch_device, init_logging

# ★ 지표는 H4에서 그대로 가져온다. 복사본을 두면 두 실험이 조용히 갈라진다.
from lerobot.scripts.H4 import (
    CHUNK,  # noqa: F401  (analyze_pair가 내부에서 쓴다)
    analyze_pair,
    estimate_task_stats,
    global_thresholds,
    load_stats,
    open_task_dataset,
    parse_floats,
    short_run_delta,
    split_episodes,
    compare_delta,
)


@dataclass
class H5Config(TrainPipelineConfig):
    """H4Config와 같은 인자 + CL 체크포인트 경로."""

    # ── 어떤 태스크들을 볼 것인가 (H4와 동일) ────────────────────────────────
    num_tasks: int = 4
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    holdout_episodes: int = 5

    # ── CL 궤적: E0가 만든 체크포인트 트리 ───────────────────────────────────
    # 예: outputs/E0/libero_spatial/seed_42/lam100
    # stage k의 앵커는 task_{k-1}/checkpoints/last/pretrained_model 이다.
    cl_root: str = ""
    compare_stored: bool = True    # <cl_root>/task_{k-1}/ewc_state.pt 와 비교

    # ── [A] Fisher / 그래디언트 추정 (H4와 동일) ─────────────────────────────
    fisher_batches: int = 100
    fisher_batch_size: int = 8
    stats_dir: str = ""            # 비면 output_dir/stats. 캐시 키에 stage가 들어간다.
    recompute: bool = False

    # ── [B] 분석 (H4와 동일한 기본값이어야 비교가 된다) ──────────────────────
    lambdas: str = ("0,1e-4,1e-3,1e-2,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,"
                    "1e4,1e5,1e6,1e7,1e8,inf")
    top_p: str = "0.0001,0.001,0.01,0.05,0.1,0.25"
    curv_damping: float = 1e-3
    subsample: int = 1_000_000
    layer_report: int = 12

    # ── [C] 실측(선택). H4와 달리 CL 앵커에서 출발한다 ───────────────────────
    measure_steps: int = 0
    measure_lambdas: str = "0,10,100,1000"
    measure_top_p: float = 0.01
    measure_stages: str = ""

    # ── 출력 ─────────────────────────────────────────────────────────────────
    run_tag: str = ""
    results_path: str = "outputs/H5/h5_results.jsonl"

    def validate(self):
        """H4Config.validate와 같은 이유로 output_dir 존재 검사만 우회한다(캐시 재사용)."""
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


# ═════════════════════════════════════════════════════════════════════════════
#  CL 앵커 다루기 (H5가 H4에 더하는 유일한 부분)
# ═════════════════════════════════════════════════════════════════════════════
def stage_anchor_ckpt(cfg: H5Config, stage: int) -> Path:
    """stage k의 EWC 앵커 = 태스크 k-1 학습을 끝낸 체크포인트."""
    return Path(cfg.cl_root) / f"task_{stage - 1}" / "checkpoints" / "last" / "pretrained_model"


def stage_stats_path(cfg: H5Config, stage: int, task: int) -> Path:
    """캐시 키에 stage가 들어간다 — 같은 태스크라도 앵커가 다르면 다른 Fisher다."""
    root = Path(cfg.stats_dir) if cfg.stats_dir else Path(cfg.output_dir) / "stats"
    return root / f"h5_stats_stage_{stage}_task_{task}.pt"


def load_policy_at(cfg: H5Config, ckpt: Path, ds_meta, device):
    """주어진 체크포인트의 파라미터로 정책을 만든다. 앵커를 옮기는 지점."""
    if not ckpt.exists():
        raise FileNotFoundError(
            f"CL 체크포인트가 없다: {ckpt}\n"
            f"  --cl_root 가 E0 산출물을 가리키는지 확인해라 "
            f"(예: outputs/E0/libero_spatial/seed_42/lam100)."
        )
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.pretrained_path = ckpt
    pcfg.device = cfg.policy.device
    return make_policy(cfg=pcfg, ds_meta=ds_meta)


def flat_params(policy) -> dict:
    return {n: p.detach().flatten().float().cpu()
            for n, p in policy.named_parameters() if p.requires_grad}


def relative_drift(cur: dict, ref: dict) -> float:
    """‖θ − θ_ref‖ / ‖θ_ref‖ (학습 가능한 파라미터 전체에 대해)."""
    num = den = 0.0
    for n, v in cur.items():
        if n in ref:
            num += float((v.double() - ref[n].double()).pow(2).sum())
            den += float(ref[n].double().pow(2).sum())
    return math.sqrt(num / max(den, 1e-30))


def stored_fisher_cosine(cfg: H5Config, stage: int, fresh_old: dict, names: list[str]) -> float | None:
    """E0가 들고 다닌 누적 Fisher와 지금 앵커에서 새로 잰 F_old의 cosine.

    E0.build_ewc_state는 파라미터 shape 그대로 저장하므로 여기서 평탄화해 맞춘다.
    cosine이라 정규화 상수는 무해하다. None이면 파일이 없다는 뜻(λ=0 팔 등).
    """
    path = Path(cfg.cl_root) / f"task_{stage - 1}" / "ewc_state.pt"
    if not (cfg.compare_stored and path.exists()):
        return None
    try:
        stored = torch.load(path, map_location="cpu", weights_only=False)["fisher"]
    except Exception as e:
        logging.warning(f"[H5] stored fisher 읽기 실패 ({path}): {e}")
        return None
    if not stored:
        return None
    dot = a2 = b2 = 0.0
    for n in names:
        if n not in stored:
            continue
        s = stored[n].flatten().double()
        f = fresh_old[n].double()
        if s.numel() != f.numel():
            return None
        dot += float((s * f).sum())
        a2 += float((s * s).sum())
        b2 += float((f * f).sum())
    return dot / (math.sqrt(a2 * b2) + 1e-30)


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (train.py / H4와 같은 [1]~ 순서)
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: H5Config):
    # ── [1] 설정 ─────────────────────────────────────────────────────────────
    cfg.validate()
    cfg.save_checkpoint = False          # H5도 정책을 저장하지 않는다
    if cfg.measure_steps > 0:
        cfg.steps = cfg.measure_steps
    if not cfg.cl_root:
        raise SystemExit(
            "--cl_root 가 필요하다. H5는 CL 궤적 위에서 재는 실험이라 E0가 만든 "
            "체크포인트 트리를 가리켜야 한다 (예: outputs/E0/libero_spatial/seed_42/lam100)."
        )
    logging.info(pformat(cfg.to_dict()))
    logging.info(colored(
        f"[H5] tasks 0..{cfg.num_tasks - 1}  CL anchors from {cfg.cl_root}", "green", attrs=["bold"]))

    # ── [2] 로거: H4와 같이 스칼라 표만 내므로 wandb를 쓰지 않는다 ────────────

    # ── [3] 재현성 ───────────────────────────────────────────────────────────
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ── [4] 디바이스 ─────────────────────────────────────────────────────────
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── [5] 데이터셋 (정책 생성용 메타) ──────────────────────────────────────
    logging.info("Creating dataset")
    dataset0 = make_dataset(cfg)

    # ── [6] 평가 환경 없음 ───────────────────────────────────────────────────

    # ── [7] 기준 파라미터 θ_pretrain — drift의 분모이자 H4가 쓴 앵커 ──────────
    logging.info("Creating policy (pretrain reference)")
    ref_policy = make_policy(cfg=cfg.policy, ds_meta=dataset0.meta)
    ref_flat = flat_params(ref_policy)
    n_train = sum(v.numel() for v in ref_flat.values())
    logging.info(f"[H5] trainable params: {n_train} ({format_big_number(n_train)})")
    del ref_policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = Path(cfg.results_path)
    results.parent.mkdir(parents=True, exist_ok=True)
    tag = cfg.run_tag or "h5"

    def emit(row: dict):
        row = {"run_tag": tag, "seed": cfg.seed, "cl_root": str(cfg.cl_root), **row}
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")

    lambdas = parse_floats(cfg.lambdas)
    ps = parse_floats(cfg.top_p)
    stages = list(range(1, cfg.num_tasks))

    # ── [A] stage마다: 앵커를 옮기고 그 자리에서 태스크별 Fisher를 잰다 ───────
    # H4와의 유일한 구조적 차이. H4는 이 루프가 없고 앵커가 하나다.
    logging.info(colored("[H5][A] per-stage Fisher at the moving CL anchor", "cyan", attrs=["bold"]))
    drift = {}
    for k in stages:
        ckpt = stage_anchor_ckpt(cfg, k)
        need = [j for j in range(k + 1)
                if cfg.recompute or not stage_stats_path(cfg, k, j).exists()]
        policy = load_policy_at(cfg, ckpt, dataset0.meta, device)
        drift[k] = relative_drift(flat_params(policy), ref_flat)
        logging.info(colored(
            f"[H5] stage k={k}: anchor={ckpt}  drift from pretrain={drift[k]:.4f}", "green"))

        for j in need:
            stats = estimate_task_stats(cfg, policy, j, device)
            path = stage_stats_path(cfg, k, j)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stats, path)
            logging.info(f"[H5] stage {k} task {j}: saved -> {path}")
            del stats
        if not need:
            logging.info(f"[H5] stage {k}: all Fisher cached")

        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── [B1] 마지막 앵커에서의 태스크 쌍 겹침 (그림 (a)용) ────────────────────
    last = stages[-1]
    logging.info(colored(
        f"[H5][B1] pairwise Fisher overlap at the k={last} anchor", "cyan", attrs=["bold"]))
    fl = [load_stats(stage_stats_path(cfg, last, j)) for j in range(last + 1)]
    names = sorted(fl[0]["fisher"].keys())
    for i in range(len(fl)):
        for j in range(len(fl)):
            if i == j:
                continue
            res, _ = analyze_pair(cfg, [fl[i]], fl[j], names, lambdas, ps, device)
            emit({"kind": "pair", "anchor_stage": last, "old": i, "new": j, **res})
    del fl

    # ── [B2] 순차 학습 그대로: 각 stage의 앵커에서 F_old=Σ_{j<k}F_j 대 F_k ────
    logging.info(colored("[H5][B2] sequential stages at their own CL anchors", "cyan", attrs=["bold"]))
    for k in stages:
        files = [load_stats(stage_stats_path(cfg, k, j)) for j in range(k + 1)]
        names = sorted(files[0]["fisher"].keys())

        # EWC가 들고 다닌 낡은 Fisher와 지금 곡률이 얼마나 어긋났나
        fresh_old = {}
        for n in names:
            v = files[0]["fisher"][n].clone()
            for f in files[1:k]:
                v += f["fisher"][n]
            fresh_old[n] = v
        cos_stored = stored_fisher_cosine(cfg, k, fresh_old, names)
        del fresh_old

        res, layers = analyze_pair(cfg, files[:k], files[k], names, lambdas, ps, device)
        emit({"kind": "stage", "stage": k, "old": list(range(k)), "new": k,
              "anchor_drift": drift[k], "stored_fresh_cosine": cos_stored, **res})
        big = sorted(layers.items(), key=lambda kv: -kv[1]["numel"])[: cfg.layer_report]
        for g, r in big:
            emit({"kind": "layer", "stage": k, "group": g, **r})

        pi = ps.index(0.01) if 0.01 in ps else 0
        logging.info(
            f"[H5] stage k={k}  drift={drift[k]:.4f}  cos={res['cosine']:.3f}  "
            f"BC={res['bhattacharyya']:.3f}  lift@{ps[pi]:g}={res['mask_lift'][pi]:.1f}x  "
            f"blocked_gain@{ps[pi]:g}={res['blocked_gain'][pi]:.3f}  "
            f"pareto_gain={res['pareto_gain']:.3f} (degenerate=0.5) at λ={res['pareto_gain_lambda']}  "
            f"degeneracy={res['degeneracy']:.2f}"
            + (f"  stored~fresh cos={cos_stored:.3f}" if cos_stored is not None else ""))
        if res["pareto_gain_at_grid_edge"]:
            logging.warning(colored(
                f"[H5] stage k={k}: λ*={res['pareto_gain_lambda']} 가 격자 끝이다. "
                f"--lambdas 를 넓혀라.", "yellow"))
        del files, layers

    # ── [C] 실측(선택): CL 앵커에서 출발해 λ별로 짧게 학습 ────────────────────
    if cfg.measure_steps > 0:
        logging.info(colored("[H5][C] measured from the CL anchor", "cyan", attrs=["bold"]))
        want = ([int(x) for x in cfg.measure_stages.split(",") if x.strip()] or stages)
        mlams = parse_floats(cfg.measure_lambdas)
        if 0.0 not in mlams:
            mlams = [0.0] + mlams
        mlams = sorted(mlams)

        for k in want:
            files = [load_stats(stage_stats_path(cfg, k, j)) for j in range(k + 1)]
            names = sorted(files[0]["fisher"].keys())
            policy = load_policy_at(cfg, stage_anchor_ckpt(cfg, k), dataset0.meta, device)
            anchor = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}

            # E0가 넘기는 것과 같은 EWC state: 누적 Fisher를 mean=1로, 앵커는 θ*_k.
            fo, tot, s = {}, 0.0, 0.0
            for n in names:
                v = files[0]["fisher"][n].clone()
                for f in files[1:k]:
                    v += f["fisher"][n]
                fo[n] = v
                s += float(v.double().sum())
                tot += v.numel()
            fo = {n: v / max(s / tot, 1e-30) for n, v in fo.items()}

            gen = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)
            rate = min(1.0, cfg.subsample / max(tot, 1))
            tau = global_thresholds(
                torch.cat([
                    v[torch.randint(0, v.numel(), (max(1, int(round(v.numel() * rate))),), generator=gen)]
                    for v in fo.values()
                ]),
                [cfg.measure_top_p],
            )[0]
            shapes = {n: p.shape for n, p in policy.named_parameters() if p.requires_grad}
            ewc_state = {"fisher": {n: v.view(shapes[n]).to(device) for n, v in fo.items()},
                         "anchor": {n: anchor[n] for n in fo}}

            dsk = open_task_dataset(cfg, k)
            train_eps, _ = split_episodes(f"{cfg.dataset_prefix}{k}", None, cfg.holdout_episodes)

            base = None
            for lam in mlams:
                logging.info(f"[H5] measuring k={k} λ={lam:g} ({cfg.measure_steps} steps)")
                run = short_run_delta(cfg, policy, anchor, dsk, train_eps, ewc_state, lam, device)
                if lam == 0.0:
                    base = run["delta"]
                    emit({"kind": "measured", "stage": k, "lambda": 0.0, "steps": cfg.measure_steps,
                          "top_p": cfg.measure_top_p, "mse": run["mse"], "penalty": run["penalty"],
                          "shrink_all": 1.0, "shrink_top": 1.0, "shrink_rest": 1.0,
                          "selectivity": 1.0, "cosine_with_free": 1.0})
                    continue
                cmp = compare_delta(base, run["delta"], fo, tau, device)
                emit({"kind": "measured", "stage": k,
                      "lambda": "inf" if math.isinf(lam) else lam, "steps": cfg.measure_steps,
                      "top_p": cfg.measure_top_p, "mse": run["mse"], "penalty": run["penalty"], **cmp})
                logging.info(f"[H5]   λ={lam:g}  shrink top={cmp['shrink_top']:.3f} "
                             f"rest={cmp['shrink_rest']:.3f}  selectivity={cmp['selectivity']:.2f}")
                del run
            del ewc_state, fo, base, dsk, policy, anchor, files
            if device.type == "cuda":
                torch.cuda.empty_cache()

    logging.info(colored(f"[H5] done -> {results}", "green", attrs=["bold"]))


# ═════════════════════════════════════════════════════════════════════════════
#  그림 (H4와 같은 6패널 + H4 겹쳐 그리기)
# ═════════════════════════════════════════════════════════════════════════════
def plot_h5(results_path: str, out_path: str, h4_results: str = "") -> None:
    rows = [json.loads(x) for x in Path(results_path).read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit(f"no rows in {results_path}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    stages = [r for r in rows if r["kind"] == "stage"]
    pairs = [r for r in rows if r["kind"] == "pair"]
    layers = [r for r in rows if r["kind"] == "layer"]
    measured = [r for r in rows if r["kind"] == "measured"]

    h4_stages = []
    if h4_results and Path(h4_results).exists():
        h4_stages = [json.loads(x) for x in Path(h4_results).read_text().splitlines() if x.strip()]
        h4_stages = [r for r in h4_stages if r.get("kind") == "stage"]

    # ── 요약 CSV ─────────────────────────────────────────────────────────────
    csv_path = str(Path(out_path).with_suffix(".csv"))
    keys = ["stage", "anchor_drift", "stored_fresh_cosine", "cosine", "bhattacharyya",
            "pareto_gain", "degeneracy", "pareto_gain_lambda", "pareto_gain_at_grid_edge",
            "ratio_log_std", "ratio_gmean", "plasticity_at_best", "forgetting_at_best",
            "gain_free", "damage_free"]
    h4_by_stage = {r["stage"]: r for r in h4_stages}
    with open(csv_path, "w") as f:
        f.write(",".join(keys + ["blocked_gain@1%", "mask_lift@1%", "h4_pareto_gain"]) + "\n")
        for r in sorted(stages, key=lambda r: r["stage"]):
            i = r["top_p"].index(0.01) if 0.01 in r["top_p"] else 0
            h4g = h4_by_stage.get(r["stage"], {}).get("pareto_gain", "")
            f.write(",".join("" if r.get(k) is None else str(r.get(k, "")) for k in keys)
                    + f",{r['blocked_gain'][i]},{r['mask_lift'][i]},{h4g}\n")
    print(f"saved table  -> {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략 (pip install matplotlib 후 --plot_only 다시)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    (a, b, c), (d, e, g) = axes

    # (a) 마지막 CL 앵커에서의 태스크 쌍 겹침 ─────────────────────────────────
    tasks = sorted({r["old"] for r in pairs} | {r["new"] for r in pairs})
    if tasks:
        M = np.full((len(tasks), len(tasks)), np.nan)
        for r in pairs:
            M[tasks.index(r["old"]), tasks.index(r["new"])] = r["bhattacharyya"]
        for i in range(len(tasks)):
            M[i, i] = 1.0
        im = a.imshow(M, cmap="magma", vmin=0, vmax=1)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       color="w" if M[i, j] < 0.6 else "k", fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046)
        anch = pairs[0].get("anchor_stage", "?")
        a.set(xticks=range(len(tasks)), yticks=range(len(tasks)),
              xticklabels=tasks, yticklabels=tasks, xlabel="task j", ylabel="task i",
              title=f"(a) Fisher overlap at the CL anchor (k={anch})\n"
                    "1.0 = identical important-parameter subspace")

    # (b) 상위 p% 마스크 교집합 / chance ──────────────────────────────────────
    for r in sorted(stages, key=lambda r: r["stage"]):
        pts = [(p, v) for p, v in zip(r["top_p"], r["mask_lift"], strict=True) if v > 0]
        if pts:
            b.plot(*zip(*pts), "-o", ms=4, label=f"k={r['stage']}")
    b.axhline(1.0, color="gray", ls="--", lw=1)
    b.text(0.02, 0.06, "chance (independent subspaces)", transform=b.transAxes,
           color="gray", fontsize=9)
    b.set(xscale="log", yscale="log", xlabel="top-p fraction of old-task Fisher",
          ylabel="overlap / chance  (lift)",
          title="(b) do the top-Fisher coordinates coincide?  (CL anchors)")
    b.grid(alpha=0.3, which="both")
    if b.get_legend_handles_labels()[0]:
        b.legend(fontsize=9, title="stage")

    # (c) ★ λ 파레토 — H4(고정 앵커)와 겹쳐 그린다 ────────────────────────────
    f_ref = np.linspace(0, 1, 200)
    c.plot(f_ref, 2 * np.sqrt(f_ref) - f_ref, "k--", lw=1.5,
           label="degenerate ($F_{old}\\propto F_{new}$)")
    c.plot([0, 0, 1], [0, 1, 1], color="gray", ls=":", lw=1.5, label="ideal (disjoint)")
    ordered = sorted(stages, key=lambda r: r["stage"])
    for r in ordered:
        (ln,) = c.plot(r["forgetting"], r["plasticity"], "-o", ms=5, label=f"H5 k={r['stage']}")
        h4r = h4_by_stage.get(r["stage"])
        if h4r:
            c.plot(h4r["forgetting"], h4r["plasticity"], ":x", ms=4, lw=1.2,
                   color=ln.get_color(), alpha=0.75, label=f"H4 k={r['stage']} (fixed)")
    c.set(xlabel="forgetting  (old-task damage, rel. to λ=0)",
          ylabel="plasticity  (new-task gain, rel. to λ=0)", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
          title="(c) Pareto of $\\lambda$ on the real CL trajectory\n"
                "solid = H5 (moving anchor), dotted = H4 (fixed anchor)")
    c.grid(alpha=0.3)
    c.legend(fontsize=7, loc="lower right", ncol=2)

    # (d) 새 태스크 update 중 옛 태스크 상위 좌표에 갇힌 몫 ────────────────────
    for r in sorted(stages, key=lambda r: r["stage"]):
        d.plot(r["top_p"], r["blocked_gain"], "-o", ms=4, label=f"k={r['stage']}")
    d.plot(f_ref, f_ref, color="gray", ls="--", lw=1)
    d.text(0.35, 0.22, "chance", transform=d.transAxes, color="gray", fontsize=9, rotation=32)
    d.set(xscale="log", xlabel="top-p fraction of old-task Fisher", ylim=(0, 1.02),
          ylabel="fraction of new-task learnable gain inside",
          title="(d) how much of the new task lives on protected weights")
    d.grid(alpha=0.3, which="both")
    if d.get_legend_handles_labels()[0]:
        d.legend(fontsize=9, title="stage")

    # (e) ★ H5의 본체: 앵커가 움직여도 퇴화가 유지되는가 ──────────────────────
    st = [r["stage"] for r in ordered]
    e.plot(st, [r["pareto_gain"] for r in ordered], "-o", ms=7, lw=2, label="H5 (CL anchor)")
    ticks = set(st)
    if h4_stages:
        h4o = sorted(h4_stages, key=lambda r: r["stage"])
        e.plot([r["stage"] for r in h4o], [r["pareto_gain"] for r in h4o], ":x", ms=7, lw=1.8,
               label="H4 (fixed anchor)")
        # H4가 H5보다 많은 스테이지를 가질 수 있다(--num_tasks가 다르면). 눈금은 합집합으로.
        ticks |= {r["stage"] for r in h4o}
    e.axhline(0.5, color="crimson", ls="--", lw=1.5)
    e.text(0.02, 0.5, " degenerate (no good $\\lambda$)", transform=e.get_yaxis_transform(),
           color="crimson", fontsize=9, va="bottom")
    e.axhline(1.0, color="seagreen", ls="--", lw=1.5)
    e.text(0.02, 1.0, " separable", transform=e.get_yaxis_transform(),
           color="seagreen", fontsize=9, va="top")
    e2 = e.twinx()
    e2.plot(st, [r.get("anchor_drift", 0.0) for r in ordered], "-s", ms=5, color="0.45", alpha=0.8)
    e2.set_ylabel("anchor drift  $\\|\\theta^*_k-\\theta_{pre}\\|/\\|\\theta_{pre}\\|$", color="0.45")
    e2.tick_params(axis="y", labelcolor="0.45")
    e.set(xlabel="CL stage k", ylabel="pareto gain", ylim=(0.4, 1.05), xticks=sorted(ticks),
          title="(e) does the conflict survive the CL trajectory?\n"
                "gray = how far the anchor actually moved")
    e.grid(alpha=0.3)
    e.legend(fontsize=8, loc="upper right")

    # (f) 실측이 있으면 실측, 없으면 EWC가 든 Fisher의 낡음 + 레이어별 ─────────
    if measured:
        def lam_x(v):
            if v == "inf" or (isinstance(v, float) and math.isinf(v)):
                return 1e5
            return 1e-3 if float(v) == 0.0 else float(v)

        for stg in sorted({r["stage"] for r in measured}):
            sub = sorted([r for r in measured if r["stage"] == stg], key=lambda r: lam_x(r["lambda"]))
            xs = [lam_x(r["lambda"]) for r in sub]
            (ln,) = g.plot(xs, [r["shrink_top"] for r in sub], "-o", ms=4,
                           label=f"k={stg} top-{sub[0]['top_p']:.0%}")
            g.plot(xs, [r["shrink_rest"] for r in sub], "--s", ms=4, color=ln.get_color(),
                   label=f"k={stg} rest")
        g.set(xscale="log", xlabel="EWC lambda",
              ylabel="$\\|\\Delta\\theta_\\lambda\\|/\\|\\Delta\\theta_0\\|$", ylim=(-0.03, 1.10),
              title="(f) measured from the CL anchor\nsolid = protected coords, dashed = rest")
        g.grid(alpha=0.3, which="both")
        g.legend(fontsize=7, ncol=2)
    elif any(r.get("stored_fresh_cosine") is not None for r in ordered):
        # EWC가 실제로 들고 다닌 Fisher가 현재 곡률과 얼마나 어긋났는가.
        # 1에서 멀어질수록 "낡은 지도를 보고 지키고 있다"는 뜻이다.
        vals = [(r["stage"], r["stored_fresh_cosine"]) for r in ordered
                if r.get("stored_fresh_cosine") is not None]
        g.plot(*zip(*vals), "-o", ms=7, lw=2, color="#c44e52")
        g.axhline(1.0, color="gray", ls="--", lw=1)
        g.text(0.02, 1.0, " EWC's Fisher is still current", transform=g.get_yaxis_transform(),
               color="gray", fontsize=9, va="top")
        g.set(xlabel="CL stage k", ylabel="cosine(stored $F_{old}$, fresh $F_{old}$)",
              ylim=(0, 1.05), xticks=[v[0] for v in vals],
              title="(f) is EWC protecting with a stale curvature map?")
        g.grid(alpha=0.3)
    elif layers:
        stg = max(r["stage"] for r in layers)
        sub = sorted([r for r in layers if r["stage"] == stg], key=lambda r: r["pareto_gain"])
        nm = [r["group"].replace("model.", "") for r in sub]
        g.barh(range(len(sub)), [r["pareto_gain"] for r in sub], color="#4c72b0")
        g.axvline(0.5, color="crimson", ls="--", lw=1.5)
        g.set(yticks=range(len(sub)), xlabel="pareto gain", xlim=(0, 1),
              title=f"(f) per-layer, stage k={stg}")
        g.set_yticklabels(nm, fontsize=7)
        g.grid(alpha=0.3, axis="x")

    gains = [r["pareto_gain"] for r in stages]
    drifts = [r.get("anchor_drift", 0.0) for r in stages]
    if gains:
        head = (f"H5 Fisher conflict on the real CL trajectory — pareto gain "
                f"{min(gains):.2f}..{max(gains):.2f}  (0.50 = degenerate, 1.00 = separable), "
                f"anchor drift up to {max(drifts):.3f}")
    else:
        head = "H5 Fisher conflict on the real CL trajectory"
    fig.suptitle(head, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out_path}")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        init_logging()
        plot_h5(kv.get("results", "outputs/H5/h5_results.jsonl"),
                kv.get("out", "outputs/H5/H5_fisher_conflict_cl.png"),
                kv.get("h4_results", ""))
    else:
        mp.set_start_method("spawn", force=True)
        init_logging()
        main()
