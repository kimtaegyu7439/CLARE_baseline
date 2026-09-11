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

"""R10 — 블록별 "조건 기여의 크기 대 task 쏠림" 지도.

무엇을 묻는가
    R9는 각 DiT 블록의 조건 기여 Δ의 **방향 분리**가 joint와 CL에서 거의 같다는 것을
    보였다(라우팅은 안 죽는다). 그러면 CL의 행동은 왜 조건과 무관하게 마지막 task로
    붕괴하는가? R10은 그 무효화 지점을 한 장으로 본다.

        밝기 = ρ    이 블록에서 조건 무관 성분이 조건 기여를 얼마나 압도하는가
        색   = lean 그 조건 무관 성분이 어느 task의 계산을 닮았는가

분해
    각 프로브·블록ℓ·서브블록에서 두 조건의 기여를 반으로 가른다.

        delta(ℓ) = ½( Δ(ℓ, c₀) − Δ(ℓ, c₁) )      조건 대비 성분
        vbase(ℓ) = ½( Δ(ℓ, c₀) + Δ(ℓ, c₁) )      조건 **평균** 기여

    ★ 함정 1 — vbase는 "조건 무관 성분"이 아니다.
      residual stream은 이미 조건을 거쳐 왔으므로 순수한 무조건 성분은 정의할 수 없다.
      vbase는 정확히 "두 조건에 걸친 평균 기여"다. 이 그림의 색은 그러므로
      "이 블록의 **조건-평균** 기여가 어느 task로 쏠렸는가"로만 읽어야 한다.
      캡션도 그렇게 쓴다. 절대 "무조건 성분"이라고 부르지 않는다.

밝기 = ρ
        rho(ℓ,t,p) = ‖vbase(ℓ)‖ / ‖delta(ℓ)‖          (토큰별로 재고 평균)

    ρ가 크다 = 조건을 바꿔도 안 변하는 부분이 조건이 만드는 차이를 압도한다.
    ★ ρ만 보면 "delta가 작아서 큰 것"과 "vbase가 커서 큰 것"을 못 가른다. 그래서
      ‖vbase‖와 ‖delta‖를 분자·분모로 따로 그린다(별도 그림 R10_rho.png).
    ★ ‖delta‖가 floor 미만이면 ρ가 정의되지 않는다 → NaN. 0으로 채우지 않는다.

색 = lean (task 쏠림)
        u0(ℓ) = normalize( mean_p Δ^{FT0}(ℓ, c₀) )     FT0에 task0 조건을 넣은 기여
        u1(ℓ) = normalize( mean_p Δ^{FT1}(ℓ, c₁) )     FT1에 task1 조건을 넣은 기여

        lean(ℓ,t,p) = cos(vbase(ℓ), u1(ℓ)) − cos(vbase(ℓ), u0(ℓ))
                      >0 → task1(마지막) 쏠림 = 빨강 · <0 → task0 = 파랑 · 0 = 흰색

    ★ 함정 2-a — u는 반드시 **조건이 들어간 채의** 기여여야 한다.
      FT의 조건평균 ½(Δ(c₀)+Δ(c₁))을 쓰면 안 된다. 그건 정의상 조건을 지운 성분이라
      "task 방향"이라 부를 근거가 없고, FT0에서 그게 task0을 가리키는 이유가 "FT0는
      뭘 넣든 task0를 한다"일 수 있어 순환 논증이 된다.
      FT의 delta ½(Δ(c₀)−Δ(c₁))를 써도 안 된다. 그건 "task를 하는 방향"이 아니라
      "두 조건을 가르는 구분축"이라 lean의 해석이 불명확해진다.

    ★ 함정 2-b — cos이 의미를 가지려면 세 좌표계가 동시에 같아야 한다.
      (1) 같은 hidden 공간: u와 vbase는 **같은 블록·같은 서브블록**에서. 블록마다
          residual 스케일과 축 의미가 다르므로 블록3의 vbase를 블록5의 u와 재면 무의미.
          여기서는 (ℓ, sub, flow time t, 관측)까지 전부 맞춘 u를 쓴다.
      (2) 같은 정규화 통계: 다섯 체크포인트가 같은 자를 써야 한다. CL은 마지막 task
          통계로, FT0는 task0 통계로 재정규화돼 있을 수 있고, 어긋나면 lean 전체가
          정규화 아티팩트가 된다. assert_shared_norm을 **가장 먼저** 돌린다.
      (3) 같은 프로브 지점: x₀뿐 아니라 a_tgt까지. R7/R8/R9와 공유하는 고정물 해시로 확인.

    측정의 비대칭은 의도된 것이다 — 재는 대상은 CL의 **조건 평균**(vbase), 기준은 FT의
    **조건부** 계산(u). 묻는 질문이 정확히 "조건 없이 굴러가는 계산이 어느 task의
    조건부 계산을 닮았나"이기 때문이다.

    ★ cos(u0, u1)을 블록별로 반드시 함께 본다. 두 기준이 애초에 비슷하면(LIBERO-Spatial은
      task가 유사해 실제로 그럴 수 있다) lean의 분모가 좁아 해석이 약해진다. 1에 가까운
      블록은 표와 캡션에 표시하고 그 블록의 lean은 약한 증거로 취급한다.

읽는 규칙 (밝기만으로 판정하지 말 것)
        옅음 (ρ↓)                  조건 기여가 눌리지 않음                건강
        진하고 흰색 (ρ↑, lean≈0)   조건은 무시하나 특정 task로 밀지 않음   무해
        진하고 빨강 (ρ↑, lean>0)   조건 무시 + 마지막 task로 밂           붕괴 유력
        진하고 파랑 (ρ↑, lean<0)   무관 성분이 옛 task로 쏠림             가설과 반대

    가설은 "CL에 진한 빨강이 나오고 joint에는 안 나온다"를 예측한다. CL의 진한 칸이
    대부분 흰색으로 나오면 그건 가설의 반증이며, 무효화 지점이 블록 기여가 아니라
    최종 판독(W)에 있다는 신호다(R9가 실제로 그 방향을 시사했다). 어느 쪽이든 그대로 그린다.

    ★ 흰색의 중의성: 이 결합 지도에서 흰 칸은 "lean≈0"일 수도 "ρ가 작아 투명"일 수도
      있다. 그래서 ρ 단독 지도를 별도 그림(R10_rho.png)으로 반드시 함께 낸다.

R9에서 그대로 물려받는 것 (재구현하지 않는다)
    프로브 고정물, capture_obs_traj(데모 재생·짝 생존), SubBlockTap과 residual 항등식,
    exec_range→tok 슬라이스, NaN 안전 누적, 팔레트·_style·타임라인 패널.

사용 예
    bash bash/E0/R10.sh
    PLOT_ONLY=1 bash bash/E0/R10.sh
"""

import hashlib
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.scripts.R3 import stage_ckpt
from lerobot.scripts.R7 import (
    assert_shared_norm,
    demo_chunks,
    exec_range,
    load_policy_at,
    minmax_normalize,
    norm_stats,
    obs_to_cond,
    task_text,
)
from lerobot.scripts.R8 import DIV_HI, DIV_LO, DIV_MID, GRID, INK, INK2, MODEL_COLORS, _style
from lerobot.scripts.R9 import (
    ALIVE,
    FINALS,  # noqa: F401  (R9와 같은 규약을 쓰는지 확인용)
    SUB_TITLE,
    SUBS,
    R9Config,
    SubBlockTap,
    _digest,
    _draw_timeline,
    _maybe_log,
    _nanmean,
    _shade_dead,
    capture_obs_traj,
)

# 재는 모델(=vbase를 분해할 대상). 기준(u)을 주는 FT0/FT1은 따로 다룬다.
MEASURED = ("pretrain", "joint", "cl")


@dataclass
class R10Config(R9Config):
    """R9Config를 그대로 상속한다 — 프로브 규약이 한 글자도 다르면 R9와 나란히 못 읽는다."""

    out_root: str = "outputs/R10"
    # task 기준 방향 u를 주는 체크포인트. ft0는 보통 순차 트리의 task_a 단계와 같다
    # (task A만 배운 상태 = FT0). ft1은 task B만 사전학습에서 바로 배운 것이어야 한다.
    ft0_ckpt: str = ""            # 비우면 stage_ckpt(ckpt_root, task_a)
    # ft1_ckpt 는 R8Config에서 상속

    # ‖delta‖가 이 값 미만이면 ρ가 정의되지 않는다 -> NaN (0으로 채우지 않는다)
    delta_floor: float = 1e-8
    # 캡션의 "붕괴 유력 블록 수"를 세는 규칙. 손으로 쓰지 않고 데이터에서 센다.
    rho_hi: float = 2.0           # ρ가 이 이상이면 "무관 성분이 압도"
    lean_hi: float = 0.05         # lean이 이 이상이면 "task1 쏠림"
    # 밝기 매핑: ρ가 이 값이면 완전 불투명. 캡션에 기록된다.
    rho_opaque: float = 4.0


def cache_name(cfg: R10Config) -> str:
    return "R10_full.npz" if cfg.cond_mode == "full" else f"R10_lang_obs{cfg.obs_task}.npz"


def model_specs(cfg: R10Config) -> tuple[list[dict], dict]:
    """(재는 모델 3개, 기준을 주는 FT 2개)."""
    a, b = cfg.task_a, cfg.task_b
    ft0 = cfg.ft0_ckpt or str(stage_ckpt(cfg.ckpt_root, a))
    table = {
        "pretrain": {"ckpt": cfg.pretrain_ckpt, "title": "pretrained  (before either task)"},
        "joint": {"ckpt": cfg.joint_ckpt, "title": f"joint  (task {a} + {b} mixed)"},
        "cl": {"ckpt": str(stage_ckpt(cfg.ckpt_root, b)), "title": f"CL  (task {a} → task {b})"},
    }
    specs = []
    for key in [k.strip() for k in cfg.models.split(",") if k.strip()]:
        if key not in table:
            raise SystemExit(f"[R10] 모르는 모델: {key!r} (가능: {list(table)})")
        if not table[key]["ckpt"]:
            raise SystemExit(f"[R10] {key} 체크포인트 경로가 비어 있다.")
        specs.append({"key": key, **table[key]})
    basis = {"ft0": {"ckpt": ft0, "cond": 0, "title": f"FT{a}  (task {a} only)"},
             "ft1": {"ckpt": cfg.ft1_ckpt, "cond": 1, "title": f"FT{b}  (task {b} only)"}}
    for k, v in basis.items():
        if not v["ckpt"]:
            raise SystemExit(f"[R10] {k} 체크포인트가 필요하다 (--ft0_ckpt / --ft1_ckpt).")
    return specs, basis


# ═════════════════════════════════════════════════════════════════════════════
#  측정
# ═════════════════════════════════════════════════════════════════════════════
def split_contrib(d0: torch.Tensor, d1: torch.Tensor, tok: slice
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(Δ(c₀), Δ(c₁)) -> (vbase, delta). 실행 구간 토큰만 남긴다.

    vbase + delta == Δ(c₀),  vbase − delta == Δ(c₁)  가 항등적으로 성립한다(§4 검사).
    """
    a, b = d0[tok].double(), d1[tok].double()
    return 0.5 * (a + b), 0.5 * (a - b)


def rho_lean(vbase: torch.Tensor, delta: torch.Tensor,
             u0: torch.Tensor, u1: torch.Tensor, floor: float
             ) -> tuple[float, float, float, float, int, int]:
    """(ρ, lean, ‖vbase‖, ‖delta‖, 유효칸, 전체칸).

    ★ 각도·비율을 (토큰, 프로브)마다 **먼저** 스칼라로 접고 그 스칼라를 평균한다.
      벡터를 먼저 평균하면 방향이 제각각인 것끼리 상쇄되어 왜곡된다.
    """
    nv = vbase.norm(dim=-1)                       # (Tk, B)
    nd = delta.norm(dim=-1)
    ok = nd > floor                               # delta가 죽으면 비율이 정의되지 않는다
    rho = torch.where(ok, nv / nd.clamp_min(1e-30), torch.full_like(nv, float("nan")))
    # u는 (Tk, H) — 토큰축을 유지한 채 브로드캐스트한다(토큰마다 다른 기준).
    c1 = (vbase * u1.unsqueeze(1)).sum(-1) / nv.clamp_min(1e-30)
    c0 = (vbase * u0.unsqueeze(1)).sum(-1) / nv.clamp_min(1e-30)
    lean = c1 - c0
    n_ok = int(ok.sum())
    return (float(rho[ok].mean()) if n_ok else float("nan"),
            float(lean.mean()), float(nv.mean()), float(nd.mean()), n_ok, int(ok.numel()))


@torch.no_grad()
def run_probe(cfg: R10Config, run_dir: Path) -> Path:
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    a, b = cfg.task_a, cfg.task_b
    specs, basis = model_specs(cfg)

    meta_a = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{a}")
    ref = load_policy_at(cfg, specs[0]["ckpt"], meta_a, device)
    pol_cfg = ref.config
    stats = norm_stats(ref)
    horizon = int(pol_cfg.horizon)
    e0, e1 = exec_range(pol_cfg, cfg.exec_slice)
    tok = slice(e0, e1)
    logging.info(colored(f"[R10] 실행 구간 = 청크 index {e0}..{e1 - 1}", "green"))

    text = {0: task_text(cfg.dataset_prefix, a), 1: task_text(cfg.dataset_prefix, b)}

    # ── [1] 프로브 고정물 — R8/R9와 **바이트 단위로 같아야** 한다 ────────────
    rng = np.random.default_rng(cfg.probe_seed)
    chunks = np.concatenate([
        minmax_normalize(demo_chunks(cfg, pol_cfg, a, cfg.demo_episodes), stats),
        minmax_normalize(demo_chunks(cfg, pol_cfg, b, cfg.demo_episodes), stats)])
    pick = rng.choice(len(chunks), size=cfg.num_probe, replace=False)
    a_tgt = torch.from_numpy(chunks[pick]).float().to(device)
    gen = torch.Generator(device="cpu").manual_seed(cfg.probe_seed)
    x0 = torch.randn(cfg.num_probe, horizon, 7, generator=gen).to(device)
    t_grid = np.linspace(0.0, float(cfg.t_max), cfg.t_steps, dtype=np.float32)
    hash_x0, hash_a = _digest(x0), _digest(a_tgt)
    logging.info(f"[R10] 프로브 {cfg.num_probe} × t {cfg.t_steps}  x₀#{hash_x0} a#{hash_a}")
    logging.info(colored("[R10] 위 해시를 R9 로그와 대조해라 — 다르면 두 실험이 서로 다른 "
                         "지점을 본 것이다.", "cyan"))

    # ── [2] 관측 (R9와 같은 규약) ────────────────────────────────────────────
    obs_steps = np.arange(0, cfg.rollout_steps + 1, cfg.obs_stride, dtype=int)
    n_step = len(obs_steps)
    status, ep_info = {}, {}
    if cfg.cond_mode == "full":
        obs_sets = {}
        for ci, task in ((0, a), (1, b)):
            obs_sets[ci], status[ci], ep_info[ci] = capture_obs_traj(
                cfg, task, cfg.num_obs, obs_steps, None, device)
    else:
        fixed, st, inf = capture_obs_traj(cfg, cfg.obs_task, cfg.num_obs, obs_steps, None, device)
        obs_sets, status, ep_info = {0: fixed, 1: fixed}, {0: st, 1: st}, {0: inf, 1: inf}
    pair_used = (status[0] == ALIVE) & (status[1] == ALIVE)
    n_live = pair_used.sum(axis=0)
    logging.info("[R10] 물리시간별 유효 짝: " +
                 "  ".join(f"{int(s)}:{int(c)}" for s, c in zip(obs_steps, n_live)))
    if not n_live.any():
        raise SystemExit("[R10] 살아있는 짝이 없다. --rollout_steps 를 줄여라.")
    del ref
    torch.cuda.empty_cache()

    # ── [3] 정규화 통계 일치 — 가장 먼저 (§4) ────────────────────────────────
    # 어긋난 채로 진행하면 lean 전체가 정규화 아티팩트가 된다.
    for key, ck in [(s["key"], s["ckpt"]) for s in specs] + \
                   [(k, v["ckpt"]) for k, v in basis.items()]:
        pol = load_policy_at(cfg, ck, meta_a, device)
        assert_shared_norm(stats, norm_stats(pol), key)
        del pol
        torch.cuda.empty_cache()
    logging.info(colored("[R10] 다섯 체크포인트 정규화 통계 일치 확인", "green"))

    n_block = None

    # ── [4] task 기준 방향 u0/u1 — FT에서 1회 계산·동결 ──────────────────────
    # u0는 FT0에 **c₀를**, u1은 FT1에 **c₁을** 넣은 조건부 기여다(함정 2-a).
    # 조건평균도 delta도 쓰지 않는다.
    U = {}          # (bkey, oi, si, ti, li, sub) -> (Tk, H) 정규화된 기준 벡터
    for bkey, spec in basis.items():
        ci = spec["cond"]
        pol = load_policy_at(cfg, spec["ckpt"], meta_a, device)
        net = pol.dit_flow.velocity_net
        tap = SubBlockTap(net)
        n_block = tap.n_blocks
        logging.info(colored(f"[R10] 기준 {bkey}: {spec['ckpt']}  (조건 c{ci} 사용)", "cyan"))
        for oi in range(cfg.num_obs):
            for si in range(n_step):
                if not pair_used[oi, si]:
                    continue
                cond = obs_to_cond(pol, obs_sets[ci][oi][si], text[ci], device)
                for ti, t in enumerate(t_grid):
                    tf = float(t)
                    xt = (1.0 - tf) * x0 + tf * a_tgt
                    tt = torch.full((cfg.num_probe,), tf, device=device)
                    net(xt, tt, cond.expand(cfg.num_probe, -1))
                    snap = tap.snapshot()
                    for li in range(n_block):
                        for s in SUBS:
                            v = snap[(li, s)][tok].double().mean(dim=1)     # 프로브 평균 (Tk,H)
                            U[(bkey, oi, si, ti, li, s)] = v / v.norm(dim=-1, keepdim=True
                                                                     ).clamp_min(1e-30)
        tap.remove()
        del pol
        torch.cuda.empty_cache()
    logging.info(f"[R10] 기준 벡터 {len(U)}개 동결")

    # 기준끼리 얼마나 닮았나 — 1에 가까우면 lean의 분해능이 낮다 (§2.3)
    u_sim = np.full((n_step, n_block, cfg.t_steps, len(SUBS)), np.nan)
    for (bk, oi, si, ti, li, s) in list(U):
        if bk != "ft0":
            continue
        k1 = ("ft1", oi, si, ti, li, s)
        if k1 in U:
            c = float((U[("ft0", oi, si, ti, li, s)] * U[k1]).sum(-1).mean())
            j = SUBS.index(s)
            u_sim[si, li, ti, j] = c if np.isnan(u_sim[si, li, ti, j]) else \
                0.5 * (u_sim[si, li, ti, j] + c)

    # ── [5] 재는 모델 3개 ────────────────────────────────────────────────────
    blob: dict[str, np.ndarray] = {}
    diag: dict[str, dict] = {}
    shape = (n_step, n_block, cfg.t_steps)
    for spec in specs:
        pol = load_policy_at(cfg, spec["ckpt"], meta_a, device)
        net = pol.dit_flow.velocity_net
        tap = SubBlockTap(net)
        logging.info(colored(f"[R10] {spec['key']}: {spec['ckpt']}", "cyan", attrs=["bold"]))

        acc = {q: {s: np.zeros(shape) for s in SUBS}
               for q in ("rho", "lean", "nv", "nd", "cnt", "rcnt")}
        ok_t = all_t = 0
        first = True
        for oi in range(cfg.num_obs):
            for si in range(n_step):
                if not pair_used[oi, si]:
                    continue
                cond = {ci: obs_to_cond(pol, obs_sets[ci][oi][si], text[ci], device)
                        for ci in (0, 1)}
                for ti, t in enumerate(t_grid):
                    tf = float(t)
                    xt = (1.0 - tf) * x0 + tf * a_tgt
                    tt = torch.full((cfg.num_probe,), tf, device=device)
                    tap.check = first
                    d = {}
                    for ci in (0, 1):
                        net(xt, tt, cond[ci].expand(cfg.num_probe, -1))
                        d[ci] = tap.snapshot()
                        if first and ci == 0:
                            err = tap.verify_residual()
                            logging.info(f"[R10]   residual 항등식 {err:.2e}")
                            tap.check = False
                    first = False
                    for li in range(n_block):
                        for s in SUBS:
                            vb, dl = split_contrib(d[0][(li, s)], d[1][(li, s)], tok)
                            if li == 0 and s == "attn" and ti == 0:
                                # §4 분해 항등식
                                assert torch.allclose(vb + dl, d[0][(li, s)][tok].double(),
                                                      atol=1e-6), "vbase+delta ≠ Δ(c₀)"
                                assert torch.allclose(vb - dl, d[1][(li, s)][tok].double(),
                                                      atol=1e-6), "vbase−delta ≠ Δ(c₁)"
                            r, ln, nv, nd, nok, nall = rho_lean(
                                vb, dl, U[("ft0", oi, si, ti, li, s)],
                                U[("ft1", oi, si, ti, li, s)], cfg.delta_floor)
                            j = (si, li, ti)
                            if not np.isnan(r):
                                acc["rho"][s][j] += r
                                acc["rcnt"][s][j] += 1
                            acc["lean"][s][j] += ln
                            acc["nv"][s][j] += nv
                            acc["nd"][s][j] += nd
                            acc["cnt"][s][j] += 1
                            ok_t += nok
                            all_t += nall
        def _m(num, cnt):
            return np.where(cnt > 0, num / np.maximum(cnt, 1.0), np.nan)
        for s in SUBS:
            k = f"{spec['key']}_{s}"
            blob[f"{k}_rho"] = _m(acc["rho"][s], acc["rcnt"][s]).astype(np.float32)
            blob[f"{k}_lean"] = _m(acc["lean"][s], acc["cnt"][s]).astype(np.float32)
            blob[f"{k}_nvbase"] = _m(acc["nv"][s], acc["cnt"][s]).astype(np.float32)
            blob[f"{k}_ndelta"] = _m(acc["nd"][s], acc["cnt"][s]).astype(np.float32)
            blob[f"{k}_rho_n"] = acc["rcnt"][s].astype(np.int32)
        excl = 1.0 - ok_t / max(all_t, 1)
        diag[spec["key"]] = {"delta_excluded_frac": excl, "ckpt": spec["ckpt"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for s in SUBS:
                k = f"{spec['key']}_{s}"
                logging.info(
                    f"[R10]   {s:>4}: ⟨ρ⟩ {np.nanmean(blob[k + '_rho']):8.3f}  "
                    f"⟨lean⟩ {np.nanmean(blob[k + '_lean']):+.3f}  "
                    f"⟨‖vbase‖⟩ {np.nanmean(blob[k + '_nvbase']):.3e}  "
                    f"⟨‖delta‖⟩ {np.nanmean(blob[k + '_ndelta']):.3e}")
        tap.remove()
        del pol
        torch.cuda.empty_cache()

    blob["t_grid"] = t_grid
    blob["obs_steps"] = obs_steps.astype(np.int32)
    blob["obs_pair_alive"] = pair_used.astype(bool)
    blob["obs_pair_used"] = pair_used.astype(bool)
    blob["obs_status_c0"] = status[0].astype(np.int8)
    blob["obs_status_c1"] = status[1].astype(np.int8)
    blob["u_sim"] = u_sim.astype(np.float32)          # (S, L, T, 2) FT 기준끼리의 cos
    blob["meta"] = np.array(json.dumps({
        "task_a": a, "task_b": b, "text_c0": text[0], "text_c1": text[1],
        "n_block": n_block, "subs": list(SUBS), "num_probe": cfg.num_probe,
        "probe_seed": cfg.probe_seed, "t_steps": cfg.t_steps, "t_max": cfg.t_max,
        "num_obs": cfg.num_obs, "obs_task": cfg.obs_task, "cond_mode": cfg.cond_mode,
        "exec_slice": [e0, e1], "demo_episodes": cfg.demo_episodes,
        "rollout_steps": cfg.rollout_steps, "obs_stride": cfg.obs_stride,
        "delta_floor": cfg.delta_floor, "rho_hi": cfg.rho_hi, "lean_hi": cfg.lean_hi,
        "rho_opaque": cfg.rho_opaque,
        "vbase_def": "0.5*(delta(c0)+delta(c1)) — 조건 평균 기여이지 무조건 성분이 아니다",
        "u_def": "u0 = normalize(mean_p delta^FT0(l, c0)) · u1 = normalize(mean_p delta^FT1(l, c1))"
                 "  — 조건부 기여. 조건평균도 delta도 쓰지 않았다.",
        "hash_x0": hash_x0, "hash_a_tgt": hash_a,
        "hash_obs_c0": _digest(obs_sets[0]), "hash_obs_c1": _digest(obs_sets[1]),
        "episodes": {"c0": ep_info[0], "c1": ep_info[1]},
        "live_pairs_by_step": [int(v) for v in n_live],
        "obs_status_legend": {"0": "alive", "1": "held", "2": "frozen"},
        "exclude_dead_obs": True, "diagnostics": diag,
        "specs": [{k: s[k] for k in ("key", "ckpt", "title")} for s in specs],
        "basis": {k: {"ckpt": v["ckpt"], "title": v["title"], "cond": v["cond"]}
                  for k, v in basis.items()},
    }))
    cache = run_dir / cache_name(cfg)
    np.savez_compressed(cache, **blob)
    logging.info(colored(f"[R10] saved -> {cache}", "green", attrs=["bold"]))
    return cache




# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def _prepare(cache: Path) -> dict:
    """npz 하나 -> 그림이 쓸 모든 것. ρ 정규화와 색 스케일은 전 모델이 **공유**한다."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    z = {k: v for k, v in np.load(cache, allow_pickle=False).items()}
    m = json.loads(str(z["meta"]))
    keys = [s["key"] for s in m["specs"]]
    q = {}
    for name in ("rho", "lean", "nvbase", "ndelta"):
        q[name] = {(k, s): z[f"{k}_{s}_{name}"].astype(np.float64)
                   for k in keys for s in SUBS}
    # 색: lean. 0이 중심(흰색)인 발산형. 낮은 쪽(task0)이 파랑, 높은 쪽(task1)이 빨강.
    cmap = LinearSegmentedColormap.from_list("lean", [DIV_HI, DIV_MID, DIV_LO])
    cmap.set_bad("#eceae5")
    # ★ 색 스케일은 **히트맵이 실제로 그리는 값**(물리시간 평균)에서 잡는다.
    #   전체 (S,L,T) 분포로 잡으면 스텝별 변동까지 범위에 들어가 요약본이 2배 이상
    #   옅어진다. 스텝별 판은 이 범위를 넘는 칸이 포화되지만, 세 모델을 나란히 읽는
    #   것이 목적이므로 하나의 스케일을 공유하는 쪽이 맞다(캡션에 명시).
    disp = np.concatenate([_nanmean(v, axis=0).ravel() for v in q["lean"].values()])
    disp = disp[np.isfinite(disp)]
    lim = max(float(np.percentile(np.abs(disp), 99)), 0.02)
    return {
        "cache": cache, "z": z, "m": m, "keys": keys,
        "titles": {s["key"]: s["title"] for s in m["specs"]},
        "short": {s["key"]: s["title"].split("  ")[0] for s in m["specs"]},
        "t_grid": z["t_grid"], "obs_steps": z["obs_steps"],
        "L": m["n_block"], "nt": len(z["t_grid"]), "ns": len(z["obs_steps"]),
        "q": q, "u_sim": z["u_sim"].astype(np.float64),
        "pair_used": z["obs_pair_used"].astype(bool),
        "n_live": z["obs_pair_used"].astype(bool).sum(axis=0),
        "dead_frac": 1.0 - z["obs_pair_used"].astype(bool).mean(axis=0),
        "n_ep": int(z["obs_pair_used"].shape[0]),
        "cmap": cmap, "norm": TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        "rho_opaque": m["rho_opaque"], "rho_hi": m["rho_hi"], "lean_hi": m["lean_hi"],
    }


def _slice(ctx: dict, si: int | None) -> dict:
    pick = (lambda d: {k: _nanmean(v, axis=0) for k, v in d.items()}) if si is None \
        else (lambda d: {k: v[si] for k, v in d.items()})
    out = {n: pick(ctx["q"][n]) for n in ctx["q"]}
    us = ctx["u_sim"]
    out["u_sim"] = _nanmean(us, axis=0) if si is None else us[si]     # (L, T, 2)
    return out


def _draw(ctx: dict, si: int | None, out: Path) -> Path:
    """결합 지도. 색 = lean(task 쏠림), 밝기 = ρ(조건 평균 성분의 지배도)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    m, keys, titles, short = ctx["m"], ctx["keys"], ctx["titles"], ctx["short"]
    t_grid, obs_steps, L, nt = ctx["t_grid"], ctx["obs_steps"], ctx["L"], ctx["nt"]
    cmap, norm = ctx["cmap"], ctx["norm"]
    r_op, r_hi, l_hi = ctx["rho_opaque"], ctx["rho_hi"], ctx["lean_hi"]
    d = _slice(ctx, si)
    a, b = m["task_a"], m["task_b"]
    summary = si is None
    lim = float(norm.vmax)
    n = len(keys)
    rows = np.arange(1, L + 1)

    # ── 붕괴 유력 블록 수: 손으로 쓰지 않고 센다 ────────────────────────────
    rho_l = {ks: _nanmean(d["rho"][ks], axis=1) for ks in d["rho"]}
    lean_l = {ks: _nanmean(d["lean"][ks], axis=1) for ks in d["lean"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hot = {ks: (rho_l[ks] >= r_hi) & (lean_l[ks] >= l_hi) for ks in rho_l}
    n_hot = {ks: int(hot[ks].sum()) for ks in hot}

    nrow = 4 if summary else 3
    ratios = [1.0, 1.0, 0.66, 0.62][:nrow]
    fig_h = 15.6 if summary else 13.0
    fig = plt.figure(figsize=(4.1 * n + 3.4, fig_h))
    hot_txt = " ;   ".join(
        f"{short[k]}: " + " · ".join(f"{s} {n_hot[(k, s)]}/{L}" for s in SUBS) for k in keys)
    usim = float(_nanmean(d["u_sim"]))
    caption = (
        f"R10: where does the conditional contribution get drowned out, and by what?   "
        f"task {a} vs task {b}\n"
        f"Each sub-block's contribution is split into  vbase = ½(Δ(c₀)+Δ(c₁))  and  "
        f"delta = ½(Δ(c₀)−Δ(c₁)).\n"
        f"vbase is the CONDITION-AVERAGED contribution, not an unconditional component — the "
        f"residual stream has already seen the condition.\n"
        f"Opacity = ρ = ‖vbase‖/‖delta‖, fully opaque at ρ ≥ {r_op:g}: how far the "
        f"condition-averaged part outweighs what the condition changes.\n"
        f"Colour = cos(vbase, u₁) − cos(vbase, u₀), where u₀/u₁ are the CONDITIONAL "
        f"contributions of FT{a}/FT{b} at the same block, sub-block, t and observation\n"
        f"(u₀ = FT{a} given task-{a} conditioning, u₁ = FT{b} given task-{b}; neither a "
        f"condition-average nor a delta, which would make the reference circular).\n"
        f"red = leans toward task {b} (the last one learned) · blue = toward task {a} · "
        f"white = neutral. Opacity spans 0.55–1 only, so ρ is a soft cue here;\n"
        f"the authoritative ρ read is the companion figure <name>_rho.png. The colour scale "
        f"(±{lim:.2f}) is set from the values this map actually draws, so a vivid cell means "
        f"the scale is tight, not that the effect is large.\n"
        f"Blocks that are both base-dominated (ρ ≥ {r_hi:g}) and task-{b} leaning "
        f"(lean ≥ {l_hi:g}):   {hot_txt}.\n"
        f"Mean cos(u₀,u₁) = {usim:+.2f}; where that approaches 1 the two references nearly "
        f"coincide and lean has little room to resolve anything.\n"
        f"This figure localises where the conditional contribution is outweighed and which "
        f"task the surviving part resembles — it does not claim that causes forgetting."
    )
    n_line = caption.count("\n") + 1
    top = min(0.86, max(0.60, 0.995 - n_line * 10 * 1.45 / 72 / fig_h - 0.045))
    fig.suptitle(caption, fontsize=10, color=INK, y=0.995, linespacing=1.45)
    gs = fig.add_gridspec(nrow, n + 2, width_ratios=[1] * n + [0.075, 1.05],
                          height_ratios=ratios, hspace=0.40, wspace=0.20,
                          left=0.088, right=0.978, top=top, bottom=0.062)

    # ── a~f: 결합 히트맵 ────────────────────────────────────────────────────
    for ri, sub in enumerate(SUBS):
        for ci, k in enumerate(keys):
            ax = fig.add_subplot(gs[ri, ci])
            ln, rh = d["lean"][(k, sub)], d["rho"][(k, sub)]
            bad = ~np.isfinite(ln)
            rgba = cmap(norm(np.where(bad, np.nan, ln)))
            rgba[bad] = to_rgb("#eceae5") + (1.0,)
            # ρ가 클수록 진하게. 다만 0..1 전 구간을 쓰면 ρ 중앙값(≈2)에서 알파가 0.5가
            # 되어 색 대비가 통째로 반감된다. 0.55~1.0 구간만 쓰면 ρ는 여전히 읽히면서
            # lean의 대비는 살아남는다. ρ의 정본은 R10_*_rho.png(불투명)다.
            al = 0.55 + 0.45 * np.clip(np.nan_to_num(rh) / r_op, 0.0, 1.0)
            rgba[..., 3] = np.where(bad, 1.0, al)
            ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest",
                      extent=(-0.5, nt - 0.5, 0.5, L + 0.5))
            tix = np.unique(np.linspace(0, nt - 1, 5).round().astype(int))
            ax.set_xticks(tix)
            ax.set_xticklabels([f"{t_grid[i]:.2f}" for i in tix])
            ax.set_yticks(rows)
            ax.set_yticklabels([f"block {i}" for i in rows] if ci == 0 else [""] * L, fontsize=8)
            ax.tick_params(colors=INK2, labelsize=8, length=3)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if ri == len(SUBS) - 1:
                ax.set_xlabel("flow time  t", color=INK2, fontsize=8.5)
            tag = "abcdef"[ri * n + ci]
            ax.set_title(f"{tag}   {titles[k]}" if ri == 0 else f"{tag}   {short[k]}",
                         fontsize=10, color=INK, pad=8, loc="left")
            if ci == 0:
                ax.set_ylabel(SUB_TITLE[sub].upper(), color=INK, fontsize=9.5,
                              fontweight="bold", labelpad=8)

    cax = fig.add_subplot(gs[0:2, n])
    pos = cax.get_position()
    w, x = pos.width * 0.42, pos.x0 - 0.012
    h_c = pos.height * 0.62
    cax.set_position([x, pos.y0 + pos.height - h_c, w, h_c])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("cos(vbase,u₁) − cos(vbase,u₀)", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    cb.ax.text(0.5, 1.02, f"leans task {b}\n(last learned)", transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=6.5, color=DIV_LO, linespacing=1.2)
    cb.ax.text(0.5, -0.02, f"leans task {a}", transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=6.5, color=DIV_HI, linespacing=1.2)
    cb.outline.set_edgecolor(GRID)

    # ── g/h: depth 단면 (ρ 실선, lean 점선 이중축) ──────────────────────────
    for ri, sub in enumerate(SUBS):
        ax = fig.add_subplot(gs[ri, n + 1])
        ax2 = ax.twiny()
        for ki, k in enumerate(keys):
            ax.plot(rho_l[(k, sub)], rows, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4,
                    zorder=4)
            ax2.plot(lean_l[(k, sub)], rows, color=MODEL_COLORS[k], lw=1.3, ls="--",
                     alpha=0.85, zorder=3)
            r = max(0, L - 1 - ki)
            if np.isfinite(rho_l[(k, sub)][r]):
                ax.annotate(short[k], (rho_l[(k, sub)][r], rows[r]),
                            textcoords="offset points", xytext=(5, 3), fontsize=8,
                            color=MODEL_COLORS[k], fontweight="bold")
        ax.axvline(r_hi, color=INK2, lw=0.9, ls=":", zorder=2)
        ax2.axvline(0.0, color=INK2, lw=0.7, ls=":", alpha=0.6, zorder=2)
        _style(ax)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7, axis="x")
        ax.set_ylim(0.5, L + 0.5)
        ax.set_yticks(rows)
        ax.set_yticklabels([f"block {i}" for i in rows], fontsize=8)
        ax.set_xlabel(f"ρ = ‖vbase‖/‖delta‖   ({sub}, solid)", color=INK2, fontsize=8.5)
        ax2.set_xlabel("lean   (dashed)", color=INK2, fontsize=8)
        ax2.tick_params(colors=INK2, labelsize=7.5)
        ax.set_title(f"{'gh'[ri]}   depth — ρ (solid) and lean (dashed), {sub}",
                     fontsize=10, color=INK, pad=34, loc="left")

    # ── i/j: flow-time 단면 · (요약본) k/l: 물리시간 단면 ───────────────────
    row2 = gs[2, :].subgridspec(1, 4, wspace=0.30)
    for i, sub in enumerate(SUBS):
        ax = fig.add_subplot(row2[0, i])
        ax2 = ax.twinx()
        for k in keys:
            ax.plot(t_grid, _nanmean(d["rho"][(k, sub)], axis=0), color=MODEL_COLORS[k],
                    lw=2.0, marker="o", ms=3.5, zorder=4)
            ax2.plot(t_grid, _nanmean(d["lean"][(k, sub)], axis=0), color=MODEL_COLORS[k],
                     lw=1.3, ls="--", alpha=0.85, zorder=3)
        ax.axhline(r_hi, color=INK2, lw=0.9, ls=":", zorder=2)
        _style(ax)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.set_xlabel("flow time  t", color=INK2, fontsize=8.5)
        ax.set_ylabel(f"ρ  ({sub}, solid)", color=INK2, fontsize=8.5)
        ax2.set_ylabel("lean (dashed)", color=INK2, fontsize=8)
        ax2.tick_params(colors=INK2, labelsize=7.5)
        ax.set_title(f"{'ij'[i]}   flow time — {sub}", fontsize=10, color=INK, pad=8,
                     loc="left")
    if summary:
        for i, sub in enumerate(SUBS):
            ax = fig.add_subplot(row2[0, 2 + i])
            ax2 = ax.twinx()
            _shade_dead(ax, obs_steps, ctx["dead_frac"])
            for k in keys:
                ax.plot(obs_steps, _nanmean(ctx["q"]["rho"][(k, sub)], axis=(1, 2)),
                        color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4, zorder=4)
                ax2.plot(obs_steps, _nanmean(ctx["q"]["lean"][(k, sub)], axis=(1, 2)),
                         color=MODEL_COLORS[k], lw=1.3, ls="--", alpha=0.85, zorder=3)
            ax.axhline(r_hi, color=INK2, lw=0.9, ls=":", zorder=2)
            _style(ax)
            ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
            ax.set_xticks(obs_steps)
            ax.set_xlabel("rollout step  (physical time)", color=INK2, fontsize=8.5)
            ax.set_ylabel(f"ρ  ({sub}, solid)", color=INK2, fontsize=8.5)
            ax2.set_ylabel("lean (dashed)", color=INK2, fontsize=8)
            ax2.tick_params(colors=INK2, labelsize=7.5)
            ax.set_title(f"{'kl'[i]}   physical time — {sub}", fontsize=10, color=INK,
                         pad=8, loc="left")

    _draw_timeline(fig.add_subplot(gs[nrow - 1, :]), ctx, si)
    handles = [Line2D([0], [0], color=MODEL_COLORS[k], lw=2.4, marker="o", ms=5,
                      label=titles[k]) for k in keys]
    handles += [Line2D([0], [0], color=INK2, lw=2.0, label="ρ  (solid)"),
                Line2D([0], [0], color=INK2, lw=1.3, ls="--", label="lean  (dashed)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.004))
    fig.savefig(out, dpi=170, facecolor="white")
    if summary:
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


def _draw_rho(ctx: dict, out: Path) -> Path:
    """ρ 단독 지도 + 분자/분모. 결합 지도의 흰 칸 중의성을 여기서 해소한다."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    m, keys, titles, short = ctx["m"], ctx["keys"], ctx["titles"], ctx["short"]
    t_grid, obs_steps, L, nt = ctx["t_grid"], ctx["obs_steps"], ctx["L"], ctx["nt"]
    d = _slice(ctx, None)
    rows = np.arange(1, L + 1)
    n = len(keys)
    seq = LinearSegmentedColormap.from_list("rho", ["#ffffff", "#2a78d6", "#0b0b0b"])
    seq.set_bad("#eceae5")
    flat = np.concatenate([v.ravel() for v in d["rho"].values()])
    flat = flat[np.isfinite(flat)]
    vmax = float(np.percentile(flat, 99))
    nrm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    fig = plt.figure(figsize=(4.1 * n + 2.2, 14.4))
    gs = fig.add_gridspec(4, n + 1, width_ratios=[1] * n + [0.09],
                          height_ratios=[1.0, 1.0, 0.7, 0.7], hspace=0.44, wspace=0.20,
                          left=0.085, right=0.955, top=0.855, bottom=0.055)
    im = None
    for ri, sub in enumerate(SUBS):
        for ci, k in enumerate(keys):
            ax = fig.add_subplot(gs[ri, ci])
            im = ax.imshow(np.ma.masked_invalid(d["rho"][(k, sub)]), aspect="auto",
                           origin="lower", cmap=seq, norm=nrm, interpolation="nearest",
                           extent=(-0.5, nt - 0.5, 0.5, L + 0.5))
            tix = np.unique(np.linspace(0, nt - 1, 5).round().astype(int))
            ax.set_xticks(tix)
            ax.set_xticklabels([f"{t_grid[i]:.2f}" for i in tix])
            ax.set_yticks(rows)
            ax.set_yticklabels([f"block {i}" for i in rows] if ci == 0 else [""] * L, fontsize=8)
            ax.tick_params(colors=INK2, labelsize=8, length=3)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if ri == len(SUBS) - 1:
                ax.set_xlabel("flow time  t", color=INK2, fontsize=8.5)
            ax.set_title(f"{'abcdef'[ri * n + ci]}   {titles[k] if ri == 0 else short[k]}",
                         fontsize=10, color=INK, pad=8, loc="left")
            if ci == 0:
                ax.set_ylabel(SUB_TITLE[sub].upper(), color=INK, fontsize=9.5,
                              fontweight="bold", labelpad=8)
    cax = fig.add_subplot(gs[0:2, n])
    cb = fig.colorbar(ScalarMappable(norm=nrm, cmap=seq), cax=cax)
    cb.set_label("ρ = ‖vbase‖ / ‖delta‖   (white = condition still dominates)",
                 color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)

    # 분자·분모를 따로 — ρ가 큰 이유가 vbase가 커서인지 delta가 작아서인지 가른다
    for ri, name in enumerate(("nvbase", "ndelta")):
        for i, sub in enumerate(SUBS):
            ax = fig.add_subplot(gs[2 + ri, i * 2 if n >= 3 else i])
            vals = []
            for ki, k in enumerate(keys):
                prof = _nanmean(d[name][(k, sub)], axis=1)
                vals.append(prof)
                ax.plot(rows, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4, zorder=4)
                j = min(L - 1, ki + 1)
                if np.isfinite(prof[j]):
                    ax.annotate(short[k], (rows[j], prof[j]), textcoords="offset points",
                                xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                                fontweight="bold", ha="center")
            _style(ax)
            ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
            _maybe_log(ax, vals)
            ax.set_xticks(rows)
            ax.set_xlabel("DiT block", color=INK2, fontsize=8.5)
            lab = "‖vbase‖  (numerator)" if name == "nvbase" else "‖delta‖  (denominator)"
            ax.set_ylabel(f"{lab}  ({sub})", color=INK2, fontsize=8.5)
            ax.set_title(f"{'ghij'[ri * 2 + i]}   {lab} — {sub}", fontsize=10, color=INK,
                         pad=8, loc="left")

    handles = [Line2D([0], [0], color=MODEL_COLORS[k], lw=2.4, marker="o", ms=5,
                      label=titles[k]) for k in keys]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.006))
    a, b = m["task_a"], m["task_b"]
    fig.suptitle(
        f"R10 · ρ alone: how far the condition-averaged contribution outweighs the "
        f"conditional one   task {a} vs task {b}\n"
        f"ρ = ‖vbase‖/‖delta‖ with vbase = ½(Δ(c₀)+Δ(c₁)) and delta = ½(Δ(c₀)−Δ(c₁)). "
        f"Dark = the condition barely changes what this sub-block adds.\n"
        f"Shown on its own because in the combined map a white cell is ambiguous — it can "
        f"mean neutral lean OR low ρ.\n"
        f"Panels g–j split the ratio into its numerator and denominator: a large ρ from a "
        f"big ‖vbase‖ is a different story from one caused by a vanishing ‖delta‖.",
        fontsize=10.5, color=INK, y=0.99, linespacing=1.45)
    fig.savefig(out, dpi=170, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


def _tables(ctx: dict) -> None:
    m, keys, short, L = ctx["m"], ctx["keys"], ctx["short"], ctx["L"]
    d = _slice(ctx, None)
    r_hi, l_hi = ctx["rho_hi"], ctx["lean_hi"]
    print("")
    print(f"{'model':<12}{'sub':<6}{'⟨ρ⟩':>9}{'⟨lean⟩':>9}{'⟨‖vbase‖⟩':>12}{'⟨‖delta‖⟩':>12}"
          f"{'hot':>8}   해석")
    print("-" * 92)
    summary = {}
    for k in keys:
        summary[k] = {"title": ctx["titles"][k], "subs": {}}
        for sub in SUBS:
            ks = (k, sub)
            rl = _nanmean(d["rho"][ks], axis=1)
            ll = _nanmean(d["lean"][ks], axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                hot = (rl >= r_hi) & (ll >= l_hi)
            note = ("무관성분 압도 + task1 쏠림" if hot.any() else
                    "압도하나 중립" if (rl >= r_hi).any() else "조건 기여가 살아있음")
            print(f"{short[k]:<12}{sub:<6}{_nanmean(d['rho'][ks]):>9.3f}"
                  f"{_nanmean(d['lean'][ks]):>+9.3f}{_nanmean(d['nvbase'][ks]):>12.3e}"
                  f"{_nanmean(d['ndelta'][ks]):>12.3e}{int(hot.sum()):>5}/{L:<2}   {note}")
            summary[k]["subs"][sub] = {
                "rho_by_layer": rl.tolist(), "lean_by_layer": ll.tolist(),
                "rho_by_t": _nanmean(d["rho"][ks], axis=0).tolist(),
                "lean_by_t": _nanmean(d["lean"][ks], axis=0).tolist(),
                "vbase_by_layer": _nanmean(d["nvbase"][ks], axis=1).tolist(),
                "delta_by_layer": _nanmean(d["ndelta"][ks], axis=1).tolist(),
                "rho_mean": float(_nanmean(d["rho"][ks])),
                "lean_mean": float(_nanmean(d["lean"][ks])),
                "hot_blocks": [int(i + 1) for i in np.flatnonzero(hot)],
            }
    us = _nanmean(ctx["u_sim"], axis=(0, 2))          # (L, 2)
    print("")
    print("FT 기준끼리의 cos(u₀,u₁) — 1에 가까우면 그 블록의 lean은 약한 증거다")
    print("  블록   ", "  ".join(f"{i:>6d}" for i in range(1, L + 1)))
    for j, sub in enumerate(SUBS):
        print(f"  {sub:<6}", "  ".join(f"{v:+6.3f}" for v in us[:, j]))
    print("")
    ctx["cache"].with_suffix(".summary.json").write_text(json.dumps({
        "t_grid": ctx["t_grid"].tolist(), "obs_steps": ctx["obs_steps"].tolist(),
        "u_sim_by_layer": {s: us[:, j].tolist() for j, s in enumerate(SUBS)},
        "rule": {"rho_hi": r_hi, "lean_hi": l_hi},
        "models": summary, "meta": m}, indent=2, ensure_ascii=False))


def plot_r10(cache: str | Path, per_step: bool = True) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략")
        return
    cache = Path(cache)
    ctx = _prepare(cache)
    _tables(ctx)
    out = cache.with_suffix(".png")
    _draw(ctx, None, out)
    print(f"saved figure -> {out}  (+ .pdf)")
    rho = _draw_rho(ctx, out.with_name(out.stem + "_rho.png"))
    print(f"saved figure -> {rho}  (+ .pdf)   ρ 단독")
    if not per_step:
        return
    stem = out.with_suffix("")
    for si, step in enumerate(ctx["obs_steps"]):
        if not int(ctx["n_live"][si]):
            print(f"skipped step {int(step):>4}: 살아있는 롤아웃이 없다")
            continue
        p = _draw(ctx, si, Path(f"{stem}_step{int(step):04d}.png"))
        print(f"saved figure -> {p}   ({int(ctx['n_live'][si])}/{ctx['n_ep']} episodes live)")


@parser.wrap()
def main(cfg: R10Config):
    cfg.validate()
    cfg.save_checkpoint = False
    logging.info(pformat(cfg.to_dict()))
    if not cfg.ckpt_root:
        raise SystemExit("--ckpt_root 가 필요하다.")
    run_dir = Path(cfg.out_root) / (cfg.run_tag or "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.seed is not None:
        set_seed(cfg.seed)
    cache = run_dir / cache_name(cfg)
    if cache.exists() and not cfg.recompute:
        logging.info(f"[R10] 캐시 재사용: {cache}")
    else:
        cache = run_probe(cfg, run_dir)
    if not cfg.no_plot:
        plot_r10(cache, per_step=cfg.per_step_figs)


if __name__ == "__main__":
    init_logging()
    if "--plot_only" in sys.argv:
        kv = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
        ps = kv.get("per_step", "1").lower() not in ("0", "false", "no")
        if "cache" in kv:
            plot_r10(kv["cache"], per_step=ps)
        elif "run_dir" in kv:
            found = sorted(Path(kv["run_dir"]).glob("R10_*.npz"))
            if not found:
                raise SystemExit(f"[R10] npz 캐시가 없다: {kv['run_dir']}")
            for c in found:
                plot_r10(c, per_step=ps)
        else:
            raise SystemExit("--plot_only 에는 --run_dir= 또는 --cache= 가 필요하다")
    else:
        main()
