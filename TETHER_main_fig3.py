#!/usr/bin/env python
"""TETHER main figure 3 — 행동과 velocity 를 한 그림에, 보존은 작은 곡선으로.

────────────────────────────────────────────────────────────────────────────
이 그림이 담는 세 주장
────────────────────────────────────────────────────────────────────────────
  (a) 작은 SR 곡선    새 task 를 배우는 동안 과거 행동이 보존되는가
  (b) 왼쪽 큰 패널    최종 checkpoint 가 **요청한 task 에 맞게** 움직이는가,
                      아니면 최신 task 의 행동으로 쏠렸는가
  (c) 오른쪽 큰 패널   같은 action 좌표 (x*,t*) 에서도 조건에 따라 속도장
                      v_θ(x_t,t | c) 가 갈라지는가  ← 여기가 직접 증거다

  (c) 만으로는 "옳은 행동"도 "최신 task 편향"도 말할 수 없다. 점이 갈라졌다는
  사실의 의미는 (b) 와 (a) 가 받쳐 준다. 반대로 (b) 의 궤적 차이는 모델이 정말
  조건을 읽는다는 증거가 못 된다(초기 상태가 열마다 다르니까). 셋이 같이 있어야
  한다는 것이 이 배치의 이유다.

배치
        왼쪽: 최종 checkpoint 의 행동        오른쪽: 같은 checkpoint 의 velocity
  위행  Seq-FT   조건별 대표 궤적 4개          조건별 velocity 응답 (공통 PCA)
  아래  TETHER   조건별 대표 궤적 4개          조건별 velocity 응답 (공통 PCA)
  맨아래  Task 0 SR vs 학습 stage             범례

색은 언제나 **입력 task** 를 뜻한다 — t0 파랑, t1 초록, t2 보라, t3 주황.
패널마다 색 의미가 바뀌지 않는다(예전 그림의 "파랑 = 그 패널의 조건" 규약을 버렸다).

────────────────────────────────────────────────────────────────────────────
왼쪽 — 행동
────────────────────────────────────────────────────────────────────────────
TETHER_main_fig2 의 캐시를 그대로 쓴다. 그 스크립트가 이미 **최종 checkpoint 하나에
조건 t0..t3 을 차례로 넣어** 롤아웃해 두었다. 여기서는 패널을 조건별로 쪼개는 대신
모델당 한 좌표계에 네 조건의 medoid 를 겹친다.

  굵은 실선   그 조건으로 실행한 rollout 의 medoid (실패 rollout 도 후보에 포함)
  옅은 점선   같은 색 task 의 expert medoid
  패널 아래   조건별 SR

rollout 다발과 파지 마커는 이 그림에서 뺐다 — appendix 의 기존 2x4 두 장이 그 역할을
계속 한다.

────────────────────────────────────────────────────────────────────────────
오른쪽 — velocity (새로 재는 부분)
────────────────────────────────────────────────────────────────────────────
EE 궤적을 시간 미분한 물리 속도가 아니다. 모델이 내는 flow velocity 를 직접 잰다.

    v_{j,b} = vec[ v_{θ,pose}(x*_t, t* | c_{j,b}) ]        c_{j,b} = (관측, 명령어)

  · x*_t 와 t* 는 **모든 조건·두 모델에 정확히 같은 값**이다. 하나를 고정 시드로
    뽑아 전부에 그대로 먹인다. 조건 말고는 아무것도 안 바뀌게 하려는 것이다.
  · c_{j,b} 는 task j 의 **holdout** 에피소드(학습에 안 쓴 5개)에서 온 관측과
    task j 의 명령어를 짝지은 것이다. b 는 그 표본 인덱스다.
  · 출력 (H,A)=(16,7) 중 pose 6 축만 펼쳐 (16*6)=96 차원으로 본다. gripper 축은
    스케일이 달라 섞으면 PC 를 혼자 끌고 간다.
  · PCA 는 그려지는 점 전체(두 모델 x 네 조건)를 모아 **한 번만** 적합한다. 두
    패널이 같은 축·같은 범위를 쓴다. 설명분산은 패널에 적는다.

  ★ 짝지은 조건이라 관측과 명령어가 같이 바뀐다. 갈라짐이 언어 때문인지 관측
    때문인지 이 패널만으로는 못 가른다. 그래서 관측을 한 task 로 고정하고 명령어만
    바꾼 통제(--cond_mode lang_only)도 같은 배치에서 같이 계산해 캐시에 넣는다.
    비용이 거의 없다(DINOv2 tail 재사용). 통계는 stats 파일에 둘 다 남는다.

  여러 (x*,t*) 를 한 그림에 섞지 않는다 — 좌표 변화가 조건 차이를 덮는다. main 은
  미리 정한 공통 질의 하나, 반복은 appendix 로.

────────────────────────────────────────────────────────────────────────────
    bash TETHER_main_fig3.sh                 # 측정 + 그림
    bash TETHER_main_fig3.sh --plot_only     # 캐시로 그림만 다시
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

# 색은 입력 task 를 뜻한다. Okabe-Ito 에서 서로 가장 잘 갈리는 네 개.
TASK_COLORS = {0: "#0072B2", 1: "#009E73", 2: "#6A3D9A", 3: "#D55E00"}
# 명령어가 전부 "pick up the black bowl <공간 참조> and place it on the plate" 라
# 범례에는 그 공간 참조만 적는다.
SHORT = {0: "between plate & ramekin", 1: "next to ramekin",
         2: "from table center", 3: "on the cookie box"}
C_SEQ, C_TET = "#4D4D4D", "#6A3D9A"      # SR 곡선에서 모델을 가르는 색
POSE_DIMS = 6                            # action = [x,y,z,r,p,y,gripper]


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--seq_run", default="outputs/SEQFT_spatial/libero_spatial_seed42_ours")
    p.add_argument("--tether_run", default="outputs/V4SEQ_spatial/libero_spatial_seed42_ours")
    p.add_argument("--row_names", default="Seq-FT,TETHER",
                   help="롤아웃 캐시의 키이기도 하다 — 캐시를 다시 만들지 않는 한 바꾸면 "
                        "안 된다. 그림에 다른 이름을 쓰려면 --display_names 를 쓴다.")
    p.add_argument("--display_names", default=None,
                   help="그림에 표시할 이름(쉼표 구분). 생략하면 --row_names 를 그대로 쓴다. "
                        "예: --display_names Sequential,TETHER")
    p.add_argument("--ckpt_task", type=int, default=3,
                   help="평가할 체크포인트 — 이 task 직후. 마지막으로 본 task 이기도 하다.")
    p.add_argument("--tasks", default="0,1,2,3", help="조건으로 넣을 task 들")
    p.add_argument("--instructions", default="results/v4_spatial_t0123/instructions.json")
    p.add_argument("--traj_cache", default="results/TETHER_main_fig2",
                   help="왼쪽 열 — 최종 ckpt x 조건 롤아웃 (TETHER_main_fig2 의 캐시)")
    p.add_argument("--sr_cache", default="results/TETHER_main_fig_spatial",
                   help="아래 곡선 — stage 별 task 0 롤아웃 (TETHER_main_fig 의 캐시)")
    p.add_argument("--n_ref_demos", type=int, default=20)
    p.add_argument("--axes", default="xy", choices=["xy", "xz", "yz"])
    # ── velocity 측정 ────────────────────────────────────────────────────────
    p.add_argument("--n_cond", type=int, default=64, help="task 당 평가 condition 표본 수")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--episodes_per_task", type=int, default=45,
                   help="학습에 쓴 에피소드 수. 나머지가 holdout 이 된다(학습 설정과 맞춰야 한다)")
    p.add_argument("--split", choices=["holdout", "train"], default="holdout")
    p.add_argument("--init_frames", type=int, default=0,
                   help="0 이면 기존 방식(episode_sampler 를 앞에서부터). N>=1 이면 holdout "
                        "에피소드 **각각의** 앞쪽 N 프레임만 쓴다 — 시점(phase)을 맞춘다. "
                        "맞추지 않으면 '동작의 어느 단계인가'가 조건 차이를 덮는다.")
    p.add_argument("--init_skip", type=int, default=1,
                   help="에피소드 앞에서 건너뛸 프레임 수. 0 번 프레임은 관측 이력이 패딩이라 "
                        "기본 1 로 둔다.")
    p.add_argument("--t_star", type=float, default=0.5, help="공통 flow time t*")
    p.add_argument("--path_every", type=int, default=5,
                   help="적분 경로를 몇 스텝마다 기록할지. 0 이면 경로를 안 남긴다. "
                        "--euler_steps 100 --path_every 5 면 21 개 점(출발점 포함)이 남는다.")
    p.add_argument("--glow_alpha", type=float, default=0.55,
                   help="조건별 밀도를 부드러운 발광으로 깐다(0 이면 끔). 찍히는 점 그대로를 "
                        "2D KDE 로 추정해 그 조건 색으로 칠하고 밀도에 비례해 불투명도를 준다.")
    p.add_argument("--glow_bw", type=float, default=0.45,
                   help="KDE 대역폭(gaussian_kde 의 bw_method). 클수록 더 부드럽게 퍼진다.")
    p.add_argument("--glow_gamma", type=float, default=0.75,
                   help="발광 감마. 1 보다 작으면 옅은 바깥쪽이 더 살아난다.")
    p.add_argument("--n_paths", type=int, default=40,
                   help="패널에 실제로 그릴 개별 경로 수(조건당). 평균 경로는 항상 그린다.")
    p.add_argument("--euler_steps", type=int, default=0,
                   help="0 = (x*,t*) 에서의 속도를 그린다(기본). N>=1 이면 policy 의 sample() "
                        "과 같은 방식으로 x* 에서 N 스텝 적분한 도달점 â 를 그린다. N=1 은 "
                        "속도와 기하가 같고(공통 상수 x* 만큼의 평행이동), N=100 은 정책이 "
                        "실제 추론에서 내놓는 action chunk 다. N>=1 이면 --t_star 는 안 쓰인다.")
    p.add_argument("--x_seed", type=int, default=20260917, help="공통 noisy action x* 의 시드")
    p.add_argument("--vel_view", choices=["centered", "raw", "raw_centered"],
                   default="centered",
                   help="centered = 표본마다 네 명령어 응답의 평균을 빼고 남는 조건 성분만 "
                        "본다(쌍 비교). raw = 응답 원값 — 관측 차이가 조건 차이를 덮는다. "
                        "두 수치 모두 stats 파일에 남는다.")
    p.add_argument("--obs_task", type=int, default=0,
                   help="stats 의 lang_only 통제에서 고정할 관측 task")
    p.add_argument("--proj", choices=["pca", "cond"], default="pca",
                   help="2 차원으로 내리는 축. pca = 그려지는 점(표본별 평균을 뺀 Δv) 전체의 "
                        "주성분 — 기본값. cond = 조건 평균 여덟 개가 펼치는 평면. 실측상 둘의 "
                        "차이가 거의 없어 반박 여지가 없는 pca 를 쓴다(project() 주석 참조).")
    p.add_argument("--n_avg", type=int, default=4,
                   help="점 하나를 조건 몇 개의 평균으로 묶을지. 평균 위치와 ‖Δv‖ 는 그대로고 "
                        "표본오차만 1/√k 로 준다. 1 이면 표본 하나가 점 하나.")
    p.add_argument("--only_velocity", action="store_true",
                   help="합본(2x2 + 곡선)을 건너뛰고 velocity 두 패널만 낸다")
    p.add_argument("--combine", default=None,
                   help="두 뷰를 한 그림에 가로로 붙인다(1x4). 예: --combine raw,raw_centered. "
                        "뷰마다 좌표계가 달라 PCA·축 범위는 **뷰별로** 잡고 같은 뷰의 두 "
                        "패널끼리만 공유한다.")
    p.add_argument("--font_scale", type=float, default=1.0,
                   help="--combine 그림의 **모든** 글자 크기에 곱하는 배율. 1x4 로 넓어지면 "
                        "논문에서 축소돼 작아 보이므로 1.3~1.5 를 권한다.")
    p.add_argument("--combine_n_avg", default=None,
                   help="--combine 의 뷰마다 쓸 n_avg(쉼표 구분). 생략하면 --n_avg 를 둘 다에.")
    p.add_argument("--bare", action="store_true",
                   help="velocity 단독 그림에서 전체 제목과 아래 주석 두 줄을 빼고 패널 + "
                        "색상/task 범례만 남긴다. 캡션이 그 역할을 하는 본문용.")
    p.add_argument("--vel_layout", choices=["col", "row"], default="col",
                   help="velocity 단독 그림에서 두 패널을 세로로 쌓을지 가로로 놓을지")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/TETHER_main_fig3")
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--measure_only", action="store_true")
    p.add_argument("--no_suptitle", action="store_true")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
#  velocity 측정
# ═════════════════════════════════════════════════════════════════════════════
def _apply_font_scale(fig, scale: float) -> None:
    """완성된 그림의 Text 를 한 번에 키운다.

    ★ 개별 fontsize= 를 일일이 고치지 않는 이유: 눈금·축 라벨·제목·조건 라벨·SR·
      범례가 전부 같은 비율로 커져야 배치가 무너지지 않는다. draw() 를 먼저 불러야
      자동 배치된 Text 들이 확정된다.
    """
    if scale == 1.0:
        return
    import matplotlib.text as mtext
    fig.canvas.draw()
    for t in fig.findobj(mtext.Text):
        t.set_fontsize(t.get_fontsize() * scale)


def measure(a, instr: dict, out: Path) -> dict:
    """V[f"{model}|obs{j}|ins{i}"] = (n_cond, 16*6). 같은 x*, t* 를 전부에 먹인다."""
    cache = out / "velocity.npz"
    if a.plot_only and cache.exists():
        z = np.load(cache, allow_pickle=True)
        logging.info(f"[FIG3] velocity 캐시 사용 {cache}")
        return z["V"].item(), (z["PATH"].item() if "PATH" in z.files else {})

    import torch
    from l2_codebook_id import (build_cfg, cond_tail, encode_lang, make_cond,
                                prep_batch, rgb_cls)
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy
    from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device

    device = torch.device(a.device)
    tasks = [int(s) for s in a.tasks.split(",")]
    rn = [x.strip() for x in a.row_names.split(",")]
    rows = [(rn[0], Path(a.seq_run)), (rn[1], Path(a.tether_run))]
    scratch = Path(os.environ.get("SCRATCH", "/tmp")) / "TETHER_fig3"

    V: dict = {}
    PATH: dict = {}                    # (n_cond, S, 96) — 적분 경로, path_every 가 0 이면 빈 채로
    x_star = None                      # (1,H,A) — 두 모델·모든 조건 공통
    for name, run in rows:
        ck = run / f"task_{a.ckpt_task}" / "checkpoints" / "last" / "pretrained_model"
        if not ck.exists():
            raise SystemExit(f"체크포인트 없음: {ck}")
        for j in tasks:
            t0 = time.perf_counter()
            ns = SimpleNamespace(suite=a.suite, device=a.device, seed=a.seed,
                                 num_workers=a.num_workers, batch_size=a.batch_size,
                                 steps_per_task=1, log_every=100,
                                 eval_episodes=1, eval_batch_size=1)
            cfg = build_cfg(ns, j, str(ck), scratch / f"{name}_{j}")
            ds = make_dataset(cfg)
            total = ds.meta.total_episodes
            tr_eps, ho_eps = split_episodes(cfg.dataset.repo_id, cfg.dataset.root,
                                            total - a.episodes_per_task)
            eps = ho_eps if a.split == "holdout" else tr_eps
            # 정규화 통계는 체크포인트에서 오지만 ds_meta 로 특징 shape 을 맞춘다.
            # task 마다 새로 만든다 — L2_condition_floor 와 같은 방식이다.
            policy = make_policy(cfg=cfg.policy, ds_meta=ds.meta)
            policy.eval()
            net = policy.dit_flow.velocity_net

            # 두 모델이 **같은 표본을 같은 순서로** 봐야 한다 -> shuffle=False
            if a.init_frames > 0:
                # ★ 에피소드마다 앞쪽 몇 프레임만 — 시점을 맞춘다.
                #   episode_sampler 를 shuffle=False 로 쓰고 앞에서 n_cond 개를 자르면
                #   **첫 에피소드의 연속 프레임**만 뽑힌다(실측 확인). 그러면 산포를
                #   지배하는 것이 조건이 아니라 동작 진행 단계가 된다.
                frm = np.asarray(ds.episode_data_index["from"])
                smp = [int(frm[e]) + k for e in eps
                       for k in range(a.init_skip, a.init_skip + a.init_frames)]
                a.n_cond = min(a.n_cond, len(smp))
            else:
                smp = episode_sampler(cfg, ds, eps, shuffle=False)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=a.batch_size, num_workers=a.num_workers,
                sampler=smp, pin_memory=True, drop_last=False,
                multiprocessing_context="spawn" if a.num_workers > 0 else None)

            if x_star is None:
                g = torch.Generator(device=device).manual_seed(a.x_seed)
                x_star = net.sample_noise(1, device, generator=g)        # (1,H,A)
                logging.info(f"[FIG3] 공통 질의 x* {tuple(x_star.shape)}  "
                             f"t*={a.t_star}  ‖x*‖={float(x_star.norm()):.3f}")

            got = {i: [] for i in tasks}
            got_path = {i: [] for i in tasks}
            n_seen = 0
            with torch.no_grad():
                for raw in loader:
                    if n_seen >= a.n_cond:
                        break
                    batch = prep_batch(policy, to_device(raw, device))
                    B = batch["observation.state"].shape[0]
                    B = min(B, a.n_cond - n_seen)
                    batch = {k: (v[:B] if hasattr(v, "__getitem__") and not isinstance(v, str)
                                 else v) for k, v in batch.items()}
                    # DINOv2 CLS 는 동결이라 배치당 한 번. 명령어 넷을 갈아 끼운다.
                    tail = cond_tail(policy, batch, rgb_cls(policy, batch))   # (B,2064)
                    xb = x_star.expand(B, -1, -1).contiguous()                # (B,H,A)
                    tb = torch.full((B,), a.t_star, device=device, dtype=xb.dtype)
                    for i in tasks:
                        cond = make_cond(encode_lang(policy, [instr[f"task{i}"]] * B), tail)
                        q, pth = query(net, xb, tb, cond, a)                  # (B,H,A)
                        got[i].append(q[..., :POSE_DIMS].reshape(B, -1).cpu().numpy())
                        if pth is not None:
                            got_path[i].append(
                                pth[..., :POSE_DIMS].reshape(B, pth.shape[1], -1).cpu().numpy())
                    n_seen += B
            for i in tasks:
                V[f"{name}|obs{j}|ins{i}"] = np.concatenate(got[i])[:a.n_cond]
                if got_path[i]:
                    PATH[f"{name}|obs{j}|ins{i}"] = np.concatenate(got_path[i])[:a.n_cond]
            logging.info(f"[FIG3] {name}  obs t{j}  표본 {n_seen}  "
                         f"({time.perf_counter() - t0:.0f}s)")
            del policy, ds, loader
            torch.cuda.empty_cache()

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, V=np.array(V, dtype=object),
                        PATH=np.array(PATH, dtype=object),
                        x_star=x_star.cpu().numpy(), t_star=np.array(a.t_star))
    logging.info(f"[FIG3] 저장 -> {cache}  (경로 {len(PATH)} 키)")
    return V, PATH


def query(net, xb, tb, cond, a):
    """그릴 양 하나를 계산한다. (B,H,A) 를 돌려준다.

    euler_steps = 0   (x*, t*) 에서의 속도 v 그 자체.
    euler_steps = N   modeling_dit_flow_mt.py:753 sample() 과 **같은 방식**으로 x* 에서
                      N 스텝 적분한 도달점 â: dt = 1/N, t_k = k/N, 매 스텝 clip.

    ★ N=1 은 â = x* + v(x*, 0) 이라 **속도와 기하가 같다**. x* 가 모든 조건·두 모델에
      공통인 상수라 점 구름 전체가 같은 벡터만큼 평행이동할 뿐이고, 중심화하면 그 상수가
      상쇄돼 값까지 같아진다(PCA 축도 분리도도 불변). 조건 차이가 실제로 누적되는 것은
      N>=2 부터다 — 두 번째 스텝부터는 조건마다 **다른 지점**에서 속도장을 쿼리한다.
    """
    import torch
    if a.euler_steps <= 0:
        return net(noisy_actions=xb, time=tb, global_cond=cond), None
    x, dt = xb, 1.0 / a.euler_steps
    keep = a.path_every > 0
    path = [x] if keep else None                 # 출발점 x* 를 포함해서 기록한다
    for k in range(a.euler_steps):
        t = torch.full((x.shape[0],), k / a.euler_steps, device=x.device, dtype=x.dtype)
        x = x + dt * net(noisy_actions=x, time=t, global_cond=cond)
        if getattr(net, "clip_sample", False):
            x = torch.clamp(x, -net.clip_sample_range, net.clip_sample_range)
        if keep and ((k + 1) % a.path_every == 0 or k == a.euler_steps - 1):
            path.append(x)
    return x, (torch.stack(path, dim=1) if keep else None)   # (B,H,A), (B,S,H,A)


def quantity_label(a) -> tuple[str, str, str]:
    """그리는 양의 (이름, 기호, 수식). 패널 제목·단독 그림 제목·모서리 상자가 같이 쓴다."""
    if a.euler_steps <= 0:
        return ("velocity", "v", r"$v_\theta(x^*_t,t^*\mid c)$")
    if a.euler_steps == 1:
        return ("1-step Euler endpoint", "â", r"$\hat a = x^* + v_\theta(x^*,0\mid c)$")
    # 적분 스텝 수는 아래 주석줄이 이미 말하므로 수식은 짧게 — 길면 제목이 잘린다.
    return (f"{a.euler_steps}-step flow endpoint", "â", r"$\hat a(c)$")


def build_groups(V: dict, rn: list[str], tasks: list[int], view: str) -> dict:
    """그림에 찍을 점들. (모델, 조건) -> (n, 96).

    raw       (관측 j, 명령어 j) 짝지은 응답 원값. 관측이 표본마다 달라서 그 산포가
              조건 차이를 덮는다 — 실측으로 관측 효과가 언어 효과의 2~8 배다.
    centered  ★ 쌍 비교. 표본 b 하나를 네 명령어에 **모두** 넣고, 그 네 응답의 평균을
              뺀다. 관측은 네 응답에 공통이므로 정확히 상쇄되고 **조건 때문에 생긴
              성분만** 남는다. 관측 task 도 전부 쓰므로 조건당 점이 4배로 늘고,
              설계가 대칭이 된다(모든 관측 x 모든 명령어).

                  Δv_{i,b} = v(o_b, ℓ_i) − (1/4) Σ_k v(o_b, ℓ_k)

              조건맹 모델이면 네 응답이 같아 Δv 가 전부 0 근처로 뭉친다. 원점에서
              얼마나 벌어지느냐가 그대로 조건 의존성의 크기다.
    """
    G: dict = {}
    for n in rn:
        if view == "raw":
            for j in tasks:
                G[(n, j)] = V[f"{n}|obs{j}|ins{j}"]
            continue
        if view == "raw_centered":
            # 대각선(관측 j + 명령어 j)인데 중심화한다. 조건마다 장면이 달라 같은
            # 프레임 안에서 뺄 수가 없으므로, **표본 인덱스가 같은 것끼리** 뺀다 —
            # 인덱스 b 는 네 task 에서 모두 (같은 holdout 에피소드 번호, 같은 프레임
            # 번호) 라서 시점이 맞춰져 있다. 그 시점에 네 task 가 공통으로 하는
            # 동작이 빠지고 task 고유 성분만 남는다.
            D = np.stack([V[f"{n}|obs{j}|ins{j}"] for j in tasks])     # (4,n,96)
            D = D - D.mean(0)
            for idx, j in enumerate(tasks):
                G[(n, j)] = D[idx]
            continue
        per = {j0: np.stack([V[f"{n}|obs{j0}|ins{i}"] for i in tasks]) for j0 in tasks}
        for idx, i in enumerate(tasks):
            G[(n, i)] = np.concatenate([per[j0][idx] - per[j0].mean(0) for j0 in tasks])
    return G


def build_path_groups(PATHS: dict, rn: list[str], tasks: list[int], view: str) -> dict:
    """적분 경로. (모델, 조건) -> (n, S, 96).

    centered 면 **스텝마다** 네 조건의 평균을 뺀다. t=0 에서는 네 조건이 전부 같은 x*
    이므로 편차가 정확히 0 이다 — 그래서 경로가 모두 원점에서 출발해 부챗살처럼
    벌어지고, 벌어진 끝이 곧 이 그림이 찍는 조건 성분이 된다. 같은 노이즈에서
    출발해 조건에 따라 서로 다른 액션으로 간다는 것을 한 장으로 보여주는 부분이다.
    """
    G = {}
    for n in rn:
        if view == "raw":
            # ★ build_groups(raw) 와 **같은 표본**이어야 한다 — 대각선(관측 j + 명령어 j).
            #   관측 4종을 다 모으면 점(대각선 40개)과 선(160개)이 어긋나, 평균 경로의
            #   끝이 조건 평균 점과 다른 곳에 찍힌다(실측 0.42~1.12 어긋남).
            for j in tasks:
                G[(n, j)] = PATHS[f"{n}|obs{j}|ins{j}"]
            continue
        if view == "raw_centered":
            D = np.stack([PATHS[f"{n}|obs{j}|ins{j}"] for j in tasks])  # (4,n,S,96)
            D = D - D.mean(0)                                           # 스텝마다 상쇄
            for idx, j in enumerate(tasks):
                G[(n, j)] = D[idx]
            continue
        per = {j0: np.stack([PATHS[f"{n}|obs{j0}|ins{i}"] for i in tasks]) for j0 in tasks}
        for idx, i in enumerate(tasks):
            G[(n, i)] = np.concatenate([per[j0][idx] - per[j0].mean(0) for j0 in tasks])
    return G


def separation_from(groups: dict, name: str, tasks: list[int]) -> dict:
    """조건 평균들이 얼마나 갈라졌나 — PCA 이전, 96 차원 원공간에서 잰다.

    between = 조건 평균끼리의 평균 거리, within = 조건 안 표본의 평균 산포.
    ratio = between/within. 1 보다 충분히 크면 조건이 응답을 가른 것이다.
    """
    G = [groups[(name, j)] for j in tasks]
    mu = [g.mean(0) for g in G]
    within = float(np.mean([np.linalg.norm(g - m, axis=1).mean() for g, m in zip(G, mu)]))
    pairs = [float(np.linalg.norm(mu[i] - mu[k]))
             for i in range(len(mu)) for k in range(i + 1, len(mu))]
    between = float(np.mean(pairs))
    return {"between": between, "within": within,
            "ratio": between / max(within, 1e-12), "pairs": pairs}


def average_points(groups: dict, k: int, seed: int = 0) -> dict:
    """점 하나를 조건 k 개의 평균으로 묶는다.

    조건 평균의 위치도, ‖Δv‖ 도 바뀌지 않는다. 바뀌는 것은 점 하나의 표본오차뿐이고
    1/√k 로 준다 — "조건 하나하나가 어디에 떨어지는가" 대신 "그 조건의 응답을 k 개로
    추정하면 어디에 떨어지는가"를 보는 것이다. 캡션에 k 를 반드시 적어야 한다.
    """
    if k <= 1:
        return groups
    rng = np.random.default_rng(seed)
    out = {}
    for key, g in groups.items():
        idx = rng.permutation(len(g))
        m = (len(g) // k) * k
        out[key] = g[idx[:m]].reshape(-1, k, g.shape[1]).mean(1)
    return out


def project(groups: dict, rn: list[str], tasks: list[int], mode: str):
    """96 차원 -> 2 차원. (proj, 축 이름, 설명분산) 을 돌려준다.

    pca   ★ 기본값. 그려지는 점(= 표본별 평균을 뺀 Δv) 전체의 주성분.
    cond  여덟 개 조건 평균(두 방법 x 네 조건)이 펼치는 평면. 조건 차이를 최대화하는
          축이라 '더 잘 갈려 보이게' 하는 선택이다.

    ★ 왜 cond 를 안 쓰는가 — 재 보니 이득이 거의 없다.
      centering 전(raw)에는 관측 분산이 PC1/PC2 를 독차지해서 조건이 정말 묻혔다.
      그런데 Δv 로 관측을 상쇄하고 나면 그 문제가 사라진다. 실측:

          조건평균 산포를 담는 비율   pca 82.0%   cond 87.9%
          조건평균 2D 쌍거리(TETHER)  pca 1.278   cond 1.330
          2D between/within (TETHER)  pca 2.05    cond 2.28   (n_avg=4 기준)

      그림이 갈려 보이게 만든 것은 축이 아니라 centering 과 --n_avg 였다
      (n_avg 1 -> 4 에서 TETHER 1.04 -> 2.05, 두 배). cond 가 더하는 10% 를 얻자고
      "주장하려는 효과에 축을 맞췄다"는 반박을 살 이유가 없다.
    """
    X = np.concatenate([groups[k] for k in groups])
    mu = X.mean(0)
    if mode == "pca":
        _, S, Wt = np.linalg.svd(X - mu, full_matrices=False)
        lab = ("PC1", "PC2")
    else:
        M = np.stack([groups[(n, j)].mean(0) for n in rn for j in tasks])
        _, S, Wt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        lab = ("CM1", "CM2")
    frac = ((S ** 2) / (S ** 2).sum())[:2]
    W = Wt[:2]
    # mu, W 도 돌려준다 — 적분 경로를 **같은 평면**에 얹어야 하기 때문이다.
    return {k: (g - mu) @ W.T for k, g in groups.items()}, lab, frac, mu, W


def glow_layer(ax, P: np.ndarray, color: str, lims, bw: float, amax: float,
               gamma: float, grid: int = 260) -> None:
    """조건 하나의 밀도를 부드러운 발광으로 깐다.

    찍히는 점 **그대로**를 gaussian_kde 로 추정해 격자에 올리고, 그 조건 색에
    밀도 비례 불투명도를 준다. 경계선(등수위선)을 그리지 않으므로 "여기까지가
    이 조건" 이라는 인상을 주지 않는다 — 실제로 분포는 겹치고, 겹침을 가리지
    않는 것이 이 그림의 규약이다.

    밀도는 **조건마다 자기 최댓값으로** 정규화한다. 조건 간 밝기를 비교하라는
    그림이 아니라 어디에 놓여 있는지를 보는 그림이고, 크기 비교는 ‖Δ‖ 상자와
    점 위치가 담당한다.
    """
    from matplotlib.colors import to_rgb
    from scipy.stats import gaussian_kde
    if amax <= 0 or len(P) < 3:
        return
    (x0, x1), (y0, y1) = lims
    xs, ys = np.linspace(x0, x1, grid), np.linspace(y0, y1, grid)
    X, Y = np.meshgrid(xs, ys)
    try:
        D = gaussian_kde(P.T, bw_method=bw)(np.vstack([X.ravel(), Y.ravel()]))
    except np.linalg.LinAlgError:      # 점이 한 곳에 뭉쳐 공분산이 특이할 때
        return
    D = (D / max(D.max(), 1e-12)).reshape(grid, grid)
    rgba = np.zeros((grid, grid, 4))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = amax * D ** gamma
    ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="lower", interpolation="bilinear",
              zorder=1, aspect="auto")


def draw_velocity(ax, name, proj, groups, vnorm, tasks, lims, a, lab, frac,
                  show_title: bool, paths=None):
    """velocity 패널 하나. 합본과 단독 그림이 같은 코드를 쓴다."""
    from matplotlib.ticker import MaxNLocator
    xl_v, yl_v = lims
    if a.vel_view != "raw":
        # 조건맹 모델의 기준점. 네 응답이 같으면 모든 점이 여기로 모인다.
        ax.axhline(0, color="0.8", lw=.8, zorder=2)
        ax.axvline(0, color="0.8", lw=.8, zorder=2)
    if paths is not None:
        # 개별 경로는 옅게, 조건 평균 경로는 굵게. 출발점은 네 조건이 공유한다.
        rng = np.random.default_rng(7)
        for j in tasks:
            P = paths[(name, j)]                                   # (n,S,2)
            for p in P[rng.choice(len(P), min(a.n_paths, len(P)), replace=False)]:
                ax.plot(p[:, 0], p[:, 1], color=TASK_COLORS[j], lw=.5, alpha=.16, zorder=3)
        for j in tasks:
            m = paths[(name, j)].mean(0)                           # (S,2)
            ax.plot(m[:, 0], m[:, 1], color=TASK_COLORS[j], lw=2.0, alpha=.95, zorder=6)
        s = paths[(name, tasks[0])][:, 0].mean(0)                  # 공통 출발점 x*
        ax.plot(s[0], s[1], "o", ms=5.5, color="0.25", mec="white", mew=.9, zorder=10)
    glow = a.glow_alpha > 0
    if glow:
        for j in tasks:
            glow_layer(ax, proj[(name, j)], TASK_COLORS[j], (xl_v, yl_v),
                       a.glow_bw, a.glow_alpha, a.glow_gamma)
    for j in tasks:
        p = proj[(name, j)]
        # 발광을 깔면 점은 작고 옅게 — 발광이 분포를, 점이 실제 표본을 담당한다.
        ax.scatter(p[:, 0], p[:, 1],
                   s=(7 if glow else (9 if a.n_avg <= 1 else 16)),
                   c=TASK_COLORS[j],
                   alpha=(.30 if glow else (.40 if a.n_avg <= 1 else .55)),
                   linewidths=0, zorder=4)
    M = np.stack([proj[(name, j)].mean(0) for j in tasks])
    gap = max(float(np.linalg.norm(M[i] - M[k]))
              for i in range(len(M)) for k in range(i + 1, len(M)))
    # 조건 평균이 붙어 버린 패널에서는 t0..t3 라벨이 한 점에 포개져 읽히지 않는다.
    # 그 경우 라벨 넷 대신 "여기서 붕괴한다" 하나를 단다 — 붕괴가 그 패널의 내용이다.
    collapsed = gap < .12 * (xl_v[1] - xl_v[0])
    for idx, j in enumerate(tasks):
        ax.scatter(M[idx, 0], M[idx, 1], s=150, c=TASK_COLORS[j], edgecolor="white",
                   linewidth=1.4, zorder=8)
        if not collapsed:
            ax.annotate(f"t{j}", (M[idx, 0], M[idx, 1]), textcoords="offset points",
                        xytext=(9, 6), fontsize=9.5, color=TASK_COLORS[j],
                        fontweight="bold", zorder=9)
        else:
            # 뭉친 패널에서도 이름은 단다. 같은 자리에 달면 넷이 포개지므로 네 방향으로
            # 일정 거리(픽셀 기준) 흩어 놓고 지시선을 연결한다 — 점이 얼마나 붙어 있든
            # 라벨은 안 겹친다.
            th = np.radians(60 + 90 * idx)
            ax.annotate(f"t{j}", (M[idx, 0], M[idx, 1]), textcoords="offset points",
                        xytext=(34 * np.cos(th), 34 * np.sin(th)), fontsize=9.5,
                        color=TASK_COLORS[j], fontweight="bold", zorder=9,
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=TASK_COLORS[j], lw=.7,
                                        alpha=.55, shrinkA=0, shrinkB=5))
    if collapsed:
        ax.annotate("all four conditions collapse here",
                    (M[:, 0].mean(), M[:, 1].mean()), textcoords="offset points",
                    xytext=(0, 84), fontsize=9, color="0.3", zorder=9,
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=.8,
                                    shrinkA=2, shrinkB=7))
    # 크기는 **평균 전** 조건 성분으로 잰다 — n_avg 로 줄어드는 것은 표본오차지
    # 조건 성분이 아니다.
    rms = float(np.sqrt(np.mean([np.square(groups[(name, j)]).sum(1).mean()
                                 for j in tasks])))
    _, sym, _ = quantity_label(a)
    if a.vel_view != "raw":
        note = (f"condition component  ‖Δ{sym}‖ = {rms:.2f}   "
                f"({100 * rms / vnorm[name]:.0f}% of ‖{sym}‖)")
    else:
        # raw 뷰에서 rms 는 ‖v‖ 그 자체라 % 를 적으면 언제나 100% 다. 대신 조건이
        # 갈라진 정도(조건 평균 사이 거리 / 조건 안 산포)를 적는다.
        mu = [groups[(name, j)].mean(0) for j in tasks]
        btw = float(np.mean([np.linalg.norm(mu[i] - mu[k])
                             for i in range(len(mu)) for k in range(i + 1, len(mu))]))
        wit = float(np.mean([np.linalg.norm(groups[(name, j)] - m, axis=1).mean()
                             for j, m in zip(tasks, mu)]))
        note = (f"‖{sym}‖ = {rms:.2f}   between/within = {btw / max(wit, 1e-12):.2f}")
    if not a.bare:      # --bare 는 캡션이 수치를 맡는다 — 패널에는 안 적는다
        # ★ SR 줄(왼쪽에서 오른쪽)과 이 상자(오른쪽 끝에서 왼쪽)가 같은 높이라,
        #   폰트를 키우면 가운데서 부딪힌다. 1.0 에서는 간신히 맞는 폭이다.
        #   --font_scale 을 올리면 한 줄 위로 올려 충돌을 피한다.
        y_note = .035 if a.font_scale <= 1.05 else .035 + .075 * a.font_scale
        ax.text(.975, y_note, note,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
                color="0.3", bbox=dict(fc="white", ec="0.85", alpha=.9, pad=2.2))
    ax.set_xlim(*xl_v); ax.set_ylim(*yl_v)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.18)
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=8.5)
    of = "of condition-mean scatter" if lab[0] == "CM1" else "of variance"
    if a.bare:          # 설명분산은 캡션으로 — 축에는 이름만
        ax.set_ylabel(lab[1], fontsize=9.5)
        ax.set_xlabel(lab[0], fontsize=9.5)
    else:
        ax.set_ylabel(f"{lab[1]}  ({100 * frac[1]:.0f}% {of})", fontsize=9.5)
        ax.set_xlabel(f"{lab[0]}  ({100 * frac[0]:.0f}% {of})", fontsize=9.5)
    if show_title:
        nm, _, expr = quantity_label(a)
        nm = nm[0].upper() + nm[1:]
        ax.set_title((f"{nm} — condition component\n" + expr +
                      " minus its mean over the conditions"
                      if a.vel_view != "raw" else
                      f"{nm} at one shared query  " + expr), fontsize=11, pad=8)


# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def prepare_view(V: dict, PATHS: dict, rn: list[str], tasks: list[int], a, view: str,
                 n_avg: int):
    """한 뷰에 필요한 것을 모두 준비한다 — groups, 찍을 점, 경로, 축 라벨.

    ★ 뷰마다 좌표계가 다르다(raw 는 원값, centered 계열은 편차). 그래서 PCA 도
      뷰마다 따로 적합하고, 축은 **같은 뷰의 두 패널끼리만** 공유한다. 서로 다른
      뷰의 패널에 같은 축을 씌우는 것은 의미가 없다.
    """
    groups = build_groups(V, rn, tasks, view)
    pts = average_points(groups, n_avg, a.seed)
    # 축은 평균 전 원자료로 적합한다(조건 간 구조로 축이 끌리지 않게).
    _, lab, frac, mu_p, W_p = project(groups, rn, tasks, a.proj)
    proj = {k: (g - mu_p) @ W_p.T for k, g in pts.items()}
    Q = np.concatenate([proj[k] for k in proj])
    pathsP = None
    if PATHS:
        pg = build_path_groups(PATHS, rn, tasks, view)
        S = next(iter(pg.values())).shape[1]
        flat = average_points({k: g.reshape(len(g), -1) for k, g in pg.items()},
                              n_avg, a.seed)
        pathsP = {k: (g.reshape(len(g), S, -1) - mu_p) @ W_p.T for k, g in flat.items()}
        if view != "raw":
            Q = np.concatenate([Q] + [p.reshape(-1, 2) for p in pathsP.values()])
    return {"groups": groups, "proj": proj, "paths": pathsP, "lab": lab,
            "frac": frac, "Q": Q}


def fit_limits(pts: np.ndarray, aspect: float, pad: float = 0.08):
    """점들을 담되 (y폭/x폭) == aspect 가 되도록 짧은 쪽을 늘린다.

    네 큰 패널을 모두 equal aspect 로 두면서 셀을 꽉 채우려는 것이다. 늘리는 쪽은
    여백만 늘 뿐 데이터는 그대로다.
    """
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    w, h = (hi - lo) * (1 + 2 * pad)
    w, h = max(w, 1e-9), max(h, 1e-9)
    if h / w < aspect:
        h = w * aspect
    else:
        w = h / aspect
    return (c[0] - w / 2, c[0] + w / 2), (c[1] - h / 2, c[1] + h / 2)


def main():
    a = parse()
    from lerobot.utils.utils import init_logging
    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = [int(s) for s in a.tasks.split(",")]
    rn = [x.strip() for x in a.row_names.split(",")]
    instr = json.loads(Path(a.instructions).read_text())

    # 캐시 키(rn)와 표시 이름(disp)을 분리한다 — 롤아웃 캐시는 rn 으로만 찾는다.
    disp = dict(zip(rn, [x.strip() for x in a.display_names.split(",")]
                    if a.display_names else rn))
    V, PATHS = measure(a, instr, out)
    if a.measure_only:
        return

    # ── 왼쪽: 최종 ckpt x 조건 롤아웃 (fig2 캐시) ────────────────────────────
    from TETHER_main_fig import expert_traj, medoid, seq_spec
    traj = np.load(Path(a.traj_cache) / "rollouts.npz", allow_pickle=True)["data"].item()
    srs = np.load(Path(a.sr_cache) / "rollouts.npz", allow_pickle=True)["data"].item()
    spec = seq_spec(a.suite, max(tasks + [a.ckpt_task]) + 1)
    ref = {j: expert_traj(spec["ds_root"][j], None, a.n_ref_demos, None) for j in tasks}
    ref_med = {j: ref[j][medoid(ref[j])] for j in tasks}
    roll_med = {(n, j): traj[f"{n}|{j}"]["traj"][medoid(traj[f"{n}|{j}"]["traj"])]
                for n in rn for j in tasks}

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
                                "axes.edgecolor": "0.7", "axes.linewidth": .8,
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ix, iy, nx, ny = {"xy": (0, 1, "x", "y"), "xz": (0, 2, "x", "z"),
                      "yz": (1, 2, "y", "z")}[a.axes]

    # 궤적 패널 공통 범위 — 그려지는 medoid 만 담는다(다발을 안 그리므로)
    P = np.concatenate([t[:, [ix, iy]] for t in list(roll_med.values()) + list(ref_med.values())])
    pad = .06 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    xl_t = (P[:, 0].min() - pad, P[:, 0].max() + pad)
    yl_t = (P[:, 1].min() - pad, P[:, 1].max() + pad)
    aspect = (yl_t[1] - yl_t[0]) / (xl_t[1] - xl_t[0])

    # ── 오른쪽: 공통 PCA. 그려지는 점 전체로 한 번만 적합한다 ────────────────
    groups = build_groups(V, rn, tasks, a.vel_view)
    # 조건 성분의 크기를 응답 자체의 크기로 재는 기준. 두 모델의 ‖v‖ 는 거의 같아서
    # (19.2) 퍼센트가 그대로 비교 가능한 수치가 된다.
    vnorm = {n: float(np.mean([np.linalg.norm(V[f"{n}|obs{j}|ins{j}"], axis=1).mean()
                               for j in tasks])) for n in rn}
    X = np.concatenate([groups[k] for k in groups])
    pts = average_points(groups, a.n_avg, a.seed)
    # ★ 축은 **평균내기 전** 원자료로 적합한다. 평균낸 점에 적합하면 조건 안의 잡음이
    #   1/n_avg 로 줄어든 상태라 축이 조건 간 구조 쪽으로 끌린다 — 보려는 효과에 축을
    #   맞추는 셈이다. 찍는 것은 평균낸 점이더라도 축은 원자료가 정한다.
    _, lab, frac, mu_p, W_p = project(groups, rn, tasks, a.proj)
    proj = {k: (g - mu_p) @ W_p.T for k, g in pts.items()}
    Q = np.concatenate([proj[k] for k in proj])
    # 적분 경로도 **같은 평면**에 얹는다. 축 범위에도 반영해야 경로가 잘리지 않는다.
    pathsP = None
    if PATHS:
        pg = build_path_groups(PATHS, rn, tasks, a.vel_view)
        # ★ 점과 **같은 묶음**으로 평균낸다. 키 순서·길이·seed 가 같으므로
        #   average_points 가 두 dict 에 똑같은 순열을 써서, 경로의 끝점이 찍히는
        #   점과 정확히 일치한다. 평균하지 않으면 경로만 표본 하나짜리라 점보다
        #   두 배 넓게 퍼져 별빛처럼 뻗고 축 범위까지 늘린다.
        S = next(iter(pg.values())).shape[1]
        flat = average_points({k: g.reshape(len(g), -1) for k, g in pg.items()},
                              a.n_avg, a.seed)
        pathsP = {k: (g.reshape(len(g), S, -1) - mu_p) @ W_p.T          # (n,S,2)
                  for k, g in flat.items()}
        if a.vel_view != "raw":
            # 중심화한 경로는 원점에서 출발하므로 끝점 범위 안에 들어온다 — 같이 담는다.
            Q = np.concatenate([Q] + [p.reshape(-1, 2) for p in pathsP.values()])
        # raw 는 경로가 x* (노름 ~10) 에서 출발해 끝점 군집(~4) 밖에서 들어온다.
        # 같이 담으면 정작 볼 끝점 구조가 뭉개진다 — 축은 끝점으로만 잡고 경로는
        # 화면 밖에서 들어오게 둔다.
    xl_v, yl_v = fit_limits(Q, aspect)

    # ── 레이아웃: 큰 패널 2x2 + 아래 (SR 곡선 | 범례) ────────────────────────
    pw = 4.3
    ph = pw * aspect
    fw = 2 * pw + 1.5
    sr_h = 2.0
    fh = 2 * ph + sr_h + (0.85 if a.no_suptitle else 1.75) + 0.35
    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(3, 2, height_ratios=[ph, ph, sr_h],
                          left=1.05 / fw, right=1 - .25 / fw,
                          top=1 - (0.75 if a.no_suptitle else 1.62) / fh,
                          bottom=.35 / fh, wspace=.17, hspace=.30)

    for r, name in enumerate(rn):
        # ── 행동 ─────────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[r, 0])
        for j in tasks:
            c = TASK_COLORS[j]
            e = ref_med[j]
            ax.plot(e[:, ix], e[:, iy], color=c, lw=1.5, ls=(0, (4, 2.2)), alpha=.55, zorder=3)
            t_ = roll_med[(name, j)]
            ax.plot(t_[:, ix], t_[:, iy], color=c, lw=2.3, zorder=6)
            ax.plot(t_[0, ix], t_[0, iy], "o", color=c, ms=5, mec="white", mew=.7, zorder=7)
        ax.set_xlim(*xl_t); ax.set_ylim(*yl_t)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.18)
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=8.5)
        ax.set_ylabel(f"{disp[name]}\nEE {ny} (m)", fontsize=10.5)
        if r == 0:
            ax.set_title("Behaviour — final checkpoint, one condition per colour",
                         fontsize=11, pad=8)
        else:
            ax.set_xlabel(f"EE {nx} (m)", fontsize=9.5)
        # 조건별 SR — 패널 아래쪽에 색으로 구분해 나란히
        ax.text(.025, .045, "SR", transform=ax.transAxes, fontsize=8.5, color="0.35",
                ha="left", va="bottom",
                bbox=dict(fc="white", ec="none", alpha=.75, pad=1.4))
        for n_, j in enumerate(tasks):
            ax.text(.115 + .105 * n_, .045, f"{traj[f'{name}|{j}']['sr']:.0f}%",
                    transform=ax.transAxes, fontsize=8.8, color=TASK_COLORS[j],
                    ha="left", va="bottom", fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=1.4))

        # ── velocity ─────────────────────────────────────────────────────────
        draw_velocity(fig.add_subplot(gs[r, 1]), name, proj, groups, vnorm, tasks,
                      (xl_v, yl_v), a, lab, frac, show_title=(r == 0), paths=pathsP)

    # ── 아래 왼쪽: 보존 곡선 ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    xs = list(range(len(tasks)))
    for name, col, ls, mk in ((rn[0], C_SEQ, (0, (4, 2)), "X"), (rn[1], C_TET, "-", "o")):
        ys = [srs[f"{name}|{k}"]["sr"] for k in tasks]
        ax.plot(xs, ys, ls=ls, color=col, lw=2.0, marker=mk, ms=7, mec="white",
                mew=1.0, label=disp[name], zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["After\nTask 0"] + [f"Tasks\n0–{k}" for k in tasks[1:]], fontsize=8.6)
    ax.set_ylim(-6, 108); ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Task 0 SR (%)", fontsize=9.5)
    ax.set_title("Retention of Task 0 while learning t1–t3", fontsize=10, pad=6)
    ax.grid(alpha=.18, axis="y")
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=9, frameon=False, loc="center left", bbox_to_anchor=(.02, .45))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── 아래 오른쪽: 범례 ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    h = [plt.Line2D([], [], color="0.35", lw=2.3),
         plt.Line2D([], [], color="0.35", lw=1.5, ls=(0, (4, 2.2)), alpha=.7),
         plt.Line2D([], [], ls="none", marker="o", ms=4, color="0.45", alpha=.6),
         plt.Line2D([], [], ls="none", marker="o", ms=9, color="0.35", mec="white")]
    _, sym_l, _ = quantity_label(a)
    lb = ["rollout medoid under that condition", "expert medoid for that task",
          (f"one eval condition (Δ{sym_l})" if a.vel_view == "centered"
           else f"one eval condition ({sym_l})"), "condition mean"]
    if pathsP is not None:
        h.append(plt.Line2D([], [], color="0.5", lw=1.0, alpha=.75))
        lb.append("integration path from the shared start")
    # 경로 항목이 붙으면 9 줄이 된다 — 폰트·간격을 줄여야 아래 색상 범례와 안 겹친다.
    leg1 = ax.legend(h, lb, loc="upper left", bbox_to_anchor=(-.02, 1.04), fontsize=8.0,
                     frameon=False, handlelength=2.2, handletextpad=.7, labelspacing=.40)
    ax.add_artist(leg1)
    hc = [plt.Line2D([], [], color=TASK_COLORS[j], lw=3) for j in tasks]
    lc = [f"Task {j} — “…{SHORT[j]}…”" for j in tasks]
    ax.legend(hc, lc, loc="lower left", bbox_to_anchor=(-.02, -.06), fontsize=8.0,
              frameon=False, handlelength=1.8, handletextpad=.7, labelspacing=.40,
              title="colour = task given as the condition",
              title_fontsize=8.0, alignment="left")

    if not a.no_suptitle:
        qs = (rf"$(x^*_t,\ t^*\!=\!{a.t_star})$" if a.euler_steps <= 0 else
              rf"start $x^*$ ({a.euler_steps} Euler steps)")
        fig.suptitle(
            f"One checkpoint (after Tasks 0–{a.ckpt_task}), four conditions\n"
            f"{a.suite} · 10 rollouts per condition · "
            f"{a.n_cond} held-out conditions per task from one shared " + qs,
            fontsize=10.5, y=1 - .26 / fh)

    if not a.only_velocity:
        for ext in ("png", "pdf"):
            fp = out / f"TETHER_main_fig3.{ext}"
            fig.savefig(fp, dpi=200 if ext == "png" else None)
            print(f"saved -> {fp}")
    plt.close(fig)

    # ── 두 뷰를 한 그림에 가로로 (1x4) ───────────────────────────────────────
    if a.combine:
        views = [v.strip() for v in a.combine.split(",")]
        navs = ([int(x) for x in a.combine_n_avg.split(",")] if a.combine_n_avg
                else [a.n_avg] * len(views))
        # 쌍 안은 붙이고 쌍 사이만 띄운다 — 사이에 빈 열을 하나 끼운 그리드로 만든다.
        # (같은 뷰의 두 패널은 축이 같으므로 오른쪽 패널의 눈금 라벨을 생략한다.)
        pw3, top3, bot3, gap = 5.4, 0.60, 1.00, .14
        wr, cols, c = [], [], 0
        for vi in range(len(views)):
            if vi:
                wr.append(gap); c += 1
            wr += [1, 1]; cols += [c, c + 1]; c += 2
        fw3 = sum(wr) * pw3 + 1.0
        fh3 = pw3 + top3 + bot3
        f3 = plt.figure(figsize=(fw3, fh3))
        gs3 = f3.add_gridspec(1, len(wr), width_ratios=wr, wspace=.08,
                              left=.80 / fw3, right=1 - .15 / fw3,
                              top=1 - top3 / fh3, bottom=bot3 / fh3)
        ax3 = [f3.add_subplot(gs3[0, i]) for i in cols]
        for vi, (view, nav) in enumerate(zip(views, navs)):
            a_v = argparse.Namespace(**vars(a))
            a_v.vel_view, a_v.n_avg = view, nav        # draw_velocity 가 뷰를 보고 분기한다
            D = prepare_view(V, PATHS, rn, tasks, a, view, nav)
            lims3 = fit_limits(D["Q"], 1.0)            # 축은 이 뷰 안에서만 공유
            for r, name in enumerate(rn):
                ax = ax3[2 * vi + r]
                draw_velocity(ax, name, D["proj"], D["groups"], vnorm, tasks, lims3,
                              a_v, D["lab"], D["frac"], show_title=False,
                              paths=D["paths"])
                # 중심화 여부를 제목에 붙인다 — 별도 (a)/(b) 줄을 두지 않는다.
                ax.set_title(disp[name] + ("" if view == "raw" else " (Centered)"),
                             fontsize=11.5, pad=6,
                             color=(C_SEQ if r == 0 else C_TET))
                box = dict(fc="white", ec="none", alpha=.75, pad=1.3)
                ax.text(.025, .035, "SR", transform=ax.transAxes, fontsize=7.5,
                        color="0.4", ha="left", va="bottom", bbox=box)
                # 패널이 좁아 100% 가 네 번 들어가면 붙는다 — 간격을 넓히고 글자를 줄인다.
                for i, j in enumerate(tasks):
                    ax.text(.135 + .145 * i, .035, f"{traj[f'{name}|{j}']['sr']:.0f}%",
                            transform=ax.transAxes, fontsize=8, color=TASK_COLORS[j],
                            ha="left", va="bottom", fontweight="bold", bbox=box)
                # y 라벨은 맨 왼쪽 패널만. 쌍의 오른쪽 패널은 왼쪽과 축이 같으므로
                # 눈금 라벨도 생략해 사이를 붙인다.
                if not (vi == 0 and r == 0):
                    ax.set_ylabel("")
                if r == 1:
                    ax.set_yticklabels([])
        hc = [plt.Line2D([], [], ls="none", marker="o", ms=8, color=TASK_COLORS[j],
                         mec="white") for j in tasks]
        leg3 = f3.legend(hc, [f"t{j}: “…{SHORT[j]}…”" for j in tasks], loc="lower center",
                  ncol=4, frameon=False, fontsize=8.8,
                  bbox_to_anchor=(.5, .10 / fh3),
                  handletextpad=.5, columnspacing=2.0)
        # 뷰 키로 이름을 만든다 — SH 의 표시 문자열을 쓰면 raw 와 raw_centered 가
        # 둘 다 'cond' 로 잘려 같은 이름이 된다.
        FN = {"raw": "cond", "raw_centered": "condc", "centered": "lang"}
        st3 = "TETHER_fig3_" + "_".join(FN[v] for v in views)
        # ★ 여기서는 범례 크기를 줄이지 않는다. combine 은 패널이 4 개라 그림이
        #   가로로 두 배 길고(예: 4677px), 한 줄 범례가 넘칠 걱정이 없다.
        #   가로가 좁은 단일 뷰(2 패널) 경로에서만 상한을 둔다.
        _apply_font_scale(f3, a.font_scale)
        bb = "tight" if a.font_scale >= 1.2 else None
        for ext in ("png", "pdf"):
            fp = out / f"{st3}.{ext}"
            f3.savefig(fp, dpi=200 if ext == "png" else None, bbox_inches=bb)
            print(f"saved -> {fp}")
        plt.close(f3)
        return

    # ── velocity 두 패널만 따로 ──────────────────────────────────────────────
    # 합본의 오른쪽 열과 **같은 데이터·같은 축**이다. 정사각 패널이라 범위만 다시 잡는다.
    col = a.vel_layout == "col"
    # 파일명을 뷰가 정하게 한다 — 예전처럼 렌더 뒤 손으로 mv 하지 않는다.
    stem = {"raw": "TETHER_fig3_cond",              # 관측+state+언어, 각 task 자기 조건
            "raw_centered": "TETHER_fig3_cond_centered",   # 위를 시점별로 중심화
            "centered": "TETHER_fig3_lang"}[a.vel_view]    # 장면 고정, 언어만
    lims2 = fit_limits(Q, 1.0)
    pw2 = 4.6
    # 아래 여백은 범례 + 주석 두 줄이 x 축 라벨과 겹치지 않을 만큼만. 가로 배치는
    # 범례가 한 줄(ncol=4)이라 세로 배치보다 덜 필요하다 — 같은 값을 쓰면 빈다.
    # --bare 면 전체 제목과 주석 두 줄이 없으므로 그만큼 위아래를 줄인다.
    top2 = 0.50 if a.bare else 1.05
    if a.bare:
        # 범례를 x 축 라벨에서 떨어뜨리려면 아래를 조금 더 잡아야 한다.
        # (조건별 SR 은 패널 **안쪽**에 넣으므로 여기에 자리를 낼 필요가 없다.)
        bot2 = 1.10 if col else 0.95
    else:
        bot2 = 1.95 if col else 1.25
    if col:
        fw2, fh2 = pw2 + 1.2, 2 * pw2 + top2 + bot2
        f2, axs = plt.subplots(2, 1, figsize=(fw2, fh2))
        f2.subplots_adjust(left=1.05 / fw2, right=1 - .22 / fw2,
                           top=1 - top2 / fh2, bottom=bot2 / fh2, hspace=.30)
    else:
        fw2, fh2 = 2 * pw2 + 1.6, pw2 + top2 + bot2
        f2, axs = plt.subplots(1, 2, figsize=(fw2, fh2))
        f2.subplots_adjust(left=1.0 / fw2, right=1 - .22 / fw2,
                           top=1 - top2 / fh2, bottom=bot2 / fh2, wspace=.20)
    for r, name in enumerate(rn):
        draw_velocity(axs[r], name, proj, groups, vnorm, tasks, lims2, a,
                      lab, frac, show_title=False, paths=pathsP)
        axs[r].set_title(disp[name], fontsize=12.5, pad=7,
                         color=(C_SEQ if r == 0 else C_TET))
        # 조건별 rollout SR — 패널 **안쪽** 왼쪽 아래. 색이 곧 조건이라 숫자에 색만
        # 입힌다. 흰 배경을 깔아 발광/경로 위에서도 읽히게 한다.
        ax = axs[r]
        box = dict(fc="white", ec="none", alpha=.75, pad=1.4)
        ax.text(.025, .035, "SR", transform=ax.transAxes, fontsize=8.5,
                color="0.4", ha="left", va="bottom", bbox=box)
        # ★ 간격도 폰트 배율만큼 벌려야 한다. 고정 .105 면 1.5 배에서 숫자가 붙는다.
        dx = .105 * max(1.0, a.font_scale)
        for i, j in enumerate(tasks):
            ax.text(.105 * max(1.0, a.font_scale) + dx * i, .035,
                    f"{traj[f'{name}|{j}']['sr']:.0f}%",
                    transform=ax.transAxes, fontsize=9, color=TASK_COLORS[j],
                    ha="left", va="bottom", fontweight="bold", bbox=box)
    hc = [plt.Line2D([], [], ls="none", marker="o", ms=8, color=TASK_COLORS[j],
                     mec="white") for j in tasks]
    # 패널 안에 이미 t0..t3 라벨이 붙으므로 범례는 "어느 색이 어느 명령어인지"만 짧게.
    leg2 = f2.legend(hc, [f"t{j}: “…{SHORT[j]}…”" for j in tasks], loc="lower center",
              ncol=(2 if col else 4), frameon=False, fontsize=8.8,
              bbox_to_anchor=(.5, (0.10 if a.bare
                                   else ((bot2 - 1.02) if col else 0.50)) / fh2),
              handletextpad=.5, columnspacing=1.6, labelspacing=.45)
    if a.bare:
        # 제목도 주석도 빼고 패널 + 색상/task 범례만 남긴다 — 캡션이 그 역할을 한다.
        _apply_font_scale(f2, a.font_scale)
        # ★ 범례는 한 줄(ncol=4)을 유지해야 읽기 좋다. 본문과 같은 비율로 키우면
        #   가로를 넘치므로, 스케일을 되돌려 조금만 키운다.
        if a.font_scale > 1.0:
            for t in leg2.get_texts():
                t.set_fontsize(8.8 * min(a.font_scale, 1.15))
        bb = "tight" if a.font_scale >= 1.2 else None
        for ext in ("png", "pdf"):
            fp = out / f"{stem}.{ext}"
            f2.savefig(fp, dpi=200 if ext == "png" else None, bbox_inches=bb)
            print(f"saved -> {fp}")
        plt.close(f2)
        return
    dot = (f"each dot = mean of {a.n_avg} held-out conditions" if a.n_avg > 1
           else "each dot = one held-out condition")
    axtxt = ("axes = the plane spanned by the eight condition means"
             if a.proj == "cond" else
             "axes = principal components of the plotted points")
    f2.text(.5, .30 / fh2, f"{dot} · {axtxt}", ha="center", va="bottom",
            fontsize=8.2, color="0.45")
    qtxt = (rf"$(x^*_t,\ t^*\!=\!{a.t_star})$" if a.euler_steps <= 0 else
            rf"the same start $x^*$, {a.euler_steps} Euler steps")
    f2.text(.5, .12 / fh2, "both panels share axes, scale and " + qtxt,
            ha="center", va="bottom", fontsize=8.2, color="0.45")
    nm2, _, expr2 = quantity_label(a)
    nm2 = nm2[0].upper() + nm2[1:]
    f2.suptitle((f"{nm2} — condition component, one shared start\n" + expr2 +
                 " minus its mean over the four conditions"
                 if a.vel_view != "raw" else
                 f"{nm2}, each task under its own condition\n" + expr2 +
                 ",  $c_j$ = (observation, state, language) of task $j$"),
                fontsize=11.5, y=1 - .14 / fh2)
    _apply_font_scale(f2, a.font_scale)
    bb = "tight" if a.font_scale >= 1.2 else None
    for ext in ("png", "pdf"):
        fp = out / f"{stem}.{ext}"
        f2.savefig(fp, dpi=200 if ext == "png" else None, bbox_inches=bb)
        print(f"saved -> {fp}")
    plt.close(f2)

    # ── 통계 ─────────────────────────────────────────────────────────────────
    L = [f"{quantity_label(a)[0]} — 공통 출발 x* (시드 {a.x_seed})"
         + (f", t* = {a.t_star}" if a.euler_steps <= 0
            else f", Euler {a.euler_steps} 스텝 적분"),
         f"pose {POSE_DIMS} 축 x 청크 16 = {X.shape[1]} 차원, task 당 {a.n_cond} 개 "
         f"{a.split} condition", "",
         f"그린 것: --vel_view {a.vel_view}", "",
         f"투영 --proj {a.proj}  축 {lab[0]}/{lab[1]}  "
         f"{100*frac[0]:.1f}% / {100*frac[1]:.1f}%  (두 패널 공통 축·공통 범위)",
         f"점 하나 = 조건 {a.n_avg} 개의 평균 (--n_avg; 평균 위치와 ‖Δv‖ 는 불변, "
         f"표본오차만 1/√{a.n_avg})", "",
         "── 조건이 응답을 가르는가 ──",
         "between = 조건 평균끼리의 평균 거리, within = 조건 안 평균 산포 (96 차원 원공간)",
         "ratio 가 1 보다 충분히 커야 조건이 응답을 가른 것이다.", ""]
    for view in ("raw", "centered"):
        tag = ("응답 원값 — 관측 산포가 그대로 섞여 있다" if view == "raw" else
               "표본별로 네 조건의 평균을 뺀 조건 성분 — 관측이 상쇄된다")
        if view == a.vel_view:
            tag += "  ← main 패널"
        L.append(f"[{view}] {tag}")
        g = build_groups(V, rn, tasks, view)
        for name in rn:
            s = separation_from(g, name, tasks)
            L.append(f"  {name:<8} between {s['between']:.4f}  within {s['within']:.4f}  "
                     f"ratio {s['ratio']:.2f}")
        L.append("")
    L += ["── 무엇이 velocity 를 움직이는가 (질의 x*, t* 는 고정) ──",
          "lang = 관측을 고정하고 명령어만 바꿨을 때의 평균 변화량 (표본별 쌍 비교)",
          "obs  = 명령어를 고정하고 관측만 바꿨을 때의 평균 변화량 (표본 인덱스를 맞춘 크기 추정)",
          "언어가 속도장을 가른다면 lang 이 커야 한다. obs 는 비교용 척도다.", ""]
    for name in rn:
        lang = float(np.mean([
            np.linalg.norm(V[f"{name}|obs{j}|ins{i}"] - V[f"{name}|obs{j}|ins{k}"], axis=1).mean()
            for j in tasks for i in tasks for k in tasks if i < k]))
        obs = float(np.mean([
            np.linalg.norm(V[f"{name}|obs{j}|ins{i}"] - V[f"{name}|obs{j2}|ins{i}"], axis=1).mean()
            for i in tasks for j in tasks for j2 in tasks if j < j2]))
        nv = float(np.mean([np.linalg.norm(V[f"{name}|obs{j}|ins{j}"], axis=1).mean()
                            for j in tasks]))
        L.append(f"  {name:<8} lang {lang:.4f} ({100 * lang / nv:.1f}% of ‖v‖)   "
                 f"obs {obs:.4f} ({100 * obs / nv:.1f}%)   ‖v‖ {nv:.4f}")
    L.append("")
    L += ["조건별 Task SR (왼쪽 패널, 10 rollout 기준)", ""]
    for name in rn:
        L.append(f"  {name:<8}" + "".join(
            f"  t{j}: {traj[f'{name}|{j}']['sr']:5.0f}%" for j in tasks))
    L += ["", "Task 0 보존 (아래 곡선, 10 rollout 기준)", ""]
    for name in rn:
        L.append(f"  {name:<8}" + "".join(
            f"  after 0–{k}: {srs[f'{name}|{k}']['sr']:5.0f}%" for k in tasks))
    (out / "fig3_stats.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
