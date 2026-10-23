#!/usr/bin/env python
"""LOTUS 셋팅 lifelong 학습 — L2_codebook v3 (grad) 앵커.

l2_codebook_id.py 와 다른 점은 **출발점**뿐이다. 원본은 태스크 0 부터 하나씩 배우며
태스크가 끝날 때마다 코드북을 만드는데, LOTUS 셋팅은 base 6 태스크를 **joint 로**
먼저 배운 체크포인트에서 출발한다. 그래서 두 가지를 앞에 덧붙인다.

  (1) base 코드북을 **태스크별로 6개** 만든다.
      base 를 통째로 코드북 하나로 만들면 안 된다 — 앵커는 코드북에서 뽑은 (s̃, õ) 에
      instructions[task j] 를 물리는데, 6 태스크가 섞인 코드북에서 뽑으면
      "task 2 장면인데 명령어는 task 5" 같은 조합이 생긴다. 원본이 시간-bin 을
      버리면서 없애려던 바로 그 불일치다. base 데이터셋의 task_index 로 갈라
      태스크마다 build_codebook 을 부르면 s̃·õ·ℓ 이 전부 같은 태스크에서 나온다.

  (2) teacher 는 base 체크포인트 스냅샷이다.
      원본은 태스크 종료 시 snapshot(policy) 로 teacher 를 뜬다. 여기서는 그
      "태스크 0 종료 시점"이 base phase 종료 시점이고, 그 가중치가 이미 디스크에
      있다. 로드한 정책을 그대로 스냅샷하면 stage 1 부터 앵커가 걸린다.

수집은 저수지 표본을 쓰지 않고 **전 프레임**을 담는다. lifelong 스테이지가 10 demo
(약 1,500 프레임)뿐이라 n_pairs=8000 을 쓰면 같은 프레임이 5배 넘게 중복되고, 셀당
유효 표본이 표시된 n_k 의 1/5 로 떨어진다. grad_min_frames 같은 임계가 부풀려진 수
위에서 걸리게 되므로 여기서는 원본 프레임을 그대로 쓴다.

SR 행렬
    행 = 스테이지 (0 = base 직후, 1..4 = lifelong 각 단계)
    열 = 스위트 태스크 0..9   (base 가 0~5, stage k 가 5+k 를 새로 배운다)
    행 k 에서는 그때까지 본 태스크 0..5+k 만 평가한다 (하삼각).

    python Lotus_l2cb_cl.py --suite goal   --gpu 0
    python Lotus_l2cb_cl.py --suite object --gpu 1
    python Lotus_l2cb_cl.py --suite 50     --gpu 2

스위트별 구조 (lotus_setting.json 에서 읽는다 — 코드에 상수로 박지 않는다)
    goal/object   base 6 task,  4 stage x 1 task  = 10 task
    50            base 25 task, 5 stage x 5 task  = 50 task
                  ★ 스테이지마다 새 태스크가 5개다. 한 행에 FWT 가 5개씩 들어간다.
                  ★ 평가 핸들이 Libero_90_*(46) 와 Libero_10_*(4) 로 섞인다.
                  ★ 코드북은 task_index 가 아니라 에피소드 블록으로 가른다
                    (stage3 은 명령어가 같은 두 태스크가 한 task_index 에 겹친다).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# 알고리즘은 전부 독립 실행본에서 가져온다 — 한 곳에만 정의가 있게 한다.
import l2_codebook_id as L2                                            # noqa: E402
from l2_codebook_id import (                                           # noqa: E402
    build_codebook, cond_tail, encode_lang, fm_loss, make_cond, prep_batch,
    rgb_cls, sample_codebook, sample_fm, snapshot, to_device_cb,
)

from lerobot.configs.default import DatasetConfig, EvalConfig          # noqa: E402
from lerobot.configs.policies import PreTrainedConfig                  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig                  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps          # noqa: E402
from lerobot.datasets.lerobot_dataset import (                         # noqa: E402
    LeRobotDataset, LeRobotDatasetMetadata,
)
from lerobot.datasets.utils import cycle                               # noqa: E402
from lerobot.envs.configs import LiberoEnv                             # noqa: E402
from lerobot.optim.factory import make_optimizer_and_scheduler         # noqa: E402
from lerobot.policies.factory import make_policy                       # noqa: E402
from lerobot.scripts.E0 import to_device                               # noqa: E402
from lerobot.utils.random_utils import set_seed                        # noqa: E402
from lerobot.utils.train_utils import (                                # noqa: E402
    get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging    # noqa: E402

# 경로는 환경변수를 먼저 보고, 없으면 서버1 의 기존 값을 쓴다 — 서버1 거동은
# 그대로이고 다른 머신(쿠버네티스 파드 등)에서는 bash/clare/env.sh 가 세운
# HF_LEROBOT_HOME / CLARE_MODEL_ROOT 를 따라간다.
LOTUS_ROOT = Path(os.environ.get("LOTUS_DATA_ROOT")
                  or Path(os.environ.get("HF_LEROBOT_HOME",
                                         "/home/sa090180/Datasets/lerobot")) / "LOTUS_setting")
MODEL_ROOT = Path(os.environ.get("CLARE_MODEL_ROOT", "/home/sa090180/Models"))
SETTING_JSON = LOTUS_ROOT / "lotus_setting.json"
LAMBDA_ANCHOR = 1.0       # L = L_FM + λ_a · L_anchor  (l2_codebook_id 와 같다)

SUITE_DIR = {"goal": "LIBERO-GOAL", "object": "LIBERO-OBJECT", "50": "LIBERO-50"}

# gym_libero 핸들 접두사. lotus_setting.json 의 entry["source"] 로 고른다.
_SRC_PREFIX = {"libero_goal": "Libero_Goal_Task_",
               "libero_object": "Libero_Object_Task_",
               "libero_spatial": "Libero_Spatial_Task_",
               "libero_90": "Libero_90_Task_",
               "libero_10": "Libero_10_Task_"}


def _env_handle(e: dict) -> str:
    """lotus_setting.json 의 태스크 항목 -> gym_libero 핸들.

    goal/object 는 번호가 스위트 안에서 닫혀 있어 suite_index 를 그대로 쓴다.
    LIBERO-50 은 LIBERO-100 인덱스라 90 미만이면 libero_90 의 그 번호,
    90 이상이면 libero_10 의 (index-90) 번이다 (lotus_setting.json 의 verified 항목).
    한 스위트 안에서 핸들 접두사가 **섞인다** — goal/object 와 결정적으로 다른 점이다.
    """
    pre = _SRC_PREFIX[e["source"]]
    if "suite_index" in e:                      # goal / object
        return f"{pre}{e['suite_index']}"
    i = e["libero100_index"]                    # LIBERO-50
    return f"{pre}{i - 90 if e['source'] == 'libero_10' else i}"


def _benchmark_of(handle: str) -> str:
    """핸들 -> benchmark 이름. LiberoEnv.resolved_benchmark 와 같은 표를 쓴다."""
    from lerobot.envs.configs import LIBERO_TASK_PREFIX_TO_BENCHMARK
    return LIBERO_TASK_PREFIX_TO_BENCHMARK.get(handle.rsplit("_Task_", 1)[0], "libero_90")


def suite_spec(suite: str) -> dict:
    """suite -> 실험 구조 전부. lotus_setting.json 이 유일한 출처다.

    스위트 태스크 id 는 base 를 0..n_base-1 로 두고 그 뒤에 stage1, stage2, ...
    순서로 이어 붙인 번호다. goal/object 에서는 기존 번호(base 0..5, stage k -> 5+k)
    와 정확히 같고, LIBERO-50 에서는 base 0..24, stage1 이 25..29, stage5 가 45..49 다.

    ★ LIBERO-50 이 goal/object 와 다른 점 세 가지
      1. base 가 25 태스크 (goal/object 는 6).
      2. stage 가 5 개이고 **스테이지마다 새 태스크가 5 개**다 (goal/object 는 1 개).
         lotus_setting.json 의 assumptions 에 적힌 LOTUS 분할 그대로다.
      3. 평가 핸들이 Libero_90_* 와 Libero_10_* 로 섞인다.

    반환 키
      dir/root      LOTUS_setting 하위 디렉터리
      n_base        base phase 가 joint 로 배운 태스크 수
      stage_tasks   [stage1 이 새로 배우는 태스크 id 리스트, stage2 .., ..]
      n_task        전체 태스크 수
      env_task      태스크 id -> gym 핸들
      instr         태스크 id -> 명령어
      base_demos    base 스플릿의 **태스크당** demo 수
      stage_demos   stage 스플릿의 태스크당 demo 수
      learned_at    태스크 id -> 그 태스크를 배운 스테이지 (base 는 전부 0)
    """
    if suite not in SUITE_DIR:
        raise SystemExit(f"suite 는 {'|'.join(SUITE_DIR)} 중 하나 (받은 값: {suite})")
    d = SUITE_DIR[suite]
    exp = json.loads(SETTING_JSON.read_text())["experiments"][d]

    env_task: dict[int, str] = {}
    instr: dict[int, str] = {}
    learned: dict[int, int] = {}
    for t, e in enumerate(exp["base"]):
        env_task[t], instr[t], learned[t] = _env_handle(e), e["instruction"], 0
    stage_tasks: list[list[int]] = []
    nxt = len(exp["base"])
    for s in exp["lifelong_stages"]:
        ids = []
        for e in s["tasks"]:
            env_task[nxt] = _env_handle(e)
            instr[nxt] = e["instruction"]
            learned[nxt] = s["stage"]
            ids.append(nxt)
            nxt += 1
        stage_tasks.append(ids)
    if len(set(env_task.values())) != nxt:
        raise SystemExit("핸들이 중복이다 — lotus_setting.json 의 인덱스를 확인하라")
    return {"dir": d, "root": LOTUS_ROOT / d,
            "n_base": len(exp["base"]), "stage_tasks": stage_tasks, "n_task": nxt,
            "env_task": env_task, "instr": instr,
            "base_demos": exp["base_n_demos"], "stage_demos": exp["lifelong_n_demos"],
            "learned_at": [learned[t] for t in range(nxt)]}


def load_split(root: Path, policy_cfg):
    """LOTUS_setting 의 한 스플릿을 연다.

    ★ delta_timestamps 를 반드시 넣어야 한다. 없으면 observation.state 가 (B,8) 로,
      action 이 (B,7) 로 나와 (B,2,8) / (B,16,7) 을 기대하는 정책과 어긋난다.
      make_dataset(cfg) 이 내부에서 하는 일과 같다(datasets/factory.py:78-127).
    """
    meta = LeRobotDatasetMetadata("local/lotus", root=str(root))
    dt = resolve_delta_timestamps(policy_cfg, meta)
    return LeRobotDataset("local/lotus", root=str(root), delta_timestamps=dt)


def instructions_of(root: Path) -> dict[int, str]:
    """스플릿의 task_index -> 명령어."""
    return {json.loads(l)["task_index"]: json.loads(l)["task"]
            for l in (root / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()}


# ═════════════════════════════════════════════════════════════════════════════
#  (s, o) 수집 — 저수지 없이 전 프레임
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_pairs(policy, dataset, device, batch_size: int, num_workers: int,
                  n_demos: int):
    """데이터셋을 한 바퀴 돌며 (s 16, o 3072) 쌍을 모은다.

    그래디언트가 없고 DINOv2 는 동결이라 forward 만 돈다. 저수지를 쓰지 않으므로
    **모든 프레임이 정확히 한 번씩** 들어간다.
    반환은 (태스크별 (s,o), 태스크별 demo 수) 두 개다 — demo 수로 K 를 정한다.

    ★ 태스크 구분은 task_index 가 아니라 **에피소드 블록**(episode_index // n_demos)
      으로 한다. LOTUS 스플릿은 태스크마다 n_demos 개 에피소드가 연속으로 들어 있고
      그 순서가 lotus_setting.json 의 태스크 순서와 같다(전 스플릿 검증함).
      task_index 를 쓰면 LIBERO-50/stage3 처럼 **명령어가 같은 서로 다른 두 태스크**가
      한 인덱스로 합쳐져(10+10=20 ep) 코드북 하나로 뭉개진다. 앵커가 없애려던
      "장면과 명령어가 어긋나는 조합"이 그 안에서 되살아난다.
      goal/object 에서는 두 방식의 결과가 완전히 같다(episode_index//50 == task_index).
    """
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False,
        pin_memory=device.type == "cuda", drop_last=False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    buf: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    act: dict[int, list[torch.Tensor]] = {}   # 태스크별 액션 -> c_j 의 speed_med2
    eps: dict[int, set] = {}          # 태스크별 에피소드 집합 -> demo 수
    policy.eval()
    for raw in loader:
        batch = prep_batch(policy, to_device(raw, device))
        cls = rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        s = batch["observation.state"].detach().flatten(1).float().cpu()   # (B,16)
        o = cls.detach().reshape(n, -1).float().cpu()                      # (B,3072)
        a = batch["action"].detach().float().cpu()                         # (B,H,A)
        ei = raw["episode_index"].cpu()
        ti = ei // n_demos                      # 에피소드 블록 = 스플릿 내 태스크 위치
        for t in ti.unique().tolist():
            m = ti == t
            buf.setdefault(int(t), []).append((s[m], o[m]))
            act.setdefault(int(t), []).append(a[m])
            eps.setdefault(int(t), set()).update(ei[m].tolist())
    policy.train()
    return ({t: (torch.cat([x for x, _ in v]), torch.cat([y for _, y in v]))
             for t, v in buf.items()},
            {t: len(v) for t, v in eps.items()},
            {t: torch.cat(v) for t, v in act.items()})


def make_books(policy, root: Path, device, args, task_ids: list[int],
               n_demos: int) -> dict[int, dict]:
    """스플릿 하나에서 코드북을 만든다. task_ids 는 스위트 기준 태스크 번호로,
    스플릿 안의 에피소드 블록 순서(0,1,2,..)와 같은 순서여야 한다.

    ★ 셀 수 K 는 **demo 수에 비례**해서 정한다 (K = k_per_demo x demo).
      LOTUS 셋팅은 base 가 태스크당 50 demo, lifelong 이 10 demo 로 5배 차이가 난다.
      둘에 같은 K 를 쓰면 lifelong 쪽 셀당 표본이 10개 안팎으로 떨어지고,
      grad_min_frames=24 에 전부 걸려 **A_k 가 모든 셀에서 0 으로 강등된다**
      (실측: 997 프레임 / K=96 -> 88셀 전부 강등). 그러면 이름만 v3 이고 실제로는
      v1(0차) 앵커가 된다.

      k_per_demo=2 면  50 demo -> K=100,  10 demo -> K=20.
      실측 강등 비율(K=20): GOAL 3~5/20, OBJECT 0~1/20 로 떨어진다.
    """
    ds = load_split(root, policy.config)
    pairs, ndemo, acts = collect_pairs(policy, ds, device, args.batch_size,
                                       args.num_workers, n_demos)
    if len(pairs) != len(task_ids):
        raise SystemExit(f"{root.name}: 에피소드 블록 {len(pairs)}개인데 "
                         f"태스크 {len(task_ids)}개다 — n_demos={n_demos} 를 확인하라")
    books = {}
    for local_idx, (s, o) in sorted(pairs.items()):
        suite_task = task_ids[local_idx]
        K = (args.codebook_k if getattr(args, "codebook_k", 0) > 0
             else max(args.k_min, args.k_per_demo * ndemo[local_idx]))
        cb = build_codebook(s.to(device), o.to(device), K, seed=args.seed,
                            h_scale=args.h_scale, ridge_rho=args.ridge_rho,
                            grad_min_frames=args.grad_min_frames)
        # c_j 의 재료. 액션은 MIN_MAX 로 [-1,1] 이라 스케일이 태스크마다 다르지 않다.
        cb["speed_med2"] = action_speed_med2(acts[local_idx])
        nk = cb["n_k"]
        logging.info(
            f"[LOTUS] 코드북 task{suite_task}  demo={ndemo[local_idx]} -> K={K}  "
            f"N={cb['N']} (전 프레임)  "
            f"K_eff={cb['K_eff']} (병합 {cb['n_merged']})  "
            f"n_k min/med/max={int(nk.min())}/{int(nk.median())}/{int(nk.max())}  "
            f"h={cb['h']:.4f}  강등 {cb['n_demoted']}셀  "
            f"speed_med2={cb['speed_med2']:.4f}")
        books[suite_task] = to_device_cb(cb, device)
    del ds
    return books


# ═════════════════════════════════════════════════════════════════════════════
#  v4 — 세 가지 변경 (전부 플래그. 기본값이면 아래 함수를 아예 타지 않는다)
# ═════════════════════════════════════════════════════════════════════════════
POSE_DIMS = 6      # action = [x,y,z,roll,pitch,yaw,gripper] -> 0..5 pose, 6 gripper
                   # (LOTUS_setting meta/info.json features.action.names 로 확인)


def sample_codebook_hard(cb: dict, n: int, gen=None):
    """(1) 경성 배정 + 1차 항만. 커널 가중 w·대역폭 h·잡음 항 σ_o 를 쓰지 않는다.

        k* ~ Cat(π)
        s̃  = μ_{s,k*} + σ_{s,k*} ⊙ ε          (기존과 동일)
        z̃  = (s̃ − s̄)/std(s)
        õ  = m_{k*} + A_{k*}(z̃ − z̄_{k*})      (강등 셀은 A=0 이라 õ = m_{k*})

    기존 soft_kernel 은 Σ_j w_j(m_j + A_jδ_j) + sqrt(Σ_j w_j σ_o,j²)⊙ε′ 이다.
    코드북 생성은 건드리지 않아 파일 호환이 유지된다.
    """
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]
    k = torch.multinomial(pi, n, replacement=True, generator=gen)      # (n,)
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)
    s = mu_s[k] + sig_s[k] * eps                                      # (n,16)
    z = (s - cb["mean_s"]) / cb["std_s"]                              # (n,16)
    d = (z - cb["zbar"][k]).unsqueeze(2)                               # (n,16,1)
    A = cb["A"][k].float()                                             # (n,3072,16)
    o = cb["m"][k] + torch.bmm(A, d).squeeze(2)                        # (n,3072)
    return s, o


def speed_weights(v_ref: torch.Tensor, eps_noise: torch.Tensor, c: float,
                  w_clip: float) -> torch.Tensor:
    """(2)(3) 공통 — 기준 속도의 역제곱 가중 w̃. (B,H,A) 를 받아 (B,H) 를 돌려준다.

        ṽ_ref = v_ref + ε                     de-noised 프레임
        ρ[b,s] = ‖ṽ_ref[b,s,0:6]‖             pose 성분의 스텝 속도
        w      = 1 / (ρ² + c)
        w      = min(w, median(w)·w_clip)      위로만 자른다
        w̃      = w / mean(w)                   배치 평균 1 -> λ 스케일 불변

    θ 와 무관한 상수라 detach 한다. 느린 스텝(ρ 작음)에 큰 가중이 가서
    "정지 구간의 미세한 속도 차이"가 손실에 제대로 반영되게 하는 것이 목적이다.
    """
    with torch.no_grad():
        rho = (v_ref + eps_noise)[..., :POSE_DIMS].norm(dim=-1)        # (B,H)
        w = 1.0 / (rho.pow(2) + c)
        w = torch.minimum(w, w.median() * w_clip)
        return (w / w.mean().clamp_min(1e-12)).detach()                # (B,H)


def weighted_sq(resid: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """가중 제곱오차. pose 6차원에만 w 를 곱하고 gripper 는 그대로 둔다.

        L = (1/(H·A)) · mean_b Σ_s [ w[b,s]·Σ_{d<6} r² + r[...,6]² ]

    mean(w)=1 이므로 평균 스케일이 기존 MSE 와 같아 λ_lvl 을 다시 훑지 않아도 된다.
    """
    H, A = resid.shape[1], resid.shape[2]
    pose = (resid[..., :POSE_DIMS].pow(2).sum(-1) * w).sum(1)          # (B,)
    grip = resid[..., POSE_DIMS:].pow(2).sum(-1).sum(1)                # (B,)
    return ((pose + grip) / (H * A)).mean()


def action_speed_med2(actions: torch.Tensor) -> float:
    """c_j 의 재료 — median_{데모·스텝} ‖a_s[0:6]‖². actions: (N, A) 또는 (N,H,A)."""
    a = actions.reshape(-1, actions.shape[-1])
    return float(a[:, :POSE_DIMS].pow(2).sum(-1).median())
# ═════════════════════════════════════════════════════════════════════════════
#  앵커 손실 — l2_codebook_id.L2CodebookAnchor.loss 와 같은 계산
# ═════════════════════════════════════════════════════════════════════════════
def anchor_loss(policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
                lam_lvl, device, args=None, diag=None):
    """과거 태스크마다 코드북에서 (s̃, õ) 를 뽑아 teacher 와 속도장을 맞춘다.

    청크 backward 고정 — j 마다 즉시 backward 하고 그래프를 버린다. 그래서 앵커
    그래디언트는 과거에 대한 **합**이 된다(평균이 아니다). 반환값은 로그용이다.

    args 가 없거나 기본값이면 v3 와 **비트 단위로 같은 경로**를 탄다.
      --sampler_mode hard_linear  -> 샘플러를 경성 배정 + 1차 항으로
      --anchor_weight speedinv    -> L_j 를 teacher 속도 역제곱으로 가중
    FM 손실은 항상 기존 MSE 다 — 속도 정규화도 EMA 도 쓰지 않는다.
    """
    sampler = getattr(args, "sampler_mode", "soft_kernel") if args else "soft_kernel"
    aw = getattr(args, "anchor_weight", "mse") if args else "mse"
    n = x_t.shape[0]
    tcol = t[:, None, None]
    lvl = []
    for j in sorted(books):
        cb = books[j]
        s_t, o_t = (sample_codebook_hard(cb, n) if sampler == "hard_linear"
                    else sample_codebook(cb, n))
        s_t, o_t = s_t.detach(), o_t.detach()
        bb = {"observation.state": s_t.view(st_shape).to(x_t.dtype)}
        cls_j = o_t.reshape(-1, cls_dim).to(x_t.dtype)
        past = [instructions[j]] * n

        tl_S = cond_tail(policy, bb, cls_j)
        with torch.no_grad():
            tl_T = cond_tail(teacher, bb, cls_j)
            cond_T = make_cond(encode_lang(teacher, past), tl_T)
        cond_S = make_cond(encode_lang(policy, past), tl_S)

        def vel(pol, xx, tt, cond):
            return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

        # teacher 부트스트랩 x_t — teacher 가 실제로 지나가는 지점에서 잰다
        with torch.no_grad():
            eps0 = torch.randn_like(x_t)
            a_hat = (eps0 + vel(teacher, eps0, torch.zeros_like(t), cond_T)).detach()
        eps1 = torch.randn_like(x_t)
        x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()

        with torch.no_grad():
            vt0 = vel(teacher, x_j, t, cond_T)
        vs0 = vel(policy, x_j, t, cond_S)
        resid = vs0 - vt0.to(vs0.dtype)                       # (B,H,A)
        if aw == "speedinv":
            # ṽ_T = v_T + ε₁ 프레임에서 잰 teacher 속도로 가중.
            # 잔차는 프레임을 옮겨도 같다: ṽ_S − ṽ_T = v_S − v_T.
            w = speed_weights(vt0, eps1, cb["speed_med2"] * args.kappa, args.w_clip)
            L_j = weighted_sq(resid, w)
            if diag is not None:
                diag.setdefault("w_mean", []).append(float(w.mean()))
                diag.setdefault("w_max", []).append(float(w.max()))
        else:
            L_j = resid.flatten(1).pow(2).mean(1).mean()
        (LAMBDA_ANCHOR * lam_lvl * L_j).backward()
        lvl.append(L_j.detach())
        if diag is not None and diag.get("sanity_pending"):
            # ṽ_T ≈ â 여야 한다. 0.2 를 넘으면 잡음 규약이 어긋난 것이다.
            with torch.no_grad():
                r = float((vt0 + eps1 - a_hat).norm(dim=-1).mean()
                          / a_hat.norm(dim=-1).mean().clamp_min(1e-9))
            diag["sanity_ratio"] = r
            diag["sanity_pending"] = False
    if not lvl:
        return torch.zeros((), device=device)
    return (lam_lvl * (sum(lvl) / len(lvl))).detach()


# ═════════════════════════════════════════════════════════════════════════════
#  평가
# ═════════════════════════════════════════════════════════════════════════════
def build_eval_cfg(args, policy_path: str, env_task: str) -> TrainPipelineConfig:
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.push_to_hub = False
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/lotus"),
        # LIBERO-50 은 핸들이 Libero_90_*/Libero_10_* 로 섞여 "libero_50" 같은
        # 벤치마크 이름이 없다. 핸들에서 되찾는다(LiberoEnv.resolved_benchmark 와 동일).
        env=LiberoEnv(benchmark=_benchmark_of(env_task), task=env_task),
        policy=policy_cfg, output_dir=None, job_name="lotus_eval",
        seed=args.eval_seed, num_workers=args.num_workers,
        batch_size=args.batch_size, steps=args.steps, eval_freq=0,
        eval=EvalConfig(n_episodes=args.eval_episodes,
                        batch_size=args.eval_batch_size, max_episodes_rendered=0),
    )
    cfg.optimizer = policy_cfg.get_optimizer_preset()
    cfg.scheduler = policy_cfg.get_scheduler_preset()
    cfg.checkpoint_path = None
    return cfg


def rollout_sr(policy, cfg, env_task: str, args) -> float | None:
    """시뮬레이터 롤아웃 성공률(%). 시작 시드는 --eval_seed."""
    import copy
    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import eval_policy
    env = None
    try:
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.task = env_task
        env = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)
        policy.eval()
        with torch.no_grad():
            info = eval_policy(env, policy, args.eval_episodes, start_seed=args.eval_seed)
        return float(info["aggregated"]["pc_success"])
    except Exception as e:
        logging.warning(f"[LOTUS] SR 실패 ({env_task}): {type(e).__name__}: {e}")
        return None
    finally:
        if env is not None:
            env.close()
        policy.train()


# ═════════════════════════════════════════════════════════════════════════════
#  지표 — tools/plot_sr_matrix.py · l2_family_report.py 와 같은 LIBERO 정의
# ═════════════════════════════════════════════════════════════════════════════
def metrics(R, learned_at, n_task):
    """FWT / NBT / AUC / AvgSR.

        FWT_t = R[learned_at(t)][t]              배운 직후
        NBT_t = mean_{k>learned} (FWT_t − R[k][t])   이후 잃은 양 (+면 망각)
        AUC_t = mean_{k>=learned} R[k][t]        시퀀스 내내 낸 평균
    """
    fwt, nbt, auc = [], [], []
    for t in range(n_task):
        col = [R[k][t] for k in range(learned_at[t], len(R)) if R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    last = [R[-1][t] for t in range(n_task) if R[-1][t] is not None]
    mean = lambda v: (sum(v) / len(v)) if v else None                  # noqa: E731
    return {"AvgSR": mean(last), "FWT": mean(fwt), "NBT": mean(nbt), "AUC": mean(auc)}


def write_report(out: Path, R, learned_at, instr, args, n_task, spec):
    m = metrics(R, learned_at, n_task)
    n_stage = len(spec["stage_tasks"])
    per_stage = len(spec["stage_tasks"][0])
    bd, sd = spec["base_demos"], spec["stage_demos"]
    W = max(86, 8 + 6 * n_task)                 # 태스크가 50개면 표가 그만큼 넓어진다
    L = ["=" * W,
         f"LOTUS 셋팅 lifelong — libero_{args.suite}  (L2_codebook v3 앵커)",
         "=" * W, "",
         f"base phase   {spec['n_base']} task joint, 200,000 step (체크포인트 재사용)",
         f"lifelong     {n_stage} stage x {per_stage} task x {sd} demo, "
         f"stage 당 {args.steps:,} step",
         f"평가         칸당 {args.eval_episodes} rollout, eval seed {args.eval_seed}",
         f"코드북       K = {args.k_per_demo} x demo  (base {bd} demo -> {args.k_per_demo*bd}, "
         f"lifelong {sd} demo -> {args.k_per_demo*sd}), 전 프레임 수집(저수지 없음), "
         f"ridge_rho={args.ridge_rho}, λ_lvl={args.lambda_level}", "",
         "지표 (LIBERO 벤치마크 Liu et al. 2023 — L2_SR_matrix.txt 와 같은 정의)",
         "  FWT = mean_t R[배운 스테이지][t]                높을수록 좋음",
         "  NBT = mean_t mean_{k>배운} (FWT_t − R[k][t])    **낮을수록** 좋음",
         "  AUC = mean_t mean_{k>=배운} R[k][t]             높을수록 좋음",
         "  AvgSR = 마지막 행 평균", "",
         "-" * W,
         "SR matrix   행 = 스테이지(0 = base 직후), 열 = 스위트 태스크",
         "-" * W,
         f"{'stage':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(n_task))]
    for k in range(len(R)):
        L.append(f"{k:<8}" + "".join(
            (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
            for t in range(n_task)))
    f1 = lambda v: "-" if v is None else f"{v:.1f}"                    # noqa: E731
    fs = lambda v: "-" if v is None else f"{v:+.1f}"                   # noqa: E731
    L += ["-" * W,
          f"AvgSR {f1(m['AvgSR'])}   FWT↑ {f1(m['FWT'])}   "
          f"NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}", "",
          "태스크별 (배운 스테이지 / 명령어)"]
    for t in range(n_task):
        L.append(f"  t{t:<3} stage{learned_at[t]}  {spec['env_task'][t]:<20} "
                 f"{instr.get(t, '')[:60]}")
    L += ["", "주의",
          f"  · 칸당 {args.eval_episodes} rollout 이라 이항 표준오차가 개별 칸 최대 "
          f"±{50/ (args.eval_episodes ** 0.5):.0f}%p 다. 행/열 평균으로 읽어야 한다.",
          f"  · base {spec['n_base']} 태스크는 joint 로 배웠으므로 stage 0 이 그 "
          f"{spec['n_base']}개의 FWT 다.",
          "  · 앵커는 청크 backward 라 과거 태스크 수에 비례해 실효 가중이 커진다.", ""]
    if per_stage > 1:
        L += [f"  · 이 스위트는 스테이지마다 {per_stage} 태스크를 **joint 로** 배운다. "
              f"따라서 한 행에 새 태스크 {per_stage}개의 FWT 가 함께 들어간다.", ""]
    out.write_text("\n".join(L) + "\n")
    return m


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["goal", "object", "50"], required=True)
    ap.add_argument("--steps", type=int, default=10000, help="lifelong 스테이지당 스텝")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--k_per_demo", type=int, default=2,
                    help="코드북 셀 수를 demo 수에 비례해 정한다: K = k_per_demo x demo. "
                         "base 50 demo -> K=100, lifelong 10 demo -> K=20.")
    ap.add_argument("--k_min", type=int, default=8, help="셀 수 하한")
    ap.add_argument("--codebook_k", type=int, default=0,
                    help="0 이면 K = k_per_demo x demo (기본 규칙). >0 이면 **모든 태스크**의 "
                         "셀 수를 이 값으로 고정한다(base 도 포함). demo 수에 비례시키지 않고 "
                         "셀 수를 직접 통제하고 싶을 때 쓴다.")
    ap.add_argument("--skip_base_eval", action="store_true",
                    help="base 태스크의 롤아웃을 건너뛴다. lifelong 구간만 재면 될 때 쓴다 — "
                         "libero_50 은 base 가 25 태스크라 stage 마다 25 칸을 더 돌게 되고, "
                         "그게 전체 시간의 대부분이다. ★ base 열은 비워지므로 리포트의 "
                         "지표도 lifelong 태스크만으로 계산된다(base 는 애초에 안 잰다).")
    # ── v4 세 가지 변경. 기본값이면 v3 와 완전히 같은 경로를 탄다 ──────────
    ap.add_argument("--sampler_mode", choices=["soft_kernel", "hard_linear"],
                    default="soft_kernel",
                    help="hard_linear: 경성 배정 + 1차 항만 (커널 가중·잡음 항 없음)")
    ap.add_argument("--anchor_weight", choices=["mse", "speedinv"], default="mse",
                    help="speedinv: teacher 예측 속도의 역제곱으로 앵커 손실 가중")
    ap.add_argument("--kappa", type=float, default=1.0, help="c = kappa x speed_med2")
    ap.add_argument("--w_clip", type=float, default=20.0, help="w 를 median(w)x이 값에서 자름")
    ap.add_argument("--n_stages", type=int, default=0,
                    help="lifelong 스테이지 수를 앞에서 이만큼만 쓴다. 0 = 전부. "
                         "스모크에서 1 로 줄이는 용도이고 본 run 에서는 건드리지 않는다.")
    ap.add_argument("--h_scale", type=float, default=1.0)
    ap.add_argument("--ridge_rho", type=float, default=0.05)
    ap.add_argument("--grad_min_frames", type=int, default=24)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42, help="학습/k-means 시드")
    ap.add_argument("--eval_seed", type=int, default=7, help="롤아웃 시작 시드")
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # GPU 지정은 셸 래퍼(Lotus_l2cb_cl.sh)가 torch import 전에 export 한다.
    # 여기서 세우면 CUDA 가 이미 초기화된 뒤라 늦을 수 있다.
    init_logging()
    spec = suite_spec(args.suite)
    if args.n_stages and args.n_stages < len(spec["stage_tasks"]):
        # 스모크용 축소. 잘린 뒤쪽 태스크는 평가·명령어 목록에서도 함께 빼야
        # SR 행렬의 열과 learned_at 이 어긋나지 않는다.
        spec["stage_tasks"] = spec["stage_tasks"][:args.n_stages]
        keep = spec["n_base"] + sum(len(x) for x in spec["stage_tasks"])
        spec["n_task"] = keep
        spec["env_task"] = {t: v for t, v in spec["env_task"].items() if t < keep}
        spec["instr"] = {t: v for t, v in spec["instr"].items() if t < keep}
        spec["learned_at"] = spec["learned_at"][:keep]
        logging.info(f"[LOTUS] ★ 스모크 축소: stage {args.n_stages}개, task {keep}개만 쓴다")
    root = spec["root"]
    n_base, n_task = spec["n_base"], spec["n_task"]
    stage_tasks = spec["stage_tasks"]
    n_stages = len(stage_tasks)
    env_task, instr = spec["env_task"], spec["instr"]
    base_ckpt = MODEL_ROOT / f"Lotus_clare_libero_{args.suite}_pretrain"
    if not (base_ckpt / "model.safetensors").exists():
        raise SystemExit(f"base 체크포인트가 없다: {base_ckpt}")
    out_dir = Path(args.out) if args.out else REPO / "results" / f"lotus_l2cb_{args.suite}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2, ensure_ascii=False)

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    # 태스크 t 를 배운 스테이지. base 태스크는 전부 스테이지 0.
    learned_at = spec["learned_at"]
    R: list[list[float | None]] = [[None] * n_task for _ in range(n_stages + 1)]

    # 명령어는 lotus_setting.json 에서 온다. 앵커가 이 문자열로 조건을 만들기 때문에
    # 데이터셋이 내놓는 batch["task"] 와 어긋나면 조용히 틀린 앵커가 된다 — 대조한다.
    for split, ids in [("base", list(range(n_base)))] + [
            (f"stage{i+1}", ids) for i, ids in enumerate(stage_tasks)]:
        have = set(instructions_of(root / split).values())
        miss = [instr[t] for t in ids if instr[t] not in have]
        if miss:
            raise SystemExit(f"{split}: lotus_setting.json 의 명령어가 데이터셋에 없다 "
                             f"({len(miss)}개, 예: {miss[0]!r})")
    logging.info(f"[LOTUS] suite={args.suite}  base {n_base} task  "
                 f"{n_stages} stage x {len(stage_tasks[0])} task  총 {n_task} task")

    # ── stage 0: base 체크포인트 로드 ────────────────────────────────────────
    cfg = build_eval_cfg(args, str(base_ckpt), env_task[0])
    base_meta = LeRobotDatasetMetadata("local/lotus", root=str(root / "base"))
    policy = make_policy(cfg=cfg.policy, ds_meta=base_meta)
    policy.to(device)
    logging.info(f"[LOTUS] base 체크포인트 로드 -> {base_ckpt}")

    t0 = time.perf_counter()
    books = make_books(policy, root / "base", device, args,
                       task_ids=list(range(n_base)), n_demos=spec["base_demos"])
    logging.info(f"[LOTUS] base 코드북 {len(books)}개 ({time.perf_counter()-t0:.0f}s)")
    teacher = snapshot(policy)

    if getattr(args, "skip_base_eval", False):
        logging.info(f"[LOTUS] base 롤아웃 건너뜀 (--skip_base_eval) — base {n_base}개 열은 빈 칸")
    else:
        for t in range(n_base):
            R[0][t] = rollout_sr(policy, cfg, env_task[t], args)
            logging.info(f"[LOTUS] SR stage0 task{t} ({env_task[t]}) = {R[0][t]}")
            write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    # ── stage 1..N ───────────────────────────────────────────────────────────
    for stage in range(1, n_stages + 1):
        new_tasks = stage_tasks[stage - 1]       # 이 스테이지가 새로 배우는 태스크들
        n_seen = new_tasks[-1] + 1               # 여기까지 본 태스크 수 (평가 범위)
        sroot = root / f"stage{stage}"
        ds = load_split(sroot, policy.config)
        loader = torch.utils.data.DataLoader(
            ds, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,
            pin_memory=device.type == "cuda", drop_last=False,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0)
        dl = cycle(loader)

        cfg.steps = args.steps
        optimizer, sched = make_optimizer_and_scheduler(cfg, policy)
        policy.train()
        logging.info(f"══ [LOTUS] stage {stage}  새 태스크 "
                     f"t{new_tasks[0]}..t{new_tasks[-1]} ({len(new_tasks)}개)  "
                     f"{ds.num_frames} frame  보유 코드북 {len(books)}개")

        diag = {"sanity_pending": args.anchor_weight == "speedinv"}
        for it in range(args.steps):
            batch = prep_batch(policy, to_device(next(dl), device))
            cls = rgb_cls(policy, batch)
            tail = cond_tail(policy, batch, cls)
            x_t, t_, target = sample_fm(policy, batch)   # target = a − ε
            cond = make_cond(encode_lang(policy, list(batch["task"])), tail)
            pred = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t_, global_cond=cond)

            loss_fm = fm_loss(pred, target)
            loss_fm.backward()                       # 앵커도 j 마다 즉시 backward 한다
            la = anchor_loss(policy, teacher, books, instr, x_t, t_,
                             batch["observation.state"].shape, cls.shape[-1],
                             args.lambda_level, device, args=args, diag=diag)
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.optimizer.grad_clip_norm,
                                           error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad()
            if sched is not None:
                sched.step()
            if it % args.log_every == 0 or it == args.steps - 1:
                extra = ""
                if "w_mean" in diag:
                    extra += (f" anc_w={sum(diag['w_mean'])/len(diag['w_mean']):.2f}"
                              f"/{max(diag['w_max']):.1f}")
                    diag["w_mean"], diag["w_max"] = [], []
                if "sanity_ratio" in diag:
                    extra += f" sanity(ṽ_T≈â)={diag.pop('sanity_ratio'):.3f}"
                logging.info(f"[LOTUS] stage{stage} step={it:6d} "
                             f"fm={float(loss_fm.detach()):.4f} anc={float(la):.4f} "
                             f"lr={optimizer.param_groups[0]['lr']:.2e}{extra}")
                with (out_dir / "diag.csv").open("a") as f:
                    f.write(f"{stage},{it},{float(loss_fm.detach()):.6f},"
                            f"{float(la):.6f}\n")

        # 체크포인트 저장 -> 코드북 추가 -> teacher 갱신
        sdir = out_dir / "checkpoints" / f"stage{stage}"
        ck = get_step_checkpoint_dir(sdir, args.steps, args.steps)
        save_checkpoint(ck, args.steps, cfg, policy, optimizer, sched)
        update_last_checkpoint(ck)
        del loader, dl, ds
        torch.cuda.empty_cache()

        books.update(make_books(policy, sroot, device, args,
                                task_ids=new_tasks, n_demos=spec["stage_demos"]))
        del teacher
        teacher = snapshot(policy)
        torch.cuda.empty_cache()

        for t in range(n_seen):
            if getattr(args, "skip_base_eval", False) and learned_at[t] == 0:
                continue          # base 태스크는 아예 안 잰다 — 시간의 대부분이 여기에 쓰인다
            R[stage][t] = rollout_sr(policy, cfg, env_task[t], args)
            logging.info(f"[LOTUS] SR stage{stage} task{t} ({env_task[t]}) = {R[stage][t]}")
            write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    m = write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)
    json.dump({"suite": args.suite, "R": R, "learned_at": learned_at,
               "stage_tasks": stage_tasks,
               "env_task": {str(k): v for k, v in env_task.items()},
               "metrics": m, "instructions": {str(k): v for k, v in instr.items()}},
              (out_dir / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    print((out_dir / "report.txt").read_text())
    logging.info(f"[LOTUS] 완료 -> {out_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
