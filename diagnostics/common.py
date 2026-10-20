"""진단 공통 유틸 — 체크포인트/코드북 로드, 앵커 주소 재현.

학습 코드는 건드리지 않는다. 앵커 관련 로직은 전부 B1.py / l2_codebook.py 에서
import 해 쓰고, 복제한 부분에는 원본 라인 번호를 주석으로 남긴다.

진단 대상 (2026-09-08 확정)
    results/l2_codebook_k96_grad_seed41  — libero_spatial, seed 41, stage 0..8
    원래 지정된 l2_codebook_k96_grad(seed42) 는 체크포인트가 덮어써져 없다.
    ckpt_root 경로 규칙 <suite>_seed<seed>_<mode> 에 팔 이름이 없어서
    l2cb_fullcov_grad(9/5) 가 같은 자리에 덮어썼다. REPORT.md 1절 참조.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import B1                                                       # noqa: E402
from l2_codebook import sample_codebook                         # noqa: E402
from lerobot.configs.policies import PreTrainedConfig           # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy                # noqa: E402

# ── 대상 런 ────────────────────────────────────────────────────────────────
RUN = REPO / "results" / "l2_codebook_k96_grad_seed41"
CKPT_ROOT = REPO / "outputs" / "L2_codebook" / "libero_spatial_seed41_ours"
OUT = REPO / "results" / "diag_l2_codebook_k96_grad"
SUITE = "libero_spatial"
SEED_RUN = 41
N_STAGES = 9              # stage 0..8 (stage 9 는 이 런에 없다)
STEPS_PER_TASK = 5000

# 행동 청크 규격 — 데이터셋 메타로 확인함 (2026-09-08)
#   action names = ['x','y','z','roll','pitch','yaw','gripper']
H, A_DIM = 16, 7          # horizon, action dim
POSE = slice(0, 6)        # pose delta (translation 0..2 + orientation 3..5)
TRANS = slice(0, 3)       # translation 만 (D3 rotate 교란용)
GRIP = 6                  # gripper
N_OBS, D_STATE = 2, 8     # observation.state = (B, 2, 8)
D_CLS = 768               # DINOv2 CLS 차원


def stage_ckpt(k: int) -> Path:
    """stage k 종료 체크포인트. teacher_prev(stage i) = stage i-1 체크포인트다
    — 학습 루프가 체크포인트를 저장한 **뒤** snapshot 을 뜨기 때문."""
    return CKPT_ROOT / f"task_{k}" / "checkpoints" / f"{STEPS_PER_TASK:06d}" / "pretrained_model"


def instructions() -> dict[int, str]:
    import json
    d = json.loads((RUN / "instructions.json").read_text())
    return {int(k.replace("task", "")): v for k, v in d.items()}


def load_codebook(j: int, device) -> dict:
    cb = torch.load(RUN / "codebooks" / f"codebook_task{j}.pt", map_location="cpu")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}


class PolicyCache:
    """stage 체크포인트를 GPU 에 얹어 두고 재사용한다 (9개 x 739MB ≈ 6.6GB)."""

    def __init__(self, device, ds_meta=None):
        self.device = device
        self._p: dict[int, object] = {}
        # 정규화 버퍼는 체크포인트 안에 있으므로(safetensors 확인) 어떤 태스크의
        # 메타로 만들어도 pretrained 가중치가 덮어쓴다.
        self.ds_meta = ds_meta or LeRobotDatasetMetadata(
            f"continuallearning/{SUITE}_image_task_0")

    def get(self, k: int):
        if k not in self._p:
            path = str(stage_ckpt(k))
            cfg = PreTrainedConfig.from_pretrained(path)
            cfg.pretrained_path = path
            cfg.device = str(self.device)
            cfg.push_to_hub = False
            pol = make_policy(cfg=cfg, ds_meta=self.ds_meta)
            self._p[k] = pol.to(self.device).eval().requires_grad_(False)
        return self._p[k]


def vel(pol, x, t, cond):
    """velocity_net forward. (n,H,A) x (n,) x (n,2576) -> (n,H,A).
    l2_codebook.py:402-403 의 vel() 과 같다."""
    return pol.dit_flow.velocity_net(noisy_actions=x, time=t, global_cond=cond)


def anchor_cond(pol, s_t: torch.Tensor, o_t: torch.Tensor, instr: str) -> torch.Tensor:
    """코드북 표본 (s̃, õ) + 명령어 -> cond (n,2576).

    l2_codebook.py:385-400 과 같은 순서다. cond 는 **모델마다 따로** 만든다 —
    rgb_embedding_projection 과 language_embedding_projection 이 학습 대상이라
    학생과 teacher 가 서로 다른 값을 내기 때문이고, 앵커 손실도 그렇게 잰다
    (v_S(x,t,cond_S) vs v_T(x,t,cond_T)).
    """
    n = s_t.shape[0]
    bb = {"observation.state": s_t.view(n, N_OBS, D_STATE)}
    cls_j = o_t.reshape(-1, D_CLS)
    tail = B1.cond_tail(pol, bb, cls_j)
    return B1.make_cond(B1.encode_lang(pol, [instr] * n), tail)


@torch.no_grad()
def anchor_addresses(cb: dict, n: int, gen: torch.Generator, device):
    """코드북에서 앵커 주소 n 개를 뽑는다. l2_codebook.py:236 sample_codebook 재사용."""
    s_t, o_t = sample_codebook(cb, n, gen)
    return s_t.detach(), o_t.detach()


@torch.no_grad()
def bootstrap_xt(teacher_prev, cond_Tp, t, eps0, eps1):
    """teacher 부트스트랩 x_t. l2_codebook.py:410-415 그대로.
        â   = ε₀ + v_Tp(ε₀, t=0, cond_Tp)
        x_t = (1−t)·ε₁ + t·â
    반환 (x_t, â) 둘 다 (n,H,A).
    """
    a_hat = eps0 + vel(teacher_prev, eps0, torch.zeros_like(t), cond_Tp)
    tcol = t[:, None, None]
    return ((1 - tcol) * eps1 + tcol * a_hat), a_hat
