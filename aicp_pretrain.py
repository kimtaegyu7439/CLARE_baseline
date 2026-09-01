#!/usr/bin/env python
"""AICP base phase — CLARE 정책(DiT-Flow MT)을 LIBERO-90 데이터로 from scratch 사전학습.

lerobot_lsy/src/lerobot/scripts/train.py 의 학습 루프를 그대로 가져왔다. 손실은
**정책의 기본 flow-matching MSE 하나뿐**이다. 앵커/패널티/정규화 항을 일절 넣지
않는다 — B1 계열(CLARE 연속학습)과 달리 여기는 base phase 다.

from scratch 의 범위
  DINOv2(vision) 와 CLIP(language) 는 각자 사전학습 가중치를 그대로 쓰고 **동결**
  된다(CLARE 미세조정과 같은 조건). 무작위 초기화되어 학습되는 것은
    - rgb_embedding_projection
    - language_embedding_projection
    - velocity_net (DiT) 와 그 주변 투영
  즉 --policy.path 를 주지 않는 bash/clare/pretrain.sh 와 같은 상태다.

데이터셋이 다르다 — 그래서 로더를 새로 썼다
  /home/sa090180/Datasets/lerobot/ASDL_CL_libero90_dataset 은 **codebase v3.0** 이고
  이 레포의 LeRobotDataset 은 v2.1 전용이라 그대로는 못 읽는다. 차이:

      항목            ASDL(v3.0)                     CLARE LIBERO(v2.1)
      data 레이아웃   data/chunk-000/file-NNN.parquet  episode_NNNNNN.parquet
                      (에피소드 여러 개가 한 파일)      (1 에피소드 = 1 파일)
      meta/episodes   디렉토리의 parquet (2800 행)     episodes.jsonl
      카메라 키       image, image2                    image, wrist_image
      state.joint     없음                             있음(7, 모델 미사용)
      fps             10                               20

  ★ 데이터 적재는 전부 load_pretrain_dataset() 하나에서 처리한다.
    이 함수만 보면 v3 -> 학습 루프가 기대하는 배치까지의 전 과정이 들어 있다.

  image2 는 손목(eye-in-hand) 카메라임을 프레임을 렌더해 확인했다 -> wrist_image
  로 이름만 바꿔 준다. observation.state.joint 는 DiT-Flow 가 참조하지 않으므로
  (modeling_dit_flow_mt.py:1142-1143) 없어도 무방하며 input_features 에서 뺀다.

  fps 가 10 이라 horizon 16 이 1.6 초에 해당한다(다운스트림 LIBERO 는 20 Hz 라
  0.8 초). horizon/n_action_steps/n_obs_steps 는 **바꾸지 않는다** — 바꾸면 출력층
  모양이 달라져 CLARE 미세조정으로 전이가 안 된다. 프레임 인덱스 기준 창 구성은
  양쪽이 동일하므로 코드 경로는 같고, 실제 시간 폭만 2배다.

사용법
    bash aicp_pretrain.sh            # GPU 0, seed 7, 200k step
    python aicp_pretrain.py --help
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image  # noqa: E402
import pyarrow.parquet as pq
import torch
from termcolor import colored
from torch.amp import GradScaler

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import get_policy_class
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_identifier
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ROOT = "/home/sa090180/Datasets/lerobot/ASDL_CL_libero90_dataset"
DEFAULT_OUT = "/home/sa090180/Models/aicp_clare_pretrain"
REF_CKPT = "/home/sa090180/Models/dit_flow_mt_libero_90_pretrain"   # 구조 참조용(가중치 미사용)

# v3 카메라 키 -> 정책이 기대하는 이름
CAM_RENAME = {"observation.images.image2": "observation.images.wrist_image"}
# 데이터에 없고 모델도 안 쓰는 키
DROP_KEYS = ("observation.state.joint",)


# ═════════════════════════════════════════════════════════════════════════════
#  ★ 데이터 적재 — 이 함수 하나가 v3 데이터셋 -> 학습 배치까지 전부 담당한다
# ═════════════════════════════════════════════════════════════════════════════
def load_pretrain_dataset(root: str | Path,
                          policy_cfg: PreTrainedConfig,
                          arrow_cache: str | Path | None = None,
                          rebuild: bool = False):
    """ASDL_CL_libero90(v3.0) -> (dataset, meta).

    하는 일 다섯 가지.
      1. meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/*.parquet 을 읽는다.
      2. data/chunk-*/file-*.parquet 를 **memory-map 가능한 Arrow IPC** 로 1회 변환해
         캐시한다(재실행 시 재사용). parquet 은 임의 행 접근이 느리고, 이 레포에 깔린
         datasets 패키지는 v3 파일의 'List' feature 타입을 몰라 load_dataset 이 실패한다.
      3. 카메라 키를 image2 -> wrist_image 로 바꾸고 state.joint 를 뺀 feature 표를 만든다.
      4. 이미지 통계를 ImageNet 값으로 갈아 끼운다(datasets/factory.py 와 같은 처리).
      5. 정책의 delta_indices(관측 [-1,0], 액션 [-1..14])대로 창을 구성하는 torch Dataset
         을 만들어 돌려준다.

    반환하는 dataset 이 내놓는 항목은 train.py 가 기대하는 것과 같다.
        observation.images.image        (2, 3, 256, 256)  float32 [0,1]
        observation.images.wrist_image  (2, 3, 256, 256)
        observation.state               (2, 8)
        action                          (16, 7)
        action_is_pad                   (16,)  bool
        task                            str
        episode_index / frame_index / index / task_index
    """
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    stats = json.loads((root / "meta" / "stats.json").read_text())

    # ── (1) 에피소드 표 ──────────────────────────────────────────────────────
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep = pq.read_table(ep_files, columns=[
        "episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index",
    ]).to_pandas().sort_values("episode_index").reset_index(drop=True)
    ep_from = ep["dataset_from_index"].to_numpy()
    ep_to = ep["dataset_to_index"].to_numpy()
    ep_len = ep["length"].to_numpy()
    # tasks 는 에피소드당 문자열 리스트(보통 1개). 첫 항목을 언어 지시로 쓴다.
    ep_task = [t[0] if len(t) else "" for t in ep["tasks"]]

    # ── (2) Arrow IPC 변환 + memory-map ─────────────────────────────────────
    cache = Path(arrow_cache) if arrow_cache else root / "_arrow_cache"
    cache.mkdir(parents=True, exist_ok=True)
    src = sorted((root / "data").rglob("*.parquet"))
    arrow_files, t0 = [], time.perf_counter()
    for i, sp in enumerate(src):
        dst = cache / (sp.parent.name + "__" + sp.stem + ".arrow")
        if rebuild or not dst.exists():
            t = pq.read_table(sp).replace_schema_metadata(None)   # v3 메타데이터 제거
            with pa.OSFile(str(dst), "wb") as f:
                with pa.ipc.new_file(f, t.schema) as w:
                    w.write_table(t)
            del t
            if (i + 1) % 50 == 0 or i + 1 == len(src):
                logging.info(f"[data] Arrow 변환 {i+1}/{len(src)}  "
                             f"{time.perf_counter()-t0:.0f}s")
        arrow_files.append(str(dst))

    # 행 순서가 전역 index 와 같은지 한 번만 확인한다(다르면 창 구성이 어긋난다).
    # index 컬럼만 읽으므로 이미지 버퍼는 건드리지 않는다.
    tbl = open_arrow(arrow_files)
    idx0 = tbl.column("index").to_numpy(zero_copy_only=False)
    if tbl.num_rows != info["total_frames"] or not np.array_equal(idx0, np.arange(len(idx0))):
        raise RuntimeError(f"행 순서/개수 불일치: {tbl.num_rows} vs {info['total_frames']}")
    del tbl, idx0

    # ── (3) feature 표 ──────────────────────────────────────────────────────
    feats = {}
    for k, v in info["features"].items():
        if k in DROP_KEYS:
            continue
        feats[CAM_RENAME.get(k, k)] = v

    # ── (4) 통계 — 이미지는 ImageNet 값으로 ─────────────────────────────────
    st = {}
    for k, v in stats.items():
        if k in DROP_KEYS:
            continue
        key = CAM_RENAME.get(k, k)
        st[key] = {kk: torch.tensor(vv, dtype=torch.float32) for kk, vv in v.items()
                   if kk in ("mean", "std", "min", "max")}
    for k, v in feats.items():
        if v["dtype"] in ("image", "video"):
            st[k] = {kk: torch.tensor(vv, dtype=torch.float32)
                     for kk, vv in IMAGENET_STATS.items()}

    meta = PretrainMeta(fps=info["fps"], features=feats, stats=st,
                        total_episodes=int(info["total_episodes"]),
                        total_frames=int(info["total_frames"]),
                        tasks=sorted(set(ep_task)))

    ds = PretrainDataset(arrow_files, ep_from, ep_to, ep_len, ep_task,
                         feats, policy_cfg, meta)
    logging.info(f"[data] {root.name}  ep {meta.total_episodes}  frame {meta.total_frames}  "
                 f"task {len(meta.tasks)}  fps {meta.fps}  "
                 f"obs_idx {policy_cfg.observation_delta_indices} "
                 f"act_idx [{policy_cfg.action_delta_indices[0]}..{policy_cfg.action_delta_indices[-1]}]")
    return ds, meta


def open_arrow(files):
    """memory-map 으로 Arrow IPC 를 열어 하나의 Table 로 잇는다. zero-copy 라 RAM 에 안 올라온다."""
    return pa.concat_tables(
        [pa.ipc.open_file(pa.memory_map(f, "r")).read_all() for f in files])


class PretrainMeta:
    """make_policy 가 쓰는 최소 메타. LeRobotDatasetMetadata 대체."""

    def __init__(self, fps, features, stats, total_episodes, total_frames, tasks):
        self.fps = fps
        self.features = features
        self.stats = stats
        self.total_episodes = total_episodes
        self.total_frames = total_frames
        self.tasks = tasks


class PretrainDataset(torch.utils.data.Dataset):
    """전역 프레임 인덱스 -> 정책이 먹는 시간 창 하나."""

    def __init__(self, arrow_files, ep_from, ep_to, ep_len, ep_task, feats, policy_cfg, meta):
        self.arrow_files = list(arrow_files)
        self.ep_from, self.ep_to, self.ep_len, self.ep_task = ep_from, ep_to, ep_len, ep_task
        self.meta = meta
        self.obs_idx = list(policy_cfg.observation_delta_indices)      # [-1, 0]
        self.act_idx = list(policy_cfg.action_delta_indices)           # [-1 .. 14]
        self.cam_src = {v: k for k, v in CAM_RENAME.items()}           # 새이름 -> 원본 컬럼
        self.img_keys = [k for k, v in feats.items() if v["dtype"] in ("image", "video")]
        self.num_frames = int(ep_to[-1])
        self.num_episodes = int(len(ep_from))
        self._cols = None          # ★ 워커마다 지연 개방한다 (아래 _ensure 참조)

    def _ensure(self):
        """memory-map 은 pickle 되면 데이터를 통째로 복사한다. 그래서 부모가 아니라
        **각 워커 프로세스가 처음 쓸 때** 연다. 그 전까지 이 객체는 경로 목록뿐이다."""
        if self._cols is not None:
            return
        t = open_arrow(self.arrow_files)
        self._cols = {
            "img": {k: t.column(self.cam_src.get(k, k)) for k in self.img_keys},
            "state": t.column("observation.state"),
            "action": t.column("action"),
            "task_index": t.column("task_index"),
        }

    def __len__(self):
        return self.num_frames

    def _rows(self, base_local, ep_i, offsets):
        """에피소드 안에서 clamp 한 전역 행 번호와 패딩 마스크."""
        L = int(self.ep_len[ep_i])
        loc = np.array(offsets) + base_local
        pad = (loc < 0) | (loc >= L)
        loc = np.clip(loc, 0, L - 1)
        return (loc + int(self.ep_from[ep_i])), pad

    def __getitem__(self, i):
        self._ensure()
        c = self._cols
        i = int(i)
        ep_i = int(np.searchsorted(self.ep_to, i, side="right"))
        local = i - int(self.ep_from[ep_i])

        obs_rows, _ = self._rows(local, ep_i, self.obs_idx)
        act_rows, act_pad = self._rows(local, ep_i, self.act_idx)

        out = {}
        for k in self.img_keys:
            col = c["img"][k]
            imgs = []
            for r in obs_rows:
                b = col[int(r)].as_py()["bytes"]
                a = np.array(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
                imgs.append(torch.from_numpy(a).permute(2, 0, 1))
            out[k] = torch.stack(imgs).float() / 255.0            # (2,3,H,W) [0,1]

        out["observation.state"] = torch.stack(
            [torch.tensor(c["state"][int(r)].as_py(), dtype=torch.float32) for r in obs_rows])
        out["action"] = torch.stack(
            [torch.tensor(c["action"][int(r)].as_py(), dtype=torch.float32) for r in act_rows])
        out["action_is_pad"] = torch.from_numpy(act_pad)
        out["task"] = self.ep_task[ep_i]
        out["episode_index"] = torch.tensor(ep_i, dtype=torch.int64)
        out["frame_index"] = torch.tensor(local, dtype=torch.int64)
        out["index"] = torch.tensor(i, dtype=torch.int64)
        out["task_index"] = torch.tensor(c["task_index"][i].as_py(), dtype=torch.int64)
        return out


# ═════════════════════════════════════════════════════════════════════════════
#  정책 — 구조는 참조 체크포인트에서, 가중치는 무작위 (인코더만 사전학습+동결)
# ═════════════════════════════════════════════════════════════════════════════
def build_policy(meta, device: str, ref_ckpt: str = REF_CKPT):
    cfg = PreTrainedConfig.from_pretrained(ref_ckpt)
    cfg.pretrained_path = None            # ★ from scratch — 가중치를 불러오지 않는다
    cfg.device = device
    cfg.push_to_hub = False

    feats = {}
    for k, v in meta.features.items():
        if v["dtype"] in ("image", "video"):
            h, w, c = v["shape"]
            feats[k] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
        elif k == "action":
            feats[k] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(v["shape"]))
        elif k.startswith("observation.state"):
            feats[k] = PolicyFeature(type=FeatureType.STATE, shape=tuple(v["shape"]))
    cfg.output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    cfg.input_features = {k: f for k, f in feats.items() if k not in cfg.output_features}

    policy = get_policy_class(cfg.type)(config=cfg, dataset_stats=meta.stats)
    policy.to(device)
    return policy, cfg


def build_train_cfg(policy_cfg, args, out_dir: Path) -> TrainPipelineConfig:
    """make_optimizer_and_scheduler 가 보는 것만 채운다 (B1.build_cfg 와 같은 방식)."""
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/ASDL_CL_libero90_dataset"),
        env=None, policy=policy_cfg, output_dir=out_dir,
        job_name=args.job_name, seed=args.seed, num_workers=args.num_workers,
        batch_size=args.batch_size, steps=args.steps, eval_freq=0,
        log_freq=args.log_freq, save_freq=args.steps,
    )
    cfg.optimizer = policy_cfg.get_optimizer_preset()
    cfg.scheduler = policy_cfg.get_scheduler_preset()
    cfg.checkpoint_path = None
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
def update_policy(tracker, policy, batch, optimizer, grad_clip_norm, grad_scaler,
                  lr_scheduler=None, use_amp=False):
    """train.py:220 과 같다. 손실은 policy.forward 하나뿐 — 추가 항 없음."""
    t0 = time.perf_counter()
    device = next(policy.parameters()).device
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm,
                                               error_if_nonfinite=False)
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    grad_norm_after_clip = torch.nn.utils.clip_grad._get_total_norm(grads, 2)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()
    tracker.loss = loss.item()
    tracker.grad_norm = grad_norm.item()
    tracker.grad_norm_after_clip = grad_norm_after_clip.item()
    tracker.lr = optimizer.param_groups[0]["lr"]
    tracker.update_s = time.perf_counter() - t0
    return tracker, output_dict


def cycle(dl):
    it = iter(dl)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(dl)
            yield next(it)


def collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT, help="최종 가중치 저장 위치")
    ap.add_argument("--ref_ckpt", default=REF_CKPT, help="구조만 참조. 가중치는 안 쓴다.")
    ap.add_argument("--arrow_cache", default=None)
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--log_freq", type=int, default=500)
    ap.add_argument("--job_name", default="aicp_clare_pretrain")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="20 스텝만 돌려 배치/손실만 확인")
    args = ap.parse_args()

    init_logging()
    if args.smoke:
        args.steps, args.log_freq, args.num_workers = 20, 5, 2
    out_dir = Path(args.out)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    device = get_safe_torch_device(args.device, log=True)

    # ── 정책 구조를 먼저 알아야 delta_indices 를 쓸 수 있다 ──────────────────
    ref = PreTrainedConfig.from_pretrained(args.ref_ckpt)
    ds, meta = load_pretrain_dataset(args.root, ref, args.arrow_cache, args.rebuild_cache)
    policy, policy_cfg = build_policy(meta, args.device, args.ref_ckpt)

    n_learn = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in policy.parameters())
    logging.info(colored(
        f"[aicp] from scratch  학습 {format_big_number(n_learn)} / 전체 "
        f"{format_big_number(n_all)} 파라미터 (인코더 동결)  seed={args.seed}  "
        f"steps={args.steps}  batch={args.batch_size}  "
        f"loss=flow-matching only (패널티 없음)", "green", attrs=["bold"]))

    cfg = build_train_cfg(policy_cfg, args, out_dir)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=policy_cfg.use_amp)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True, collate_fn=collate,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)
    dl_iter = cycle(loader)

    metrics = {"loss": AverageMeter("loss", ":.3f"),
               "grad_norm": AverageMeter("grdn", ":.3f"),
               "grad_norm_after_clip": AverageMeter("grdn_after_clip", ":.3f"),
               "lr": AverageMeter("lr", ":0.1e"),
               "update_s": AverageMeter("updt_s", ":.3f"),
               "dataloading_s": AverageMeter("data_s", ":.3f")}
    tracker = MetricsTracker(args.batch_size, ds.num_frames, ds.num_episodes, metrics,
                             initial_step=0)

    logging.info("Start offline training on a fixed dataset")
    t_start = time.perf_counter()
    for step in range(args.steps):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        tracker.dataloading_s = time.perf_counter() - t0
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device, non_blocking=device.type == "cuda")
        tracker, _ = update_policy(tracker, policy, batch, optimizer,
                                   cfg.optimizer.grad_clip_norm, grad_scaler,
                                   lr_scheduler, policy_cfg.use_amp)
        tracker.step()
        if args.log_freq > 0 and (step + 1) % args.log_freq == 0:
            el = time.perf_counter() - t_start
            sps = (step + 1) / el
            logging.info(f"{tracker}  {sps:.2f} step/s  "
                         f"남은 {(args.steps-step-1)/max(sps,1e-9)/3600:.1f}h")
            tracker.reset_averages()

    # ── 저장 (마지막에 한 번) ────────────────────────────────────────────────
    del dl_iter          # 워커를 먼저 정리한다. 안 하면 종료 시 shutdown 잡음이 뜬다.
    del loader
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    json.dump({"job_name": args.job_name, "seed": args.seed, "steps": args.steps,
               "batch_size": args.batch_size, "dataset_root": str(args.root),
               "dataset_version": "v3.0", "fps": meta.fps,
               "total_episodes": meta.total_episodes, "total_frames": meta.total_frames,
               "num_tasks": len(meta.tasks),
               "cam_rename": CAM_RENAME, "dropped_keys": list(DROP_KEYS),
               "encoders": "DINOv2 / CLIP pretrained + frozen (CLARE 와 동일)",
               "loss": "flow-matching MSE only (no anchor / no penalty)",
               "ref_ckpt_for_architecture_only": args.ref_ckpt,
               "wall_hours": round((time.perf_counter() - t_start) / 3600, 2)},
              (out_dir / "aicp_pretrain_meta.json").open("w"), indent=2, ensure_ascii=False)
    logging.info(colored(f"[aicp] 저장 완료 -> {out_dir}  "
                         f"({(time.perf_counter()-t_start)/3600:.2f}h)", "green", attrs=["bold"]))
    logging.info(f"[aicp] CLARE 에서 쓰려면:  PRETRAIN_PATH={out_dir} bash B1.sh ...")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
