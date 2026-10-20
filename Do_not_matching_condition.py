#!/usr/bin/env python
"""앵커가 정말 **입력에 대한 반응성**을 만드는가 — 조건 불일치 롤아웃.

    CK=checkpoints/stage4/checkpoints/010000/pretrained_model
    python Do_not_matching_condition.py --suite object \
        --model tether:42=results/l2_codebook_v4_hard_sf_object_k96/$CK \
        --model tether:7=results/l2_codebook_v4_object_seed7/$CK \
        --model tether:1000=results/l2_codebook_v4_object_seed1000/$CK \
        --model seq:42=results/lotus_seq_object_seed42/$CK

train seed 7/42/1000 은 **이미 학습된 체크포인트를 그대로** 쓴다(롤아웃만 새로 한다).
eval seed 는 7 로 고정한다 — 기존 SR 행렬과 같은 조건이어야 matched 가 재현된다.
GROUP 이 같은 것끼리 묶어 train seed 간 **평균 ± 표본표준편차**를 낸다.

LIBERO-OBJECT 를 순차로 다 배운 모델을 **자기 태스크 환경에서** 굴리면서, 입력 한
쪽만 다른 스위트(LIBERO-GOAL task0)의 것으로 바꿔치기한다. 성공률이 떨어지면 그
입력을 실제로 읽고 있다는 뜻이고, 안 떨어지면 무시하고 있다는 뜻이다.

조건 3개 (각 조건에서 OBJECT 10 태스크 전부 평가, 칸당 --eval_episodes 롤아웃)

  matched       지시문·관측 모두 그 태스크의 것. 기존 SR 행렬 마지막 행의 재현이다.
  instr_xtask   **지시문만** GOAL task0 것으로 고정("open the middle drawer of the
                cabinet"). 관측·상태는 진짜 OBJECT 환경 그대로.
                  -> OBJECT 지시문은 전부 "pick up the X and place it in the basket"
                     인데 이건 서랍 열기다. 언어를 읽고 있으면 무너져야 한다.
  obs_xtask     **이미지 2대만** GOAL task0 데모(ep0, 138프레임) 재생으로 교체.
                지시문과 observation.state 는 진짜 것 그대로.
                  -> 언어·관절은 맞고 장면만 틀린 상황. 정지 프레임이 아니라 시간적으로
                     움직이는 데모를 쓴다(상수 프레임보다 공정하다). 롤아웃이 데모보다
                     길면 마지막 프레임을 유지한다.

모델 2개
  tether  L_FM + 앵커 (= L2_codebook v4)
  seq     L_FM 만 (아무 대책 없는 순차 미세조정)

읽는 법: matched 대비 낙폭(Δ)이 그 입력에 대한 반응성이다. Δ 가 클수록 반응성이 크다.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import L2_V4 as V4                                                     # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata    # noqa: E402
from lerobot.policies.factory import make_policy                       # noqa: E402
from lerobot.utils.random_utils import set_seed                        # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging    # noqa: E402

CONDITIONS = ["matched", "instr_xtask", "obs_xtask"]


def _w(t: str) -> int:
    """모노스페이스 표시 폭. 한글·전각 기호는 두 칸을 차지한다."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in t)


def pad(t: str, n: int) -> str:
    """표시 폭 기준 좌측 정렬 — f"{t:<n}" 은 글자 수로 세어 한글에서 어긋난다."""
    return t + " " * max(0, n - _w(t))

IMAGE_KEYS = ["observation.images.image", "observation.images.wrist_image"]


# ═════════════════════════════════════════════════════════════════════════════
#  바꿔치기할 재료 — LIBERO-GOAL task0
# ═════════════════════════════════════════════════════════════════════════════
def xtask_source(suite_of_model: str) -> dict:
    """불일치 조건에 쓸 '남의 태스크'. OBJECT 모델에는 GOAL task0 을 쓴다."""
    other = "LIBERO-GOAL" if suite_of_model != "goal" else "LIBERO-OBJECT"
    exp = json.loads(V4.SETTING_JSON.read_text())["experiments"][other]
    e = exp["base"][0]
    return {"suite": other, "instruction": e["instruction"], "dataset": Path(e["dataset"])}


def load_demo_frames(ds_root: Path, episode: int = 0) -> dict[str, torch.Tensor]:
    """데모 한 에피소드의 이미지를 (T,3,256,256) float32 [0,1] 로 읽는다.

    환경이 정책에 주는 형식(preprocess_observation: CHW float32 [0,1])과 **같게** 만들어야
    한다. select_action 내부의 normalize_inputs 가 그다음을 처리한다.

    이 데이터셋(v2.1)은 이미지를 parquet 안에 PNG 바이트로 들고 있다. LeRobotDataset 을
    거치지 않고 직접 읽는 이유는 delta_timestamps 없이 단일 프레임만 필요하기 때문이다.
    """
    import pyarrow.parquet as pq
    from PIL import Image
    p = ds_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not p.exists():
        raise SystemExit(f"데모 parquet 이 없다: {p}")
    tb = pq.read_table(p)
    out = {}
    for k in IMAGE_KEYS:
        arr = np.stack([np.array(Image.open(io.BytesIO(tb.column(k)[i].as_py()["bytes"])))
                        for i in range(tb.num_rows)])            # (T,256,256,3) uint8
        out[k] = (torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0).contiguous()
    logging.info(f"[xtask] 데모 프레임 로드 {ds_root.name} ep{episode}: "
                 f"{out[IMAGE_KEYS[0]].shape[0]} 프레임")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  정책 패치 — 관측을 select_action 직전에 바꿔치기한다
# ═════════════════════════════════════════════════════════════════════════════
def patch_policy(policy, mode: str, instruction: str | None = None,
                 frames: dict[str, torch.Tensor] | None = None):
    """policy 의 select_action / reset 을 **인스턴스 수준에서** 갈아끼운다.

    래퍼 클래스를 쓰지 않는 이유: eval.py:124 가 isinstance(policy, nn.Module) 을
    단정하고 get_device_from_parameters(policy) 로 디바이스를 캐낸다. 인스턴스 속성으로
    메서드를 덮으면 클래스 메서드를 가리면서 객체의 정체성은 그대로 남는다.

    반환한 함수를 부르면 원상복구된다.
    """
    cls = type(policy)
    orig_select = cls.select_action.__get__(policy)
    orig_reset = cls.reset.__get__(policy)
    state = {"t": 0, "logged": False}
    n_demo = frames[IMAGE_KEYS[0]].shape[0] if frames else 0

    def reset():
        state["t"] = 0                     # 롤아웃마다 데모 재생 위치를 되감는다
        return orig_reset()

    def select_action(batch):
        batch = dict(batch)                # 환경이 들고 있는 dict 를 건드리지 않는다
        if mode == "instr_xtask":
            was = batch["task"][0]
            batch["task"] = [instruction] * len(batch["task"])
            if not state["logged"]:        # 바꿔치기가 실제로 일어났다는 증거를 남긴다
                logging.info(f"[patch] instr: {was!r} -> {instruction!r}")
                state["logged"] = True
        elif mode == "obs_xtask":
            i = min(state["t"], n_demo - 1)          # 데모보다 길면 마지막 프레임 유지
            b = batch[IMAGE_KEYS[0]].shape[0]
            if not state["logged"]:
                d = float((batch[IMAGE_KEYS[0]][0] - frames[IMAGE_KEYS[0]][0]
                           .to(batch[IMAGE_KEYS[0]].device)).abs().mean())
                logging.info(f"[patch] obs: env 이미지 -> 데모 재생 "
                             f"(|Δpixel| 평균 {d:.4f}, 프레임 {n_demo}개)")
                state["logged"] = True
            for k in IMAGE_KEYS:
                f = frames[k][i].to(batch[k].device, batch[k].dtype)
                batch[k] = f.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        state["t"] += 1
        return orig_select(batch)

    policy.select_action = select_action
    policy.reset = reset

    def unpatch():
        policy.__dict__.pop("select_action", None)
        policy.__dict__.pop("reset", None)
    return unpatch


# ═════════════════════════════════════════════════════════════════════════════
#  리포트
# ═════════════════════════════════════════════════════════════════════════════
def write_report(path: Path, SR: dict, spec: dict, xt: dict, args, models: dict,
                 done: list[str]) -> None:
    """태스크별 SR + train seed 간 평균/편차 matrix."""
    n_task = spec["n_task"]
    instr = spec["instr"]
    mean = lambda v: (sum(v) / len(v)) if v else None                   # noqa: E731

    def avg(model, cond):
        """한 모델·조건의 태스크 평균 SR. 다 안 찼으면 None."""
        v = [SR[model][cond][t] for t in range(n_task)]
        return mean(v) if all(x is not None for x in v) else None

    groups: dict[str, list[str]] = {}
    for m, info in models.items():
        groups.setdefault(info["group"], []).append(m)

    L = ["=" * 104,
         "조건 불일치 롤아웃 — 앵커가 입력에 대한 반응성을 만드는가?",
         "=" * 104, "",
         f"스위트        libero_{args.suite}  ({n_task} 태스크, 마지막 스테이지 체크포인트)",
         f"시드          train 7/42/1000 (기존 체크포인트 그대로) / "
         f"eval {args.eval_seed} 고정, 칸당 {args.eval_episodes} rollout",
         f"불일치 출처   {xt['suite']} task0",
         f"              지시문  \"{xt['instruction']}\"",
         f"              관측    {xt['dataset'].name} ep{args.demo_episode} 데모 재생 "
         f"(이미지 2대만, state 는 진짜 것)", "",
         "조건",
         "  matched       지시문·관측 모두 그 태스크의 것 (기존 SR 마지막 행의 재현)",
         "  instr_xtask   지시문만 남의 것. 관측·state 는 진짜",
         "  obs_xtask     이미지만 남의 데모 재생. 지시문·state 는 진짜", "",
         "모델"]
    for g, ms in groups.items():
        L.append(f"  {g:<8} {models[ms[0]]['label']}")
        for m in ms:
            L.append(f"    train seed {models[m]['seed']:<5} {models[m]['path']}")
    L += ["", "-" * 104, "태스크별 SR (%)", "-" * 104,
          f"{'model':<14}{'condition':<14}"
          + "".join(f"{'t'+str(t):>6}" for t in range(n_task)) + f"{'평균':>9}"]
    for g, ms in groups.items():
        for m in ms:
            for cond in CONDITIONS:
                row = "".join(
                    (f"{SR[m][cond][t]:6.0f}" if SR[m][cond][t] is not None else f"{'':>6}")
                    for t in range(n_task))
                a = avg(m, cond)
                L.append(f"{m:<14}{cond:<14}{row}"
                         + (f"{a:9.1f}" if a is not None else f"{'':>9}"))
            L.append("")

    # ── 요청받은 비교 matrix: 마지막 스테이지 평균 SR, train seed 간 평균±편차 ──
    def cell(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return f"{'n/a':>16}"
        if len(v) == 1:
            return f"{v[0]:16.1f}"
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        body = f"{mu:.1f}±{sd:.1f}"
        return " " * max(0, 16 - _w(body)) + body

    L += ["-" * 104,
          "★ 마지막 스테이지 평균 SR matrix (%)  — train seed 간 평균 ± 표본표준편차",
          "-" * 104, "",
          f"{'':<14}" + "".join(f"{c:>16}" for c in CONDITIONS)]
    for g, ms in groups.items():
        for m in ms:
            L.append(f"  {('seed '+models[m]['seed']):<12}"
                     + "".join((f"{avg(m,c):16.1f}" if avg(m, c) is not None
                                else f"{'n/a':>16}") for c in CONDITIONS))
        tag = f"{g} 평균±편차" if len(ms) > 1 else f"{g} (시드 1개)"
        L.append(pad(tag, 14) + "".join(cell([avg(m, c) for m in ms])
                                        for c in CONDITIONS))
        L.append("")

    L += ["matched 대비 Δ (반응성)", "",
          f"{'':<14}" + "".join(f"{c:>16}" for c in CONDITIONS[1:])]
    for g, ms in groups.items():
        for m in ms:
            b = avg(m, "matched")
            L.append(f"  {('seed '+models[m]['seed']):<12}" + "".join(
                (f"{avg(m,c)-b:+16.1f}" if (avg(m, c) is not None and b is not None)
                 else f"{'n/a':>16}") for c in CONDITIONS[1:]))
        deltas = {c: [avg(m, c) - avg(m, "matched")
                      for m in ms if avg(m, c) is not None and avg(m, "matched") is not None]
                  for c in CONDITIONS[1:]}
        tag = f"{g} 평균±편차" if len(ms) > 1 else f"{g} (시드 1개)"
        L.append(pad(tag, 14) + "".join(cell(deltas[c]) for c in CONDITIONS[1:]))
        L.append("")

    # ── 상대 낙폭 — matched 바닥이 다른 모델을 공정하게 비교하려면 필요하다 ──
    L += ["matched 대비 상대 낙폭 (%)  = (조건 − matched) / matched x 100", "",
          "  seq-FT 처럼 matched 자체가 낮으면 Δ 절대값이 작아질 수밖에 없다(바닥 효과).",
          "  상대 낙폭은 그 영향을 뺀다. 두 지표가 같은 방향이면 결론이 단단하다.", "",
          f"{'':<14}" + "".join(f"{c:>16}" for c in CONDITIONS[1:])]
    for g, ms in groups.items():
        for m in ms:
            b = avg(m, "matched")
            L.append(f"  {('seed '+models[m]['seed']):<12}" + "".join(
                (f"{(avg(m,c)-b)/b*100:+16.1f}" if (avg(m, c) is not None and b)
                 else f"{'n/a':>16}") for c in CONDITIONS[1:]))
        rel = {c: [(avg(m, c) - avg(m, "matched")) / avg(m, "matched") * 100
                   for m in ms if avg(m, c) is not None and avg(m, "matched")]
               for c in CONDITIONS[1:]}
        tag = f"{g} 평균±편차" if len(ms) > 1 else f"{g} (시드 1개)"
        L.append(pad(tag, 14) + "".join(cell(rel[c]) for c in CONDITIONS[1:]))
        L.append("")

    L += ["-" * 104, "읽는 법", "-" * 104,
          "  · Δ 가 크게 음수 = 그 입력을 실제로 읽고 있다(반응성 있음).",
          "  · Δ ≈ 0 = 그 입력을 무시하고 나머지만으로 행동한다.",
          "  · instr_xtask 의 Δ 가 얕으면 언어 조건이 사실상 죽어 있다는 뜻이다 —",
          "    OBJECT 태스크는 지시문의 목적어만 다르므로, 언어를 못 읽으면 태스크를",
          "    구분할 수 없고 '바구니에 뭔가 담기'로 뭉개진다.",
          "  · obs_xtask 는 장면이 완전히 남의 것이라 낙폭이 큰 게 정상이다. 여기서",
          "    Δ 가 얕다면 정책이 이미지를 거의 안 보고 state 만으로 움직이는 것이다.",
          "  · matched 가 낮은 모델은 Δ 절대값과 상대 낙폭을 **둘 다** 봐야 한다. 절대값만",
          "    보면 바닥 효과 때문에 '입력을 덜 읽는다'로 잘못 읽힌다.", "",
          "주의",
          f"  · 칸당 {args.eval_episodes} rollout 이라 개별 칸의 이항 표준오차가 최대 "
          f"±{50 / (args.eval_episodes ** 0.5):.0f}%p, {n_task}칸 평균은 "
          f"±{50 / ((args.eval_episodes * n_task) ** 0.5):.1f}%p 다. 그래서 train seed",
          "    3개의 편차를 함께 본다 — 편차가 이 크기와 비슷하면 시드 차이를 확정할 수 없다.",
          "  · 편차는 시드 3개의 **표본표준편차**다(n-1). 3점 추정이라 그 자체도 불안정하다.",
          "  · matched 는 기존 SR 행렬 마지막 행과 같은 조건이다. 원본은 seed42 91.0 /",
          "    seed7 83.5 / seed1000 91.5 였다. 크게 다르면 평가 경로가 어긋난 것이다.",
          "  · obs_xtask 의 데모 프레임은 학습 데이터셋에서 온 것이다. 정책이 그 분포로",
          "    학습되고 환경 이미지로 90% 안팎을 내므로 두 분포는 서로 맞는다고 본다.", "",
          "태스크별 지시문"]
    L += [f"  t{t:<3} {instr[t]}" for t in range(n_task)]
    n_cell = len(models) * len(CONDITIONS) * n_task
    n_filled = sum(1 for m in SR for c in CONDITIONS
                   for t in range(n_task) if SR[m][c][t] is not None)
    L += ["", f"진행 상태  {n_filled}/{n_cell} 칸 완료",
          f"생성 시각  {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    path.write_text("\n".join(L) + "\n")

# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["goal", "object", "50"], default="object")
    ap.add_argument("--model", action="append", default=[], metavar="GROUP:SEED=PATH",
                    help="평가할 체크포인트. 여러 번 준다. 예: "
                         "--model tether:42=results/.../pretrained_model "
                         "GROUP 이 같은 것끼리 묶어 train seed 간 평균±편차를 낸다.")
    ap.add_argument("--demo_episode", type=int, default=0)
    ap.add_argument("--eval_seed", type=int, default=7)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "results" / "Do_not_matching_condition_SR.txt"))
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--max_tasks", type=int, default=0,
                    help="앞에서 이만큼 태스크만 평가한다(스모크용). 0 = 전부")
    ap.add_argument("--resume", action="store_true",
                    help="--json_out 에 이미 있는 칸은 건너뛴다. 모델을 나눠 돌리거나 "
                         "중단된 run 을 이어갈 때 쓴다.")
    ap.add_argument("--merge_json", action="append", default=[],
                    help="다른 run 의 JSON 에서 칸을 끌어온다. 여러 번 줄 수 있다. "
                         "시드별로 GPU 를 나눠 돌린 뒤 하나로 합칠 때 쓴다.")
    ap.add_argument("--report_only", action="store_true",
                    help="롤아웃을 하지 않고 모아진 칸으로 리포트만 쓴다.")
    args = ap.parse_args()

    init_logging()
    spec = V4.suite_spec(args.suite)
    if args.max_tasks:
        spec["n_task"] = min(args.max_tasks, spec["n_task"])
        logging.info(f"[xcond] ★ 스모크 축소: 앞 {spec['n_task']} 태스크만")
    n_task, env_task = spec["n_task"], spec["env_task"]
    xt = xtask_source(args.suite)
    frames = load_demo_frames(xt["dataset"], args.demo_episode)

    GROUP_LABEL = {"tether": "L_FM + 앵커 (L2_codebook v4)",
                   "seq": "L_FM 만 (seq-FT 기준선)"}
    models: dict[str, dict] = {}
    for spec_str in args.model:
        try:
            head, path = spec_str.split("=", 1)
            group, seed = head.split(":", 1)
        except ValueError:
            raise SystemExit(f"--model 형식은 GROUP:SEED=PATH 다 (받은 값: {spec_str!r})")
        name = f"{group}_s{seed}"
        if not (Path(path) / "model.safetensors").exists():
            raise SystemExit(f"{name}: 체크포인트가 없다 -> {path}")
        models[name] = {"path": path, "group": group, "seed": seed,
                        "label": GROUP_LABEL.get(group, group)}
    if not models:
        raise SystemExit("--model 을 하나 이상 줘야 한다")

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jout = Path(args.json_out) if args.json_out else out.with_suffix(".json")

    SR = {m: {c: [None] * n_task for c in CONDITIONS} for m in models}
    srcs = ([jout] if (args.resume and jout.exists()) else []) \
        + [Path(x) for x in args.merge_json]
    for src in srcs:
        if not src.exists():
            raise SystemExit(f"--merge_json 파일이 없다: {src}")
        prev = json.load(src.open())["SR"]
        # 이름이 바뀐 초기 run 을 이어받는다: 단일 모델이던 "tether"/"seq" -> "<group>_s42"
        alias = {"tether": "tether_s42", "seq": "seq_s42"}
        for old, new in alias.items():
            if old in prev and new not in prev and new in SR:
                prev[new] = prev.pop(old)
                logging.info(f"[xcond] resume: 기존 키 {old!r} -> {new!r} 로 이어받음")
        n_kept = 0
        for m in SR:
            for c in CONDITIONS:
                for t in range(min(n_task, len(prev.get(m, {}).get(c, [])))):
                    if prev[m][c][t] is not None:
                        SR[m][c][t] = prev[m][c][t]
                        n_kept += 1
        logging.info(f"[xcond] {src.name} 에서 {n_kept}칸 이어받음")
    done: list[str] = []
    base_meta = LeRobotDatasetMetadata("local/lotus", root=str(spec["root"] / "base"))

    if args.report_only:
        write_report(out, SR, spec, xt, args, models, done)
        print(out.read_text())
        logging.info(f"[xcond] 리포트만 생성 -> {out}")
        return

    for mname, minfo in models.items():
        if all(SR[mname][c][t] is not None for c in CONDITIONS for t in range(n_task)):
            logging.info(f"[xcond] {mname}: 모든 칸이 이미 차 있다 — 건너뜀")
            continue
        set_seed(args.eval_seed)                       # 모델마다 같은 곳에서 출발
        cfg = V4.build_eval_cfg(args, minfo["path"], env_task[0])
        policy = make_policy(cfg=cfg.policy, ds_meta=base_meta).to(device)
        logging.info(f"[xcond] {mname} 로드 -> {minfo['path']}")

        for cond in CONDITIONS:
            unpatch = patch_policy(
                policy, cond,
                instruction=xt["instruction"] if cond == "instr_xtask" else None,
                frames=frames if cond == "obs_xtask" else None)
            try:
                for t in range(n_task):
                    if SR[mname][cond][t] is not None:
                        continue                       # --resume 으로 이어받은 칸
                    t0 = time.perf_counter()
                    SR[mname][cond][t] = V4.rollout_sr(policy, cfg, env_task[t], args)
                    logging.info(f"[xcond] {mname}/{cond} task{t} ({env_task[t]}) = "
                                 f"{SR[mname][cond][t]}  ({time.perf_counter()-t0:.0f}s)")
                    write_report(out, SR, spec, xt, args, models, done)
                    json.dump({"suite": args.suite, "SR": SR, "conditions": CONDITIONS,
                               "xtask": {"suite": xt["suite"],
                                         "instruction": xt["instruction"],
                                         "dataset": str(xt["dataset"]),
                                         "episode": args.demo_episode},
                               "models": {k: v["path"] for k, v in models.items()},
                               "eval": {"seed": args.eval_seed,
                                        "episodes": args.eval_episodes}},
                              jout.open("w"), indent=2, ensure_ascii=False)
            finally:
                unpatch()
            done.append(f"{mname}/{cond}")
            write_report(out, SR, spec, xt, args, models, done)

        del policy
        torch.cuda.empty_cache()

    write_report(out, SR, spec, xt, args, models, done)
    print(out.read_text())
    logging.info(f"[xcond] 완료 -> {out}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
