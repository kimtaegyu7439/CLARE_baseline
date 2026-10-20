#!/usr/bin/env python
"""LOTUS 셋팅 lifelong — **CLARE**(어댑터 라우팅·확장) 기준선 드라이버.

    python L2_clare_cl.py --suite goal --seed 7 --out results/lotus_clare_goal_seed7

L2_seq_cl.py(seq-FT) · L2_er_cl.py(ER) 와 같은 자리에 놓이는 세 번째 기준선이다.
행렬 R · learned_at · metrics.json 스키마가 셋 다 같아 LOTUS_seq_cl_report.py 로
그대로 읽힌다.

════════════════════════════════════════════════════════════════════════════
이 파일이 하는 일과 하지 않는 일
════════════════════════════════════════════════════════════════════════════
**학습은 저자 코드가 한다.** 이 드라이버는 lerobot_lsy/src/lerobot/scripts/clare.py
를 서브프로세스로 부르기만 한다. CLARE 알고리즘(분포 이동 감지 -> 확장 결정 ->
어댑터 학습 -> 판별기 학습)은 한 줄도 옮겨 쓰지 않았다. "저자 구현을 그대로
돌렸다"고 말할 수 있어야 기준선으로서 의미가 있다.

이 드라이버가 맡는 것은 셋뿐이다.
  (1) LOTUS 의 계단식 구조를 clare.py 의 "태스크 하나 = 프로세스 하나" 로 번역
  (2) base 태스크용 라우터 보정 (아래 [B])
  (3) 스테이지마다 R 행렬을 채우고 metrics.json 을 쓰기 — seq/ER 과 **같은**
      평가 경로(V4.rollout_sr)를 쓴다. 평가가 같아야 세 기준선을 비교할 수 있다.

════════════════════════════════════════════════════════════════════════════
[B] base 태스크 라우터 보정 — 왜 필요한가
════════════════════════════════════════════════════════════════════════════
CLARE 는 추론 때 태스크 id 를 안 받고, 판별기(오토인코더)들의 복원 손실을 비교해
argmin 으로 어댑터를 고른다(peft/tuners/clare/layer.py 의 eval 분기).
**"해당 없음" 선택지가 없다.**

원 CLARE 셋팅은 task 0 부터 배우니 문제가 없다. 그런데 LOTUS 는 base 태스크
(goal/object 6개, 50 은 25개)를 **이미 배운 상태**의 체크포인트에서 출발한다.
이 태스크들에는 판별기가 없으므로, 보정을 안 하면 stage 1 이후 base 관측이
전부 lifelong 어댑터 중 하나로 강제 라우팅된다 — 가중치는 하나도 안 변했는데
SR 이 떨어진다. 방법의 실력이 아니라 셋팅이 만든 핸디캡이다.

그래서 lifelong 을 시작하기 전에 base 태스크마다 clare.py 를 **--steps 0** 으로
한 번씩 돌린다. 그러면:
    * 어댑터 학습 구간([C])이 통째로 건너뛰어진다
    * 판별기 학습 구간([D])만 돈다
    * 어댑터는 생성 직후 상태로 남는다 = lora_b 가 zeros_ 초기화라(layer.py:363)
      ΔW = B·A·scaling = 0, 즉 **정확히 항등**이다
결과적으로 "base 태스크를 가리키는 라우터 항목"만 생기고 정책 출력은 base
체크포인트 그대로다. 저자 코드 수정이 필요 없다는 점이 중요하다.

★ 이 보정은 CLARE 논문에 없는 **우리가 추가한 것**이다. 논문에 쓸 때 반드시
  밝혀야 한다. 다만 방향은 기준선에 유리한 쪽이다 — 안 하면 CLARE 가 부당하게
  나빠 보인다.

★ 확장 판정과 at_least_expand — 저자 기본값 그대로 간다
  층 l 에서 z-score 가 expand_threshold 를 넘으면 새 어댑터를 단다. 못 넘으면
  **가장 가까운 기존 어댑터를 재사용**하고 판별기만 새로 달아 거기 연결한다
  (layer.py add_discriminator -> connected_adapter_indices). 재사용되는 기존
  어댑터는 옵티마이저에 안 들어가므로 동결된다. "태스크를 건너뛴다"가 아니다.
  어느 층도 확장 신호를 못 내면 가장 얕은 층 하나를 강제로 확장한다 — 논문이
  말하는 최소 task-specific capacity 보장이다. 이 레포는 target_modules 가
  velocity_net.cond_proj **하나**뿐이라 이 장치가 특히 중요하다: 층이 하나면
  "재사용"과 "새 용량 0" 이 같은 상황이 되어, 보험이 없으면 그 태스크가 새
  용량을 전혀 못 받는다.
  실측: 원본 10-태스크 런 세 시드 모두 structure=[10,10] 이고, 로그를 보면
  보험이 발동한 적은 한 번도 없다 — z-score 가 스스로 문턱(1.0)을 다 넘었다.
  즉 실제로는 태스크당 어댑터 1개가 판정으로 결정되고 있다.
  base 보정 때는 태스크마다 항등 어댑터가 하나씩 생기는데, 전부 항등이라
  base 관측이 어느 쪽으로 라우팅되든 출력은 같다.

════════════════════════════════════════════════════════════════════════════
CLARE 태스크 하나 = LIBERO 태스크 하나 — 스플릿을 태스크별로 만든다
════════════════════════════════════════════════════════════════════════════
CLARE 의 확장 판정은 **태스크마다 한 번** 일어난다. LIBERO-50 의 스테이지에는
태스크가 5개 들어 있으므로, 스테이지를 통째로 한 번 부르면 확장 판정이 5번이
아니라 1번만 일어나 방법이 달라진다. 그러면 안 된다.

★ --dataset.episodes 로 자르는 방법은 **쓸 수 없다**. 설치된 lerobot 의 버그다:
    datasets/utils.py  get_episode_data_index()  -> 길이 len(episodes) 의 0-기반 배열
    datasets/lerobot_dataset.py:689              -> 그걸 **원래** ep_idx 로 인덱싱
  episodes=[50..99] 를 주면 배열 크기는 50 인데 ep_idx 가 50..99 로 들어와
  IndexError 가 난다(실측: ep0-49 통과, ep50-99 실패). episodes=[0..n-1] 일 때만
  우연히 맞는다. 공용 설치본이라 여기서 고치면 이미 끝난 seq-FT/ER/TETHER 런과
  환경이 달라지므로 손대지 않는다.

그래서 **태스크마다 독립 데이터셋을 물리적으로 만든다**. lotus_build_setting.py 의
build_split() 을 그대로 쓴다 — LOTUS 스플릿 자체를 만든 함수라 episode_index /
index / task_index 재번호 규칙이 이미 검증돼 있다. 매니페스트의 태스크 엔트리
하나만 넘기면 그 태스크짜리 데이터셋이 나온다.

용량 때문에 **쓰고 나면 지운다**. LIBERO-50 의 base 는 25 태스크 19GB 라 전부
남겨 두면 디스크가 위험하다. 한 번에 한 태스크분(<=800MB)만 존재한다.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import torch                                                            # noqa: E402

import L2_V4 as V4                                                      # noqa: E402
import lotus_build_setting as LBS                                       # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata     # noqa: E402
from lerobot.policies.factory import make_policy                        # noqa: E402
from lerobot.utils.random_utils import set_seed                         # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging     # noqa: E402

CLARE_PY = REPO / "lerobot_lsy" / "src" / "lerobot" / "scripts" / "clare.py"
PEFT_BLUEPRINT = REPO / "peft_lsy" / "peft_config" / "clare_dit_flow_encoder_adapter"
NEVER = 10 ** 9      # eval_freq/save_freq 를 사실상 끄는 값
SUITE_EXP = {"goal": "LIBERO-GOAL", "object": "LIBERO-OBJECT", "50": "LIBERO-50"}


# ═════════════════════════════════════════════════════════════════════════════
#  보고서
# ═════════════════════════════════════════════════════════════════════════════
def write_report(path: Path, R, learned_at, instr, args, n_task, spec, units):
    L = ["LOTUS 셋팅 lifelong — CLARE (어댑터 라우팅 기준선)", "=" * 92, "",
         f"  suite      libero_{args.suite}",
         f"  학습       clare.py(저자 코드) 서브프로세스. 태스크당 "
         f"{args.steps} step 어댑터 + {args.disc_steps} step 판별기",
         f"  단위       CLARE 태스크 {len(units)}개 "
         f"(base 보정 {sum(1 for u in units if u['kind'] == 'base')}개 + "
         f"lifelong {sum(1 for u in units if u['kind'] == 'life')}개). "
         f"CLARE 태스크 하나 = LIBERO 태스크 하나",
         f"  확장       expand_threshold={args.expand_threshold}, "
         f"at_least_expand=shallowest (clare.py 기본값)",
         f"  base 보정  --steps 0 으로 판별기만 학습. 어댑터는 zeros 초기화 그대로 = 항등",
         f"  평가       칸당 {args.eval_episodes} rollout, eval seed {args.eval_seed}",
         f"  train seed {args.seed}", "",
         "  행 = 스테이지 k 까지 배운 뒤, 열 = 평가 태스크. 단위 %. 빈 칸 = 아직 안 배움/안 잼", ""]
    w = 4 if n_task > 20 else 6
    L.append("stage" + "".join(f"{('t' + str(t)):>{w}}" for t in range(n_task)))
    for k, row in enumerate(R):
        cells = "".join(f"{'':>{w}}" if row[t] is None else f"{row[t]:>{w}.0f}"
                        for t in range(n_task))
        L.append(f"{k:<5d}" + cells)
    L.append("")
    m = V4.metrics(R, learned_at, n_task)
    if m:
        L.append("  지표  " + "   ".join(
            f"{k} {m[k]:.1f}" for k in ("AvgSR", "FWT", "NBT", "AUC")
            if isinstance(m.get(k), (int, float))))
    path.write_text("\n".join(L) + "\n")
    return m


# ═════════════════════════════════════════════════════════════════════════════
#  CLARE 태스크 단위로 쪼개기
# ═════════════════════════════════════════════════════════════════════════════
def build_units(spec, args, man_exp) -> list[dict]:
    """clare.py 를 한 번 부를 때마다 하나씩 대응하는 "CLARE 태스크" 목록.

    CLARE 태스크 하나 = LIBERO 태스크 하나다. 확장 판정이 태스크마다 일어나야
    하므로 스테이지를 뭉치지 않는다.

    각 항목:
        kind     "base"(보정, steps 0) | "life"(실제 학습)
        entry    매니페스트의 태스크 엔트리. build_split 에 [entry] 로 넘긴다
        ready    이미 디스크에 있는 스플릿 이름. None 이면 만들어 써야 한다
        stage    이 단위를 끝낸 시점의 LOTUS 스테이지 (평가 시점 판정용)
        tag      출력 폴더 이름
    """
    units: list[dict] = []

    if not args.skip_base_calib:
        # base 태스크마다 판별기 하나. 어댑터는 zeros 초기화 그대로라 항등이고,
        # 태스크마다 하나씩 생겨도 전부 항등이므로 출력은 base 체크포인트 그대로다.
        for b, entry in enumerate(man_exp["base"]):
            units.append({"kind": "base", "entry": entry, "ready": None,
                          "stage": 0, "tag": f"base{b:02d}"})

    for s in man_exp["lifelong_stages"]:
        si, tasks = s["stage"], s["tasks"]
        for j, entry in enumerate(tasks):
            # 스테이지에 태스크가 하나뿐이면(goal/object) 이미 있는 스플릿을 그대로 쓴다.
            ready = f"stage{si}" if len(tasks) == 1 else None
            units.append({"kind": "life", "entry": entry, "ready": ready,
                          "stage": si, "tag": f"s{si}t{j:02d}"})
    return units


def prepare_split(unit, root, out_dir, cache) -> tuple[Path, bool]:
    """이 단위가 쓸 데이터셋 경로. (경로, 끝나고_지울지)

    ready 가 있으면 기존 스플릿을 그대로 쓴다(지우지 않는다).
    없으면 매니페스트 엔트리 하나로 태스크짜리 데이터셋을 새로 만든다.
    """
    if unit["ready"]:
        return root / unit["ready"], False
    dst = out_dir / "_split" / unit["tag"]
    if dst.exists():
        shutil.rmtree(dst)
    t0 = time.perf_counter()
    r = LBS.build_split(unit["tag"], dst, [unit["entry"]], cache, dry=False)
    logging.info(f"[clare]   스플릿 생성 {unit['tag']}: 에피소드 {r['episodes']} "
                 f"프레임 {r.get('frames', '?')}  ({time.perf_counter() - t0:.0f}s)")
    return dst, True


def run_clare(unit, prev_adapter, data_root, base_ckpt, out_dir, args, env_handle) -> Path:
    """clare.py 를 한 번 돌리고, 저장된 adapter 디렉터리를 돌려준다."""
    work = out_dir / "clare" / unit["tag"]
    if work.exists():
        # clare.py 는 output_dir 이 이미 있으면 FileExistsError 를 낸다.
        raise SystemExit(f"이미 있는 작업 폴더다: {work}  (지우고 다시 돌려라)")
    steps = 0 if unit["kind"] == "base" else args.steps

    cmd = [sys.executable, "-u", str(CLARE_PY),
           f"--seed={args.seed}",
           f"--job_name=lotus_clare_{args.suite}_{unit['tag']}",
           f"--output_dir={work}",
           "--dataset.repo_id=local/lotus",
           f"--dataset.root={data_root}",
           f"--policy.path={base_ckpt}",
           "--policy.push_to_hub=false",
           f"--batch_size={args.batch_size}",
           f"--num_workers={args.num_workers}",
           f"--steps={steps}",
           # env 는 설정 검증 때문에 필요하지만 eval_freq=0 이라 실제로 안 쓴다.
           # 평가는 이 드라이버가 V4.rollout_sr 로 따로 한다(seq/ER 과 같은 경로).
           "--env.type=libero",
           f"--env.benchmark={V4._benchmark_of(env_handle)}",
           f"--env.task={env_handle}",
           "--eval.batch_size=2", "--eval.n_episodes=2",
           "--eval_freq=0",
           f"--save_freq={NEVER}",
           f"--log_freq={args.log_every}",
           f"--expand_threshold={args.expand_threshold}",
           f"--detect_distribution_shift_steps={args.detect_steps}",
           f"--detect_distribution_shift_batch_size={args.batch_size}",
           f"--detect_distribution_shift_num_workers={args.num_workers}",
           f"--train_discriminators_steps={args.disc_steps}",
           f"--train_discriminators_log_freq={args.log_every}",
           f"--train_discriminators_save_freq={args.disc_steps}",
           f"--train_discriminators_eval_freq={NEVER}",
           "--wandb.enable=false"]
    # 스테이지 사슬: 첫 호출만 설계도, 이후는 직전 어댑터를 이어받는다.
    if prev_adapter is None:
        cmd.append(f"--peft_cfg_path={PEFT_BLUEPRINT}")
    else:
        cmd.append(f"--peft_weight_path={prev_adapter}")

    logging.info(f"[clare] ▶ {unit['tag']}  {unit['kind']}  "
                 f"어댑터 {steps} step + 판별기 {args.disc_steps} step")
    (out_dir / "cmd.log").open("a").write(" ".join(cmd) + "\n\n")
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        raise SystemExit(f"clare.py 가 실패했다 (unit={unit['tag']}, exit={rc})")
    adapter = work / "checkpoints" / "last" / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"어댑터가 저장되지 않았다: {adapter}")
    logging.info(f"[clare] ◀ {unit['tag']}  {(time.perf_counter() - t0) / 60:.1f}분")
    return adapter


def load_peft_policy(cfg, ds_meta, adapter: Path | None, device):
    """평가용 정책. adapter=None 이면 base 체크포인트 그대로(어댑터 없음).

    ★ 평가에 넘기는 것은 **안쪽 policy** 다 (peft_policy 가 아니다). PEFT 는
      cond_proj 자리에 CLARELayer 를 꽂아 두므로 policy 객체 하나만 있으면
      라우팅이 동작한다. clare.py 자신도 eval 에 policy 를 넘긴다.
    """
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    policy.eval()
    if adapter is not None:
        from peft import PeftModel
        from lerobot.scripts.clare import PeftWrapperPolicy
        wrapper = PeftWrapperPolicy(policy=policy)
        PeftModel.from_pretrained(wrapper, str(adapter),
                                  is_trainable=False, autocast_adapter_dtype=False)
    policy.to(device)
    policy.eval()
    return policy


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["goal", "object", "50"], required=True)
    ap.add_argument("--steps", type=int, default=10000,
                    help="CLARE 태스크당 어댑터 학습 스텝. seq-FT/ER 과 맞춘 값")
    ap.add_argument("--disc_steps", type=int, default=2000, help="판별기 학습 스텝")
    ap.add_argument("--detect_steps", type=int, default=200, help="분포 이동 감지 스텝")
    ap.add_argument("--expand_threshold", type=float, default=1.0,
                    help="원 논문 셸 스크립트와 같은 값")
    # ★ at_least_expand 는 **일부러 CLI 로 넘기지 않는다.** draccus 가 그 필드의
    #   Literal 타입을 디코딩하지 못해(No decoding function for type Literal[...])
    #   값을 주는 순간 파싱이 실패한다. 원본 bash/clare/*.sh 도 넘기지 않고
    #   기본값 "shallowest"(= 논문 동작)에 의존한다. 여기서도 그대로 둔다.
    ap.add_argument("--skip_base_calib", action="store_true",
                    help="base 라우터 보정을 건너뛴다 (보정의 효과를 재는 대조군용)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--n_stages", type=int, default=0, help="0 = 전부. 스모크용 축소")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_seed", type=int, default=7)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    # V4.build_eval_cfg 가 getattr 로 읽는 이름들 — 평가 설정에만 쓰인다
    ap.add_argument("--sampler_mode", default="clare")
    ap.add_argument("--anchor_weight", default="none")
    ap.add_argument("--lambda_level", type=float, default=0.0)
    ap.add_argument("--k_per_demo", type=int, default=0)
    ap.add_argument("--k_min", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--w_clip", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse()
    init_logging()
    spec = V4.suite_spec(args.suite)
    if args.n_stages and args.n_stages < len(spec["stage_tasks"]):
        spec["stage_tasks"] = spec["stage_tasks"][:args.n_stages]
        keep = spec["n_base"] + sum(len(x) for x in spec["stage_tasks"])
        spec["n_task"] = keep
        spec["env_task"] = {t: v for t, v in spec["env_task"].items() if t < keep}
        spec["instr"] = {t: v for t, v in spec["instr"].items() if t < keep}
        spec["learned_at"] = spec["learned_at"][:keep]
        logging.info(f"[clare] ★ 스모크 축소: stage {args.n_stages}개, task {keep}개")

    root = spec["root"]
    n_base, n_task = spec["n_base"], spec["n_task"]
    stage_tasks = spec["stage_tasks"]
    n_stages = len(stage_tasks)
    env_task, instr = spec["env_task"], spec["instr"]
    learned_at = spec["learned_at"]

    base_ckpt = V4.MODEL_ROOT / f"Lotus_clare_libero_{args.suite}_pretrain"
    if not (base_ckpt / "model.safetensors").exists():
        raise SystemExit(f"base 체크포인트가 없다: {base_ckpt}")
    out_dir = Path(args.out) if args.out else REPO / "results" / f"lotus_clare_{args.suite}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2, ensure_ascii=False)

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    R: list[list[float | None]] = [[None] * n_task for _ in range(n_stages + 1)]

    # 명령어 대조 — seq/ER 과 같은 사전 검사
    for split, ids in [("base", list(range(n_base)))] + [
            (f"stage{i + 1}", ids) for i, ids in enumerate(stage_tasks)]:
        have = set(V4.instructions_of(root / split).values())
        miss = [instr[t] for t in ids if instr[t] not in have]
        if miss:
            raise SystemExit(f"{split}: lotus_setting.json 의 명령어가 데이터셋에 없다 "
                             f"({len(miss)}개, 예: {miss[0]!r})")

    man = json.loads(LBS.MANIFEST.read_text())
    man_exp = man["experiments"][SUITE_EXP[args.suite]]
    units = build_units(spec, args, man_exp)
    if args.n_stages:
        units = [u for u in units if u["stage"] <= args.n_stages]
    split_cache: dict = {}
    logging.info(f"[clare] suite={args.suite}  base {n_base} task  {n_stages} stage  "
                 f"총 {n_task} task  CLARE 태스크 {len(units)}개  seed={args.seed}")

    cfg = V4.build_eval_cfg(args, str(base_ckpt), env_task[0])
    base_meta = LeRobotDatasetMetadata("local/lotus", root=str(root / "base"))

    # ── stage 0: base 체크포인트 그대로 (어댑터 없음) ────────────────────────
    policy = load_peft_policy(cfg, base_meta, None, device)
    for t in range(n_base):
        R[0][t] = V4.rollout_sr(policy, cfg, env_task[t], args)
        logging.info(f"[clare] SR stage0 task{t} ({env_task[t]}) = {R[0][t]}")
        write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec, units)
    del policy
    torch.cuda.empty_cache()

    # ── CLARE 태스크를 순서대로. 스테이지가 끝날 때마다 평가 ─────────────────
    t_start = time.perf_counter()
    prev_adapter: Path | None = None
    for i, unit in enumerate(units):
        data_root, disposable = prepare_split(unit, root, out_dir, split_cache)
        try:
            prev_adapter = run_clare(unit, prev_adapter, data_root, base_ckpt,
                                     out_dir, args, env_task[0])
        finally:
            # 태스크짜리 스플릿은 바로 지운다 — LIBERO-50 base 는 전부 두면 19GB 다.
            if disposable and data_root.exists():
                shutil.rmtree(data_root, ignore_errors=True)
        # 이 단위가 그 스테이지의 마지막인가?
        last_of_stage = (unit["stage"] > 0 and
                         (i + 1 == len(units) or units[i + 1]["stage"] != unit["stage"]))
        if not last_of_stage:
            continue
        stage = unit["stage"]
        n_seen = stage_tasks[stage - 1][-1] + 1
        policy = load_peft_policy(cfg, base_meta, prev_adapter, device)
        for t in range(n_seen):
            R[stage][t] = V4.rollout_sr(policy, cfg, env_task[t], args)
            logging.info(f"[clare] SR stage{stage} task{t} ({env_task[t]}) = {R[stage][t]}")
            write_report(out_dir / "report.txt", R, learned_at, instr, args,
                         n_task, spec, units)
        del policy
        torch.cuda.empty_cache()

    m = write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec, units)
    json.dump({"suite": args.suite, "method": "clare", "R": R, "learned_at": learned_at,
               "stage_tasks": stage_tasks,
               "env_task": {str(k): v for k, v in env_task.items()},
               "metrics": m, "instructions": {str(k): v for k, v in instr.items()},
               "n_clare_tasks": len(units),
               "base_calibrated": not args.skip_base_calib,
               "wall_hours": (time.perf_counter() - t_start) / 3600},
              (out_dir / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    logging.info(f"[clare] 끝. {out_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
