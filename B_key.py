#!/usr/bin/env python
"""어느 키가 라우팅하는가 — 관측 o 인가 명령어 ℓ 인가.

배경
  협업자 분석의 핵심 주장: ER 은 요구마다 좌표가 달라서(o_j 에 v*_j) 언어 경로가
  전혀 없어도 성공할 수 있다. 반사실 앵커는 같은 좌표 (o_new, x_t^new) 에 K+2개
  요구를 겹쳐 놓고 구분 키를 ℓ 하나로 몰아넣는다. 그래서 ℓ 이 공선이면 무너진다.

  이게 맞다면 ER 모델은 **명령어를 바꿔 끼워도 성공률이 유지**돼야 한다 — o 가
  이미 어느 태스크인지 말해 주기 때문이다. 반대로 ℓ 이 실제로 하중을 지고 있다면
  명령어를 바꾸는 순간 무너진다.

측정 (명령어 교체 롤아웃)
  stage 3 체크포인트로 태스크 j 를 롤아웃하되 명령어만 ℓ_i 로 강제한다.
  4 태스크 x 4 명령어 = 16 칸, 칸당 20 롤아웃.

    대각선 >> 비대각선  ->  ℓ 이 하중을 진다
    행 전체가 평평       ->  o 가 라우팅한다. ℓ 은 무시된다
    전부 낮다            ->  그 태스크 자체를 잃었다

  ★ 이건 SR 로 재는 것이지 velocity 거리로 재는 것이 아니다. B_merge 는 후자를
    이미 쟀고(ER 의 라우팅은 velocity 공간에서 정상), 그게 롤아웃으로 이어지는지는
    별개 문제라는 것을 B_chunk 에서 확인했다.

정적 측정도 함께 남긴다
  err[i]  = ‖v(o_j, ℓ_i) − v*_j‖ / ‖v*_j‖
  pair    = ‖v(o_j, ℓ_i) − v(o_j, ℓ_j)‖ / ‖v*_j‖
"""
from __future__ import annotations

import argparse, json, sys
from contextlib import contextmanager
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B_merge import ARMS, _ns

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402


@contextmanager
def force_lang(policy, text: str, lang_dim: int):
    """추론 내내 조건 벡터의 언어 슬롯을 text 로 덮어쓴다.

    B1.cfg_guidance 와 같은 방식이다 — velocity_net.sample() 이 self.forward 를
    부르므로(modeling_dit_flow_mt.py:779) 인스턴스 수준에서 감싸면 100 스텝 적분
    전체에 적용된다. 환경이 주는 batch["task"] 는 그대로 두고 조건만 갈아 끼운다.
    """
    net = policy.dit_flow.velocity_net
    base = net.forward
    vec = B1.encode_lang(policy, [text]).detach()

    def patched(noisy_actions, time, global_cond):
        g = global_cond.clone()
        g[:, :lang_dim] = vec.to(g.dtype).expand(g.shape[0], lang_dim)
        return base(noisy_actions=noisy_actions, time=time, global_cond=g)

    net.forward = patched
    try:
        yield
    finally:
        net.forward = base


def rollout(policy, cfg, env_task, text, lang_dim, a):
    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import eval_policy

    env = None
    try:
        env_cfg = __import__("copy").deepcopy(cfg.env)
        env_cfg.task = env_task
        env = make_env(env_cfg, n_envs=a.eval_batch_size, use_async_envs=False)
        policy.eval()
        with torch.no_grad(), force_lang(policy, text, lang_dim):
            info = eval_policy(env, policy, a.eval_episodes, start_seed=a.seed)
        return float(info["aggregated"]["pc_success"])
    except Exception as e:
        print(f"[key] 롤아웃 실패 {env_task}/{text[:30]}: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        if env is not None:
            env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ER,B2λ3,seq-FT")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--num_tasks", type=int, default=4)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--static_only", action="store_true")
    ap.add_argument("--out", default="results/B_key")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    K = a.num_tasks
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, env_prefix = B1.suite_prefixes(a.suite)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = [B1.task_instruction(f"{ds_prefix}{i}") for i in range(K)]

    def ckpt(root, k):
        if isinstance(root, dict):
            return REPO / root["tmpl"].format(k=k)
        return (REPO / root / f"{a.suite}_seed42_ours" / f"task_{k}"
                / "checkpoints" / a.steps_tag / "pretrained_model")

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_key"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    sr_path = out / "swap_sr.jsonl"
    done = set()
    if sr_path.exists():                       # 재시작 대비
        for line in sr_path.read_text().splitlines():
            r = json.loads(line); done.add((r["arm"], r["task"], r["instr"]))

    # ── 정적 측정용 고정 배치 (모든 팔이 같은 좌표를 본다) ──────────────────
    seed_pol = load(ckpt(ARMS[arms[0]], a.stage))
    data = {}
    for j in range(K):
        cfg = B1.build_cfg(_ns(a), j, str(ckpt(ARMS[arms[0]], a.stage)), Path("/tmp/b_key"))
        ds = make_dataset(cfg)
        sp = EpisodeAwareSampler(ds.episode_data_index,
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sp,
                                         num_workers=0, drop_last=True)
        torch.manual_seed(a.seed); it = iter(dl)
        bs, fm = [], []
        for i in range(a.n_batches):
            b = B1.prep_batch(seed_pol, B1.to_device(next(it), device))
            torch.manual_seed(a.seed * 7 + j * 131 + i)
            bs.append(b); fm.append(B1.sample_fm(seed_pol, b))
        data[j] = (bs, fm)
        del ds, dl, it
    del seed_pol; torch.cuda.empty_cache()

    static = []
    for name in arms:
        pol = load(ckpt(ARMS[name], a.stage))
        lang_dim = pol.dit_flow.language_embedding_projection.out_features
        net = pol.dit_flow.velocity_net
        with torch.no_grad():
            for j in range(K):
                vs, scale = {}, 0.0
                for i in range(K):
                    acc = []
                    for b, (x_t, t, tgt) in zip(*data[j]):
                        n = x_t.shape[0]
                        cond = B1.make_cond(B1.encode_lang(pol, [instr[i]] * n),
                                            B1.cond_tail(pol, b))
                        acc.append(net(noisy_actions=x_t, time=t, global_cond=cond).clone())
                    vs[i] = acc
                scale = sum(float(tgt.flatten(1).norm(dim=1).mean())
                            for _, _, tgt in data[j][1]) / a.n_batches
                row = {"arm": name, "task": j, "scale": scale}
                for i in range(K):
                    err = sum(float((v - tgt).flatten(1).norm(dim=1).mean())
                              for v, (_, _, tgt) in zip(vs[i], data[j][1])) / a.n_batches
                    pair = sum(float((v - w).flatten(1).norm(dim=1).mean())
                               for v, w in zip(vs[i], vs[j])) / a.n_batches
                    row[f"err{i}"] = err / max(scale, 1e-8)
                    row[f"pair{i}"] = pair / max(scale, 1e-8)
                static.append(row)
                print(f"[key/static] {name:>7} obs task{j}  "
                      + "  ".join(f"err{i}={row[f'err{i}']:.3f}" for i in range(K)), flush=True)
        del pol; torch.cuda.empty_cache()
    json.dump(static, (out / "static.json").open("w"), indent=2)

    if a.static_only:
        print("static_only — 롤아웃 생략"); return

    # ── 명령어 교체 롤아웃 ───────────────────────────────────────────────────
    for name in arms:
        p = ckpt(ARMS[name], a.stage)
        if not p.is_dir():
            print(f"[key] 체크포인트 없음: {name}"); continue
        pol = load(p)
        lang_dim = pol.dit_flow.language_embedding_projection.out_features
        for j in range(K):
            cfg = B1.build_cfg(_ns(a), j, str(p), Path("/tmp/b_key"))
            for i in range(K):
                if (name, j, i) in done:
                    continue
                sr = rollout(pol, cfg, f"{env_prefix}{j}", instr[i], lang_dim, a)
                with sr_path.open("a") as f:
                    f.write(json.dumps({"arm": name, "task": j, "instr": i, "sr": sr}) + "\n")
                print(f"[key/swap] {name:>7} task{j} <- ℓ{i}  SR={sr}", flush=True)
        del pol; torch.cuda.empty_cache()
    print(f"완료 -> {sr_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
