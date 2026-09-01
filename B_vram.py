"""앵커가 과거 태스크 수 K 에 따라 쓰는 최대 메모리.

torch 할당자 기준(max_memory_allocated / reserved)이라 CUDA 컨텍스트와 시뮬레이터
메모리가 빠져 있다. 논문에 "이 방법이 필요로 하는 메모리"로 적을 값은 이쪽이다.
nvidia-smi 숫자는 컨텍스트 + 할당자 예약 + (평가 중이면) LIBERO 환경까지 포함한다.

사용법:  CUDA_VISIBLE_DEVICES=2 python B_vram.py
결과는 results/VRAM.txt 에 손으로 옮겨 적었다.
"""
import sys, argparse, torch
from pathlib import Path
sys.path.insert(0,"/home/sa090180/clare")
import B1
from B_merge import _ns
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.utils.utils import init_logging, get_safe_torch_device
init_logging()
dev=get_safe_torch_device("cuda",log=False)
a=argparse.Namespace(suite="libero_spatial",device="cuda",seed=1,num_workers=0,batch_size=32,
    steps_per_task=1,log_every=100,mode="mem",eval_episodes=1,eval_batch_size=1,
    p_drop=0.0,lambda_anchor=1.0)
ck="outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model"
meta=LeRobotDatasetMetadata("continuallearning/libero_spatial_image_task_0")
cfg=B1.build_cfg(a,0,ck,Path("/tmp/rmem"))
pol=make_policy(cfg=cfg.policy,ds_meta=meta); pol.train()
ds=make_dataset(cfg)
dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
b=B1.prep_batch(pol,B1.to_device(next(iter(dl)),dev))
teach=B1.snapshot(pol)
instr=B1.task_instruction("continuallearning/libero_spatial_image_task_0")
import argparse as _ap
_a=_ap.ArgumentParser(); _a.add_argument("--mode", choices=["batch","chunk","both"], default="both")
_args,_ = _a.parse_known_args()

def run(K, chunked):
    """K개 과거 태스크에 대한 앵커 한 스텝. chunked면 j마다 즉시 backward."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cls = B1.rgb_cls(pol, b); tail = B1.cond_tail(pol, b, cls)
    x_t, t, tgt = B1.sample_fm(pol, b)
    cond = B1.make_cond(B1.encode_lang(pol, list(b["task"])), tail)
    l_fm = B1.fm_loss(pol.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond), tgt)
    n = x_t.shape[0]; past = [instr] * n

    def anchor_j():
        """과거 하나에 대한 level+structure 두 점."""
        acc = 0.0
        for _ in range(2):
            tl = B1.cond_tail(pol, b, cls)
            c = B1.make_cond(B1.encode_lang(pol, past), tl)
            vs = pol.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=c)
            with torch.no_grad():
                tt = B1.teacher_tail(pol, teach, b, cls)
                tc = B1.make_cond(B1.encode_lang(teach, past), tt)
                vt = teach.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=tc)
            acc = acc + (vs - vt).pow(2).mean()
        return acc

    if chunked:
        # ★ 항마다 즉시 backward. 그래프가 하나씩만 살아 있다.
        #   L = L_FM + Σ_j L_j 이고 미분은 선형이므로 결과 그래디언트는 동일하다.
        l_fm.backward()
        for _ in range(K):
            anchor_j().backward()
    else:
        loss = l_fm
        for _ in range(K):
            loss = loss + anchor_j()
        loss.backward()
    pol.zero_grad(set_to_none=True)
    return (torch.cuda.max_memory_allocated() / 2**20,
            torch.cuda.max_memory_reserved() / 2**20)

KS = (1, 3, 5, 7, 9)
print(f"{'K(과거수)':>10}{'grad fwd':>10}{'일괄 alloc':>13}{'청크 alloc':>13}{'절감':>9}")
for K in KS:
    a = run(K, False) if _args.mode in ("batch", "both") else (float("nan"),)*2
    c = run(K, True) if _args.mode in ("chunk", "both") else (float("nan"),)*2
    save = (1 - c[0]/a[0]) * 100 if a[0] == a[0] and c[0] == c[0] else float("nan")
    print(f"{K:>10}{1+2*K:>10}{a[0]:>10.0f} MiB{c[0]:>10.0f} MiB{save:>8.0f}%")

# 그래디언트가 실제로 같은지 확인 — 청크 backward 가 수학적으로 동일한가
torch.manual_seed(0)
def grads(chunked):
    pol.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    run(3, chunked)
    return None
print("\n[검증] 청크 backward 가 일괄과 같은 그래디언트를 주는지")
pol.zero_grad(set_to_none=True); torch.manual_seed(1234)
cls = B1.rgb_cls(pol, b); tail = B1.cond_tail(pol, b, cls)
x_t, t, tgt = B1.sample_fm(pol, b)
cond = B1.make_cond(B1.encode_lang(pol, list(b["task"])), tail)
n = x_t.shape[0]; past = [instr] * n
def one():
    tl = B1.cond_tail(pol, b, cls)
    c = B1.make_cond(B1.encode_lang(pol, past), tl)
    vs = pol.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=c)
    with torch.no_grad():
        tt = B1.teacher_tail(pol, teach, b, cls)
        tc = B1.make_cond(B1.encode_lang(teach, past), tt)
        vt = teach.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=tc)
    return (vs - vt).pow(2).mean()
L = B1.fm_loss(pol.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond), tgt)
t1, t2 = one(), one()
(L + t1 + t2).backward()
g_batch = [p.grad.detach().clone() for p in pol.parameters() if p.grad is not None]
pol.zero_grad(set_to_none=True)
L2 = B1.fm_loss(pol.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond), tgt)
L2.backward(); one().backward(); one().backward()
g_chunk = [p.grad.detach().clone() for p in pol.parameters() if p.grad is not None]
d = max(float((a - c).abs().max()) for a, c in zip(g_batch, g_chunk))
r = max(float((a - c).abs().max() / a.abs().max().clamp_min(1e-12)) for a, c in zip(g_batch, g_chunk))
print(f"  최대 절대차 {d:.3e}   최대 상대차 {r:.3e}")
pol.zero_grad(set_to_none=True)
