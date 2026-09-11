"""D2 — 옛 task 실패의 단계 귀속 (rollout).

가설: 앵커가 느린 구간의 방향을 덜 보호한다면, stage j 이후 **새로 생긴** 실패는
정밀한 방향이 필요한 grasp·place 단계에 몰려야 한다 (P4). 그리고 그 실패가 최신
task 로 끌려간 것(recency)이 아니라면, reach 단계에서 다른 목표로 가 버린 비율은
낮아야 한다 (P5).

롤아웃은 lerobot/scripts/eval.py 의 rollout()/eval_policy() 를 복제한다 — 스텝마다
sim 내부 상태를 읽어야 해서 eval_policy 를 그대로 쓸 수 없다. 시드 규칙
(start_seed + batch_ix*n_envs ...) 과 종료 조건은 eval.py:330-341 과 같게 맞췄다.

목표 물체는 bddl 에서 온다: robosuite env 의 obj_of_interest = [목표 그릇, 접시]
(bddl goal_state = [['on', <그릇>, <접시>]] 와 일치함을 전 task 에서 확인했다).
따라서 "데모의 파지 시점 위치로 대체" 하는 우회는 필요 없었다.

단계 판정 (임계는 아래 상수, 실제 사용값은 REPORT 4절에 기록)
    파지 시도   gripper 액션 > 0 이 CLOSE_RUN 스텝 연속인 첫 지점
    운반        파지 시점 대비 물체가 CARRY_D 이상 이동하는 동안 eef 가 HOLD_TOL 이내
    접시 도달   운반 이후 물체가 접시 중심 NEAR_PLATE 이내
    reach 임계  성공 에피소드 d_close 의 REACH_Q 분위수로 **보정**한다 (고정값 아님)

자기검증: 성공 에피소드가 물리적으로 말이 되는지 본 뒤에만 판정을 낸다.
통과하지 못하면 P4·P5 를 자동으로 '판단불가' 로 강등한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch

from common import OUT, SUITE, SEED_RUN, N_STAGES, PolicyCache, stage_ckpt

ENV_PREFIX = "Libero_Spatial_Task_"
GRIP_CLOSE = 0.0      # gripper 액션 > 0 = 닫기 (robosuite OSC_POSE). 사실상 ±1 뱅뱅이다.
CLOSE_RUN = 10        # 이만큼 연속으로 닫기 명령이 나와야 "파지 시도"로 본다
HOLD_TOL = 0.12       # 물체를 '붙들고 있다'고 볼 eef-물체 거리 상한 (m)
CARRY_D = 0.03        # 파지 후 물체가 이만큼 움직이면 "들어서 옮겼다"로 본다 (m)
LIFT_DZ = 0.02        # (참고용) 절대 z 상승. 판정에는 쓰지 않는다 — 아래 ★ 참조
NEAR_PLATE = 0.05     # 접시 중심 반경 5cm 이내면 도달
REACH_Q = 0.95        # reach 임계는 **성공 에피소드에서 관측된 파지 거리**의 이 분위수
REACH_FLOOR = 0.08    # 성공 표본이 부족할 때 쓰는 하한 (m)
STAGES = ["reach", "grasp", "transport", "place"]

# ★ 왜 임계를 데이터에서 뽑는가
#   eef 좌표는 robot0_eef_pos(손목), 물체 좌표는 body_xpos(질량중심)이라 둘 사이에
#   구조적 오프셋이 있다. 실측(task7 성공 4 에피소드): 파지 시점 거리 4.8~6.7cm,
#   에피소드 전체 최소거리도 4.1~4.9cm 였다. 따라서 고정 4cm 임계는 **성공한
#   에피소드조차 만족하지 못해** 모든 실패를 reach 로 몰아넣는다(초판의 버그).
#   그래서 임계를 같은 런의 성공 에피소드 분포에서 정하고 그 값을 보고서에 적는다.
#
# ★ 왜 "절대 z 상승" 을 파지 판정에 쓰지 않는가
#   task 3("black bowl **on the cookie box**") 과 task 5("**on the ramekin**") 는
#   그릇이 다른 물체 위에서 시작한다. 접시는 그보다 낮으므로 로봇이 들어 옮겨도
#   z 가 시작값을 넘지 않는다. 실측: 이 두 task 의 성공 에피소드에서 z_max − z0 가
#   각각 0.0000 / 0.0090 m 로, LIFT_DZ=0.02 를 **성공해도** 만족하지 못했다
#   (나머지 7 task 는 0.042~0.100 m 로 정상). 그래서 판정 기준을
#   "파지 시점 대비 물체의 3D 이동량 + 그때 그리퍼가 물체를 붙들고 있는가" 로 바꿨다.
#   시작 높이에 의존하지 않는다.


def _raw(venv, i):
    """SyncVectorEnv 의 i 번째 robosuite env."""
    return venv.envs[i].unwrapped._env


def _bodies(venv, i):
    """(목표 그릇 body id, 접시 body id, 같은 씬의 다른 그릇 body id 들)."""
    raw = _raw(venv, i)
    sim = raw.sim
    bid = lambda n: sim.model.body_name2id(f"{n}_main")          # noqa: E731
    tgt, plate = raw.obj_of_interest[0], raw.obj_of_interest[1]
    others = [o for o in raw.objects_dict
              if o != tgt and "bowl" in o]                       # 같은 씬의 다른 그릇
    return bid(tgt), bid(plate), [bid(o) for o in others]


@torch.no_grad()
def rollout_record(venv, policy, seeds, max_steps, device):
    """eval.py:128-205 의 rollout 을 복제하되 스텝마다 sim 상태를 기록한다.

    원본과 맞춘 것
      · policy.reset() -> env.reset(seed=seeds)
      · reset 직후 _init_state_id 를 num_envs 만큼 회전 (eval.py:131-134).
        이걸 빼면 매 배치가 같은 초기 상태를 보게 되어 SR 이 달라진다.
      · preprocess_observation 으로 관측을 정책 키로 바꾼 뒤 텐서만 device 로.
        "task" 는 문자열이라 그대로 통과한다 (envs/utils.py: preprocess_observation).
      · 성공 판정은 final_info / is_success 두 규약을 모두 처리 (eval.py:194-201).
    """
    from lerobot.envs.utils import preprocess_observation

    policy.reset()
    obs, _ = venv.reset(seed=seeds)
    n = venv.num_envs
    for i in range(n):                                    # eval.py:131-134
        e = venv.envs[i].env.env
        e._init_state_id = (e._init_state_id + n) % 50

    ids = [_bodies(venv, i) for i in range(n)]
    rec = {k: [] for k in ("eef", "grip_cmd", "grip_q", "tgt", "plate", "other")}
    rec["valid"] = []          # ★ env 별 done 이후 프레임을 무효로 표시한다
    done = np.array([False] * n)
    success = np.zeros(n, dtype=bool)
    step = 0
    while not np.all(done) and step < max_steps:
        ob = preprocess_observation(obs)
        ob = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in ob.items()}
        with torch.inference_mode():
            action = policy.select_action(ob)
        a = action.to("cpu").numpy()

        # sim 상태는 **행동을 넣기 직전** 시점으로 기록한다
        eef, gq, tg, pl, ot = [], [], [], [], []
        for i in range(n):
            sim = _raw(venv, i).sim
            bt, bp, bo = ids[i]
            xp = sim.data.body_xpos
            eef.append(np.asarray(obs["agent_pos"][i][:3], dtype=float))   # robot0_eef_pos
            gq.append(np.asarray(obs["agent_pos"][i][6:8], dtype=float))   # gripper qpos
            tg.append(xp[bt].copy())
            pl.append(xp[bp].copy())
            ot.append(np.stack([xp[b] for b in bo]) if bo else np.zeros((0, 3)))
        rec["eef"].append(np.stack(eef)); rec["grip_q"].append(np.stack(gq))
        rec["tgt"].append(np.stack(tg)); rec["plate"].append(np.stack(pl))
        rec["other"].append(ot); rec["grip_cmd"].append(a[:, 6].copy())
        # 이 프레임이 아직 해당 env 의 **이번** 에피소드인가.
        # gymnasium autoreset 은 done 이 된 env 를 즉시 새 에피소드로 되돌리므로,
        # 마스킹하지 않으면 다음 에피소드의 물체 위치가 이번 궤적에 섞인다.
        rec["valid"].append(~done.copy())

        obs, _, term, trunc, info = venv.step(a)
        if "final_info" in info:
            s = np.array([bool(x["is_success"]) if x is not None else False
                          for x in info["final_info"]])
        elif "is_success" in info:
            valid = info.get("_is_success", np.ones(n, dtype=bool))
            s = np.array([bool(x) and bool(m)
                          for x, m in zip(info["is_success"], valid)])
        else:
            s = np.zeros(n, dtype=bool)
        success |= (s & ~done)
        done = done | np.asarray(term, bool) | np.asarray(trunc, bool)
        step += 1
    return rec, success, step


def episode_traces(rec, success, n):
    """에피소드별 요약 지표. done 이후 프레임은 잘라낸다. 분류는 임계 보정 후에 한다."""
    T = len(rec["grip_cmd"])
    out = []
    for i in range(n):
        valid = np.array([rec["valid"][t][i] for t in range(T)])
        T_i = int(valid.sum())                     # 이 env 의 실제 에피소드 길이
        if T_i == 0:
            out.append({"success": bool(success[i]), "t_close": -1, "d_close": None,
                        "d_min": None, "d_other_close": None, "z0": None, "z_max": None,
                        "carried": False, "near_plate_after_carry": False,
                        "obj_disp_max": None, "steps": 0})
            continue
        sl = slice(0, T_i)
        cmd = np.array([rec["grip_cmd"][t][i] for t in range(T)])[sl]
        eef = np.array([rec["eef"][t][i] for t in range(T)])[sl]
        tgt = np.array([rec["tgt"][t][i] for t in range(T)])[sl]
        pla = np.array([rec["plate"][t][i] for t in range(T)])[sl]
        d_t = np.linalg.norm(eef - tgt, axis=1)
        d_p = np.linalg.norm(tgt[:, :2] - pla[:, :2], axis=1)
        oth = [rec["other"][t][i] for t in range(T)][:T_i]
        d_o = np.array([np.min(np.linalg.norm(o - eef[t], axis=1)) if len(o) else np.inf
                        for t, o in enumerate(oth)])
        # 지속 close: cmd>0 이 CLOSE_RUN 스텝 연속인 첫 지점
        pos = (cmd > GRIP_CLOSE).astype(int)
        c = -1
        if len(pos) >= CLOSE_RUN:
            run = np.convolve(pos, np.ones(CLOSE_RUN, int), "valid")
            if (run == CLOSE_RUN).any():
                c = int(np.argmax(run == CLOSE_RUN))
        # 파지 후 "들어서 옮김": 물체가 파지 시점 대비 CARRY_D 이상 움직이는 동안
        # 그리퍼가 여전히 물체를 붙들고 있는가. 시작 높이에 의존하지 않는다.
        carried, near_after, disp_max = False, False, 0.0
        if c >= 0:
            disp = np.linalg.norm(tgt[c:] - tgt[c], axis=1)
            disp_max = float(disp.max())
            held = d_t[c:] <= HOLD_TOL
            k = np.where((disp > CARRY_D) & held)[0]
            if len(k):
                carried = True
                near_after = bool((d_p[c + int(k[0]):] < NEAR_PLATE).any())
        out.append({"success": bool(success[i]), "t_close": c,
                    "d_close": float(d_t[c]) if c >= 0 else None,
                    "d_min": float(d_t.min()),
                    "d_other_close": float(d_o[c]) if c >= 0 and np.isfinite(d_o[c]) else None,
                    "z0": float(tgt[0, 2]), "z_max": float(tgt[:, 2].max()),
                    "carried": carried, "near_plate_after_carry": near_after,
                    "obj_disp_max": disp_max, "steps": T_i})
    return out


def calibrate_reach(df):
    """성공 에피소드의 파지 시점 거리 분포에서 reach 임계를 정한다."""
    ok = df[df.success & df.d_close.notna()]
    if len(ok) < 20:
        return REACH_FLOOR, len(ok), None
    q = float(np.quantile(ok.d_close, REACH_Q))
    return max(q, REACH_FLOOR), len(ok), q


def assign_stage(row, tol):
    """실패 에피소드 하나에 단계 하나. 성공이면 None.

    물체 상태를 우선으로 본다 — 거리 임계에 덜 기대는 순서다.
      carry 없음 + 파지 시도가 목표에서 멀다(또는 아예 없다) -> reach
      carry 없음 + 파지 시도가 목표 근처                     -> grasp
      carry 있음 + 접시 근처 도달 없음                        -> transport
      carry 있음 + 접시 근처 도달                             -> place
    """
    if row["success"]:
        return None
    if not row["carried"]:
        if row["t_close"] < 0 or row["d_close"] is None or row["d_close"] > tol:
            return "reach"
        return "grasp"
    return "place" if row["near_plate_after_carry"] else "transport"


def build_env(task_id, args):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    cfg = LiberoEnv(benchmark=SUITE, task=f"{ENV_PREFIX}{task_id}")
    return make_env(cfg, n_envs=args.eval_batch_size, use_async_envs=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--eval_start_seed", type=int, default=SEED_RUN)
    ap.add_argument("--tasks", default=",".join(str(t) for t in range(N_STAGES)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    pc = PolicyCache(device)
    tasks = [int(x) for x in args.tasks.split(",")]
    final = N_STAGES - 1                      # 이 런의 최종 stage (seed41 은 8)

    rows, t0 = [], time.perf_counter()
    for task in tasks:
        for tag, ck in (("diag", task), ("final", final)):
            if ck < task:                     # 아직 배우기 전 -> 의미 없음
                continue
            policy = pc.get(ck)
            policy.eval()
            venv = build_env(task, args)
            try:
                nb = args.eval_episodes // venv.num_envs + \
                    int(args.eval_episodes % venv.num_envs != 0)
                maxs = venv.call("_max_episode_steps")[0]
                for b in range(nb):
                    seeds = list(range(args.eval_start_seed + b * venv.num_envs,
                                       args.eval_start_seed + (b + 1) * venv.num_envs))
                    rec, succ, nst = rollout_record(venv, policy, seeds, maxs, device)
                    for i, r in enumerate(episode_traces(rec, succ, venv.num_envs)):
                        r.update({"task": task, "ckpt": tag, "ckpt_stage": ck,
                                  "seed": seeds[i]})
                        rows.append(r)
            except Exception as e:
                print(f"[D2] 실패 task{task} {tag}: {type(e).__name__}: {e}", flush=True)
            finally:
                venv.close()
            df = pd.DataFrame(rows)
            df.to_csv(OUT / "d2_episodes.csv", index=False)
            sub = df[(df.task == task) & (df.ckpt == tag)]
            f = sub[~sub.success]
            print(f"[D2] task{task} {tag}(stage{ck})  n={len(sub)} "
                  f"SR={100*sub.success.mean():.0f}%  "
                  f"실패 {len(f)} (carry 성공 {int(f.carried.sum())}, "
                  f"파지시도거리 중앙값 "
                  f"{f.d_close.median() if len(f) and f.d_close.notna().any() else float('nan'):.3f}m)  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    aggregate()
    make_plot()


def aggregate():
    """임계 보정 -> 단계 배정 -> 자기검증 -> 집계. 실패 0 건이어도 동작해야 한다."""
    df = pd.read_csv(OUT / "d2_episodes.csv")

    # (1) reach 임계를 성공 에피소드에서 보정한다
    tol, n_ok, q95 = calibrate_reach(df)
    # (2) 자기검증 — 성공 에피소드가 물리적으로 말이 되는가.
    #     ★ reach_ok 는 **게이트에 쓰지 않는다**. tol 을 성공 d_close 의 REACH_Q
    #       분위수로 정해 놓고 성공 중 몇 %가 tol 이하인지 세는 것이라 정의상
    #       항상 ≈REACH_Q 이고, 표본이 적으면 분위수 이산성만으로 오르내린다
    #       (실측: n=45 에서 42/45=0.933). 정보가 없는 검사라 참고로만 남긴다.
    #     게이트는 **실제로 틀릴 수 있는** 것만 본다:
    #       carry_ok      성공했는데 물체를 들어 옮긴 적이 없다 -> 파지 지표가 깨진 것
    #                     (LIFT_DZ 절대 z 기준의 버그를 이 검사가 잡아냈다: 0.787)
    #       near_plate_ok 성공했는데 접시 근처에 간 적이 없다 -> 접시 지표가 깨진 것
    #       tol_sane      보정된 임계가 물리적으로 그럴듯한 범위인가
    #       separation    성공과 reach 실패의 d_close 가 실제로 갈리는가
    ok = df[df.success]
    fail = df[~df.success]
    med_ok = float(ok.d_close.median()) if ok.d_close.notna().any() else None
    med_bad = float(fail.d_close.median()) if fail.d_close.notna().any() else None
    chk = {
        "n_success": int(len(ok)),
        "carry_ok": float(ok.carried.mean()) if len(ok) else None,
        "near_plate_ok": float(ok.near_plate_after_carry.mean()) if len(ok) else None,
        "tol_used_m": tol, "reach_q95_of_success_m": q95,
        "n_success_with_close": int(n_ok),
        "median_d_close_success_m": med_ok,
        "median_d_close_failure_m": med_bad,
        "reach_ok_informational": (
            float((ok.d_close.notna() & (ok.d_close <= tol)).mean()) if len(ok) else None),
    }
    chk["tol_sane"] = bool(0.03 <= tol <= 0.25)
    chk["separation_ok"] = bool(med_ok is not None and med_bad is not None and med_bad > med_ok)
    chk["passed"] = bool(
        len(ok) == 0 or
        (chk["carry_ok"] >= 0.95 and chk["near_plate_ok"] >= 0.90
         and chk["tol_sane"] and chk["separation_ok"]))
    (OUT / "d2_selfcheck.json").write_text(json.dumps(chk, indent=2, ensure_ascii=False))
    print("[D2][자기검증] " + json.dumps(chk, ensure_ascii=False), flush=True)
    # (3) 단계 배정
    df["stage"] = [assign_stage(r, tol) for r in df.to_dict("records")]
    df.to_csv(OUT / "d2_episodes.csv", index=False)
    fails = df[~df.success]
    idx = pd.MultiIndex.from_product(
        [sorted(df.task.unique()), ["diag", "final"]], names=["task", "ckpt"])
    if len(fails):
        piv = (fails.groupby(["task", "ckpt", "stage"]).size()
               .unstack("stage").fillna(0).astype(int))
    else:
        piv = pd.DataFrame(index=idx[:0])
    piv = piv.reindex(idx, fill_value=0)
    for s in STAGES:
        if s not in piv.columns:
            piv[s] = 0
    piv = piv[STAGES].fillna(0).astype(int)
    piv.to_csv(OUT / "d2_stage_counts.csv")

    # 최종 − 대각
    d = []
    for task in sorted(df.task.unique()):
        if (task, "final") not in piv.index or (task, "diag") not in piv.index:
            continue
        fin, dia = piv.loc[(task, "final")], piv.loc[(task, "diag")]
        d.append({"task": task, **{f"d_{s}": int(fin[s] - dia[s]) for s in STAGES}})
    D = pd.DataFrame(d, columns=["task"] + [f"d_{s}" for s in STAGES])
    D.to_csv(OUT / "d2_delta.csv", index=False)

    tot = {s: int(D[f"d_{s}"].clip(lower=0).sum()) if len(D) else 0 for s in STAGES}
    new_tot = sum(tot.values())
    v = {"P4": {"new_failures_by_stage": tot,
                "share_grasp_place": (tot["grasp"] + tot["place"]) / new_tot if new_tot else None,
                "n_new_failures": new_tot}}
    rf = df[(~df.success) & (df.stage == "reach")]
    near_other = (rf.d_other_close.notna() & (rf.d_other_close <= tol))
    v["P5"] = {"n_reach_failures": int(len(rf)),
               "frac_near_other_bowl": float(near_other.mean()) if len(rf) else None,
               "note": "각 task 는 서로 다른 bddl 씬이라 '다른 task 의 목표 그릇'이 "
                       "그 씬에 없다. 같은 씬의 **비목표** 그릇(akita_black_bowl_2 등)에 "
                       "4cm 이내인 비율로 대체했다."}
    v["selfcheck"] = chk
    if not chk["passed"]:
        v["P4"]["verdict_override"] = "판단불가 — 분류기 자기검증 실패"
        v["P5"]["verdict_override"] = "판단불가 — 분류기 자기검증 실패"
    (OUT / "d2_verdicts.json").write_text(json.dumps(v, indent=2, ensure_ascii=False))
    print(json.dumps(v, indent=2, ensure_ascii=False), flush=True)


def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = pd.read_csv(OUT / "d2_stage_counts.csv", index_col=[0, 1])
    tasks = sorted({t for t, _ in piv.index})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, tag in zip(axes, ("diag", "final")):
        bot = np.zeros(len(tasks))
        for s in STAGES:
            vals = np.array([piv.loc[(t, tag), s] if (t, tag) in piv.index else 0
                             for t in tasks], dtype=float)
            ax.bar(tasks, vals, bottom=bot, label=s)
            bot += vals
        ax.set_title(tag); ax.set_xlabel("task"); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("failed episodes"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "d2_stage_stack.png", dpi=140); plt.close(fig)
    print("[D2] 그림 저장", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
