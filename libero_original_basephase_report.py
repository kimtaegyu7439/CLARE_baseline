#!/usr/bin/env python
"""우리가 학습한 libero_90 base phase 위에서 잰 원본 셋팅 SR —
results/libero_original_basephase_v3_SR_matrix.txt.

무엇이 다른가
    지금까지의 L2_codebook 실행은 전부 저장소가 주던 사전학습 체크포인트
    (Models/dit_flow_mt_libero_90_pretrain 또는 aicp_clare_pretrain)에서 출발했다.
    이 표는 **우리가 직접 200,000 step 학습한** Models/Lotus_clare_libero90_pretrain
    을 base 로 쓴 실행만 모은다. 팔·프로토콜은 기존과 같으므로, 차이가 나면 그것은
    **base phase 차이**다.

    셋팅은 LOTUS 가 아니라 **원본**이다 — task 0 부터 하나씩 순차로 배우고
    태스크가 끝날 때마다 코드북을 만든다(base joint 없음).

지표는 LIBERO 벤치마크(Liu et al. 2023) 정의로 L2_SR_matrix.txt 와 같다.

    FWT_t = R[t][t]                        갓 배운 직후 (= 대각)
    NBT_t = mean_{k>t} (R[t][t] − R[k][t])   배운 뒤 잃은 양 (+ 면 망각)
    AUC_t = mean_{k>=t} R[k][t]
    AvgSR = 마지막 **완성된** 행의 평균

    python libero_original_basephase_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "libero_original_basephase_v3_SR_matrix.txt"
K = 10

# (표시명, 결과 디렉터리, base phase, 설명)
ARMS = [
    ("v3+grad", "l2cb_v3_libero90base_spatial",
     "우리 libero_90 (Lotus_clare_libero90_pretrain, 200K step)",
     "l2_codebook v3 — 셀별 선형 기울기 A_k, 대각 p(s|j), 커널 가중"),
]
# 같은 팔·같은 프로토콜인데 base 만 다른 기존 실행. 대조로 함께 싣는다.
REF = [
    ("v3+grad", "l2_codebook_k96_grad",
     "저장소 사전학습 (dit_flow_mt_libero_90_pretrain)",
     "동일 팔. 이 표의 비교 기준이다."),
]


def read_csv(d: Path) -> dict[tuple[int, int], float]:
    p = d / "sr_matrix.csv"
    if not p.exists():
        return {}
    c = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if not f[0].strip().isdigit():
            continue
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip():
                c[(k, t)] = float(v)
    return c


def last_full_row(c) -> int | None:
    best = None
    for k in range(K):
        if all((k, t) in c for t in range(k + 1)):
            best = k
    return best


def metrics(c, kmax):
    fwt, nbt, auc = [], [], []
    for t in range(kmax + 1):
        col = [c[(k, t)] for k in range(t, kmax + 1) if (k, t) in c]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    last = [c[(kmax, t)] for t in range(kmax + 1)]
    mean = lambda v: (sum(v) / len(v)) if v else None            # noqa: E731
    return {"avg": mean(last), "fwt": mean(fwt), "nbt": mean(nbt), "auc": mean(auc),
            "bwt": (sum(c[(kmax, t)] - c[(t, t)] for t in range(kmax)) / kmax
                    if kmax else None)}


def block(name, d, base, desc, c) -> list[str]:
    kmax = last_full_row(c)
    filled = len(c)
    if kmax is None:
        return [f"[{name}] 아직 완성된 행이 없다 ({d}, {filled}/55칸)", ""]
    m = metrics(c, kmax)
    done = kmax == K - 1
    L = ["-" * 84,
         f"{name}   ({d})   "
         + ("완료 55/55칸" if done else f"진행 중 {filled}/55칸 (행 {kmax} 까지 완성)"),
         f"  base   {base}",
         f"  팔     {desc}", "-" * 84,
         f"{'after':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(K))]
    for k in range(K):
        L.append(f"{k:<8}" + "".join(
            (f"{c[(k,t)]:6.0f}" if (k, t) in c else f"{'':>6}") for t in range(K)))
    f1 = lambda v: "-" if v is None else f"{v:.1f}"              # noqa: E731
    fs = lambda v: "-" if v is None else f"{v:+.1f}"             # noqa: E731
    L.append(f"  지표  AvgSR {f1(m['avg'])}" + ("" if done else f" (행{kmax})")
             + f"   FWT↑ {f1(m['fwt'])}   NBT↓ {fs(m['nbt'])}"
             + f"   AUC↑ {f1(m['auc'])}   BWT↑ {fs(m['bwt'])}")
    cfg = d if isinstance(d, Path) else RES / d
    p = cfg / "l2_codebook_config.json"
    if p.exists():
        j = json.loads(p.read_text())
        L.append(f"  설정  seed={j.get('seed', 42)}  codebook_k={j.get('codebook_k')}  "
                 f"n_pairs={j.get('n_pairs')}  grad_enable={j.get('grad_enable')}  "
                 f"lambda_level={j.get('lambda_level')}")
    L.append("")
    return L


def main() -> None:
    L = ["=" * 84,
         "원본 셋팅 SR — 우리가 학습한 libero_90 base phase 위에서",
         "=" * 84, "",
         "셋팅   libero_spatial 10 task 를 **task 0 부터 순차로** 학습(LOTUS 셋팅 아님).",
         "       태스크가 끝날 때마다 (s,o) 코드북을 만들고 다음 태스크의 앵커로 쓴다.",
         "프로토콜  5000 step/task · 45 에피소드(뒤 5개 hold-out) · seed 42 ·",
         "          칸당 20 rollout · 하삼각 55칸 · 과거 원시 데이터 미사용", "",
         "이 표의 목적은 **base phase 를 바꾸면 무엇이 달라지는가**다.",
         "팔과 프로토콜을 고정하고 출발 체크포인트만 갈아 끼웠다.", "",
         "지표 (LIBERO 벤치마크 Liu et al. 2023 — L2_SR_matrix.txt 와 같은 정의)",
         "  AvgSR = 마지막 완성 행의 평균                    높을수록 좋음",
         "  FWT   = mean_t R[t][t]                          높을수록 좋음",
         "  NBT   = mean_t mean_{k>t} (R[t][t] − R[k][t])   **낮을수록** 좋음, + 면 망각",
         "  AUC   = mean_t mean_{k>=t} R[k][t]              높을수록 좋음",
         "  BWT   = mean_t (R[마지막][t] − R[t][t])         높을수록 좋음", ""]

    rows = []
    for name, d, base, desc in ARMS:
        c = read_csv(RES / d)
        L += block(name, d, base, desc, c)
        if c and last_full_row(c) is not None:
            rows.append((name + "  (우리 base)", c))

    L += ["", "=" * 84, "대조 — 같은 팔, 저장소 사전학습 base", "=" * 84, ""]
    for name, d, base, desc in REF:
        c = read_csv(RES / d)
        L += block(name, d, base, desc, c)
        if c and last_full_row(c) is not None:
            rows.append((name + "  (저장소 base)", c))

    if len(rows) >= 1:
        L += ["-" * 84, "base phase 대조", "-" * 84,
              f"{'base':<22}{'AvgSR':>8}{'FWT↑':>8}{'NBT↓':>8}{'AUC↑':>8}{'BWT↑':>8}  상태"]
        for nm, c in rows:
            kmax = last_full_row(c)
            m = metrics(c, kmax)
            f1 = lambda v: f"{'-':>8}" if v is None else f"{v:8.1f}"     # noqa: E731
            fs = lambda v: f"{'-':>8}" if v is None else f"{v:+8.1f}"    # noqa: E731
            L.append(f"{nm:<22}" + f1(m["avg"]) + f1(m["fwt"]) + fs(m["nbt"])
                     + f1(m["auc"]) + fs(m["bwt"])
                     + "  " + ("완료" if kmax == K - 1 else f"행{kmax}까지"))
        L.append("")

    L += ["읽을 때 주의",
          "  · 칸당 20 rollout 이라 개별 칸의 이항 표준오차가 최대 ±11%p, 10칸 평균도",
          "    ±3.5%p 다. 시드 하나(42)뿐이라 수 %p 차이는 확정할 수 없다.",
          "  · 진행 중인 팔의 AvgSR 은 **마지막 완성 행** 기준이라 최종값이 아니다.",
          "    태스크가 더 붙으면 대체로 내려간다.",
          "  · 두 base 는 데이터셋이 다르다. 우리 것은 yzembodied/libero_90_image",
          "    (4,500 ep, 669K frame, 9.6 epoch), 저장소 것은 출처가 다른 사전학습이다.",
          "    base phase 학습 조건까지 같지는 않으므로 '데이터가 좋아서'와",
          "    '학습을 더 해서'를 이 표만으로 가를 수는 없다.", ""]

    OUT.write_text("\n".join(L) + "\n")
    print(f"saved -> {OUT}")
    for nm, c in rows:
        kmax = last_full_row(c)
        m = metrics(c, kmax)
        f = lambda v: "-" if v is None else f"{v:.1f}"                   # noqa: E731
        print(f"  {nm:<22} AvgSR {f(m['avg']):>5}  FWT {f(m['fwt']):>5}  "
              f"NBT {f(m['nbt']):>5}  AUC {f(m['auc']):>5}  "
              + ("완료" if kmax == K - 1 else f"행{kmax}까지"))


if __name__ == "__main__":
    main()
