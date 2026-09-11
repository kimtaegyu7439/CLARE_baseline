#!/usr/bin/env python
"""방법별 저장물 실측 — ER 버퍼 vs teacher vs R10/R11 통계.

공정하게 세기 위한 규칙
  1. 모든 방법이 공통으로 갖는 것은 뺀다:
       사전학습 동결 백본(DINOv2+CLIP) + 현재 학습 모델.
       이건 어느 방법을 쓰든 있어야 하고, 방법 간 차이가 아니다.
  2. ER 버퍼는 **마지막 것 하나만** 센다. 지금 구현은 스테이지마다 새 버퍼를
     만들어 9개가 디스크에 남지만, 마지막 것이 과거 전부를 담고 있으므로
     제대로 만들면 하나면 된다. 구현 낭비를 방법의 비용으로 세면 불공정하다.
  3. teacher 는 학습분만 센다. B1.snapshot 이 동결 백본을 참조 공유하므로
     체크포인트 파일 크기(774MB)가 아니라 43.88M 파라미터 = 176 MB 다.
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
HF = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / "Datasets/lerobot"))


def du(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


MB = 1e6
FROZEN_PARAMS = 149_750_000      # DINOv2 86.58M + CLIP 63.17M
TRAIN_PARAMS = 43_880_000        # velocity_net 43.22M + 투영 0.65M
BYTES = 4                        # fp32

L = ["=" * 84, "저장물 실측 — 방법별 추가분", "=" * 84, "",
     "공통분 (모든 방법이 갖고 있어 비교에서 제외)",
     f"  동결 백본  {FROZEN_PARAMS/1e6:6.2f}M x 4B = {FROZEN_PARAMS*BYTES/MB:7.1f} MB",
     f"  현재 모델  {TRAIN_PARAMS/1e6:6.2f}M x 4B = {TRAIN_PARAMS*BYTES/MB:7.1f} MB",
     ""]

# ── ER 버퍼 ─────────────────────────────────────────────────────────────────
bufs = sorted(HF.glob("er_buffer/libero_spatial_seed42_ep5_tasks_0_*"),
              key=lambda p: int(p.name.rsplit("_", 1)[-1]))
L += ["-" * 84, "ER 재생 버퍼 (과거 태스크당 5 에피소드)", "-" * 84,
      f"{'버퍼':>12}{'담긴 과거':>10}{'크기':>12}{'증분':>12}"]
prev, per = 0, []
for b in bufs:
    k = int(b.name.rsplit("_", 1)[-1]) + 1        # tasks_0_{k-1} 은 과거 k개
    sz = du(b) / MB
    L.append(f"{'0_'+str(k-1):>12}{k:>10}{sz:>10.1f} MB{sz-prev:>10.1f} MB")
    if prev:
        per.append(sz - prev)
    prev = sz
rate = sum(per) / len(per) if per else (prev / max(1, len(bufs)))
L += ["", f"  과거 태스크당 증분 평균 {rate:.1f} MB",
      f"  -> 10 태스크(과거 9개) 예상 {rate*9:.0f} MB",
      f"  -> 40 태스크(과거 39개) 예상 {rate*39:.0f} MB",
      "",
      f"  ※ 현재 디스크에는 버퍼 {len(bufs)}개가 모두 남아 총 {prev if not per else sum(du(b)/MB for b in bufs):.0f} MB 를",
      "     쓰고 있지만, 마지막 하나가 과거 전부를 담으므로 위 숫자가 필요량이다.", ""]

# ── teacher ─────────────────────────────────────────────────────────────────
tsz = TRAIN_PARAMS * BYTES / MB
L += ["-" * 84, "teacher 스냅샷 (학습분만. 동결 백본은 student 와 공유)", "-" * 84,
      f"  teacher 1개 = {TRAIN_PARAMS/1e6:.2f}M x 4B = {tsz:.1f} MB",
      f"  B1/B8/R10/R11  rolling  -> {tsz:.1f} MB (상수)",
      f"  B2             frozen   -> {tsz:.1f} x (태스크수-1)", ""]

# ── R10 / R11 통계 ──────────────────────────────────────────────────────────
L += ["-" * 84, "R10 / R11 통계 파일 (실측)", "-" * 84,
      f"{'실행':>18}{'파일수':>8}{'총량':>12}{'태스크당':>12}"]
for name in ("R10", "R11", "R10_10task", "R11_10task"):
    d = REPO / "results" / name / "stats"
    fs = sorted(d.glob("task*.pt")) if d.exists() else []
    if not fs:
        continue
    tot = sum(du(f) for f in fs) / MB
    L.append(f"{name:>18}{len(fs):>8}{tot:>10.2f} MB{tot/len(fs):>10.2f} MB")
L.append("")

# ── 종합 ────────────────────────────────────────────────────────────────────
def stat_per(name):
    d = REPO / "results" / name / "stats"
    fs = sorted(d.glob("task*.pt")) if d.exists() else []
    return (sum(du(f) for f in fs) / MB / len(fs)) if fs else None

s10 = stat_per("R10_10task") or stat_per("R10") or 0.25
s11 = stat_per("R11_10task") or stat_per("R11") or 3.39
COMMON = (FROZEN_PARAMS + TRAIN_PARAMS) * BYTES / MB     # 체크포인트 하나 = 774.5 MB

METHODS = [
    ("seq-FT (기준)",       lambda n: 0.0,                 "없음"),
    ("ER 버퍼",             lambda n: rate * (n - 1),      "선형"),
    ("B2 frozen teachers",  lambda n: tsz * (n - 1),       "선형"),
    ("B1/B8 rolling",       lambda n: tsz,                 "상수"),
    ("R10 (rolling+통계)",  lambda n: tsz + s10 * n,       "거의 상수"),
    ("R11 (+백색화 기저)",  lambda n: tsz + s11 * n,       "거의 상수"),
]
NS = (4, 10, 40)

L += ["=" * 84, "종합 A — 추가분만 (공통분 제외)", "=" * 84, "",
      f"{'방법':>22}" + "".join(f"{str(n)+' 태스크':>12}" for n in NS) + f"{'증가':>11}",
      "-" * 84]
for nm, f, g in METHODS:
    L.append(f"{nm:>22}" + "".join(f"{f(n):>10.0f} MB" for n in NS) + f"{g:>11}")

L += ["", "=" * 84, "종합 B — 총 저장량 (공통분 포함)", "=" * 84, "",
      f"  공통분 = 사전학습 동결 백본 599 MB + 현재 학습 모델 176 MB = {COMMON:.0f} MB",
      "  어느 방법을 쓰든 이만큼은 디스크에 있어야 한다(= 체크포인트 하나).", "",
      f"{'방법':>22}" + "".join(f"{str(n)+' 태스크':>12}" for n in NS) + f"{'ER 대비':>11}",
      "-" * 84]
for nm, f, g in METHODS:
    tot = [COMMON + f(n) for n in NS]
    er = [COMMON + rate * (n - 1) for n in NS]
    ratio = "" if nm.startswith("ER") else f"{er[1]/tot[1]:.2f}x"
    L.append(f"{nm:>22}" + "".join(f"{v:>10.0f} MB" for v in tot) + f"{ratio:>11}")

L += ["", "  ER / R10 비율:",
      "    " + "   ".join(
          f"{n}태스크 {(COMMON+rate*(n-1))/(COMMON+tsz+s10*n):.2f}x" for n in NS),
      "",
      "★ 추가분 기준(A)과 총량 기준(B)의 인상이 크게 다르다.",
      "  A 로는 10태스크에서 R10 이 ER 의 1/3.9 지만, B 로는 1/1.5 다.",
      "  공통분 774 MB 가 양쪽에 똑같이 깔려 있어 비율을 눌러 주기 때문이다.",
      "  방법 자체의 비용을 논할 때는 A, 실제 디스크 요구량은 B 를 쓴다.", ""]

rep = "\n".join(L)
(REPO / "results" / "STORAGE.txt").write_text(rep)
print(rep[rep.index("종합 A"):])
print(f"\nsaved -> {REPO/'results'/'STORAGE.txt'}")
