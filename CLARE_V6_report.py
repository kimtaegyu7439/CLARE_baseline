#!/usr/bin/env python
"""L2_V6 세 팔의 SR -> results/direction error aware anchor_V6_SR.txt

    python CLARE_V6_report.py

읽는 것  results/clare_v6_<form>_goal_seed7/sr_matrix.csv   (form = dea|sqerr|unifmd)
         results/clare_v5_goal_c0.1_n50_seed7               (대조: 기존 V5 최고 팔)
         results/clare_mar_goal_median_seed7                (대조: V4 median)

행렬·지표 정의는 clare_v4_report 와 공유한다.
"""
from __future__ import annotations

from pathlib import Path

from clare_v4_report import metrics, read_matrix

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "direction error aware anchor_V6_SR.txt"
W = 94

# (폴더 태그, 표시 이름, e_j 계수, 설명)
ARMS = [
    ("clare_v6_sqerr_goal_seed7",  "V6 sqerr",  "2·r_s·r̄ / N",
     "균일 제곱오차. ‖v_s−v̄‖² 와 항등 — 기존 MSE 앵커와 같은 양이다."),
    ("clare_v6_unifmd_goal_seed7", "V6 unifmd", "β / N",
     "균일 크기-방향. 방향 감독을 크기에서 떼어내 모든 과거에 똑같이 준다."),
    ("clare_v6_dea_goal_seed7",    "V6 dea",    "β·w_j (detach)",
     "방향오차 인지. w_j ∝ e_j + ε_w, 태스크 축에서 합 1. 제안 방식."),
    ("clare_v6_dea_eps1e-5_goal_seed7", "V6 dea ε1e-5", "β·w_j (detach)",
     "같은 dea 인데 ε_w 만 1e-3 -> 1e-5. 적응 비율 ρ=ē/(ē+ε_w) 가 0.7 -> 0.99 로 오른다."),
    # ── V7: 분해를 **액션 공간**에서 (a = x_t + (1−t)v). m 과 e 둘 다 ──────────
    ("clare_v7_unifmd_goal_seed7", "V7 unifmd", "β / N",
     "액션 공간. 공통 노이즈 ε 가 빠져 각도 압축이 풀린다(전체 2.2 배)."),
    ("clare_v7_dea_goal_seed7",    "V7 dea",    "β·w_j (detach)",
     "액션 공간 + 방향오차 인지 재배분."),
    # ── V8: OMP 의 Directional Alignment L_DA 를 붙인 2 팔 (속도 각도) ───────
    ("clare_v8_mse_da_goal_seed7", "V8 mse+DA", "MSE + λ·L_DA",
     "‖v_s−v_T‖² 를 그대로 두고 L_DA 를 더한다. = V6 sqerr + 방향 항."),
    ("clare_v8_mag_da_goal_seed7", "V8 mag+DA", "m + λ·L_DA",
     "크기 (r_s−r_T)² 만 남기고 방향 감독을 전부 L_DA 에 맡긴다."),
    # ── V9: V8 과 같은 손실을 **액션 공간**에서 (V6->V7 과 같은 관계) ─────────
    ("clare_v9_mse_da_goal_seed7", "V9 mse+DA", "MSE + λ·L_DA (act)",
     "V8 mse+DA 를 액션 공간으로. 유지 항은 속도 버전의 (1−t)² 배다."),
    ("clare_v9_mag_da_goal_seed7", "V9 mag+DA", "m + λ·L_DA (act)",
     "V8 mag+DA 를 액션 공간으로. 크기 항은 (1−t)² 비례가 아니라 다른 양이다."),
    # ── V7 dea β 스윕 (액션 공간). 기존 β=2.0 아래쪽을 훑는다 ─────────────────
    ("clare_v7_dea_b0.1_goal_seed7", "V7 dea β0.1", "β·w_j  (act)",
     "방향 감독을 거의 끈 쪽. 실측 β=2 일 때 dea 방향 항이 sqerr 의 12 배였다."),
    ("clare_v7_dea_b0.5_goal_seed7", "V7 dea β0.5", "β·w_j  (act)", ""),
    ("clare_v7_dea_b1.0_goal_seed7", "V7 dea β1.0", "β·w_j  (act)", ""),
    ("clare_v7_dea_b1.5_goal_seed7", "V7 dea β1.5", "β·w_j  (act)", ""),
    ("clare_v7_dea_b2.5_goal_seed7", "V7 dea β2.5", "β·w_j  (act)",
     "β=2.0 위쪽. 2.0 에서 1.5 로 내릴 때 5.4%p 가 꺾여 문턱을 확인한다."),
    ("clare_v7_dea_b3.0_goal_seed7", "V7 dea β3.0", "β·w_j  (act)", ""),
    ("clare_v7_dea_b3.5_goal_seed7", "V7 dea β3.5", "β·w_j  (act)",
     "β 를 3.0 위로 더 밀어본다. 액션 공간 보정값 2r₀²=1.35 보다 한참 크지만\n      seed7 에서 β3.0 이 AvgSR 91.2 로 가장 좋았어서 더 가도 되는지 본다."),
    ("clare_v6_dea_b0.5_goal_seed7", "V6 dea β0.5", "β·w_j  (vel)", ""),
    ("clare_v6_dea_b1.0_goal_seed7", "V6 dea β1.0", "β·w_j  (vel)", ""),
    ("clare_v6_dea_b1.5_goal_seed7", "V6 dea β1.5", "β·w_j  (vel)", ""),
    ("clare_v6_dea_b2.5_goal_seed7", "V6 dea β2.5", "β·w_j  (vel)",
     "속도 공간 β 스윕. 액션 공간(V7)과 짝을 이뤄 공간 효과와 세기 효과를 가른다."),
    ("clare_v6_dea_b3.0_goal_seed7", "V6 dea β3.0", "β·w_j  (vel)", ""),
    # ── V10: DEA 가중을 **크기 항에도** (액션 공간) ──────────────────────────
    ("clare_v10_dea_goal_seed7", "V10 dea m+e", "w^m_j / β·w^e_j (act)",
     "크기 항 계수를 1/N -> w^m_j ∝ m_j+ε_w 로. 방향과 같은 꼴, 통계만 m 이다."),
]
def _discover_beta_arms(arms):
    """results/ 에 있는 v6/v7 dea β 런 중 ARMS 에 없는 것을 자동으로 붙인다.

    전에는 β 를 하나 추가할 때마다 이 목록에 손으로 넣어야 했고, 서버가
    여러 대면 서버마다 넣어야 했다. 폴더 이름 규칙이 고정이므로
    (clare_v{6,7}_dea_b<β>_goal_seed<N>) 긁어서 붙이는 게 맞다.
    β 순으로 정렬해 넣는다 — 커브를 읽을 때 순서가 중요하다.
    """
    import re as _re
    known = {a[0] for a in arms}
    pat = _re.compile(r"^clare_(v6|v7)_dea_b([0-9.]+)_goal_seed(\d+)$")
    found = []
    for d in sorted(RES.glob("clare_v[67]_dea_b*_goal_seed*")):
        if not d.is_dir() or d.name in known:
            continue
        m = pat.match(d.name)
        if not m:
            continue                      # _b24bs 같은 배치 변형은 여기 안 넣는다
        ver, beta, seed = m.group(1), float(m.group(2)), m.group(3)
        space = "act" if ver == "v7" else "vel"
        found.append((beta, (d.name,
                             f"{ver.upper()} dea \u03b2{m.group(2)}",
                             f"\u03b2\u00b7w_j  ({space})",
                             f"\uc790\ub3d9 \ud0d0\uc9c0 \u2014 seed{seed}")))
    return arms + [a for _, a in sorted(found, key=lambda x: x[0])]


REFS = [
    ("clare_v5_goal_c0.1_n50_seed7", "참고: V5 c=0.1",  "속도 역제곱 가중, 정규화 없음"),
    ("clare_mar_goal_median_seed7",  "참고: V4 median", "속도 역제곱 가중, 배치평균 1 정규화"),
]


def hyper(tag: str) -> str:
    """런 폴더의 config 에서 β(와 λ_DA)를 읽어 표기용 문자열로.

    표시 이름에 적어두면 오타가 나거나 실제 설정과 어긋날 수 있다. 기록된
    config 를 그대로 읽어야 표와 런이 반드시 일치한다.
    """
    import json
    d = RES / tag
    for f in ("v6_config.json", "v8_config.json"):
        p = d / f
        if not p.exists():
            continue
        try:
            c = json.loads(p.read_text())
        except Exception:
            continue
        if "lambda_da" in c:
            return f"λ_DA {c['lambda_da']:g}"
        if c.get("beta") is not None:
            return f"β {c['beta']:g}"
    # V4/V5 참고 행은 β 가 아니라 분모 상수 c 를 쓴다. n50 병합 폴더는
    # l2_config.json 이 없고 config.json 만 있으므로 둘 다 본다.
    for f in ("l2_config.json", "config.json"):
        p = d / f
        if not p.exists():
            continue
        try:
            c = json.loads(p.read_text())
        except Exception:
            continue
        if c.get("anchor_c"):
            return f"c {c['anchor_c']:g}"
    # 폴더 이름에 c 가 박혀 있으면 거기서 (median 버전은 c 가 태스크마다 다르다)
    if "_c" in tag:
        return "c " + tag.split("_c")[1].split("_")[0]
    if "median" in tag:
        return "c med"
    return "-"


def block(tag: str, name: str, note: str, L: list, rows: list) -> None:
    d = RES / tag
    R, n = read_matrix(d)
    L.append("-" * W)
    L.append(f"{name}   [{hyper(tag)}]   ({tag})")
    if note:
        L.append(f"   {note}")
    L.append("-" * W)
    if not R:
        L.append("  아직 SR 칸이 없다"); L.append("")
        rows.append((name, None, "미시작", hyper(tag))); return
    m = metrics(R, n)
    filled = sum(1 for r in R for v in r if v is not None)
    total = n * (n + 1) // 2
    done = filled >= total
    L.append("  " + f"{'stage':<7}" + "".join(f"{'t'+str(t):>6}" for t in range(n)))
    for k in range(n):
        L.append("  " + f"{k:<7}" + "".join(
            (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}") for t in range(n)))
    f1 = lambda v: "-" if v is None else f"{v:.1f}"        # noqa: E731
    fs = lambda v: "-" if v is None else f"{v:+.1f}"       # noqa: E731
    L.append(f"  지표  AvgSR {f1(m['AvgSR'])}"
             + (f" (행{m['row']})" if m["row"] is not None and not done else "")
             + f"   FWT↑ {f1(m['FWT'])}   NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}"
             + f"   [{'완주' if done else f'진행 중 {filled}/{total}칸'}]")
    L.append("")
    rows.append((name, m, "완주" if done else f"{filled}/{total}칸", hyper(tag)))


def main() -> None:
    L: list[str] = []
    add = L.append
    add("=" * W)
    add("방향 감독 앵커 비교 (L2_V6 / V7 / V8 / V9) — CLARE 셋팅 · libero_goal")
    add("=" * W)
    add("")
    add("손실  속도를 크기와 방향으로 쪼개 따로 감독한다.")
    add("")
    add("      v_θ = r_θ·u_θ ,  v̄ = r̄·ū")
    add("      m_j = (r_θ,j − r̄_j)²            크기 오차")
    add("      e_j = 1 − u_θ,j·ū_j ∈ [0,2]     방향 오차")
    add("")
    add("      L_anchor = (1/H) Σ_h [ (1/N) Σ_j m_j + Σ_j (계수_j)·e_j ]")
    add("      L        = L_FM + λ · L_anchor")
    add("")
    add("      V6/V7 네 팔은 **e_j 의 계수만** 다르다. V8 은 방향 항 자체가 다르다:")
    for _tag, name, coef, desc in _discover_beta_arms(ARMS):
        add(f"        {name:<12}{coef:<18}{desc}")
    add("")
    add("      w_j = sg[ (e_j + ε_w) / (Σ_m e_m + N·ε_w) ],  Σ_j w_j = 1")
    add("")
    add("V8 (OMP 의 Directional Alignment)  방향 항을 e=1−cos 대신 로그 형태로 쓴다.")
    add("")
    add("      L_DA = −log( (cos α + 1) / 2 ) ,  cos α = (v_T·v_s)/((‖v_T‖+ε_dir)·‖v_s‖)")
    add("")
    add("      mse_da   ‖v_s − v_T‖²        + λ_DA·L_DA     (= V6 sqerr + 방향 항)")
    add("      mag_da   (‖v_s‖₂ − ‖v_T‖₂)²  + λ_DA·L_DA     (크기만 + 방향은 L_DA)")
    add("")
    add("      왜 로그인가. 크기 ρ 의 두 벡터에서 ∂‖ρu−ρ*u*‖²/∂α = 2ρρ*·sinα 라")
    add("      **ρ→0 이면 방향 기울기가 같이 죽는다**. 실측: ρ=1 에서 1.29, ρ=1e-4")
    add("      에서 1.3e-8 (ρ² 로). 반면 ∂L_DA/∂α = sinα/(cosα+1) 은 ρ 와 무관해")
    add("      같은 조건에서 0.365 로 일정하다. LIBERO 는 파지 전환 ±8 스텝에서")
    add("      ‖a[0:6]‖ 이 43~48% 떨어지므로, 보호가 필요한 구간이 바로 ρ 가 작은")
    add("      구간이다. α→π 에서 발산해 정반대 예측에 큰 벌점도 준다.")
    add("      ★ V8 의 각도는 **속도** v 로 잰다(V7 의 액션 공간이 아니다).")
    add("")
    add("      V9 는 V8 과 같은 손실을 **액션 공간**에서 낸다 (V6->V7 과 같은 관계).")
    add("      대수적으로 확인된 것:")
    add("        · a_s − a_T = (1−t)(v_s − v_T)  — 공통 x_j 가 빠진다. 따라서")
    add("          mse_da 의 유지 항은 속도 버전의 **(1−t)² 배**다. 새로운 양이")
    add("          아니라 t 의존 재가중이다.")
    add("        · 크기 항은 비례하지 않는다 — a = x_j + (1−t)v 가 공통 오프셋")
    add("          x_j 를 더하므로 ‖a_s‖−‖a_T‖ ≠ (1−t)(‖v_s‖−‖v_T‖) 다")
    add("          (측정: m_act/(m_vel·(1−t)²) 중앙 1.48).")
    add("        · 항등식 ‖·‖² = m + 2·r·r·e 는 액션 공간에서도 성립한다(오차 9e-15).")
    add("      즉 V9 가 새로 보는 것은 **방향·크기의 분해 방식**이고, mag_da 쪽이")
    add("      V8 과 실질적으로 다른 팔이다.")
    add("")
    add("β 스윕 (V7 dea, 액션 공간)  방향 항의 세기만 바꾼다. 실측(stage 9, 3200 프레임,")
    add("      노트의 계수표 그대로 1/N 포함):")
    add("")
    add("        공간      sqerr 방향   unifmd(β=2)   dea(β=2)    L_DA(λ=1)")
    add("        속도       .00019      .00005       .00009      .00001")
    add("        액션       .00003      .00011       .00037      .00003")
    add("")
    add("      즉 액션 공간에서 dea β=2 의 방향 항은 sqerr 의 **12 배**다. β 를 0.1~1.5")
    add("      로 내리면 0.6~9 배 구간을 훑는다. 기존 V7 dea 는 β=2.0 이다.")
    add("      ★ Σ_j w_j = 1 이어도 dea 가 unifmd 보다 크다 — 손실에 들어가는 건")
    add("        E[w_j·e_j] = E[e_j]/N + Cov(w_j, e_j) 이고 w 는 e 와 양의 상관이라")
    add("        공분산 항이 더해진다(속도 1.8 배, 액션 3.4 배). 재배분만이 아니라")
    add("        **총량도 늘린다**.")
    add("")
    add("분해 공간  V6 는 **속도** v 로, V7 은 **액션** a = x_t + (1−t)·v 로 m·e 를 낸다.")
    add("      선형 경로에서 v = a − ε 이고 student·teacher 가 같은 x_t 를 쓰므로 ε 를")
    add("      공유한다. 차이 벡터는 상쇄되지만(v_s−v_t = a_s−a_t) **각도는 상쇄되지")
    add("      않는다** — ‖ε‖/‖a‖ ≈ 2.6 이라 두 속도가 둘 다 −ε 쪽을 가리킨다.")
    add("      실측(stage 9): e_act/e_vel 이 t<0.2 에서 14 배, t>0.8 에서 0.2 배,")
    add("      전체 2.2 배. 즉 두 공간이 경로의 서로 다른 구간을 강조한다.")
    add("      ★ m 과 e 는 항상 같은 공간이다. 섞으면 ‖·‖²=m+2·r·r·e 가 깨진다.")
    add("")
    add("      β 보정(노트: β≈2r₀²)도 공간마다 다르다 — 속도 r₀=2.49 -> 12.4,")
    add("      액션 r₀=0.82 -> 1.35. 전 팔이 β=2.0 이므로 V6 는 보정값의 1/6,")
    add("      V7 은 1.35 에 가깝다. V6 의 unifmd/dea 는 방향 감독이 약한 설정이었다.")
    add("")
    add("프로토콜  태스크당 10,000 step, 50 demo, 10 task = 55칸, 코드북 K=96,")
    add("          칸당 **50 rollout** (eval seed 7~56), train seed 7, β=2.0, ε_w=1e-3")
    add("          과거 태스크 평균은 1/N (노트 정의). 기존 V4/V5 는 합이었다.")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("      AvgSR = 마지막 **완성된** 행의 평균")
    add("")

    rows: list = []
    arms = _discover_beta_arms(ARMS)
    for tag, name, _coef, desc in arms:
        block(tag, name, desc, L, rows)
    for tag, name, desc in REFS:
        block(tag, name, desc, L, rows)

    add("=" * W)
    add("대조")
    add("=" * W)
    add(f"  {'팔':<16}{'세기':>10}{'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}{'AUC↑':>9}   상태")
    for name, m, state, hp in rows:
        if m is None:
            add(f"  {name:<16}{hp:>10}" + "".join(f"{'-':>9}" for _ in range(4))
                + f"   {state}")
            continue
        f1 = lambda v: f"{'-':>9}" if v is None else f"{v:9.1f}"      # noqa: E731
        fs = lambda v: f"{'-':>9}" if v is None else f"{v:+9.1f}"     # noqa: E731
        add(f"  {name:<16}{hp:>10}" + f1(m["AvgSR"]) + f1(m["FWT"])
            + fs(m["NBT"]) + f1(m["AUC"]) + f"   {state}")
    add("")
    add("읽을 때 주의")
    add("  · 시드가 1 개(7)다. 같은 셋팅의 V5 측정에서 시드 간 AvgSR 이 1.2~1.6%p,")
    add("    NBT 가 최대 3.7%p 흔들렸다. 그보다 작은 차이는 효과로 읽으면 안 된다.")
    add("  · sqerr 는 수학적으로 MSE 앵커와 같은 양이다. 따라서 V6 안에서 sqerr 가")
    add("    기준선이고, unifmd/dea 가 그보다 나아야 '분리'와 '재배분'이 값을 한 것이다.")
    add("  · V4/V5 참고 행은 **가중 방식이 전혀 다르다**(속도 역제곱). 그리고 과거")
    add("    태스크 합산이 V6 는 1/N 평균, V4/V5 는 합이다 — 실효 λ 가 다르므로")
    add("    직접 비교보다 '같은 프로토콜의 다른 계열' 정도로 읽어라.")
    add("")

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
