# K10 게이트 — Langevin 표본이 manifold 에 접근하는가

판정  PASS = d̂_after ≤ 1.4 AND diversity ≥ 0.7   |   STRONG = d̂ ≤ 1.2 동일 조건
d̂ = med NN(b → 실제, 같은 bin) / 실제 LOO NN 중앙값.  bin {2,5,8} 평균.

기준선 (실제 프레임 대비)

| bin | d_real | 가우시안 d̂ | 가우시안 Ê_wit | 가우시안 Ê_U |
|---|---|---|---|---|
| 2 | 26.3 | 1.643 | 83.40 | 1.26 |
| 5 | 22.7 | 2.036 | 48.53 | 1.39 |
| 8 | 24.0 | 1.867 | 19.65 | 1.04 |

| arm | T | coords | d̂_before | d̂_after | diversity | Ê_wit b→a | Ê_U b→a | 판정 | s |
|---|---|---|---|---|---|---|---|---|---|
| U | T0 | collective | 1.451 | 1.274 | 0.648 | 20.90→118.31 | 1.10→1.01 | — | 31 |
| U | const | collective | 1.386 | 7.544 | 6.139 | 20.15→2469479.46 | 0.99→30.92 | — | 21 |
| prod | T0 | collective | 1.451 | 1.391 | 0.989 | 20.90→15.15 | 1.10→1.00 | PASS | 38 |
| wit | T0 | collective | 1.451 | 1.392 | 0.989 | 20.90→15.03 | 1.10→1.01 | PASS | 6 |

비정상 조합 19개: wit|const|collective|bin2(diverged), wit|const|collective|bin5(diverged), wit|const|collective|bin8(diverged), wit|anneal|collective|bin2(diverged), wit|anneal|collective|bin5(diverged), wit|anneal|collective|bin8(diverged), U|const|collective|bin5(diverged), U|anneal|collective|bin2(diverged), U|anneal|collective|bin5(diverged), U|anneal|collective|bin8(diverged), prod|const|collective|bin2(diverged), prod|const|collective|bin5(diverged)

## 핵심 대조

- **교집합 기여**: 최선 단독 ('U', 'T0') d̂ 1.274  vs  최선 prod ('prod', 'T0') d̂ 1.391  (+0.116)

**선택: arm=prod, T=T0, coords=collective**  (d̂ 1.391, diversity 0.989, PASS)
