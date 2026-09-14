# Exhaustive full-benchmark regression/rescue audit

The frozen result is primarily **preservation-limited in pooled accounting**, with benchmark-specific treatment failure and inactivity. Stage-1 admits 496/4,380 Dense-W and 405/15,580 Dense-C. Stage-2 then uses at least one non-FULL action on 119/496 triggered W and 92/405 triggered C—similar conditional activation rates of 0.2399 and 0.2272. The decisive asymmetry is outcome: only 3/119 W interventions rescue, while 19/92 C interventions regress.

## Primary funnel

| Family | P(trigger|W) | P(trigger|C) | P(nonFULL|trig,W) | P(nonFULL|trig,C) | W→C | C→W | Net |
|---|---|---|---|---|---|---|---|
| chartqa | 0.1388 | 0.0270 | 0.2245 | 0.1552 | 1 | 3 | -2 |
| textvqa | 0.2612 | 0.0595 | 0.1398 | 0.2235 | 2 | 12 | -10 |
| mmmu_pro | 0.1168 | 0.0751 | 0.3142 | 0.2826 | 0 | 4 | -4 |
| pope | 0.0000 | 0.0000 | NA | NA | 0 | 0 | 0 |
| overall | 0.1132 | 0.0260 | 0.2399 | 0.2272 | 3 | 19 | -16 |

## Answers to the plan questions

1. Dense-W reaching Stage-1 trigger: ChartQA 49/353, TextVQA 186/712, MMMU-Pro 261/2,235, POPE 0/1,080; pooled 496/4,380.
2. Non-FULL among triggered W: ChartQA 11/49, TextVQA 26/186, MMMU-Pro 82/261, POPE 0/0; pooled 119/496.
3. Non-FULL among triggered C: ChartQA 9/58, TextVQA 57/255, MMMU-Pro 26/92, POPE 0/0; pooled 92/405.
4. Conditional W→C among non-FULL triggered W is 3/119 (0.0252) pooled.
5. Conditional C→W among non-FULL triggered C is 19/92 (0.2065) pooled.
6. The 19 regressions have first action WRITE_ONLY=13, IGNORE=6, READ_ONLY=0; 4 are single-intervention, 15 multiple, 2 immediate, and 17 delayed. WRITE_ONLY occurs in 13 regression traces and IGNORE dominates 9; no regression uses READ_ONLY. These patterns are exhaustive associations, not causal isolation.
7. The 3 rescues start with WRITE_ONLY=2 and IGNORE=1; all are delayed by 2–5 layers. Two are exact one-non-FULL trajectories, while the other uses WRITE_ONLY at three consecutive layers.
8. TextVQA dominates because Stage-2 acts on 57 triggered C but only 26 triggered W, then regresses 12/57 C versus rescuing 2/26 W. It contributes -10 of the -16 net.
9. MMMU-Pro is not merely admission-limited: 261 W trigger and 82 receive non-FULL, but 0 are rescued; 4/26 treated C regress. This is treatment-quality limited under the fixed rule.
10. POPE is completely inactive because no score crosses P90. Its maximum score is 0.828423, 0.077710 below threshold; Stage-2 performance on POPE is unobserved.
11. Pooled failure is preservation-limited by the fixed accounting rule, with a mixed family picture: ChartQA/TextVQA preservation-limited, MMMU-Pro treatment-quality-limited, and POPE inactive.
12. TextVQA contributes most: 12/19 regressions and -10/−16 net correction.
13. This audit does not prove any individual action caused a transition, justify lowering Stage-1 threshold, show POPE preservation, show dynamic routing is impossible, or validate a replacement method.

## Development-calibration context

The P90 development expectation was pooled W recall 0.3572; external pooled W admission is 0.1132. This calibration shift is real but is not the main current net-loss accounting: among W already admitted and treated, success is only 2.52%, while C treatment risk is 20.65%. Expanding admission without improving treatment/preservation is therefore not supported.
