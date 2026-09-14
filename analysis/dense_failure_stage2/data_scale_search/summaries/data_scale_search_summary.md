# Stage-2 data-scale corrective-search summary

Contract: `d85b5e9b45dbc51ff9cec40268375624c26807e297cdea94c89d03c9068c4d94`. Candidate selection was frozen before dense outcomes and contains no legacy UID or SHA-256 image-group overlap.

1. **New candidates processed:** 4,000 completed dense/Stage-1 rows from 4,000 frozen candidates; 0 dense skips are retained in the bound dense audit.
2. **New triggered Dense-W:** 257.
3. **SINGLE_FIXABLE:** 75 bases and 950 exact-replay routes.
4. **Additional MCTS_ONLY_FIXABLE:** 33 bases and 66 retained exact-replay routes.
5. **UNRESOLVED:** 149; this means unresolved under the frozen bounded search, not unfixable.
6. **New triggered Dense-C preservation:** 1,691 bases/routes.
7. **Expanded corpora:** A=1,730 routes/1,730 bases; B=8,578 routes/773 bases; C=508 routes/242 bases.
8. **Old label-yield pattern:** reproduced within the prospectively frozen ±0.10; new-minus-old bounded fixability is -0.0620. This is a canonical-source expansion, not a pure scale-only replication.
9. **GQA supervision:** 44 new bounded-fixable GQA bases; material by the prospective ≥25 new-base criterion (old GQA bounded-fixable bases: 85).
10. **Trigger-depth variation:** new total bounded-fixability rates span 0.4670; exact counts/costs are in `metrics/trigger_depth_breakdown.csv`.
11. **Single-search compute:** 12,225 exhaustive single terminal routes.
12. **Additional MCTS compute:** 31,923 iterations and 25,197 physical terminal routes; total search sample time was 5.477 GPU-hours.
13. **Integrity:** PASS. All 2,707 retained new routes are current-LMMS-correct exact-token replays; all 1,799 new state shards are contract/model/code/schema/source-bound. Old Phase-56 inputs passed their frozen artifact-manifest verification. No validation/test sample was searched.

No Stage-2 model was trained, Stage 1 was not retuned, and no validation/test/external evaluation was run.
