# Stage-1 ALL-source robust threshold calibration protocol

- Candidate: frozen Phase-62 Shared Random-4 ALL-source five-fold system.
- Head identity: `672632831f556fe45a4312a831a819609797f83bdab987633cca3d1a098a2422`.
- Threshold calibration data: Historical validation (800) plus Canonical OOF (4,000).
- Historical test (800) is excluded from threshold selection and used only after freeze.
- Trigger rule: first layer 0-27 with strict `score > tau`; sample admission is the true any-layer rule.
- Sweep: every observed calibration trajectory-maximum breakpoint plus the all-trigger terminal point.
- Primary rule: maximize pooled W recall subject to Historical and Canonical C preservation both >=98%; use the plan's fixed tie-break order.
- References: most permissive thresholds satisfying 99%, 98%, and 95% worst-source C preservation.
- Useful-W criterion for Decision A: pooled W recall >= 5% and nonzero W detections in both calibration sources.
- Adequately supported catastrophic-cell check: N_C >= 30; flag C preservation <90%.
- Cross-fit stability is declared only when threshold IQR <= 0.05, range <= 0.10, the full threshold lies inside the fold range, and every held-out Canonical fold preserves >= 95% of C.
- Bootstrap: 5,000 source-stratified image-group cluster replicates; seed 20260903.
- Canonical TextVQA rates receive Wilson 95% intervals because only 19 W records exist.
- Old reference threshold: `0.92538392806010306`, applied to these same frozen ALL-head trajectories.
- No threshold is optimized for trigger depth or downstream Stage-2 outcomes.
