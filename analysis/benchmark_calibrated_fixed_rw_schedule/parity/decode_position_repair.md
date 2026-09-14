# Dense continuation repair
Supported cause: native first decode rotary position194 versus cached323 on Psychology128; all28 prefill layers and initial logits are exactly equal. Native E/EOS score0; old cached long answer score1; changing only continuation delta restores native E/EOS score0. Evidence: dense_scorer_mismatch_diagnostic.json and archived failed_attempt3_decode_position/logs.

Reviewer verdict stable, confidence high. Ranking: Phase88-only adapter and full Dense/q regeneration > stop inconclusive > relax scorer gate. Strongest objection: generation and q must use identical position convention. Reconciled: baseline metadata is shared by both paths; no shared historical executor change. Final-plane and unpadded assertions fail closed. Seven CAL variants and both Dense-only regression sentinels precede full Dense census.

All older partial Dense evidence is archived, not eligible for resumed analysis. Frozen splits and selection protocol unchanged; this amendment precedes all calibration and TEST interventions.
