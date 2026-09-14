# Invalid BF16-router attempt

The first full V1 attempt was stopped after epoch 5 reported a non-finite aggregate loss at global update 1,310. No checkpoint was emitted and validation did not start.

A bounded diagnostic checked every tensor in all 698 unique Phase-56 Corpus-B routed-state shards; all were finite. This rules out a non-finite value in those transferred summaries, but does not prove the source of the full-token failure. Router autocast/backward numerical instability remains only a suspected diagnosis.

The attempt is historical failure evidence and must not be used as a Stage-2 result.
