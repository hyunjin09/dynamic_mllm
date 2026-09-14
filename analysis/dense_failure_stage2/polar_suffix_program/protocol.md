# Phase-74 frozen protocol

- Contract: `0c1696ed895353f31ba40201633215f235bacc7175624dccd2ac8d7fdb597a49`
- Population: 569 P90-triggered UIDs / 4948 unique programs.
- Dense-C: only all-FULL suffixes. Dense-W: every unique exact-replay-valid complete suffix.
- Objective: length-normalized teacher-forced program NLL, then dataset × source-regime × outcome / UID / 1-K weights.
- Trigger cache: raw BF16 all-FULL P90 token states and masks; exact cache/live branch, context, and initial-logit parity is mandatory for every UID.
- Decoder: 2 blocks, width 256, 4 heads, FFN 1024; beam-8 sum-log-probability primary.
- Internal dev selects only epoch count; the final checkpoint is reinitialized and refit on 100% of the corpus.
- Evaluation: exact Phase-69 19,960-row ChartQA/TextVQA/MMU-Pro/POPE protocol.
