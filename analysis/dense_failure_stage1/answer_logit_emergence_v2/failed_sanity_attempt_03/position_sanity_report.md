# Corrected assistant answer-position sanity

- Frozen contract: `2a53c337a3116908dc019350c5168233b9e0271c4d33d6c3cd12947cf18abf48`
- Result: **FAIL**
- Exact layer-27 top-1 / stored first-token agreement: 72/72 (100.00%)
- Cached shared-prefix plus next-token replay agreement: 12/13 (92.31%)
- Raw layer-27 readout / raw model-logit top-1 agreement: 13/13
- Diagnostic only—raw top-1 / repetition-penalized generated-token agreement: 9/13 (69.23%)
- Execution failures: 0
- Position: final non-padding token of the complete prompt after `add_generation_prompt=true`.
- Literal suffix: `<|im_end|>\n<|im_start|>assistant\n`; the final newline is the state used to predict answer token one.

## Token boundary from the first sanity record

- `654`: ID `5267`, token `?Ċ`, decoded `?\n`
- `655`: ID `16141`, token `Answer`, decoded `Answer`
- `656`: ID `279`, token `Ġthe`, decoded ` the`
- `657`: ID `3405`, token `Ġquestion`, decoded ` question`
- `658`: ID `1667`, token `Ġusing`, decoded ` using`
- `659`: ID `264`, token `Ġa`, decoded ` a`
- `660`: ID `3175`, token `Ġsingle`, decoded ` single`
- `661`: ID `3409`, token `Ġword`, decoded ` word`
- `662`: ID `476`, token `Ġor`, decoded ` or`
- `663`: ID `17133`, token `Ġphrase`, decoded ` phrase`
- `664`: ID `13`, token `.`, decoded `.`
- `665`: ID `151645`, token `<|im_end|>`, decoded `<|im_end|>`
- `666`: ID `198`, token `Ċ`, decoded `\n`
- `667`: ID `151644`, token `<|im_start|>`, decoded `<|im_start|>`
- `668`: ID `77091`, token `assistant`, decoded `assistant`
- `669`: ID `198`, token `Ċ`, decoded `\n` **← position used**

## Dataset/outcome cells

- chartqa/correct: 12/12 exact
- chartqa/wrong: 12/12 exact
- gqa/correct: 12/12 exact
- gqa/wrong: 12/12 exact
- textvqa/correct: 12/12 exact
- textvqa/wrong: 12/12 exact

## Teacher-forced shared-prefix divergence checks

- chartqa: cached replay 4/4; raw readout parity 4/4; raw-top1/processed-token diagnostic 3/4
- gqa: cached replay 4/4; raw readout parity 4/4; raw-top1/processed-token diagnostic 3/4
- textvqa: cached replay 4/5; raw readout parity 5/5; raw-top1/processed-token diagnostic 3/5

The full analysis is authorized only when every base answer-start record matches, every cached prefix plus next-token replay matches the stored processed generation, and every layer-27 readout matches the corresponding raw model logits. Raw top-1 need not equal a token selected after the frozen repetition penalty.
