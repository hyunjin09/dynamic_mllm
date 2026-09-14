# Corrected assistant answer-position sanity

- Frozen contract: `5703b57d2ba24e10409a17088c88cb7c90caae1ec994e0c514bfb861b38ab4de`
- Result: **FAIL**
- Exact layer-27 top-1 / stored first-token agreement: 72/72 (100.00%)
- Teacher-forced shared-prefix divergence agreement: 9/13 (69.23%)
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

- chartqa: 3/4 exact
- gqa: 3/4 exact
- textvqa: 3/5 exact

The full analysis is authorized by this gate only when every sanity record matches exactly.
