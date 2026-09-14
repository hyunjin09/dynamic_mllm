# Failed sanity attempt 02

- Contract: `1e912c2681de87ee09595a5610c21ce650973da40c6682e9fa5c8c4c0ce3548d`
- Base answer-start validation: 72/72 exact layer-27 raw top-1 matches.
- Cached processed generation: 13/13 common prefixes and stored next generated
  tokens replayed exactly.
- Raw LM-head top-1 versus processed generated divergence token: 9/13.
- Direct cause: the frozen Qwen generation configuration applies
  `repetition_penalty=1.05`. Generation chooses from processed scores, whereas
  this analysis intentionally measures raw `final_norm + lm_head` logits.
  Therefore raw top-1 and the processed generated token need not coincide after
  an answer prefix.
- Correction: validate cached generated token sequences against stored generated
  tokens, and separately validate the reconstructed layer-27 raw readout against
  the model's raw, pre-processor logits. Retain raw-top1/processed-token agreement
  as a diagnostic only.

The raw GT-minus-generated-token margin definition is unchanged. The full
population run was not launched under this failed gate.
