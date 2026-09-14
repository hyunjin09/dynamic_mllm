# READ-operation feature contract

- Frozen contract: `eaaef86021f06f032a4e809b3db468fb4ea281f5aa0ce921ee15f8e0e256e4d6`
- Target: `H_R = q_WRITE_ONLY - q_FULL`; positive means READ is harmful.
- Query row: final valid text/control row that predicts the first assistant answer token.
- Token-update scope: valid text/control rows after the last visual row.
- Attention: projected and multimodal-rotary q/k/v, float32 q-k and summary accumulation, scaled softmax over all valid keys.
- READ output: FULL attention output minus text-only attention output before the residual update.
- F1: READ update magnitude and residual-relative ratios.
- F2: READ-update cosine alignment with pre, WRITE_ONLY, and FULL text states.
- F3: text/control update concentration, top-1/top-5 share, and entropy.
- F4: visual attention mass, conditional entropy, top weights, effective tokens, and head dispersion.
- F5: query-to-visual-key compatibility and gaps.
- F6: projected READ-output and visual-value magnitudes, residual ratio, and head dispersion.
- F7: visual token/spatial concentration summaries.

No suffix state, generated answer, final correctness, dataset/source identity,
route identity, future action, or ground-truth answer is included as a feature.
The 24-state smoke passed exact state/branch/feature-repeat checks and
same-shape causal SDPA operation reconstruction before bulk extraction.
