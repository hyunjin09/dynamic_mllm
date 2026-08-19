# Proposed v3 Held-Out Confirmation Protocol

Status: primary design frozen conditionally; entry gate not satisfied. The
final manifest, structured-null draw count, and real-donor caliper/index remain
closed pending the repairs in `reports/v3_preflight_report.md`.

## Primary object

Primary utility is the frozen per-token accepted-reference log-likelihood.
Sequence accepted-reference likelihood is secondary. For each record:

\[
S_i=\max_{l\in\{0,4,8,12,16,20,24\},\;a\in
\{IGNORE,READ\_ONLY,WRITE\_ONLY\}}
[Q_{i,l}(a)-Q_{i,l}(FULL)].
\]

Each branch starts from the identical dense pre-layer state, applies exactly
one action, uses the unchanged dense suffix, and uses identical accepted-answer
scoring. Layer 27 is excluded only because WRITE is structurally terminal-silent
there and creates deterministic duplicate action cells. No narrower grid is
justified by the discovery medians, trimmed means, interaction, and tie audit.

- Numerical epsilon: `1e-6 nats/token` (`1e-5` sequence).
- Practical threshold: `0.05 nats/token`.
- The raw maximum is retained; epsilon changes labels, not the numeric score.
- Exact/epsilon tie preference: `FULL > READ_ONLY > WRITE_ONLY > IGNORE`.
- Joint primary estimator: equal-weight mean of the GQA and TextVQA means.
- Dataset-specific estimates: secondary, not alternative primaries.
- Resampling unit: image. The primary manifest permits one record per image.

## Proposed data construction

Target `1,600` records: `800 GQA + 800 TextVQA`, all unique images. Exclude by
record ID, question ID, image ID, canonical RGB hash, normalized path, and
image-question identity against Stage A, Stage B/v3 discovery, v2 Stage C, and
all null-calibration data.

Rank candidates by ascending `SHA256(2026080602:record_id)` and retain the first
technically valid record per image. TextVQA is restricted to the remaining
singleton images so all remaining multi-question images stay available for
Stage C2. Apply the existing frozen validity rules for readable image, prompt,
normalization, answer span, context length, image-token indices, and pinned
processor/tokenizer. Stop rather than resize if 800 valid records in either
dataset are unavailable.

The audit found 10,234 metadata/image-valid GQA images and 2,362 fully audited
TextVQA images. The 1,600-record identity preview is disjoint, but it is not a
frozen manifest and no intervention value has been opened.

## Endpoints and reporting

Primary replication requires the stratified image-bootstrap 95% CI for the
joint mean `S` to lie entirely above zero. Report mean, SD, median, 5% and 20%
trimmed means, quantiles, best-action frequencies, conditional effects,
interaction prevalence, FULL-correct regressions, FULL-wrong improvements, and
fractions above epsilon and `0.05`.

Tail-robust confirmation additionally requires all of:

- the median `S` 95% CI lower bound is above `0.05`;
- the 20% trimmed-mean `S` 95% CI lower bound is above `0.05`;
- the 95% CI lower bound for `Pr(S > 0.05)` is above `0.5`.

This prevents a large oracle mean driven by numerical ties or a few extremes
from determining success.

Structured-null specificity requires all three paired real-minus-null mean
CIs to lie above zero under the exact rules in
`workspace/v3_structured_null_spec.md`. The conjunction is an
intersection-union test; no family-wise alpha division is applied because all
three component alternatives must pass. Ordinary 95% CIs are reported for
every family.

Accepted-answer variants, sequence scoring, deterministic generation, and a
predefined prompt-paraphrase subset are secondary robustness analyses and may
not replace the primary endpoint. The active plan also requires a valid
visual-grounding control. A same-question/different-official-answer GQA margin
control is viable, but independent review found it is not a minimal visual
counterfactual and should not be the sole grounding gate. Its final pairing and
success rule are therefore not frozen here.

## Success ladder

1. **Primary replication:** joint mean `S` CI entirely above zero.
2. **Structured-null specificity:** every paired `S_real - S_null` CI entirely
   above zero.
3. **Distributional robustness:** all median, trimmed-mean, and prevalence
   gates above pass.
4. **Answer-content and grounding robustness:** prospectively frozen controls
   pass.
5. **Confirmed answer-misaligned dense participation:** permitted only when
   steps 1–4 all pass.

Failure at any tier cannot be rescued by a lower tier, a dataset subgroup, a
different layer, or a different action. No deployable policy, acceleration,
accuracy, or mechanism claim follows from this protocol.
