# Stage B Exploratory Conclusion

Status: Stage B complete. These findings are discovery evidence only; they are
not held-out prevalence estimates and do not establish harmful visual
participation.

## Scope and Integrity

- All 400 frozen candidates completed: 200 GQA and 200 TextVQA.
- No record met a predefined technical-invalid rule; no sample was replaced.
- Every sample has all four states at layers `[0,4,8,12,16,20,24,27]`.
- FULL sequence scores and generated token IDs matched the cached unmodified
  baseline across every layer.
- The source plan SHA-256 remains
  `d476736dde6d5d7d44ab3e18794ebc3c4e988d703829657271bc285ecd5171d1`.

Pinned FULL relabeling yielded 98 correct / 102 wrong GQA records and 104
strictly correct / 96 not-strictly-correct TextVQA records. Six TextVQA records
had partial consensus scores (three at `1/3`, three at `2/3`) and are retained
numerically but classified as not correct under the frozen strict threshold.

## Supported Exploratory Findings

1. **Early WRITE is strongly answer-aligned on average.** At layer 0, enabling
   WRITE increased mean reference support under both READ conditions. GQA
   per-token mean effects were `+0.544` (95% sample-bootstrap CI
   `[+0.306,+0.806]`) and `+0.599` (`[+0.314,+0.943]`). TextVQA effects were
   `+1.245` (`[+0.964,+1.547]`) and `+1.219` (`[+0.925,+1.523]`). This rejects
   an exploratory hypothesis of broadly answer-misaligned early WRITE.

2. **The strongest candidate answer-misaligned READ effect is TextVQA layer 0
   conditional on WRITE being enabled.** The all-sample `read_w1` effect was
   `-0.196` sequence log-likelihood (`[-0.374,-0.050]`) and `-0.053` per-token
   mean (`[-0.116,-0.008]`). It was stronger among FULL-wrong records
   (`n=96`, mean `-0.098`, `[-0.233,-0.006]`; 59.4% negative beyond numerical
   epsilon). The `w=0` conditional mean and the interaction CI crossed zero, so
   the two READ effects must remain separate.

3. **GQA has a weaker, correctness-stratified terminal READ candidate.** At
   layer 27, WRITE and interaction are structurally answer-silent because no
   suffix layer can consume the modified visual rows. Among FULL-wrong GQA
   records (`n=102`), READ had per-token mean `-0.077`
   (`[-0.146,-0.004]`; 54.9% negative). Among FULL-correct records it was
   positive (`+0.022`, `[+0.011,+0.036]`). The all-sample mean and the
   FULL-wrong sequence-sum CI crossed zero, so this is tentative rather than a
   cross-metric confirmed discovery pattern.

4. **READ is heterogeneous rather than globally negative.** Across-layer
   sample means for READ were near zero with CIs spanning zero in each dataset,
   whereas WRITE across-layer means were positive. No common negative layer
   band was supported across GQA and TextVQA.

5. **Sequence-sum and per-token-mean effects usually agree in direction, but
   not perfectly.** Threshold-label agreement was generally high. At the
   strongest TextVQA candidate it was 89.5% with Pearson `r=0.637`; at the GQA
   layer-27 candidate it was 97% with `r=0.924`. Layer-averaged effect versus
   answer-length correlations were weak (absolute `r <= 0.17`).

## Tentative Observations

- Negative READ evidence is concentrated in dataset-, layer-, conditioning-,
  and correctness-specific tails. Medians are near zero even where bootstrap
  mean CIs are negative.
- The TextVQA layer-0 READ candidate is the only negative effect supported in
  both sequence-sum and per-token-mean aggregate CIs over all samples.
- FULL-wrong stratification appears to enrich negative READ effects, but that
  stratum is defined by the pinned model and discovery data; it is not a
  confirmatory prevalence estimate.

## Secondary Greedy Behavior

Greedy behavior changes were sparse and mixed relative to the continuous
likelihood effects. Removing READ at TextVQA layer 0 produced five strict
FULL-wrong-to-correct transitions and four strict FULL-correct-to-wrong
transitions. Removing READ at GQA layer 27 produced three corrections and two
regressions. Many likelihood improvements remained wrong. These are secondary
behavioral observations, not causal corrections or evidence of a harmful
mechanism.

## Failed or Unsupported Hypotheses

- A broadly answer-misaligned WRITE effect is not supported; early WRITE is
  strongly answer-aligned on average.
- A single shared negative READ/WRITE layer band across both datasets is not
  supported.
- The numerical-noise epsilon alone does not yield a useful substantive
  answer-silent class: outside terminal WRITE, almost every nonzero effect
  exceeds the very small exact-noise floor.
- Top-1 correction counts do not provide a reliable substitute for the
  reference-likelihood diagnostic.

## Unresolved Scientific Risks

- Stage B selected the strongest candidate from multiple layers, operations,
  conditioning states, datasets, and correctness strata. Any confirmation must
  freeze that search or reproduce the identical search under every structured
  null.
- Heavy tails and near-zero medians make the tiny numerical epsilon a noise
  threshold, not automatically a practically meaningful effect cutoff.
- No search-adjusted covariance/subspace-matched or real-residual null was run
  in Stage B; this is required before a confirmed answer-misaligned claim.
- The current source plan's Stage C endpoint is multiple-choice margin based and
  requires option/label permutations. Carrying this open-ended reference metric
  into Stage C is a core protocol amendment, not an implementation detail.
- The discovery pool is train-derived and cannot be reused as held-out
  confirmatory prevalence evidence.

## Evidence Index

- Versioned sample results: `outputs/stage_b/stage_b_results_v1.jsonl`
- Integrity/checksums: `outputs/stage_b/analysis_v1/analysis_manifest.json`
- Effects and CIs: `outputs/stage_b/analysis_v1/layer_signed_effects.csv`
- Frozen-threshold fractions:
  `outputs/stage_b/analysis_v1/layer_threshold_fractions.csv`
- Greedy outcomes: `outputs/stage_b/analysis_v1/greedy_behavior_counts.csv` and
  `likelihood_behavior_relation.csv`
- Robustness: `sequence_mean_agreement.csv`,
  `answer_length_sensitivity.csv`, and `answer_length_correlations.csv`
- Plots: `outputs/stage_b/analysis_v1/plots/`
