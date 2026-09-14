# Corrected dense answer-logit emergence analysis

- Frozen contract: `c9a6d6302779c0dfa07886397106bd23d1d7dbbf323ee9008906b0b524cbdcce`
- Population: 7,999 current-runtime dense records (3,999 correct, 4,000 wrong).
- Wrong comparisons: 3,993 usable, 7 excluded with logged reasons; 587 use a teacher-forced shared prefix.
- No conclusion from the invalid `text_final` readout is reused here.

## Position validation

The final prompt token after `<|im_end|>\n<|im_start|>assistant\n` reproduced the stored generated first token for 72/72 sanity samples (100.00%) and 7,926/7,999 full-population samples (99.09%). All GQA samples matched. The 73 full-population mismatches were 70 ChartQA and 3 TextVQA samples, predominantly dense-wrong; these compare raw LM-head top-1 with tokens selected after the frozen generation processors, including `repetition_penalty=1.05`, and are not evidence of the old user-token position error.

Cached shared-prefix plus next-token replay matched 12/13 sanity cases, with at least four exact cases per dataset; raw layer-27 readouts matched the corresponding raw model logits in 13/13. In the full population, 587 wrong comparisons used cached shared-prefix replay and 7 additional collision comparisons were logged as unreplayable and excluded rather than called ambiguous.

## Main curves

- Layer 0: correct GT-minus-competitor mean/median -11.382/-11.249; wrong GT-minus-generated mean/median -0.199/-0.137.
- Layer 4: correct GT-minus-competitor mean/median -14.238/-13.730; wrong GT-minus-generated mean/median 0.061/0.172.
- Layer 9: correct GT-minus-competitor mean/median -8.718/-9.027; wrong GT-minus-generated mean/median -0.011/0.062.
- Layer 13: correct GT-minus-competitor mean/median -8.493/-8.500; wrong GT-minus-generated mean/median -0.003/0.053.
- Layer 18: correct GT-minus-competitor mean/median -9.226/-9.062; wrong GT-minus-generated mean/median -0.198/-0.137.
- Layer 23: correct GT-minus-competitor mean/median -6.072/-6.188; wrong GT-minus-generated mean/median -2.233/-1.918.
- Layer 27: correct GT-minus-competitor mean/median 4.304/4.375; wrong GT-minus-generated mean/median -3.099/-2.125.

## Persistent emergence and crossover

- Correct emergence: median layer 24 (IQR 23–25); no persistent emergence in 3,148/3,999.
- Wrong crossover: median layer 2 (IQR 0–11); no persistent crossover in 272/3,993 usable comparisons.
- Population 50% coverage: correct layer not reached, wrong layer 3.
- Population 75% coverage: correct layer not reached, wrong layer 15.

The correct persistent statistic is strongly right-censored: a crossing must be followed by two more observed positive layers, so an answer first becoming preferred at layer 26 or 27 cannot satisfy it. Without that persistence requirement, the correct GT first becomes raw top-1 at median layer 26 (IQR 26–27), and 94.95% are GT top-1 at layer 27. Thus correct-answer emergence is a late, mostly layer-26/27 phenomenon; the layer-24 median describes only the 851 early-enough persistent cases, not the full correct population.

The wrong median crossover at layer 2 must not be read as formed wrong answers. Through the early and middle layers the GT and eventual-wrong tokens both have very low ranks and their margin is centered close to zero: at layer 0 the median margin is -0.137 and the two sign fractions are 52.0%/47.9%. Separation becomes materially answer-like only in the late stack. The generated-wrong token is rank ≤10 for 22.9%, 64.1%, 86.8%, and 100% of usable wrong samples at layers 23, 25, 26, and 27; it is raw top-1 for 4.2%, 20.3%, 42.9%, and 91.2% at those layers.

## Wrong trajectory taxonomy

- `ambiguous`: 272/3,993 (6.8%)
- `answer_erosion`: 1,711/3,993 (42.8%)
- `early_wrong`: 1,977/3,993 (49.5%)
- `progressive_wrong`: 33/3,993 (0.8%)

These are the requested fixed zero-threshold taxonomy counts, but the large `early_wrong` and `answer_erosion` categories inherit the near-zero early-margin instability above. They quantify sign trajectories; they do not establish that a semantic wrong answer was already represented in early layers.

## Dataset breakdown

| Dataset | Correct first GT-top1 layer, median (IQR) | Correct GT top-1 at L27 | Wrong margin < 0 at L23 / L27 | Wrong generated token top-1 at L27 |
|---|---:|---:|---:|---:|
| GQA | 27 (26–27) | 99.4% | 69.1% / 98.2% | 96.2% |
| ChartQA | 25 (23–27) | 95.9% | 75.2% / 89.3% | 84.5% |
| TextVQA | 26 (25–27) | 85.1% | 68.3% / 92.7% | 87.8% |

All three tasks show late answer formation. TextVQA's lower canonical-GT top-1 rate is consistent with its multi-reference semantics: 151/1,000 evaluator-correct TextVQA samples have a generated first token different from the canonical GT first token.

## Final questions

1. **Yes, essentially all.** Layer 27 at the true answer-start position reproduced 72/72 stored first generated tokens in the stratified gate and 7,926/7,999 over the full population (99.09%). The residual 73 are generation-processor/raw-top1 differences, not the previous position error.
2. **Late.** The correct GT first becomes raw top-1 at median layer 26 (IQR 26–27), with the population top-1 fraction rising from 22.3% at layer 25 to 47.1% at layer 26 and 94.9% at layer 27. The fixed three-layer persistent rule is defined for only 851/3,999 because late crossings are right-censored; among those defined cases its median is layer 24.
3. **The literal fixed-rule answer is layer 2, but it is not a defensible answer-formation depth.** Early margins are near-zero and both target tokens rank in the tens of thousands. Material wrong-answer dominance emerges around layers 23–27, becoming strong at 25–27 and overwhelming at layer 27.
4. Over 3,993 usable wrong comparisons: early-wrong 49.5%, progressive-wrong 0.8%, answer erosion 42.8%, and ambiguous/unstable 6.8%. Seven unreplayable collision comparisons are excluded. The early/erosion proportions should be interpreted as zero-threshold sign patterns, not semantic answer presence.
5. **Yes for this final-head answer-token lens, with an important limit.** Early correct GT margins are uniformly negative, while wrong GT-versus-generated margins are near-zero with roughly even signs and negligible target top-1 rates. This is strong evidence that early layers are too answer-ambiguous to justify supervision *because an answer token has already emerged*. It does not prove that the full early hidden state lacks predictive information about eventual correctness.
6. **The defensible candidate region from this analysis is the late stack, approximately layers 25–27, strongest at 26–27.** Layer 25 is where the wrong generated token first reaches rank ≤10 for a majority (64.1%); by layer 26 this is 86.8%, while correct GT top-1 reaches 47.1%, and both become dominant at layer 27. This is evidence for the next strategy decision, not a selection of all-layer versus post-emergence supervision.

## Reference and interpretation limits

Canonical GT first-token and generated first-token differ in 189/3,999 evaluator-correct records. Those records remain in the requested canonical-GT analysis and are explicitly auditable.
Raw logits from the frozen final norm and LM head are not calibrated probabilities. Multi-reference and evaluator-equivalent answers can differ tokenically from the canonical GT string.

## Stop status

The corrected answer-position analysis is complete. No Stage-1 predictor training, supervision-range choice, W→C work, four-action routing, MCTS, or external evaluation was performed.
