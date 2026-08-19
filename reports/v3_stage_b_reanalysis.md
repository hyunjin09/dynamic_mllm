# v3 Stage B Four-Action Reanalysis

Date: 2026-08-06  
Status: complete, inspected discovery analysis  
Active plan: `plans/dynamic_mllm_read_write_policy_conditional_plan_v3.md`

## Scope and validity

This analysis deterministically re-expresses the preserved v2 Stage B results;
it does not load the model, collect an intervention outcome, train anything, or
reuse the inspected v2 Stage C population as confirmation. The 400 records
(200 GQA, 200 TextVQA) yield all 3,200 expected sample-layer matrices and all
12,800 action cells at layers `[0,4,8,12,16,20,24,27]`. No value was filled or
imputed.

The mapping is `IGNORE=Q(0,0)`, `READ_ONLY=Q(1,0)`,
`WRITE_ONLY=Q(0,1)`, and `FULL=Q(1,1)`. Static implementation evidence and the
preserved Stage A/Stage B validity artifacts verify the same image/question,
identical cached FULL pre-layer state, one action at the selected layer,
unchanged dense suffix, and identical accepted-answer scoring. Recomputed FULL
score parity and stored-effect error are exactly zero. Maximum recorded READ
and WRITE reconstruction discrepancies are respectively `5.96e-08` and
`2.98e-08`.

Per-token accepted-reference log-likelihood is the cross-sample metric.
Sequence log-likelihood is secondary. Exact `argmax Q` defines the best action;
exact ties prefer `FULL > READ_ONLY > WRITE_ONLY > IGNORE` only to make the
label deterministic. Numerical near-ties are separately defined by the frozen
Stage B no-op thresholds (`1e-6` nats/token and `1e-5` nats/sequence). A
secondary practical band uses `0.05` nats/token. Confidence intervals use
2,000 deterministic sample-clustered percentile-bootstrap draws.

## Four-action value landscape

The following entries are per-token `mean / median`; the complete distribution
(standard deviation, 5% and 20% trimmed means, quantiles, and clustered CIs) is
in `layer_q_summary_v1.csv`.

### GQA Q values

| Layer | IGNORE | READ_ONLY | WRITE_ONLY | FULL |
|---:|---:|---:|---:|---:|
| 0 | -2.189 / -1.041 | -2.245 / -0.962 | -1.645 / -0.564 | -1.646 / -0.668 |
| 4 | -1.660 / -0.653 | -1.643 / -0.685 | -1.652 / -0.704 | -1.646 / -0.668 |
| 8 | -1.631 / -0.688 | -1.630 / -0.688 | -1.654 / -0.709 | -1.646 / -0.668 |
| 12 | -1.711 / -0.780 | -1.663 / -0.733 | -1.681 / -0.684 | -1.646 / -0.668 |
| 16 | -1.651 / -0.683 | -1.658 / -0.689 | -1.646 / -0.711 | -1.646 / -0.668 |
| 20 | -1.654 / -0.632 | -1.657 / -0.636 | -1.638 / -0.667 | -1.646 / -0.668 |
| 24 | -1.644 / -0.659 | -1.644 / -0.663 | -1.637 / -0.696 | -1.646 / -0.668 |
| 27 | -1.617 / -0.636 | -1.646 / -0.668 | -1.617 / -0.636 | -1.646 / -0.668 |

### TextVQA Q values

| Layer | IGNORE | READ_ONLY | WRITE_ONLY | FULL |
|---:|---:|---:|---:|---:|
| 0 | -2.445 / -1.724 | -2.473 / -1.713 | -1.200 / -0.681 | -1.253 / -0.700 |
| 4 | -1.273 / -0.728 | -1.285 / -0.748 | -1.263 / -0.728 | -1.253 / -0.700 |
| 8 | -1.287 / -0.686 | -1.294 / -0.687 | -1.266 / -0.687 | -1.253 / -0.700 |
| 12 | -1.298 / -0.731 | -1.296 / -0.724 | -1.272 / -0.694 | -1.253 / -0.700 |
| 16 | -1.279 / -0.693 | -1.271 / -0.684 | -1.265 / -0.695 | -1.253 / -0.700 |
| 20 | -1.254 / -0.683 | -1.256 / -0.681 | -1.251 / -0.692 | -1.253 / -0.700 |
| 24 | -1.276 / -0.697 | -1.272 / -0.687 | -1.254 / -0.692 | -1.253 / -0.700 |
| 27 | -1.305 / -0.796 | -1.253 / -0.700 | -1.305 / -0.796 | -1.253 / -0.700 |

The repeated FULL value across layers is expected: each instrumented FULL
branch reproduces the same dense model. The non-FULL cells vary with layer.

### Suppression gain, best actions, ties, and interaction

`Best I/R/W/F` gives exact-argmax counts for
`IGNORE/READ_ONLY/WRITE_ONLY/FULL`. `G` gives mean/median/20%-trimmed mean.
`Near` is the fraction with `|G| <= 0.05`. `Rrev/Wrev` are strict conditional
sign-reversal fractions. `IndFail` is failure of the independent-main-effect
action to recover the exact best cell.

| Dataset | L | Best I/R/W/F | G mean / median / trim20 | Near | Rrev / Wrev | IndFail |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 0 | 25/38/68/69 | .165/.004/.028 | .440 | .345/.165 | .315 |
| GQA | 4 | 58/50/47/45 | .090/.015/.038 | .510 | .315/.365 | .260 |
| GQA | 8 | 41/66/44/49 | .112/.017/.045 | .510 | .305/.255 | .235 |
| GQA | 12 | 42/42/59/57 | .101/.015/.044 | .520 | .325/.225 | .225 |
| GQA | 16 | 45/48/45/62 | .120/.012/.042 | .465 | .220/.300 | .230 |
| GQA | 20 | 54/44/57/45 | .082/.013/.038 | .520 | .305/.340 | .270 |
| GQA | 24 | 47/38/53/62 | .076/.006/.029 | .550 | .310/.325 | .300 |
| GQA | 27 | 0/0/69/131 | .085/.000/.010 | .735 | .000/.000 | .000 |
| TextVQA | 0 | 10/19/85/86 | .146/.001/.018 | .600 | .495/.070 | .395 |
| TextVQA | 4 | 36/55/44/65 | .060/.003/.016 | .655 | .290/.155 | .185 |
| TextVQA | 8 | 51/35/49/65 | .040/.002/.014 | .650 | .370/.155 | .300 |
| TextVQA | 12 | 39/47/54/60 | .042/.003/.014 | .660 | .340/.165 | .255 |
| TextVQA | 16 | 43/47/46/64 | .055/.003/.016 | .630 | .225/.230 | .240 |
| TextVQA | 20 | 57/44/58/41 | .061/.007/.021 | .630 | .260/.205 | .180 |
| TextVQA | 24 | 37/29/72/62 | .040/.003/.013 | .700 | .350/.250 | .250 |
| TextVQA | 27 | 0/0/76/124 | .029/.000/.001 | .870 | .000/.000 | .000 |

Except at layer 27, exact-epsilon multi-action ties are at most 2%. Layer 27
has a structural WRITE tie for every record and is mainly
answer-silent/redundant (65.5% GQA, 62.0% TextVQA under the no-op threshold).
The practical picture is more cautious: 44–55% of GQA and 60–70% of TextVQA
pairs at layers 0–24 have best-suppression gain within 0.05 nats/token.
Nevertheless, `G > 0.05` occurs in 37.2% of all GQA pairs and 25.6% of all
TextVQA pairs, so the landscape is not explained entirely by practical ties.

Heavy tails matter. Layerwise mean `G` is 0.076–0.165 for GQA and 0.040–0.146
for TextVQA at layers 0–24, whereas medians are 0.001–0.017. The 20%-trimmed
means remain positive (0.028–0.045 GQA; 0.013–0.021 TextVQA), but much smaller.
Effects are concentrated in FULL-wrong records: averaging the layerwise
stratum summaries, FULL-wrong `G` has mean/median/trim20 of
0.191/0.098/0.117 for GQA and 0.108/0.035/0.045 for TextVQA, versus
0.013/0.001/0.005 and 0.014/0.0004/0.002 for FULL-correct records. These are
descriptive strata, not prevalence estimates.

Sequence/per-token agreement supports the ranking qualitatively without being
an independent replication. Across layers, `G` agreement is GQA Pearson
0.944, Spearman 0.994, sign-label 0.978; TextVQA Pearson 0.786, Spearman 0.867,
sign-label 0.890. Complete quantity-level agreement is in
`sequence_per_token_agreement_v1.csv`.

## Fixed-policy test

All fixed policies are fitted and evaluated on the same inspected discovery
set; their values are optimistic descriptions, not held-out policy estimates.

| Per-token policy, joint | Utility vs FULL (95% CI) | Oracle match | Regret to oracle (95% CI) |
|---|---:|---:|---:|
| One global action (`FULL`) | 0 | .340 | .0976 [.0825,.1152] |
| One action per layer | .0045 [-.0014,.0105] | .348 | .0931 [.0782,.1091] |
| One action per dataset/layer | .0075 [-.0005,.0159] | .333 | .0902 [.0768,.1040] |
| Always FULL | 0 | .340 | .0976 [.0817,.1156] |
| Sample-layer oracle | .0976 [.0814,.1139] | 1.000 | 0 |

The best per-layer schedule selects WRITE_ONLY at layers 0, 20, and 24 and
FULL elsewhere. The dataset/layer schedule differs between tasks, yet closes
only 0.0075 of the 0.0976 discovery oracle gap. Best actions vary across
layers for 198/200 GQA and 197/200 TextVQA samples. Thus neither one global
action nor a fixed layer/task schedule substantially explains the observed
landscape.

Likelihood-selected fixed schedules have both behavioral gains and costs. The
per-dataset/layer policy has 14 FULL-correct regression pairs (10 unique
samples) and 29 FULL-wrong improvement pairs (24 unique samples). The oracle
has 8 regression pairs (4 samples) and 109 improvement pairs (45 samples).
These are repeated sample-layer observations and must not be read as
independent accuracy prevalence. Enabled-bit counts in the CSV are descriptive
action rankings only, not FLOPs or acceleration claims.

## Policy-conditional interaction

Across all layers, READ strictly reverses sign between WRITE conditions for
26.6% of GQA and 29.1% of TextVQA pairs; WRITE reverses across READ conditions
for 24.7% and 15.4%. Independent main effects miss the exact best action for
22.9% and 22.6% (22.6% and 22.4% even after allowing the numerical epsilon).
Interaction exceeds the no-op tolerance in about 87% of pairs and exceeds
`0.05` nats/token in 40.5% of GQA and 27.9% of TextVQA pairs. Medians are
usually near zero, so interaction is heterogeneous rather than a uniform
signed shift.

Concrete high-interaction examples were selected only after aggregates were
complete and are discovery illustrations in `interaction_examples_v1.json`.
For example, GQA `gqa_gh_00256240` at layer 0 has READ effects `+5.027` with
WRITE off and `+0.193` with WRITE on, while WRITE changes from `+4.818` to
`-0.016`; TextVQA `textvqa_train_1585` has READ effects `-2.056` and `-5.102`
and WRITE effects `+1.453` and `-1.593`. These cases are not nominated for
future replication.

## Reinterpretation of v2 Stage B and Outcome B

1. The TextVQA layer-0 `FULL` versus `WRITE_ONLY` result is one narrow edge of
   a broader four-cell landscape. The layer is dominated by the two WRITE-on
   cells; both WRITE-off cells are much worse on average.
2. At TextVQA layer 0, exact best actions are FULL 86, WRITE_ONLY 85,
   READ_ONLY 19, and IGNORE 10. There is no single winning action.
3. WRITE_ONLY wins through the conditional READ contrast, not because all
   suppressions behave similarly. Its advantage over FULL is mean `0.05295`,
   median `0.0000035`, and 20%-trimmed mean `0.00565`; only 10 of its 85 wins
   have IGNORE or READ_ONLY within 0.05.
4. The old structured-null failures are a direct warning for v3 `G`: selecting
   the maximum across more actions and layers increases the search advantage
   available to generic residual perturbations. A v3 endpoint must beat nulls
   with exactly the same action/layer search budget.
5. Early WRITE is strongly FULL-critical, not merely conditionally valuable.
   At layer 0 its mean/median effects are `0.544/0.047` and `0.599/0.023` for
   GQA, and `1.245/0.384` and `1.219/0.393` for TextVQA, under READ off/on.
   Later-layer WRITE effects are much smaller and condition-dependent; layer
   27 WRITE is exactly silent in these cells.
6. Layers 0–24 are heterogeneous mixtures of FULL-critical and candidate
   FULL-misaligned discovery cells; layer 27 is mainly answer-silent/redundant.
   No positive `G` is called harmful participation.

The v2 Stage C conclusion remains exactly Outcome B: its held-out
reference-support contrast replicated, but actual READ removal did not beat
either frozen structured residual null. It is historical risk evidence, not
v3 confirmation.

## Same-image query-dependence feasibility

The frozen Stage B selector has 400 unique effective image assets, so it
contains no same-image multi-question Q group and supports no within-image
discovery summary. Structurally, the pinned architecture places visual tokens
before question tokens and the causal mask blocks visual-query access to later
question tokens. Thus visual prefix/WRITE is query-invariant for the same image
and identical preceding prefix; a numerical cross-question sanity check remains
required before C2.

Metadata show ample prospective groups: GQA validation has 9,800 images with
at least two metadata-eligible questions (131,628 records); TextVQA validation,
after excluding the 800 inspected Stage C images and frozen invalid records,
has 1,243 two-question images (2,486 records). GQA still needs processor/token
eligibility checks. Seventeen Stage B TextVQA records lack an official image
ID; exact future image-disjointness for them must be resolved with normalized
paths or image hashes in preflight. No prospective manifest is frozen here.

## Artifact index and preservation

The versioned per-matrix Q table, advantages/contrasts, policy table,
interaction table, feasibility audit, and analysis manifest are under
`outputs/v3_discovery/`. The manifest records source/artifact checksums,
software versions, seeds, tie rules, and integrity status.

Preserved source hashes are unchanged:

- v3 plan: `8612e1b22d76dcfcc8a5f63780493d9d29e1f9c0288d986ee21047890c129605`
- v2 Stage B results: `411cb24899973ad19d7c3186bfa9f88ae59f0a1ca09366b889b57053c32ba4d4`
- v2 Stage C results: `fc5a011ad20735d6b5b2ed0a41e8f5060b31d5e921600d33ebebf2bba0346c05`
- frozen v2 archive: `65ac4d07fcf348a55475266479d3786dcbb9e8914711faa988677fcb7515df92`

