# Cross-Dataset Direct Visual-Access Analysis

Date: 2026-08-22  
Plan: `plans/motivation_check4.md`  
Status: **PASS**

## Executive result

The four task families differ strongly in both the prevalence of direct visual
dependence and the amount of direct visual access needed among samples that
need it. They do **not** show a comparably strong difference in where access is
placed across decoder depth.

Under the matched 200-simulation FULL-correct prefix, the fraction of current
FULL-correct samples classified V+ was 57.3% for GQA, 93.3% for TextVQA, 82.3%
for ChartQA, and 50.9% for WeMath2.0-Pro. Among those V+ samples, mean minimum
positive VISUAL_ON increased from 8.66 layers for GQA to 10.74 for TextVQA,
12.47 for ChartQA, and 13.86 for WeMath2.0-Pro. The associated clustered
intervals are well separated except for partial ChartQA/WeMath overlap.

Placement differences were much smaller. Exact-min normalized centroid ranged
only from 0.473 to 0.492, a maximum difference of about half a decoder layer.
Every aggregate profile remained very similar in shape: pairwise cosine
similarity was 0.982--0.996 at exact minimum and 0.994--0.999 at min+4. Some
ChartQA/WeMath centroid and late-fraction coefficients remain statistically
nonzero after amount adjustment, but their practical magnitude is small and
the placement models explain little centroid variance. The defensible result
is therefore an amount effect, not a strong task-specific depth schedule.

This is observational, search-conditioned evidence. GQA/TextVQA/ChartQA came
from historically balanced correct/wrong pools, while WeMath uses its full
technically valid Pro pool; prompts, scorers, answer formats, and native visual
token counts also differ. Dataset identity is not treated as calibrated
difficulty.

## Execution semantics and source integrity

The same pinned Qwen2.5-VL-7B-Instruct snapshot
`cc594898137f460bfe9f0759e9844b3ce807cfb5` and the same verified
`BinaryQwen25VL` executor were used by both source contracts. All sources use
28 unrestricted binary layer actions:

- VISUAL_ON: visual and text/control rows execute normally and text can attend
  to visual K/V;
- VISUAL_OFF: visual rows bypass the layer while text/control rows execute
  without visual K/V.

Thus ALL-OFF means no decoder layer has **direct visual access**, not literally
no image-derived side channel. ON count is the number of decoder layers with
direct visual-token participation, not total FLOPs.

Every indexed file was re-hashed and revalidated. The audit checked mask
length and uniqueness, score/threshold validity, FULL and ALL-OFF anchors,
source binding, current FULL status, MCTS trace/candidate linkage, and route
semantics.

| Dataset | Records | Evaluated routes | Valid routes | FULL anchors | ALL-OFF anchors |
|---|---:|---:|---:|---:|---:|
| GQA | 4,000 | 1,339,399 | 353,518 | 4,000 | 4,000 |
| TextVQA | 2,000 | 652,999 | 98,344 | 2,000 | 2,000 |
| ChartQA | 2,000 | 650,600 | 76,185 | 2,000 | 2,000 |
| WeMath2.0-Pro | 4,544 | 1,658,485 | 107,671 | 4,544 | 4,544 |

All source and output checksum checks passed. The output manifest binds 31
tables/figures, including 13 SVG figures.

## Matched search opportunity

There was no early stopping. FULL and ALL-OFF anchors were evaluated outside
the simulation count. Every current FULL-correct record received exactly 200
simulations, so `B_correct=200`. Every current FULL-wrong record received at
least 400, so `B_wrong=400`.

The legacy search extended to 600 only when no correction had been found after
400. This affected 657 GQA, 279 TextVQA, and 244 ChartQA FULL-wrong records.
WeMath was hard-capped at 400. Therefore:

- the V+ primary comparison uses FULL, ALL-OFF, and the first 200 simulations;
- the A+ primary comparison uses FULL, ALL-OFF, and the first 400 simulations;
- 600-simulation legacy routes appear only in all-available sensitivity.

V+ results are identical under matched and all-available analysis because all
FULL-correct searches ended at 200. In A+, the extra legacy tail found a
correction for 43 additional GQA, 25 TextVQA, and 29 ChartQA records. It did
not materially change A+ budget or placement summaries.

## FULL/ALL-OFF taxonomy and direct visual dependence

The matched taxonomy is:

| Dataset | Eligible | FULL acc | V0 | V+ | A0 | A+ | D | ALL-OFF acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GQA | 4,000 | 50.0% | 855 | 1,145 | 151 | 1,192 | 657 | 25.2% |
| TextVQA | 2,000 | 51.7% | 69 | 965 | 35 | 652 | 279 | 5.2% |
| ChartQA | 2,000 | 50.6% | 179 | 832 | 35 | 710 | 244 | 10.7% |
| WeMath2.0-Pro | 4,544 | 18.5% | 413 | 428 | 162 | 1,263 | 2,278 | 12.7% |

Conditioning on current FULL correctness separates visual dependence from
FULL performance:

| Dataset | FULL-correct N | V0 / FULL-correct | V+ / FULL-correct |
|---|---:|---:|---:|
| GQA | 2,000 | 42.8% | 57.3% |
| TextVQA | 1,034 | 6.7% | 93.3% |
| ChartQA | 1,011 | 17.7% | 82.3% |
| WeMath2.0-Pro | 841 | 49.1% | 50.9% |

Direct visual dependence therefore differs substantially in these frozen
populations. OCR-heavy TextVQA and structured ChartQA have far higher V+
fractions than GQA or WeMath. However, these are not natural benchmark
prevalence estimates: the first three sources were constructed from
historically balanced pools, and the current outcome need not equal the
historical label used for selection.

## Minimum positive direct visual access within V+

For each V+ sample, `b_i` is the minimum positive ON count among routes found
in its matched prefix.

| Dataset | V+ N | Mean `b_i` (95% image-cluster CI) | Median | Q25--Q75 | P10--P90 | Mean removable | Cheaper than FULL |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA | 1,145 | 8.66 [8.53, 8.79] | 8 | 7--9 | 7--11 | 19.34 | 99.7% |
| TextVQA | 965 | 10.74 [10.57, 10.90] | 10 | 9--12 | 8--14 | 17.26 | 99.4% |
| ChartQA | 832 | 12.47 [12.21, 12.75] | 12 | 10--14 | 9--16 | 15.53 | 96.6% |
| WeMath2.0-Pro | 428 | 13.86 [13.18, 14.61] | 12 | 9--16 | 8--28 | 14.14 | 86.7% |

The amount ordering is clear in the mean. WeMath is heavy-tailed: its median
equals ChartQA's, but its P90 is 28 and 13.3% of V+ records have no cheaper
route in the finite prefix. This is why mean, median, and quantiles must all be
retained.

### Budget feasibility

| Dataset | C=4 | C=8 | C=12 | C=16 | C=20 | C=24 | C=28 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA | 0.4% | 53.2% | 95.9% | 99.2% | 99.7% | 99.7% | 100% |
| TextVQA | 0.1% | 16.3% | 82.1% | 97.9% | 99.4% | 99.4% | 100% |
| ChartQA | 0.1% | 8.8% | 59.1% | 91.6% | 96.3% | 96.6% | 100% |
| WeMath2.0-Pro | 0.0% | 14.3% | 53.7% | 77.8% | 85.7% | 86.7% | 100% |

The curves differ materially: GQA rises much earlier, TextVQA is intermediate,
and ChartQA/WeMath need larger budgets. The abrupt WeMath mass at C=28 reflects
the FULL-only tail, not evidence that all 28 layers are individually necessary.

## V+ placement across depth

All exact-minimum routes were retained. Route metrics were averaged within a
sample first and then across samples, so samples with many routes do not
dominate.

| Dataset | Centroid (95% CI) | First ON | Last ON | Early | Middle | Late | ON segments | Late re-entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GQA | 0.492 [0.487, 0.497] | 1.76 | 24.86 | 32.5% | 35.6% | 31.9% | 6.21 | 96.9% |
| TextVQA | 0.488 [0.484, 0.492] | 0.73 | 25.71 | 33.8% | 33.7% | 32.5% | 7.09 | 98.7% |
| ChartQA | 0.473 [0.469, 0.478] | 0.56 | 25.55 | 35.3% | 34.7% | 30.0% | 7.12 | 95.5% |
| WeMath2.0-Pro | 0.476 [0.469, 0.483] | 0.76 | 25.55 | 35.1% | 34.7% | 30.2% | 6.05 | 84.5% |

ChartQA and WeMath are modestly earlier than GQA, and TextVQA has more evenly
distributed access. But the maximum centroid gap is 0.019, equivalent to 0.51
of 27 normalized layer intervals. Coarse third fractions differ by only a few
percentage points.

### Exact-min / min+2 / min+4 sensitivity

Mean normalized centroids were:

| Dataset | min+0 | min+2 | min+4 |
|---|---:|---:|---:|
| GQA | 0.492 | 0.493 | 0.492 |
| TextVQA | 0.488 | 0.490 | 0.492 |
| ChartQA | 0.473 | 0.474 | 0.478 |
| WeMath2.0-Pro | 0.476 | 0.481 | 0.481 |

The coarse ordering is reasonably stable, while profiles converge as the route
set broadens. Pairwise cosine similarity rises from 0.982--0.996 at exact min
to 0.994--0.999 at min+4. Exact-min L1 distances range 1.54--5.20, but much of
that scale reflects the different total ON mass. At min+4, L1 distances shrink
to 0.94--4.14.

### Amount-adjusted placement

Descriptive linear models used GQA as the reference and bootstrapped image
groups within dataset. At exact minimum:

- TextVQA centroid coefficient: -0.0046, CI [-0.0111, 0.0019];
- ChartQA: -0.0194, CI [-0.0262, -0.0127];
- WeMath: -0.0169, CI [-0.0273, -0.0064].

ChartQA and WeMath remain earlier after controlling minimum ON, and the signs
remain at min+2 and min+4. The effects remain small: the exact-min centroid
model has R²=0.012. Fixed amount bins show the same modest pattern in populated
cells. Late-fraction differences are roughly one to two percentage points.

Late-reentry models show dataset coefficients even after controlling amount,
but the raw rate is saturated near one in most non-FULL-only cells and drops to
zero by definition for the uninterrupted ALL-ON mask. This makes it a fragile
summary of shape when amount differs strongly. It is preserved as descriptive
evidence, not promoted to a strong task-specific placement claim.

## Native token and format controls

Native image processing creates large input-scale differences:

| Dataset | All mean / median / P90 / max visual tokens | V+ mean / median / P90 / max |
|---|---:|---:|
| GQA | 265 / 234 / 324 / 1,452 | 265 / 234 / 324 / 1,452 |
| TextVQA | 956 / 925 / 1,073 / 1,369 | 941 / 888 / 999 / 1,369 |
| ChartQA | 522 / 580 / 638 / 1,798 | 431 / 630 / 630 / 1,419 |
| WeMath2.0-Pro | 2,515 / 900 / 8,453 / 11,342 | 2,119 / 814 / 7,276 / 10,165 |

Controlling `log1p(visual_tokens)` did not remove the V+ amount differences.
Relative to GQA, adjusted minimum-ON coefficients were +2.13 for TextVQA
[1.68, 2.58], +3.82 for ChartQA [3.51, 4.13], and +5.25 for WeMath [4.42,
6.15]. The model is descriptive and has R²=0.214; it does not establish a
causal task effect.

Adding visual-token count to the centroid model also leaves the small ChartQA
and WeMath shifts: -0.0192 and -0.0161, respectively. TextVQA remains
indistinguishable from GQA. Coarse visual-token bins are saved for audit.

Other observed context differences are substantial:

- mean prompt text tokens: GQA 40.7, TextVQA 39.4, ChartQA 45.3, WeMath 103.7;
- mean reference whitespace length: 1.04, 1.77, 1.19, and 1.39 words;
- GQA uses normalized exact matching; TextVQA uses consensus scoring at 0.5;
  ChartQA uses relaxed accuracy; WeMath uses tagged MathRuler answers;
- each record contains one image, but unique-image reuse differs by dataset.

The raw caches do not contain authoritative target-answer token IDs, so answer
length is documented as reference whitespace tokens and characters rather than
manufactured tokenizer counts. No optional within-GQA type analysis was run:
the raw authoritative cache does not bind a sufficiently clear official
question-type taxonomy for the planned categories.

## A+ correction analysis

A+ is kept separate because it measures an alternative route that corrects a
FULL failure, not a sparse correctness-preserving route.

| Dataset | FULL-wrong N | A0 | A+ | No correction | Mean min correcting ON (95% CI) | Median |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 2,000 | 7.6% | 59.6% | 32.9% | 10.05 [9.90, 10.19] | 10 |
| TextVQA | 966 | 3.6% | 67.5% | 28.9% | 12.14 [11.89, 12.39] | 12 |
| ChartQA | 989 | 3.5% | 71.8% | 24.7% | 12.40 [12.18, 12.60] | 12 |
| WeMath2.0-Pro | 3,703 | 4.4% | 34.1% | 61.5% | 10.58 [10.40, 10.77] | 10 |

The A+ ordering is not the V+ ordering: TextVQA and ChartQA corrections require
more ON layers than WeMath among corrections actually found, while WeMath has
far lower correction discovery. All-available legacy extensions modestly raise
A+ counts but leave medians and placement essentially unchanged. This argues
against reducing the result to a single easy-to-hard task scale.

## Direct answers to the plan questions

1. **Direct dependence:** It differs strongly in the frozen populations:
   TextVQA 93.3% and ChartQA 82.3% V+ versus GQA 57.3% and WeMath 50.9%.
2. **Minimum positive ON:** Yes. V+ means are 8.66, 10.74, 12.47, and 13.86
   layers in GQA, TextVQA, ChartQA, and WeMath order.
3. **Budget curves:** Yes. GQA reaches 95.9% by 12 ON layers, versus 82.1%,
   59.1%, and 53.7% for TextVQA, ChartQA, and WeMath.
4. **Placement:** Only modestly. ChartQA/WeMath routes are slightly earlier,
   but broad profiles are highly shape-similar.
5. **After amount control:** Small ChartQA and WeMath centroid shifts remain,
   but explain little variance and are not a strong placement regime.
6. **Route-set sensitivity:** Amount is unchanged; coarse placement is stable
   while profiles become still more similar from exact min to min+4.
7. **Search budget:** No for V+ and primary A+: matching 200/400 removes the
   unequal 600-simulation opportunity. The all-search A+ tail changes counts
   modestly, not conclusions.
8. **Input scale:** Visual-token and prompt-scale differences are large, but
   visual-token adjustment does not remove amount differences. It reinforces
   the observational, noncausal interpretation.
9. **A+:** Correction discovery is highest for ChartQA and lowest for WeMath;
   minimum correcting ON is highest for ChartQA/TextVQA, not WeMath.
10. **Task regime versus within-WeMath difficulty:** Broad task family explains
    V+ amount and dependence more clearly than official difficulty did within
    WeMath. It does not explain placement strongly, and cannot be interpreted
    as a scalar task-demand axis.
11. **Strongest remaining interpretation:** Different task/data/scoring regimes
    induce different probabilities of needing direct vision and different
    discovered access budgets, while the depth shape of successful access
    remains broadly shared and individual route identities remain
    heterogeneous.
12. **Classification:** amount differs robustly; placement differences are
    small relative to their high profile similarity.

## Artifact index

All required tables are in `outputs/cross_dataset_visual_access_v1/`.
`analysis_manifest.json` records source/output hashes, matched budgets,
bootstrap configuration, and execution semantics. Figures are sample-balanced
and stored under `figures/`. No model inference, MCTS, training, or new route
generation was performed.

Outcome C — task family affects visual-access amount, not placement
