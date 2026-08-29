# Four-Action Router Collapse: Label and Supervision Audit

## Scope

This is a deterministic, CPU-only audit of the frozen GQA/ChartQA/TextVQA
four-action manifest and the exact online-router sampler. It was run after both
upfront Image+Question POLAR objectives and the online state-conditioned router
converged to essentially all-`FULL` deployment. No model training, inference,
label mutation, or scheduler action was performed.

The audit asks whether the collapse is explained by:

1. the exact all-`FULL` route shared by C2C samples;
2. action and prefix imbalance within W2C supervision itself;
3. teacher-forcing states that do not cover the deployed all-`FULL` rollout;
4. objective or checkpoint-selection behavior specific to the upfront models.

## Frozen population

| Split | W2C | C2C | Total |
|---|---:|---:|---:|
| Train | 2,397 | 3,548 | 5,945 |
| Validation | 356 | 510 | 866 |

The upfront POLAR data loader therefore used the natural train ratio, 40.3%
W2C to 59.7% C2C. Only the online router used exact 50:50 W2C:C2C sampling,
with equal sampling across GQA, ChartQA, and TextVQA.

## Direct observations

### C2C has a universal complete-route shortcut

- 3,501/3,548 train C2C samples (98.675%) contain the exact all-`FULL` route.
- No W2C sample contains the exact all-`FULL` route, as expected because native
  `FULL` is wrong for W2C.
- Removing only the exact all-`FULL` route would retain at least one non-`FULL`
  correct route for 3,513/3,548 C2C samples. The remaining 35 would become
  route-empty and require exclusion or separate preservation handling.
- C2C labels contain only `FULL` and `IGNORE`. This follows the approved label
  conversion: C2C binary routes were preserved mechanically and were not
  decomposed into READ versus WRITE supervision.

The user's universal-C2C-route hypothesis is therefore real. It is especially
relevant to a complete-route predictor because all-`FULL` is the only coherent
route shared across almost the whole C2C population.

### W2C supervision is also strongly FULL-heavy

The following values reproduce the exact planned ten-epoch online sampler:

| Route type | `FULL` teacher action | `IGNORE` | `READ_ONLY` | `WRITE_ONLY` |
|---|---:|---:|---:|---:|
| W2C | 76.782% | 7.357% | 9.589% | 6.271% |
| C2C | 56.687% | 43.313% | 0.000% | 0.000% |
| Combined 50:50 | 66.735% | 25.335% | 4.795% | 3.136% |

Set-valued valid-next-action frequency is more imbalanced than the 50:50
sample ratio suggests:

| Action | Valid at a sampled prefix | Only valid action at a sampled prefix |
|---|---:|---:|
| `FULL` | 72.880% | 55.360% |
| `IGNORE` | 32.913% | 20.729% |
| `READ_ONLY` | 10.591% | 3.107% |
| `WRITE_ONLY` | 6.279% | 2.216% |

Within W2C alone, `FULL` is valid at 80.505% of sampled prefix nodes and is the
only valid action at 65.261%. Removing all C2C would increase the sampled
teacher-action share of `FULL` from 66.7% to 76.8%, although it would also
remove C2C's direct negative pressure on READ/WRITE. Thus sample-level 50:50
balancing did not produce action-level or decision-level balance.

### Greedy deployment follows a poorly covered all-FULL prefix

- At the root, `FULL` is valid for 94.451% of W2C samples and 100% of C2C
  samples.
- For W2C, the longest valid all-`FULL` prefix has mean 14.586 layers, median
  15, P95 27, and maximum 27. A greedy policy can therefore remain locally
  valid for many layers before it must make a rare corrective deviation.
- The ten-epoch sampler visits a mean 6.51 distinct teacher routes per W2C
  sample despite a mean 30.15 available routes.
- 1,045/2,397 W2C samples (43.6%) are never teacher-forced through the route
  that reaches their latest valid all-`FULL`-prefix deviation boundary.

At those 2,397 mandatory W2C boundaries, `FULL` is invalid by construction and
the corrective valid-action coverage is concentrated exactly where the normal
sampler is weakest:

| Action | Valid at mandatory boundary | Sole valid action |
|---|---:|---:|
| `IGNORE` | 27.284% | 17.814% |
| `READ_ONLY` | 43.388% | 27.076% |
| `WRITE_ONLY` | 52.733% | 34.293% |
| `FULL` | 0.000% | 0.000% |

The boundaries span every layer 0--27 rather than one narrow depth band.

This matters because teacher forcing changes the hidden state after the sampled
expert route first deviates. A deployed router that keeps choosing `FULL` does
not reach those later teacher states. The current training loss contains no
on-policy correction for an invalid or unsupported all-`FULL` prefix.

The trained online model exhibits exactly this separation. At epoch 9, its
teacher-forced node predictions over 24,248 validation nodes were:

| `FULL` | `IGNORE` | `READ_ONLY` | `WRITE_ONLY` |
|---:|---:|---:|---:|
| 21,288 | 2,950 | 10 | 0 |

Yet free-running validation executed 24,247 `FULL` decisions and one `IGNORE`.
The improving teacher-node loss/Valid-Action@1 therefore did not imply that the
policy could leave its deployed all-`FULL` trajectory.

### The upfront objectives have additional collapse incentives

- Validation contains 507 C2C samples with a valid exact all-`FULL` route.
  Therefore all-`FULL` has 507/866 = 0.585450 top-1 valid-route coverage.
- The duplicated-BCE model achieved exactly 0.585450 top-1 coverage and
  all-`FULL` decoding at every epoch. The checkpoint order uses overall route
  coverage rather than balanced W2C rescue/C2C preservation, so this constant
  solution is an especially strong selection baseline.
- Under the exact sample-balanced route weights used by training, the
  per-sample categorical-marginal modal route is invalid for 2,057/2,397 W2C
  samples (85.8%) and 3,479/3,548 C2C samples (98.1%). This directly confirms
  that duplicated per-action BCE tends to combine incompatible route modes.
- Exact-set NLL avoids the duplicated-BCE marginal target, but its factorized
  predictor still found the one coherent route shared across almost all C2C
  inputs. Its selected all-`FULL` margin was stronger than BCE, so the collapse
  is not explained solely by the duplicated-BCE hybridization defect.

## Diagnosis

### Supported contributors

1. **Action/prefix imbalance.** READ and especially WRITE have very little
   positive valid-action mass, while `FULL` is valid or uniquely valid at most
   sampled nodes.
2. **C2C vocabulary asymmetry.** Half of online sample visits come from C2C,
   where READ/WRITE are never valid because those labels contain only binary
   `FULL`/`IGNORE` routes.
3. **Teacher-forcing/free-rollout exposure gap.** The exact sampler misses the
   latest all-`FULL` deviation boundary for 43.6% of W2C samples, and the model
   predicts some non-`FULL` actions on teacher states while remaining all-`FULL`
   in free rollout.
4. **Upfront checkpoint/objective geometry.** The all-`FULL` validation baseline
   exactly explains 0.585450 Hit@1, and duplicated BCE has severe per-sample
   route hybridization.

These are supported contributors, not proof that one is the sole cause.

### Still unresolved

- Whether the online router's state/features/head can discriminate mandatory
  W2C deviation states after those states are adequately sampled and weighted.
- Whether deleting the exact C2C all-`FULL` route alone is enough to break the
  shared complete-route mode.
- Whether a corrective policy learned with stronger W2C pressure can preserve
  C2C and external FULL-correct behavior.
- Whether search-derived alternative routes are too multimodal for the current
  shared router even after exposure is repaired.

## Candidate remedies

### Remove C2C entirely

Not recommended as the main fix. It removes the universal C2C shortcut and the
C2C negative pressure on READ/WRITE, but also removes all preservation
supervision. W2C itself remains 76.8% `FULL` at sampled teacher actions and
permits `FULL` at 80.5% of prefix nodes. A W2C-only run would therefore be a
bounded diagnostic, not a defensible final training population.

### Remove only the exact all-FULL route from C2C

This is a coherent, low-cost ablation of the user's hypothesis. It retains
3,513/3,548 C2C samples and removes the universal complete route. It is unlikely
to be sufficient by itself because it creates no READ/WRITE positives in C2C
and does not repair W2C action imbalance or rollout coverage. It also changes
C2C from "preserve correctness, where doing nothing is valid" to "choose a
known-correct nontrivial intervention," which must be reported explicitly.

### Repair prefix coverage and minority-action supervision while retaining C2C

The clean first change is narrower than a combined reweighting/on-policy
redesign: guarantee one teacher-forced visit to every W2C sample's latest valid
all-`FULL`-prefix deviation boundary while leaving the loss, action weights,
and C2C population unchanged. This directly raises coverage of the measured
boundary from 1,352/2,397 (56.4%) to 2,397/2,397. Removing the C2C all-`FULL`
route alone leaves W2C sampling unchanged at 56.4%.

Only after this isolated boundary-coverage test should a prospective combined
version consider:

1. boundedly rebalancing singleton `READ_ONLY`, `WRITE_ONLY`, and
   `FULL`-invalid nodes using train-split statistics;
2. exposing the router to its own prefixes through scheduled/on-policy rollout or
   a DAgger-like corrective prefix pass;
3. keeping C2C preservation as an explicit metric/constraint rather than assuming
   equal sample counts are sufficient;
4. selecting checkpoints by routed W2C rescue subject to a stated C2C preservation
   constraint, with node likelihood secondary.

Bundling all of these changes into the first retry would make any improvement
causally ambiguous.

## Smallest decision-changing next check

Before another full eight-GPU run, use a fixed small train/validation subset to
test whether the unchanged online router and unchanged loss can overfit and
free-run a deliberately covered set of W2C mandatory-deviation prefixes. Keep
C2C and its all-`FULL` route unchanged in this first capacity/coverage test.
Report singleton READ/WRITE recall, the first predicted non-`FULL` layer on the
all-`FULL` rollout, W2C rescue, and C2C preservation.

If the router cannot overfit these explicitly covered boundary states, inspect
the state features, structured action head, and gradient allocation before any
population reweighting. If it can, run removal of the exact C2C all-`FULL`
route as a separate matched ablation before adding action weights or on-policy
training. This ordering isolates the two supported mechanisms rather than
changing them simultaneously.

An independent read-only research review revised the initial bundled-pilot
ranking in favor of this isolated boundary-coverage test. Its reason was that
C2C-route removal does not alter the directly measured W2C coverage defect,
while a bundled coverage/weighting/on-policy arm would not identify which
change mattered. Confidence in this ordering is high for the next diagnostic,
not for the eventual full-training recipe.

No such pilot is authorized or launched by this audit.

## Evidence paths

- `outputs/four_action_polar/preparation_v1/manifest_v1.jsonl`
- `configs/four_action_polar_image_question_bce_v1.yaml`
- `configs/four_action_polar_image_question_nll_v1.yaml`
- `configs/four_action_online_router_v1.yaml`
- `four_action_online_router/supervision.py`
- `four_action_policy/losses.py`
- `outputs/four_action_polar/training_bce_v1/history.json`
- `outputs/four_action_polar/training_nll_v1/history.json`
- `outputs/four_action_online_router/training_v3/history.json`
- `reports/four_action_polar_action_collapse_audit_20260829.md`
- `reports/four_action_online_router_early_stop_20260829.md`
