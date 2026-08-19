# Binary POLAR BP-1 Input-Contract Repair

Date: 2026-08-10

Decision: **INPUT GEOMETRY REPAIRED; FROZEN BP-1 STILL FAILS; TRAINING BLOCKED**.

## Authorized action

The action was limited to repairing the known label-runtime preprocessing
mismatch and rerunning the unchanged 16-fixture BP-1 executor suite. Fixtures,
thresholds, action semantics, cached token IDs, and the exact-cache pass rule
were not modified. Predictor training was permitted only if BP-1 passed.

## Repair

`experiments/binary_executor_preflight.py` previously opened each original
image and passed it to the processor without consuming the record-specific
`max_image_tokens` field. The repaired `prepare` path now:

1. constructs the same image/text chat content as the label runtime;
2. converts a positive image-token budget to
   `max_pixels = max_image_tokens * 28 * 28`;
3. passes the original image and `max_pixels` to the pinned Transformers 5.3.0
   Qwen processor, which performs factor-aligned smart resize;
4. requests multimodal token-type IDs explicitly.

No manual area-only resize is used. A regression test was added at
`tests/test_binary_executor_preflight.py`.

## Pre-run validation

The local contract regression passed. With the pinned processor, all four
DocVQA fixtures now exactly match cached geometry:

| Fixture | Visual rows | Full prompt rows |
|---|---:|---:|
| `ba41adcb...` | 2,028 | 2,061 |
| `07032c11...` | 2,028 | 2,063 |
| `7c4975fd...` | 1,989 | 2,026 |
| `9492aaf4...` | 2,040 | 2,081 |

The formerly unresolved 1,989-row path is therefore repaired without adding
or guessing a `qwen_vl_utils` version.

## Unchanged BP-1 rerun

- Slurm job: `99737`
- GPU: A6000, node03
- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Transformers: 5.3.0
- PyTorch: 2.6.0+cu124
- Precision/backend: BF16/SDPA
- Fixture manifest: unchanged `executor_fixtures_v1.json`
- Full-logit tolerance: unchanged `0.005`
- Result: `outputs/binary_polar/preflight/executor_preflight_v3.json`
- Result SHA-256: `901f10dd9147169d5d7bbf77840cd82903a9358e5ea664fe2413d0793baa19b7`

### Gate results

| Check | Result |
|---|---:|
| Records completed | 16/16 |
| Cached prompt geometry | 16/16 |
| Split/scatter identity | 16/16 |
| OFF compact-text oracle | 16/16 |
| OFF visual bypass | 16/16 |
| Deterministic FULL repeats | 16/16 |
| Deterministic arbitrary-mask repeats | 12/12 |
| Route cache lengths | all routes, 16/16 |
| ALL-ON/native maximum logit error | `0.0` |
| ALL-ON/native greedy equality | 16/16 |
| Cached ALL-ON generated IDs | 15/16 |
| Cached ALL-OFF generated IDs | 15/16 |
| Cached best-mask generated IDs | 10/12 |
| Complete fixture rows passing BP-1 | 12/16 |

The repair improved complete rows from 11/16 to 12/16, ALL-OFF cache matches
from 14/16 to 15/16, and best-mask matches from 8/12 to 10/12. It also changed
the DocVQA `ba41...` cached mismatch from its best mask to a punctuation-only
ALL-ON difference. The frozen exact-token gate nevertheless remains failed.

## Remaining mismatches

| Fixture/route | Repaired replay | Cached output | Behavioral status |
|---|---|---|---|
| ChartQA `149296...`, best mask | `Germany` | `India` | valid -> invalid |
| DocVQA `ba41...`, ALL-ON | `B-3.` | `B-3` | valid -> valid |
| DocVQA `7c497...`, best mask | `200; 200 people will be allowed to attend the workshop.` | `limited to the first 200.` | valid -> invalid |
| TextVQA `260`, ALL-OFF | `123456789` | `10` | invalid -> invalid |

The behavioral classification is a bounded validity diagnostic using the
existing cached answers and official label-generation evaluators. It did not
change the BP-1 criterion.

## Interpretation and stop

Confirmed observations:

- the preprocessing/token-layout mismatch is repaired;
- executor identities, deterministic behavior, cache geometry, and native
  ALL-ON parity pass;
- exact cached token reproduction still fails;
- two cached positive masks are invalid under the repaired target executor.

The exact remaining source of token drift is **unknown**. Incomplete source GPU,
driver/kernel, and utility provenance remains a plausible explanation, but it
is not established as the precise cause.

Deleting only the four known failures would be outcome-dependent filtering and
would not estimate label drift elsewhere. Under the unchanged plan, BP-1 has
not passed, so BP-2 and predictor training remain blocked.

Independent review refined the prospective fallback: do not automatically
revalidate the complete 184,785-mask cache. First precommit an
image-group-disjoint stratified cohort and a coverage floor, then uniformly
revalidate every cached mask admitted for that cohort and rebuild its valid
sets and route weights. This preserves target-executor label validity at lower
cost, but requires explicit protocol approval before execution.
