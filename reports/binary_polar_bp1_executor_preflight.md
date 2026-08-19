# Binary POLAR BP-1 Executor Preflight

Date: 2026-08-09

Decision: **FAIL — training remains blocked**.

## Scope

BP-1 used the frozen, outcome-blind 16-record fixture manifest: two records
from each easy/hard cell of ChartQA, DocVQA, GQA, and TextVQA. The pinned
runtime was Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, Transformers 5.3.0, BF16, and
SDPA. No predictor training or held-out evaluation was run.

## Confirmed observations

- Split/scatter reconstruction was exact on 16/16 fixtures.
- The compact text-only OFF layer matched its direct compact-text oracle
  exactly on 16/16 fixtures, and bypassed visual rows were unchanged on 16/16.
- Repeated all-ON and arbitrary-mask executions were deterministic on 16/16.
- Fresh all-ON greedy generation matched native generation and the cached
  all-ON tokens on 16/16 fixtures.
- Cached all-OFF token IDs reproduced on 14/16 fixtures.
- A cached best mask existed on 12 fixtures; 8/12 reproduced and 4/12 did not.
- The original cache-length comparator inspected a cache after autoregressive
  decode had mutated it. Its excess length equaled the generated-token count.
  A bounded one-record diagnostic changed the comparator to use immutable
  prefill layer statistics and obtained exact expected lengths at all 28
  layers for all-ON, all-OFF, and the cached best mask.
- The same diagnostic showed that all-ON native-logit parity still genuinely
  failed: maximum absolute error was 3.25 over all rows, 3.25 over visual rows,
  0.75 over text/control rows, and 0.25 at the final prompt text row. The
  frozen tolerance is 0.005.
- That diagnostic's cached best route also disagreed at the first generated
  token (`50170` fresh versus `33548` cached), although all-ON and all-OFF
  matched their cached sequences.

## Interpretation and failure classification

- **Objective/optimization failure:** not observed in BP-0A. The exact weighted
  complete-mask NLL matched an independent formula and could concentrate on a
  coherent valid mask. This was only an implementation sanity check, not
  evidence about dataset-scale optimization.
- **Predictor generalization failure:** not evaluated; no predictor was trained.
- **Incomplete or non-reproducing valid-mask cache:** unresolved but directly
  relevant. Some cached routes do not reproduce under the current executor,
  so membership in the cached valid set cannot yet be treated as executable
  correctness supervision.
- **Binary-factorization limitation:** not evaluated. BP-0 remains a
  label-structure diagnostic and is not the reason for this stop.
- **Executor/native-path mismatch:** confirmed at the output level; the exact
  internal cause is **unknown**. Explicit-mask/kernel-path numerical behavior
  is a possible explanation, but has not been established.

The observations rule out stochastic nondeterminism, Transformers-version
skew, split/scatter corruption, OFF bypass corruption, and the cache-length
comparator as the sole cause. They do not distinguish a porting mismatch from
another executor, kernel, or cached-label provenance difference.

## Gate decision and smallest follow-up

The approved BP-1 gate requires all-ON native logits within 0.005 and exact
cached generated-token reproduction. Both conditions fail. The smallest
protocol-preserving follow-up is a focused executor-equivalence repair against
the supplied `reference/binary_action_qwen` path on one non-reproducing mixed
mask, followed by a fresh frozen 16-fixture BP-1 rerun. Training must not start
until that rerun passes. The POLAR-style canonical representation is not
implicated and remains deferred.

## Evidence

- Frozen fixtures: `outputs/binary_polar/preflight/executor_fixtures_v1.json`
- Full 16-fixture result: `outputs/binary_polar/preflight/executor_preflight_v1.json`
- One-record diagnostic: `outputs/binary_polar/preflight/executor_diagnostic_v1.json`
- Objective gate: `outputs/binary_polar/preflight/bp0a_exact_set_nll_v1.json`

