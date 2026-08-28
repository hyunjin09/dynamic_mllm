# Four-Action Implementation Audit

Date: 2026-08-23

## Existing implementations

Two relevant implementations were present:

1. `interventions/four_state.py` and `interventions/read_path.py` implement
   historical four-state experiments by decomposing full eager attention and
   subtracting the visual-value contribution from nonvisual rows. Their action
   names match IGNORE/READ_ONLY/WRITE_ONLY/FULL, but their READ-off operation
   retains full-attention normalization and therefore does not exactly equal
   the current binary executor's compacted text/control-row VISUAL_OFF action.
2. `binary_policy/executor/` implements the authoritative regenerated-label
   semantics. FULL scatters text/control and visual rows into native order and
   runs the unchanged decoder layer. VISUAL_OFF runs the same layer only on
   compacted text/control rows and carries incoming visual rows unchanged.

All six core binary-executor files match the hashes frozen in the transferred
GQA/TextVQA/ChartQA label contract. The binary implementation was therefore
retained unchanged.

## Minimal extension

`binary_policy/executor/four_action.py` composes four local actions from the
full-row materialized-mask and compacted-row primitives. At the target layer,
every action performs both decoder calls in the same order from clones of the
same unified-FULL pre-layer state. The action chooses only the returned text
rows, visual rows, and cache:

| Action | Text/control output | Visual output | Target-layer decode cache |
|---|---|---|---|
| IGNORE | compacted text/control call | incoming visual rows | text/control K/V only |
| READ_ONLY | native full-row call | incoming visual rows | full text/control+visual K/V |
| WRITE_ONLY | compacted text/control call | native full-row visual output | text/control K/V only |
| FULL | native full-row call | native full-row visual output | full text/control+visual K/V |

The full-row call supplies READ-enabled text and WRITE-enabled visual rows; the
compacted call supplies READ-disabled text/control rows and the READ-disabled
cache. All suffix layers execute the same unified FULL path. Unified FULL
prompt states and caches are captured once per sample and reused across the 28
local factorial comparisons.

M11 is unified FULL, M10 is READ_ONLY, M01 is WRITE_ONLY, and M00 is IGNORE.
Native Qwen FULL is never used in a factorial contrast. It is retained only for
the frozen cohort, semantic validation, and a separately reported signed and
absolute implementation-drift distribution.

Teacher-forced scoring and deterministic greedy generation clone the
heterogeneous per-layer prompt cache, so scoring or generation of one branch
cannot mutate another branch.

## Scoring contract

- Primary and FULL-wrong controls use length-normalized teacher-forced
  `S(correct) - S(FULL generated wrong answer)`.
- GQA uses its canonical answer.
- TextVQA groups references by the official EvalAI normalization, retains only
  evaluator-valid normalization classes, chooses a deterministic raw-string
  representative before outcomes, and uses the maximum per-token likelihood
  across valid classes.
- The FULL-wrong target is the frozen raw generated answer string.
- FULL-correct vision-required controls use `S(correct)` because there is no
  FULL wrong answer contrast.
- Greedy outputs are scored with the frozen evaluator semantics and threshold.

## Static validation

- 61 focused four-action, binary-executor, cohort, eligibility, parallelism,
  ramp-monitor, target, scoring, trajectory,
  and analysis tests pass.
- IGNORE exactly matches the binary compacted layer primitive in synthetic
  tests.
- READ_ONLY text equals FULL text and its visual output equals the incoming
  visual state.
- WRITE_ONLY text equals IGNORE text and its visual output equals FULL visual.
- Every target action makes exactly two decoder calls and all target branches
  share bit-identical pre-layer text and visual states.
- Target-layer cache rows depend on READ, not WRITE.
- A local branch reuses the captured target pre-state and returns every suffix
  layer to FULL.
- Greedy and teacher-forced continuation paths clone caches rather than mutate
  reusable prompt state.

## Model-scale validation status

Historical jobs `1424`, `1482`, and `1483` established the action semantics but
showed that native-maskless versus materialized-mask BF16 score drift reached
about 0.125, which is too large to mix in the causal estimand. That evidence is
preserved under `preflight__bf16_diagnostic_v1/` and `preflight__bridge_v1/`.

The corrected unified preflight, 8-example all-worker 28-layer smoke, and
56-example validation passed all current semantic and structural gates in jobs
`1485`--`1487`. The chain gates on generated token/answer,
evaluator correctness, visual bypass/update, READ access/cache geometry,
identical branch pre-states, identical non-target FULL execution, determinism,
and cache correctness. Native/unified continuous drift is reported but is not a
gate or effect threshold. Across the eight shared validation examples, every
native/unified FULL and unified IGNORE/old-binary semantic comparison matched.
Native/unified absolute margin drift nevertheless reached 0.1875 and remains a
separate diagnostic. One validation sample's current native/unified FULL token
sequence differed from its transferred historical FULL anchor, but both old and
current answers were evaluator-wrong. The exact historical token comparison is
therefore reported as provenance-only; current native/unified token identity and
cohort correctness remain hard gates.

The first production launch (`1497`) exposed one transferred candidate whose
current native and unified FULL answers agreed and were correct although its
matched-cache FULL answer was wrong. A frozen current unified-FULL eligibility
stage now gates cohort membership without altering cached ALL-OFF or correcting-
route provenance. Job `1505` evaluated all 4,890 candidates on eight workers
with zero failures and retained 1,880 primary, 868 no-correction, and 2,084
vision-required samples. A two-replica/H100 ramp later passed correctness and
worker-layout gates but achieved only 0.8004x the one-replica sample throughput.
Production therefore resumed with one worker per H100 while retaining all eight
parallel H100 workers and every completed append-only result. No primary causal
outcome has been claimed yet.
