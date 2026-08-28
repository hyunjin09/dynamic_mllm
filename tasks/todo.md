# Exact Sequential Four-Action Label Conversion Tasks

## Task 1: Freeze the replacement contract

**Acceptance criteria:** the source manifest remains byte-identical; new output
and analysis roots are isolated; smoke/full topology and all relevant code,
config, model, and manifest hashes are bound.

**Verification:** contract tests reject changed inputs and prove no beam/cap
configuration exists.

## Task 2: Implement exact W2C sequential branching

**Acceptance criteria:** original OFF layers are processed early-to-late; FULL
short-circuits partials; both correct partials branch; neither correct partial
retains IGNORE; every branch is complete-route evaluator-correct; no branch is
ranked or pruned.

**Verification:** RED/GREEN synthetic truth-table and trajectory-context tests.

## Task 3: Preserve C2C and sample-level provenance

**Acceptance criteria:** replay-valid C2C stays mechanically FULL/IGNORE;
W2C/C2C semantics remain separate; every source route and deduplicated final
route retains provenance, action counts, scores, and evaluator output.

**Verification:** focused runner/dedup tests.

## Task 4: Add resumable smoke/full orchestration

**Acceptance criteria:** smoke uses 8 workers/8 GPUs; full uses 16 workers/8
GPUs; models load once; outputs are atomic/checksummed; resume skips completed
samples; full uses a dynamic shared queue.

**Verification:** queue, topology, contract, collision, and resume tests.

## Task 5: Pass the 8-sample smoke

**Acceptance criteria:** all five datasets and required W2C/C2C/route shapes
are covered; semantic, correctness, cache, dedup, checksum, worker, and resume
gates pass.

**Verification:** smoke audit reports `passed=true` before full inference.

## Task 6: Complete and analyze the full conversion

**Acceptance criteria:** all 12,278 samples and 545,531 source routes are
accounted for; every final label is evaluator-correct; required views, analyses,
plots, reports, and checksum ledger answer all plan questions.

**Verification:** full integrity audit, final report audit, and complete active
test suite pass.
