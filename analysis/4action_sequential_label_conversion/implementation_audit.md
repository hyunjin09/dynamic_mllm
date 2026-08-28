# Exact Sequential Four-Action Label Conversion: Implementation Audit

## Approved contract

- Plan: `plans/4way_labeling_3.md`.
- Source inventory: 12,278 samples / 545,531 positive binary routes across GQA, TextVQA, ChartQA, WeMath2.0 Standard, and WeMath2.0 Pro.
- Unified executor: the previously validated complete-route four-action Qwen2.5-VL executor; no independent attention implementation was introduced.
- W→C order: fixed early-to-late over the original binary-OFF positions.
- W→C decision: try FULL; if it fails, execute READ_ONLY and WRITE_ONLY; retain every correct partial branch; retain IGNORE only when neither partial branch is correct.
- C→C decision: mechanical ON→FULL and OFF→IGNORE only.
- Search exclusions: no binary/four-action MCTS, beam, branch cap, margin ranking, cost ranking, canonical selection, or independent per-layer composition.

## Implementation boundaries

- Exact policy core: `tools/research_analysis/four_action/sequential_label_conversion.py`.
- Isolated execution contract/queue: `tools/research_analysis/four_action/sequential_label_jobs.py`.
- GPU runner: `experiments/run_sequential_four_action_label_conversion.py`.
- Output root: `datasets/mcts_labels_4action/sequential_branching_v1/`; original binary labels and all prior conversion outputs are unchanged.
- Every complete four-action route is cached by its exact 28-action tuple within a sample. Image preprocessing, unified FULL, and model replicas are reused.
- Atomic per-sample JSON records and SHA-256 sidecars make full execution resumable. The full run uses a launch-scoped dynamic queue.

## Independent review and failure replacement

The required read-only research review rejected reuse of the old conversion core because it implemented two-order purification, margin/cost selection, beam pruning, and one-route canonical output. Only the validated executor/runtime and general scheduling machinery were retained. Jobs 1609/1610 were cancelled after the old beam gate had already become impossible to pass; their partial records remain historical provenance.

## Prospective validation

- Synthetic truth-table, branch-context, C→C, cache, deduplication, topology, contract, runner, smoke-audit, and aggregate-analysis tests are present under `tests/test_sequential_four_action_*`.
- Complete active repository test gate before submission: 424/424 passed.
- Smoke topology is frozen at 8 GPUs × 1 process/GPU for exactly 8 samples.
- Full topology is frozen at 8 GPUs × 2 processes/GPU with a shared atomic queue.
- The dependent full wrapper audits the completed smoke, records a smoke-derived compute estimate, and starts full inference only if every semantic/integrity gate passes.
