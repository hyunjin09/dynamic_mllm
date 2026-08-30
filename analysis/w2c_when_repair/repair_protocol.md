# W2C Route-Cache Repair Protocol

Frozen before any smoke or full repair outcome was observed.

## Authority

- Source plan: `plans/w2c_when_label_repair_plan.md`
- Source-plan SHA-256: `19d750c7acca5caaf37a85438f432e566dd980cbc29ddb1e6cf7d3c8e0c23e88`
- Executor implementation commit: `4a700e8c34fa9b7e82980aa63bfc848527d89390`
- Parent online config: `analysis/persistent_corrective_supervision/online_config.yaml`
- Source W2C manifest: `analysis/persistent_corrective_supervision/training_manifest.jsonl`
- Boundary manifest: `analysis/persistent_corrective_supervision/boundary_manifest.jsonl`
- Physical source-label root: `datasets/mcts_labels_4action/sequential_branching_v1/full/records`
- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Existing executor contract: `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`
- Action order: `IGNORE`, `READ_ONLY`, `WRITE_ONLY`, `FULL`.
- Evaluator/answer-normalization contract: source `metric_name`, `answer`,
  `all_answer_norms`, and `correctness_threshold`, implemented by the frozen
  evaluator code hashes in the config.

Only the 640 W2C records in the authoritative matched population
are repaired (512 train,
128 validation). C2C records and all
original cache files remain unchanged.

## Iterative repair

For each sample, compute the maximal all-FULL prefix over the versioned correct
route cache. At its next non-FULL boundary:

1. force FULL and execute every deduplicated same-sample compatible known suffix;
2. if any are correct, add every correct route, record a verified CONTINUE
   state, recompute the maximal prefix, and repeat;
3. if known suffixes are exhausted, construct every one-action mutation strictly
   after the boundary from those FULL-insertion routes;
4. select at most 96 candidates with
   deterministic layer-stratified round-robin ordering and stable within-layer
   hashing (seed 20260830);
5. add every correct route and repeat, or emit
   `FULL_UNRESCUED_UNDER_BUDGET` when none is correct.

The 96-route budget is fixed because it exhausts every possible one-position
mutation of one 27-layer suffix (maximum 81) and retains limited diversity for
multi-suffix states. It initially selects 34861
of 55001 unique variants and fully exhausts the
one-edit population for 472/640
states. No two-edit sweep or four-action MCTS is admitted. The existing MCTS is
binary-only; adapting it would introduce an unvalidated search method.

`FULL_UNRESCUED_UNDER_BUDGET` means only that no rescue was found in this frozen
known-suffix plus one-edit neighborhood. It is not global impossibility.

## Parallelism, caching, and failure handling

- Four direct GPUs, one process per GPU; no Slurm.
- Deterministic cost-balanced static shards from the frozen manifests.
- Atomic per-sample records under `/mnt/hyemin/qwen_train_eval/outputs/w2c_when_repair_v1`.
- Cache key: UID + complete 28-action route + model revision + executor/code
  contract. Exact routes execute at most once within a repair record; completed
  smoke samples are reused by the full repair.
- Runtime/evaluator failures and a correct all-FULL route for a W2C sample are
  quarantined as `UNRESOLVED`, never silently labeled.

## Smoke gate

The 12-sample smoke has four records per dataset, six
previously cache-incomplete and six previously bounded-invalid states, six
single- and six multi-suffix states, and all early/middle/late depths. It must
verify all ten checks from the plan, including old-route replay correctness,
iteration, bounded-search ordering, cache deduplication, resume consistency,
and deterministic output. Any quarantine or replay failure stops the full run.

## Post-repair decision

Rebuild `CONTINUE`, `DEVIATE_CANDIDATE`, and `UNRESOLVED` labels without
overwriting the original cache. Re-audit repaired validation candidates under
the identical cached bounded contract with 10,000 UID-group bootstrap draws
(seed 20260831).

Stage-1 gate training is not part of this phase. A future gate is eligible only
if there are at least 512 repaired train and 128 repaired validation DEVIATE
candidates, no major executor inconsistency, and the repaired label audit is
acceptable under this explicitly bounded semantics. Stop after Q1--Q6.

## Independent challenge

One read-only research reviewer ranked this strategy above adapting binary MCTS
or using known suffixes alone (confidence medium). Its strongest objection is
preserved: residual labels establish only local bounded non-rescue, not global
necessity.
