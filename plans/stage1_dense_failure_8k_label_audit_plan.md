# Stage-1 Dense-Failure Prediction: 8K Label Audit Plan

## 1. Context and Decision

The previous research path was:

```text
W→C route-cache repair
→ rebuild WHEN labels
→ CONTINUE / DEVIATE gate
→ READ_OFF / WRITE_OFF / BOTH_OFF
```

Pause the W→C repair for now.

Before spending more effort repairing four-action route labels, first exploit the stronger and cleaner evidence already observed:

> Intermediate dense hidden states contain signal about whether the model's final dense all-visual-on answer will be wrong.

The next research direction is therefore:

```text
dense all-on trajectory
→ current-layer hidden state
→ predict final dense correctness/failure
```

with the intended future policy:

```text
uncertain / likely correct
→ stay FULL

high-confidence likely failure
→ invoke four-action intervention
```

However, **this plan stops before hidden-state extraction or gate training**.

The immediate task is only:

> **Audit and freeze the 8,000 dense correct/wrong labels that will supervise Stage 1.**

Do not resume W→C repair, train a gate, extract all hidden states, or run four-action intervention experiments in this phase.

---

## 2. Stage-1 Target Definition

The future Stage-1 predictor will estimate:

```text
P(dense all-on final answer is wrong | current layer hidden state)
```

The supervision must therefore come only from the final result of the corresponding dense all-on execution.

Define:

```text
y = 0  if dense all-on final answer is correct
y = 1  if dense all-on final answer is wrong
```

Do not use:

```text
W→C membership
route-cache membership
four-action correctability
MCTS result
sparse-router outcome
benchmark identity as a target
```

The Stage-1 target is **dense failure**, not intervention benefit.

---

## 3. Expected 8K Population

The expected population is:

```text
Total:   8,000
Correct: 4,000
Wrong:   4,000
```

Datasets:

```text
GQA
ChartQA
TextVQA
```

The first audit must verify that these counts are actually true in the authoritative source files.

Do not assume the 4,000 / 4,000 balance from prior notes.

---

## 4. Phase 0 — Locate the Authoritative Label Population

Identify the exact source files/manifests containing the 8K population.

Record:

```text
absolute paths
file names
record counts
file hashes
dataset field
UID field
image identifier
question identifier
ground-truth answer field
dense generated answer field
dense correctness field
```

Also identify the code/config that originally produced the dense all-on correctness labels if recoverable.

Write:

```text
analysis/dense_failure_stage1/8k_label_audit/source_inventory.md
```

Do not mutate the source labels.

---

## 5. Phase 1 — Population Count Audit

Produce the exact dataset × correctness table.

Required table:

| Dataset | Correct | Wrong | Total |
|---|---:|---:|---:|
| GQA | | | |
| ChartQA | | | |
| TextVQA | | | |
| **Total** | **4000 expected** | **4000 expected** | **8000 expected** |

Also report:

```text
missing labels
unknown correctness values
duplicate UID count
records with missing image/question/answer fields
```

The audit fails if the actual population cannot be reconciled with the expected 8,000 records.

Do not silently drop records to recover 4K/4K balance.

---

## 6. Phase 2 — Verify What “Correct / Wrong” Means

For the authoritative 8K population, determine exactly how the stored dense correctness label was produced.

Audit:

```text
model snapshot / revision
processor/tokenizer revision
prompt template
image preprocessing
dense executor
all-on action semantics
generation configuration
greedy vs sampling
max_new_tokens
temperature
top-p
use_cache
dtype
attention implementation
evaluator
answer normalization
hardware/runtime if recorded
```

The key question is:

> Does `correct/wrong` mean correctness of the same dense all-on execution contract that will later be used to extract hidden states?

Separate findings into:

```text
verified
inferred
unresolved
```

Do not assume compatibility merely because the same model name appears.

---

## 7. Phase 3 — Current-Runtime Dense Replay Parity Smoke

Because previous work revealed runtime/execution mismatches, perform a bounded dense-only replay check before trusting the 8K labels.

This check is **dense all-on only**.

Do not invoke any four-action route.

### 7.1 Sample size

Use at least:

```text
96 records
```

Preferred:

```text
128 records
```

Stratify by:

```text
GQA / ChartQA / TextVQA
correct / wrong
```

Aim for approximately equal representation where possible.

Use deterministic UID selection and save the subset before execution.

### 7.2 Replay contract

For each selected record:

1. run the current intended dense all-on executor;
2. generate the final answer;
3. evaluate correctness with the intended evaluator;
4. compare against the stored label.

Record:

```text
stored generated answer
current generated answer
stored correctness
current correctness
exact generated-token parity if cached tokens exist
normalized-answer parity
correctness parity
```

The future hidden-state extraction must use this same current dense execution contract if the audit passes.

---

## 8. Replay Metrics

Report:

```text
correctness parity rate
generated-answer parity rate
exact-token parity rate if available
correct→wrong flips
wrong→correct flips
per-dataset mismatch rate
```

Required table:

| Dataset | Audited | Correctness matches | C→W flips | W→C flips |
|---|---:|---:|---:|---:|
| GQA | | | | |
| ChartQA | | | | |
| TextVQA | | | | |
| Overall | | | | |

If mismatches exist, list every mismatched UID.

---

## 9. Label Authority Decision

After the replay smoke, assign one of three outcomes.

### A. STORED_LABELS_AUTHORITATIVE

Use only if:

```text
stored dense correctness
and
current intended dense runtime
```

are sufficiently consistent under the frozen audit.

Then preserve the existing 8K labels.

### B. REBUILD_LABELS_FROM_CURRENT_DENSE_RUNTIME

Use if the stored label population is conceptually correct but the current hidden-state extraction runtime produces materially different dense outcomes.

In that case, do not mix:

```text
old correctness labels
+
new hidden states
```

Instead, the future Stage-1 dataset must derive hidden states and correctness labels from the same dense execution.

Do not perform the 8K rebuild in this audit unless separately authorized.

### C. EXECUTION_CONTRACT_UNRESOLVED

Use if the dense source/runtime cannot be reconciled.

Stop before Stage-1 training.

---

## 10. Phase 4 — Dataset Balance / Shortcut Audit

Even if the overall population is exactly 4K correct / 4K wrong, inspect whether correctness is confounded with dataset identity.

Report:

```text
P(wrong | GQA)
P(wrong | ChartQA)
P(wrong | TextVQA)
```

and the corresponding counts.

The future failure predictor should not succeed merely because one dataset contributes many more wrong examples.

If per-dataset class imbalance exists, record it now.

Do not rebalance the population in this audit.

---

## 11. Phase 5 — Duplicate and Leakage Audit

Before a future train/validation/test split, identify grouping keys that could cause leakage.

At minimum audit:

```text
duplicate UID
duplicate image ID
same image with multiple questions
duplicate question text
same source example under multiple records
```

Produce counts for:

```text
unique UIDs
unique image groups
image groups with >1 question
exact duplicate records
```

Determine the grouping key to use later for split construction.

Preferred future split rule:

```text
image-group disjoint
```

if the dataset metadata supports it.

Do not create the final split yet unless required only for auditing feasibility.

---

## 12. Phase 6 — Proposed Future Split Feasibility

Without yet training anything, verify that an image-group-disjoint split can approximately preserve:

```text
dataset proportions
correct/wrong balance
```

Target future split:

```text
Train: 6,400
Validation: 800
Test: 800
```

This phase should only demonstrate feasibility and generate a proposed deterministic split manifest if safe.

Requirements:

```text
no image-group overlap across splits
no UID overlap
approximately 50:50 correct/wrong per split
reasonable GQA/ChartQA/TextVQA proportions
```

If exact 6400/800/800 is incompatible with group boundaries, use the nearest valid group-disjoint counts and report them.

Do not sacrifice group disjointness to force exact counts.

---

## 13. Required Split Audit Table

If a proposed split is produced:

| Split | GQA C | GQA W | ChartQA C | ChartQA W | TextVQA C | TextVQA W | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | | | | | | | |
| Val | | | | | | | |
| Test | | | | | | | |

Also report:

```text
image-group overlap = 0 required
UID overlap = 0 required
```

The test split must be designated as untouched for later final Stage-1 evaluation.

---

## 14. Important Scientific Separation

Keep these concepts separate.

### Dense-failure label

```text
all-on final answer is wrong
```

This is the Stage-1 target.

### W→C label

```text
all-on wrong
+
some four-action route correct
```

This is a treatment/correctability property.

### Four-action route label

```text
which intervention trajectory is correct
```

This belongs to later Stage 2.

Do not use W→C or route labels in the Stage-1 label audit.

This separation is deliberate because the current goal is to exploit the previously observed hidden-state failure signal without inheriting route-cache incompleteness.

---

## 15. No Hidden-State Training Yet

This audit must stop before:

```text
extracting all 28-layer hidden states for all 8K samples
training a linear failure head
training an MLP failure head
adding learnable layer embeddings
threshold calibration
99/98/95% preservation analysis
four-action intervention
W→C repair
Stage-2 training
external evaluation
```

Those belong to the next authorized plan after the label audit is accepted.

---

## 16. Required Outputs

Suggested root:

```text
analysis/dense_failure_stage1/8k_label_audit/
```

Create:

```text
source_inventory.md
population_counts.csv
population_audit.md

replay_subset.json
dense_replay_results.jsonl
dense_replay_parity_report.md

dataset_balance.csv
duplicate_group_audit.md

proposed_split_manifest.jsonl
proposed_split_audit.md

label_audit_decision.md
```

If a proposed split cannot safely be created, do not fabricate `proposed_split_manifest.jsonl`; explain why.

---

## 17. Final Decision Report

`label_audit_decision.md` must explicitly answer:

### Q1 — Population

> Is the authoritative population exactly 8,000 records with 4,000 dense-correct and 4,000 dense-wrong?

### Q2 — Source contract

> What exact dense execution/evaluator contract produced the stored correctness labels?

### Q3 — Runtime compatibility

> Do stored dense labels agree with the current intended dense runtime on the 96-128 sample replay smoke?

### Q4 — Dataset shortcut risk

> Are correct/wrong labels strongly imbalanced by dataset?

### Q5 — Leakage risk

> What grouping key must be used to prevent train/validation/test leakage?

### Q6 — Split feasibility

> Can the 8K population support an approximately 6400/800/800 image-group-disjoint split while preserving class/dataset balance?

### Q7 — Authority decision

Choose exactly one:

```text
STORED_LABELS_AUTHORITATIVE
REBUILD_LABELS_FROM_CURRENT_DENSE_RUNTIME
EXECUTION_CONTRACT_UNRESOLVED
```

---

## 18. Stop Rule

Do not begin Stage-1 failure-predictor training until:

1. the 8K population is audited;
2. the dense correctness contract is identified;
3. the current-runtime replay smoke is complete;
4. the authoritative label decision is made;
5. split leakage risk is resolved.

The immediate research action is:

```text
pause W→C repair
→ audit the 8K dense all-on correct/wrong population
→ freeze the clean Stage-1 supervision contract
→ stop
```
