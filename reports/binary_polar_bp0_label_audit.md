# Binary Visual POLAR BP-0 Label and Representation Audit

Date: 2026-08-09

## Decision

`BP-0_FAIL_DIRECT_REPRESENTATION`

The full 4,000-record label audit completed without a schema failure, matched
the source audit exactly, and showed that the prospective image-group split is
feasible. The frozen direct-versus-canonical representation gate failed by a
large margin. Under the approved stage order, BP-1, manifest freezing, and
training remain blocked.

This audit used existing successful-route labels and binary-mask geometry only.
It did not run Qwen2.5-VL, load reference answers into a scoring routine,
generate answers, or train a model.

## Why this audit was run

The proposed direct predictor emits 28 independent ON/OFF logits and is trained
with probability mass assigned to any observed valid mask. Existing MCTS labels
contain many correlated, non-contiguous masks. BP-0 prospectively required the
direct factorization's empirical top-5 valid-set coverage to be no more than
0.02 below a canonical maximal-run factorization in the eight-cell macro
average and no more than 0.05 below it in any cell. This check prevents training
a direct head whose factorization is already poorly aligned with the observed
valid-route sets.

## Execution and integrity

- CPU-only Slurm job: `99718` (`binary_label_audit_bp0_20260809`).
- Source: `/home/hyemin/data/dataset/dynamic_mllm/mcts_v2` (read-only).
- Parsed records: 4,000 of 4,000.
- Invalid records: 0.
- Records with at least one valid route: 3,408.
- Records without a valid route: 592; these agree with the source audit and
  remain evaluation-only rather than becoming positive training examples.
- Total record-local deduplicated valid routes: 184,785.
- Source cell counts and positive-route counts: exact match in all eight cells.
- Audit artifact SHA-256:
  `48f36c9057b2a8e977720a8667b9ed854c8d95772d366cb6ea1154c66adecc63`.
- Checksum verification: passed.
- Binary-policy contract tests after the audit-tool repair: 12 passed.
- Audit-tool SHA-256:
  `ac56490e8ba08e8387db90baac6a38421a1f2f6544d5d45131373989b402ce1e`.

## Cell results

`Mean ON` and `mean transitions` describe all deduplicated valid masks. Coverage
is the fraction of positive-route records for which a top-5 mask derived from
the empirical factorized representation belongs to that record's valid set.

| Cell | Records | With valid route | Valid routes | Routes/positive record | Mean ON | Mean transitions | Direct top-5 | Canonical-run top-5 | Direct deficit | Cell gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ChartQA/easy | 500 | 500 | 27,257 | 54.51 | 15.17 | 13.28 | 0.028 | 0.636 | 0.608 | fail |
| ChartQA/hard | 500 | 352 | 5,928 | 16.84 | 14.64 | 13.62 | 0.176 | 0.207 | 0.031 | pass |
| DocVQA/easy | 500 | 500 | 23,022 | 46.04 | 15.78 | 13.16 | 0.022 | 0.682 | 0.660 | fail |
| DocVQA/hard | 500 | 421 | 9,413 | 22.36 | 15.29 | 13.57 | 0.131 | 0.145 | 0.014 | pass |
| GQA/easy | 500 | 500 | 65,961 | 131.92 | 14.36 | 13.41 | 0.004 | 0.908 | 0.904 | fail |
| GQA/hard | 500 | 300 | 9,270 | 30.90 | 14.07 | 13.54 | 0.107 | 0.170 | 0.063 | fail |
| TextVQA/easy | 500 | 500 | 35,901 | 71.80 | 15.28 | 13.41 | 0.012 | 0.768 | 0.756 | fail |
| TextVQA/hard | 500 | 335 | 8,033 | 23.98 | 14.92 | 13.50 | 0.137 | 0.158 | 0.021 | pass |

Macro coverage across the eight benchmark/difficulty cells was:

| Candidate budget | Direct independent bits | Canonical maximal runs | Direct deficit |
|---:|---:|---:|---:|
| top-1 | 0.0762 | 0.3360 | 0.2598 |
| top-5 | 0.0771 | 0.4593 | 0.3822 |
| top-10 | 0.0783 | 0.4903 | 0.4120 |

The frozen macro tolerance was 0.02 and the per-cell tolerance was 0.05. The
top-5 macro deficit was 0.3822, and five of eight cells failed their cell gate.
The failure is therefore not a numerical tie or a boundary case.

## Image-group split feasibility

The prospective deterministic split used image-content SHA-256 with seed
`20260809` and 75%/12.5%/12.5% fractions.

- Unique image groups: 3,824.
- Repeated image groups: 170.
- Maximum records per image group: 3.
- Records missing image SHA-256: 0.
- Cross-split image groups: 0.
- Smallest benchmark/difficulty/split cell: 52 records, above the prospective
  BP-2 minimum of 40.

This part of BP-0 passed. The split was audited but no compact training manifest
was frozen.

## Confirmed observation and interpretation

Confirmed: the independent direct-bit representation fails the prospectively
frozen in-sample valid-set coverage screen, while the canonical maximal-run
factorization covers substantially more observed valid sets at the same top-k
budgets.

Supported diagnosis: the observed valid masks contain strong cross-layer
structure that independent per-layer marginal decoding does not recover. The
high mean transition counts also show that this is not merely a collection of
single contiguous ON or OFF spans; the canonical boundary/action
factorization is benefiting from conditional adjacency structure.

Unresolved: this geometry-only audit does not establish learned held-out
generalization, executor correctness, or fresh-route correctness for the
canonical representation. It also does not prove that every possible direct
structured decoder would fail; it rejects the currently planned independent
direct head and decoder under the frozen gate.

## Stop and next decision

The plan's BP-0 stop condition has been reached. Do not execute BP-1, freeze the
BP-2 manifest, or start BP-3 training under the direct-head plan.

The smallest protocol-preserving next decision is a separately authorized
representation amendment centered on the already implemented canonical-run
head, followed by a revised pre-training gate. Its strongest objection is that
the apparent advantage is measured on inspected successful masks and still
does not demonstrate predictor generalization or online executor validity.

## Evidence

- Full audit:
  `/data/dataset/dynamic_mllm/binary_polar_v1/binary_polar_label_geometry_audit_v1.json`
- Full audit checksum:
  `/data/dataset/dynamic_mllm/binary_polar_v1/binary_polar_label_geometry_audit_v1.json.sha256`
- Slurm log: `runs/binary_label_audit_bp0_20260809/slurm.log`
- Audit implementation: `tools/audit_binary_polar_labels.py`
- Frozen gate: `plans/dynamic_mllm_binary_visual_polar_plan_v1.md`
