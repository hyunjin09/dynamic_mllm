# Route-Conditioned Four-Action Implementation Audit

Date: 2026-08-24 KST

## Prerequisite gate

The prerequisite experiment is complete. Slurm job `1573` exited `0:0`, all
new final-analysis SHA-256 sidecars verify, and the completed report covers all
1,880 frozen primary A+ samples. No disqualifying semantic or execution failure
remains.

Authoritative evidence:

- `analysis/4action_answer_alignment/4action_answer_unaligned_report.md`
- `analysis/4action_answer_alignment/numerical_consistency_report.md`
- `analysis/4action_answer_alignment/aggregate/analysis_summary.json`
- `logs/slurm/four-action-final-analysis-r9-target-identity-20260823-1573.log`

## Frozen cohort and route evidence

The authoritative inputs are:

- cohort manifest:
  `analysis/4action_answer_alignment/cohort/cohort_manifest_v1.jsonl`
  (`cd6e2fa74fc520508bb923499b55bd8debecb840c09dbe2cd66faafe755a5d66`)
- current unified-FULL eligibility:
  `analysis/4action_answer_alignment/cohort_eligibility__unified_v1/merged_results.jsonl`
  (`74e4578b6964748b1f542a13b07319d36b3f54f12daa1542d22be23abdd0d40e`)

Joining these sources yields exactly:

| Dataset | Frozen eligible A+ samples |
|---|---:|
| GQA | 1,222 |
| TextVQA | 658 |
| Total | 1,880 |

All compact correcting-route records pass the audited invariants:

- exactly 28 binary mask values;
- non-ALL-OFF positive-vision route;
- stored Hamming distance equals the number of OFF layers;
- stored visual-ON count equals the mask sum;
- stored evaluator score meets the sample's correctness threshold;
- route IDs and sample IDs remain deterministic.

No malformed correcting route was found.

## Deterministic anchor policy

Candidate routes will be ordered by:

1. ascending Hamming distance from FULL (fewest OFF layers);
2. descending cached evaluator score within a distance tie;
3. ascending route ID;
4. ascending 28-bit mask key as a final stable tie-break.

All candidates tied at the minimum distance will be retained in anchor
metadata. A candidate is not frozen as the anchor until its generation is
re-evaluated as correct with the current unified route executor and evaluator.
If it fails, candidates are tried in the same deterministic order. Samples
with no current-correct cached route are excluded only from the
route-conditioned analyzable set, not from the frozen A+ cohort.

Historical route geometry before current-runtime fallback:

| Quantity | Value |
|---|---:|
| Mean anchor OFF count | 9.5271 |
| Median anchor OFF count | 9 |
| Minimum / maximum OFF count | 2 / 22 |
| Samples with multiple nearest candidates | 504 |
| Samples where cached score breaks a nearest-route tie | 15 |
| Samples retaining a stable tie after score | 492 |
| Approximate new branch evaluations (`3K`) | 53,733 |

The final anchor distribution and count will be recomputed after current-
runtime route validation.

## Existing executor capabilities

The validated executor in `binary_policy/executor/four_action.py` already
provides:

- exact FULL, READ_ONLY, WRITE_ONLY, and IGNORE semantics;
- the same materialized-mask full-row and compacted text-row target calls for
  all four branches;
- per-layer heterogeneous K/V cache geometry;
- deterministic greedy generation and fixed-token teacher-forced scoring;
- fixed correct-target and frozen FULL-wrong-target scoring through the
  established runner utilities;
- semantic checks for visual-row bypass, READ access, WRITE updates, branch
  identity, cache geometry, and target identity.

The binary executor already runs arbitrary 28-layer binary routes, but its
all-ON case intentionally takes the native maskless path. Route-conditioned
causal cells must instead stay in the unified materialized-mask machinery.

## Missing capability and minimal extension

The current four-action executor captures only an all-FULL baseline and always
uses an all-FULL suffix after a local intervention. It does not yet expose:

- a cached baseline under an arbitrary FULL/IGNORE anchor schedule;
- a one-layer four-action branch starting from that anchor prefix;
- restoration of the same anchor schedule at every non-target suffix layer.

The minimal extension is therefore:

1. capture a unified route baseline by mapping binary `1 -> FULL` and
   `0 -> IGNORE`, always with materialized FULL attention;
2. retain the same pre-layer states and heterogeneous cache as the existing
   FULL baseline;
3. at an anchor OFF layer, run the existing unified target-layer four-action
   function for M00/M10/M01/M11;
4. execute every suffix layer with its unchanged anchor FULL/IGNORE action;
5. reuse the captured anchor M00 state where exact identity is established,
   while explicitly reproducing and comparing M00 in the pilot.

No new attention mechanism, mask implementation, model family, search, or
four-action MCTS is required.

## Execution and output gaps

A dedicated thin runner is still needed for:

- current-runtime candidate fallback and anchor freezing;
- stratified 48--64-sample pilot selection;
- cost-balanced work units by expected `3K` intervention cells, with more
  work units than GPUs;
- append-only/resumable cell records with worker/work-unit provenance;
- exact uniqueness, coverage, semantic, and failure summaries;
- pilot throughput comparisons and full-run launch gating;
- route-conditioned aggregate analysis and final report generation.

These will reuse the existing processor, evaluator, target, generation,
scoring, and artifact conventions rather than creating a second executor.

## Audit decision

**PASS for minimal implementation.** All required source data and scientific
anchors are present. The sole executor gap is the bounded arbitrary-route
baseline/one-layer override described above. No new search or scope change is
needed.
