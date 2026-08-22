# Phase 28: WeMath Visual-Dependence Reanalysis Memory

## Current Objective

Reanalyze the frozen hard-cap-400 WeMath2.0-Pro route cache after recognizing
that ALL-OFF is a qualitatively distinct no-direct-visual-K/V regime. Separate
whether direct vision is needed at all from how many VISUAL_ON layers are
needed among samples where ALL-OFF is behaviorally insufficient.

## Active Constraints

- Read-only analysis of the same 4,544 eligible records and all raw evaluated
  routes used by the completed difficulty report.
- Verify the exact 28-zero ALL-OFF anchor directly from every authoritative raw
  record and reconcile it with the raw-derived index.
- Primary cohort is the 841 FULL-correct records, decomposed into V0 and V+.
- Preserve all eight official difficulty strata before coarse degree or axis
  summaries.
- No MCTS, Qwen execution, route generation, predictor training, route-space
  change, threshold change, or REPEAT analysis.

## Current State

- Done: the first difficulty analysis passed and found an axis-specific
  negative minimum-ON association in the mixed V0/V+ FULL-correct cohort.
- Done: the V0/V+ decomposition and conditional visual-budget reanalysis
  authorized by `plans/motivation_check2.md` passed as Outcome A.
- Blocked: none.
- Most recent useful observation: V0 prevalence increases from 32.5% to 73.4%
  across degrees 0 to 3, while the V+-only rho is -0.057 with clustered 95% CI
  [-0.154, 0.037] and the paired V+ mean delta is -0.04 with CI [-0.63, 0.57].

## Evidence That Matters

| Evidence | Source / Path | Why It Matters | Status |
|---|---|---|---|
| ALL-OFF bypasses visual rows and excludes visual K/V from text at every layer | `binary_policy/executor/layers.py` | Makes zero-ON qualitatively different from a sparse positive visual program | confirmed |
| Previous mixed-cohort rho was -0.225 | `reports/wemath2pro_visual_compute_difficulty_v1.md` | Must be decomposed into V0 prevalence and V+-conditional depth | confirmed |
| Frozen cache has 4,544 records and 1,658,485 evaluated routes | `outputs/wemath2pro_mcts_label_analysis_v1/completion_audit_v1.json` | Defines the immutable analysis population | confirmed |

## Failed Attempts and Lessons

| Attempt | Observed Failure | Diagnosis | Evidence | Lesson / Next Implication | Do Not Repeat |
|---|---|---|---|---|---|
| Interpret zero ON as merely an extremely sparse visual route | ALL-OFF prevents direct decoder access to encoded visual K/V | supported | executor code and `plans/motivation_check2.md` | Separate V0 from V+ before visual-budget interpretation | Do not pool zero-ON with positive visual-access routes |
| Launch the analysis as a file path | Python omitted the repository root and failed importing `experiments` before data access | supported | `runs/wemath2pro_visual_dependence_reanalysis_v1.log` | Invoke repository analysis packages with `python -m` | Do not repeat direct-file invocation for this module |

## Open Candidates

| Candidate | Why Plausible | What It Resolves | Cost | Status |
|---|---|---|---|---|
| Frozen V0/V+ decomposition | Directly distinguishes composition from conditional budget effects | Whether the old trend survives among behaviorally vision-dependent samples | low | selected |

## Next-Step Decision

- Deliberation mode: standard
- Active objective and bottleneck: determine whether the previous negative
  depth trend is V0 composition or persists among V+ samples.
- Relevant memory item used: the previous result mixed a no-direct-vision mass
  at zero with positive visual-access programs.
- Confirmed observation: the exact ALL-OFF anchor exists in every raw record
  and is already represented in the checksum-bound index.
- Unverified interpretation: the x-associated trend remains after excluding
  ALL-OFF-solvable cases.
- Diagnosis: supported semantic flaw in the previous pooled interpretation;
  the conditional scientific outcome remains unknown.
- Evidence path if diagnosis is not unknown: `binary_policy/executor/layers.py`
  and `plans/motivation_check2.md`.
- Viable alternatives considered: none; the user specified one complete
  read-only reanalysis.
- Chosen action: implement, test, and run the exact raw-anchor audit and all
  V0/V+, decomposition, conditional feasibility, paired, and FULL-wrong
  analyses specified by `plans/motivation_check2.md`.
- Strongest objection: V+ minimum ON remains finite-search-dependent and the
  increasingly selected FULL-correct cohort still limits population claims.
- How this differs from failed attempts: it treats zero visual access as a
  separate behavioral regime before estimating positive visual budgets.
- Automatic execution authorized: yes.
- Authorization basis: explicit user request to perform
  `plans/motivation_check2.md`.
- Stop condition: stop on any source/hash/anchor mismatch, missing ALL-OFF
  record, V0/min-ON identity failure, or need for new route execution.

## Latest Research-Action Result

- Action taken: verified every exact raw ALL-OFF/FULL anchor and completed the
  eight-stratum V0/V+, exact mean decomposition, V+-only depth/feasibility,
  paired-family/image, 2×2 contingency, and A0/A+ analyses.
- Result: Outcome A — mostly ALL-OFF composition. Of 841 FULL-correct records,
  413 are V0 and 428 V+. Composition explains 83.7–94.9% of the degree-level
  mean decline from degree 0; the V+-only global and paired trends are null.
- Evidence saved:
  `outputs/wemath2pro_visual_dependence_reanalysis_v1/`,
  `reports/wemath2pro_visual_dependence_reanalysis_v1.md`, and
  `runs/wemath2pro_visual_dependence_reanalysis_final_v1.log`.
- Failure or issue: one pre-analysis module-import invocation failure was
  repaired without scientific output or protocol change.
- Lesson learned: direct visual dependence and positive visual-access budget
  are distinct quantities. The previous x-associated minimum-ON pattern is
  primarily a high V0-prevalence pattern, not a supported conditional depth
  requirement.
- Next implication: stop. No new search, REPEAT action, or predictor change is
  justified or authorized by this result.
