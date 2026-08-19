# Baseline-Relative Visual Router

This directory implements the revised method direction from
`docs/Research (last updated 2026-08-11).md`.

The first executable experiment is an oracle Pareto audit over deterministic,
paired all-on and sparse-policy generations on the canonical natural heldout
population. It answers whether useful treatment headroom exists before fitting
another gate.

The implemented method checkpoint freezes `sw31_bt_leg_s41` and learns its
actual all-on-relative treatment outcomes. A conservative harm gate provides
the efficiency region; a separate rescue-utility gate can override fallback
for a small high-confidence region. Benchmark identity is never an input.

Run:

```bash
PYTHONPATH=baseline_relative_visual_router/src \
  dvr_qwen/.venv/bin/python \
  baseline_relative_visual_router/scripts/analyze_oracle_pareto.py \
  --config baseline_relative_visual_router/configs/canonical_proposers.json \
  --output-dir baseline_relative_visual_router/experiments/oracle_pareto_canonical_v1
```

Primary results:

- `experiments/oracle_pareto_canonical_v1/report.md`
- `experiments/actual_policy_admission_sw31_l27_v1/report.md`
- `experiments/hierarchical_actual_policy_sw31_l27_v1/summary.json`
- `../reports/baseline_relative_natural_pareto_and_admission_20260812.md`

## Shared-prefix admission

The deployable follow-up executes a common all-on prefix of `K` layers, captures
instruction and visual state at that boundary, and then chooses between exact
all-on continuation and the frozen SW31 online policy. The sparse continuation
is generated from the same prefix state; its route mask and cost include the
forced-on prefix.

Predeclared prefix depths are `K={2,4,8}`. A fixed UID split, independent of
K-specific outcomes, is used for fitting and calibration. Calibration compares
harm-only, conservative utility-only, and hierarchical admission scores. It
selects one accuracy-first and one efficiency-first operating point using
actual generation correctness, never pair accuracy. Benchmark identity is not
an input.

The end-to-end queue is:

```bash
baseline_relative_visual_router/scripts/run_prefix_pipeline_after_canonical.sh
```

It audits `3 x 22,349` canonical outcomes, selects `K`, generates only that
candidate on the UID-disjoint MMStar/MMMU population (`n=5,807`), and evaluates
the frozen admission rule. Runtime caches are under
`/mnt/hyemin/10k_dataset_mask/baseline_relative_visual_router/`.
