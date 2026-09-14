# Stage-1 gate treatment-correctability decision

Contract: `2f935dd01763840d4e0420267b21c11980e45b8d9929054821a18f2d9c956f1e`

All rates below are current-runtime LMMS-Eval results under the fixed all-single plus 12-pair bounded search. They are lower estimates, not exhaustive four-action oracle rates.

## Validation

| Gate | Preservation | Wrong trigger recall | Trigger precision | Triggered-wrong correctability | Population oracle rescue | Median trigger |
|---|---:|---:|---:|---:|---:|---:|
| shared_random4 | 0.9425 | 0.5300 | 0.9021 | 0.3821 | 0.2025 | 0.0 |
| independent_sequential | 0.9475 | 0.5150 | 0.9075 | 0.3544 | 0.1825 | 4.0 |
| fixed_l27 | 0.9400 | 0.5400 | 0.9000 | 0.3657 | 0.1975 | 27.0 |

## Test confirmation

| Gate | Preservation | Wrong trigger recall | Trigger precision | Triggered-wrong correctability | Population oracle rescue | Median trigger |
|---|---:|---:|---:|---:|---:|---:|
| shared_random4 | 0.9400 | 0.5100 | 0.8947 | 0.4118 | 0.2100 | 0.0 |
| independent_sequential | 0.9450 | 0.5325 | 0.9064 | 0.3709 | 0.1975 | 4.0 |
| fixed_l27 | 0.9500 | 0.5275 | 0.9134 | 0.4171 | 0.2200 | 27.0 |

## Answers

1. Triggered dense-wrong correctability is reported for every gate in the tables; exact numerators are in `metrics/gate_correctability.csv`.
2. The validation-selected dynamic gate `shared_random4` has full-replay correctability enrichment 1.1211 versus all validation wrong samples.
3. Its validation correctability by trigger depth is: early=0.4189 (n=148), middle=0.2727 (n=33), late=0.3226 (n=31).
4. The validation-frozen better dynamic treatment substrate is `shared_random4` under population rescue, conditional correctability, preservability, compute, then the predeclared tie rule.
5. `shared_random4` validation population rescue is 0.2025 versus fixed-L27 replay 0.1975; compute is reported separately in `metrics/compute_comparison.csv`.
6. Triggered-correct preservability is 1.0000 for `shared_random4` and 1.0000 for fixed L27 on validation.
7. Evidence to proceed to a learned Stage-2 action head: **YES, provisionally**. This rule requires nonzero population rescue and enrichment above 1 for the selected dynamic gate; it does not claim statistical significance or exhaustive oracle coverage.

The Stage-2 future manifest contains 463 triggered states. No Stage-2 model was trained.
