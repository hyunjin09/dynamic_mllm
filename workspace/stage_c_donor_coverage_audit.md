# Stage C Outcome-Blind Donor-Coverage Audit

Status: **complete; candidate amendment not applied; Stage C sweep not resumed**

## Scope and integrity

This audit used only the frozen 800-record manifest identifiers, images, prompt
geometry, layer-0 postvisual READ-residual geometry, the frozen 200-record
TextVQA donor pool, and the frozen deterministic donor tie seeds. It preserved
the frozen maximum-multiplicative-ratio distance over residual norm,
postvisual-row count, and image-token count; same-sample and same-image
exclusions; donor count eight; hook; layer; task; and norm-matching rule.

The forward exited immediately after extraction of the layer-0 postvisual READ
residual. No accepted-reference or generated-answer value, likelihood,
correctness label, intervention effect, confidence interval, structured-null
comparison, or scientific endpoint was computed or inspected. The 93 partial
Stage C records were not loaded or inspected.

## Deterministic result

The exact minimum common caliper is

```text
c_star = 1.5833333333333333  (19/12)
```

It is determined uniquely by target
`textvqa:textvqa_validation_36174` (image `9ebbe79077ffe201`). Its
eighth-nearest distance is driven by the frozen image-token ratio
`703 / 444 = 19 / 12`; its residual-norm and row-count ratios remain below
that value.

Eighth-nearest-donor distance summary across all 800 targets:

| Statistic | Distance |
|---|---:|
| Minimum | 1.0328882410750226 |
| Mean | 1.0887126131054512 |
| Standard deviation | 0.06038496047945747 |
| 1% | 1.0372570272627966 |
| 5% | 1.0413124363741924 |
| 10% | 1.0416666666666667 |
| 25% | 1.0454545454545454 |
| 50% | 1.076055487679397 |
| 75% | 1.1058636313891594 |
| 90% | 1.1481481481481481 |
| 95% | 1.1875627011323937 |
| 99% | 1.3151137816279503 |
| Maximum | 1.5833333333333333 |

Coverage at the requested calipers:

| Caliper | Targets with at least eight donors | Fraction |
|---:|---:|---:|
| 1.5 | 798 / 800 | 99.75% |
| `c_star = 1.5833333333333333` | 800 / 800 | 100% |
| 1.75 | 800 / 800 | 100% |
| 2.0 | 800 / 800 | 100% |

Only two targets fail the original 1.5 rule:

- `textvqa:textvqa_validation_39543`: seven donors at 1.5; eighth-nearest
  distance `1.5155925155925156`.
- `textvqa:textvqa_validation_36174`: three donors at 1.5; eighth-nearest
  distance `1.5833333333333333`.

## Donor identities under the candidate caliper

The eight nearest donors for the two targets whose coverage changes are:

| Target | Rank | Donor sample ID | Donor image ID | Distance |
|---|---:|---|---|---:|
| `textvqa:textvqa_validation_39543` | 1 | `textvqa:textvqa_train_844` | `textvqa:09cd1c938e3b7ad5` | 1.0769230769230769 |
|  | 2 | `textvqa:textvqa_train_11788` | `textvqa:02ed611f88177ee4` | 1.2307692307692308 |
|  | 3 | `textvqa:textvqa_train_2865` | `textvqa:40861fe14ae57b02` | 1.3846153846153846 |
|  | 4 | `textvqa:textvqa_train_19` | `textvqa:0393c9d77b8215a3` | 1.4615384615384615 |
|  | 5 | `textvqa:textvqa_train_659` | `textvqa:ded9fafc272bf94a` | 1.4615384615384615 |
|  | 6 | `textvqa:textvqa_train_1221` | `textvqa:7287fb7f85a44890` | 1.4615384615384615 |
|  | 7 | `textvqa:textvqa_train_1174` | `textvqa:0f9a0cf7e03f66db` | 1.4615384615384615 |
|  | 8 | `textvqa:textvqa_train_5931` | `textvqa:482352df0b02e980` | 1.5155925155925156 |
| `textvqa:textvqa_validation_36174` | 1 | `textvqa:textvqa_train_844` | `textvqa:09cd1c938e3b7ad5` | 1.1666666666666667 |
|  | 2 | `textvqa:textvqa_train_11788` | `textvqa:02ed611f88177ee4` | 1.3333333333333333 |
|  | 3 | `textvqa:textvqa_train_2865` | `textvqa:40861fe14ae57b02` | 1.5 |
|  | 4 | `textvqa:textvqa_train_19` | `textvqa:0393c9d77b8215a3` | 1.5833333333333333 |
|  | 5 | `textvqa:textvqa_train_659` | `textvqa:ded9fafc272bf94a` | 1.5833333333333333 |
|  | 6 | `textvqa:textvqa_train_1221` | `textvqa:7287fb7f85a44890` | 1.5833333333333333 |
|  | 7 | `textvqa:textvqa_train_1174` | `textvqa:0f9a0cf7e03f66db` | 1.5833333333333333 |
|  | 8 | `textvqa:textvqa_train_18533` | `textvqa:4aebcf0dddbcbd37` | 1.5833333333333333 |

The complete frozen ordering and eight donor identities for every target are
stored in `targets[].nearest_eight_donors` in
`outputs/stage_c/preflight/stage_c_donor_coverage_audit_v1.json`. Because donor
selection takes the eight nearest eligible donors, increasing only the
admission cap leaves the selected donors unchanged for the 798 targets already
supported at 1.5.

## Interpretation and candidate amendment

This appears to be a **minimal local repair**, not a substantial global
weakening: `c_star` is the exact minimum, is only 5.56% above 1.5, changes
coverage for 2/800 targets, and changes no distance definition, covariate,
exclusion, donor count, ranking rule, or norm matching. The strongest objection
is local: the determining target has only three donors at 1.5 and needs five
donors at the new boundary, so its real-residual controls are visibly less
tightly image-token-matched than the bulk of the cohort. That limitation must
remain explicit if the amendment is approved.

The single deterministic candidate amendment is:

```text
replace caliper 1.5 with 1.5833333333333333;
preserve eight donors and every other frozen matching rule.
```

This audit does not apply that amendment. Freezing an amended donor index and
restarting Stage C require explicit user approval. Stage D remains
unauthorized.

