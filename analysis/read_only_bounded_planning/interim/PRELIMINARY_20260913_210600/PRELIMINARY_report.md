# PRELIMINARY — READ-only bounded planning interim analysis

**Snapshot: 2026-09-13T21:06:00.071160+09:00. This is not the final Phase 85 result.**

**Main finding:** The completed-cohort curves do not establish saturation. Beam8 gains 9.80 percentage points in candidate discovery and 8.50 points in q-selected correctness from 64 to 128 evaluations. The completed and pending W cohorts differ substantially, so these results cannot estimate pending-UID outcomes or full-population rates.

**1. PRELIMINARY cohort verification**

There are 638 fully completed UID-level records in the immutable snapshot: **612 W and 26 C**. The primary analysis uses exactly the same **612 W UIDs**, each with all 13 required algorithm/seed sessions complete. The other 695 W UIDs are pending and are excluded from all primary search numerators and denominators. The completion rule was frozen before inspecting search outcomes. The source files and copied result histories are hash recorded in [snapshot_manifest.json](snapshot_manifest.json).

| Characteristic              | Completed        | Pending          |
|:----------------------------|:-----------------|:-----------------|
| dataset: chartqa            | 159/612 (25.98%) | 113/695 (16.26%) |
| dataset: gqa                | 188/612 (30.72%) | 501/695 (72.09%) |
| dataset: textvqa            | 265/612 (43.30%) | 81/695 (11.65%)  |
| source: canonical           | 101/612 (16.50%) | 213/695 (30.65%) |
| source: historical          | 511/612 (83.50%) | 482/695 (69.35%) |
| Mean trigger layer          | 13.72            | 20.10            |
| Mean suffix length          | 14.28            | 7.90             |
| Hamming-1 rescue prevalence | 24.02%           | 10.94%           |

Within dataset/source cells, completed W are 42.81% Historical TextVQA versus 11.65% pending, 16.50% Historical GQA versus 43.88% pending, and 14.22% Canonical GQA versus 28.20% pending. Median trigger is 14 versus 22; median suffix length is 14 versus 6. Trigger and suffix length are the same structural variable under T=28−L*. Completion is determined by the scheduled workload and elapsed runtime, not random sampling. The completed cohort is also richer in already known single-off rescues. Its observed rates are not extrapolated or reweighted to the full population. All six source/dataset cells and full layer/suffix distributions are retained in [cohort_categorical_comparison.csv](cohort_categorical_comparison.csv); ALL/W/C comparisons are separate.

**2. PRELIMINARY common-cohort planning curves**

Each cell is **ANY_CORRECT% / SELECTED_CORRECT%**, with n=612 W throughout. ANY means at least one evaluated route generated an LMMS-correct answer. SELECTED means the highest frozen gold-answer-q route did so; ties use earliest evaluated. Random/MCTS results average three seeds within UID, not a best-seed or seed-union rate. Dense is candidate 1 and cached evaluations count if independently requested by the algorithm.

| Algorithm      | 4             | 8             | 16            | 32            | 64            | 128           |
|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| greedy         | 14.54 / 10.62 | 27.12 / 21.57 | 38.56 / 31.86 | 42.32 / 35.78 | 42.32 / 35.78 | 42.32 / 35.78 |
| beam2          | 10.62 / 7.52  | 19.44 / 14.22 | 31.37 / 24.51 | 43.63 / 35.78 | 45.92 / 38.40 | 45.92 / 38.40 |
| beam4          | 10.62 / 7.52  | 16.67 / 11.27 | 23.69 / 16.99 | 36.11 / 28.59 | 46.41 / 37.42 | 48.20 / 39.05 |
| beam8          | 10.62 / 7.52  | 16.67 / 11.27 | 20.92 / 14.22 | 29.25 / 20.42 | 40.69 / 31.21 | 50.49 / 39.71 |
| random_uniform | 19.55 / 12.47 | 28.16 / 18.36 | 34.26 / 23.80 | 38.78 / 27.23 | 42.86 / 30.99 | 46.57 / 34.15 |
| random_sparse  | 17.32 / 12.04 | 23.97 / 16.72 | 29.85 / 20.75 | 34.42 / 24.40 | 37.64 / 27.02 | 40.63 / 29.58 |
| binary_mcts    | 20.53 / 14.60 | 27.89 / 19.77 | 34.42 / 24.78 | 39.92 / 28.98 | 44.99 / 32.52 | 48.47 / 35.35 |

B is a **maximum unique complete-route budget**. A naturally completed algorithm with fewer than B evaluations is reported as an at-most-B policy; no unexecuted route is counted. This differs from partially completed UIDs, which are wholly excluded. On this cohort the complete fixed single-off census has ANY=147/612 (24.02%) and q-selected correctness=101/612 (16.50%), including Dense in selection. Dense itself is wrong on all 612 by construction. The single-off census is a separate complete baseline, not inserted into every algorithm's candidate list.

[PRELIMINARY candidate-discovery curve](figures/PRELIMINARY_any_correct.png) · [PRELIMINARY selected-correct curve](figures/PRELIMINARY_selected_correct.png)

**3. PRELIMINARY beam versus random comparisons**

These ANY differences are in **percentage points**, using a shared per-UID actual route cap across the compared beam and all three random seeds. Thus natural early termination cannot give one side additional evaluations in the comparison. Full selected-correct differences and exploratory paired 5,000-draw UID bootstrap intervals are in [beam_vs_random.csv](beam_vs_random.csv).

| Comparison             |     4 |      8 |     16 |    32 |    64 |   128 |
|:-----------------------|------:|-------:|-------:|------:|------:|------:|
| beam2 − random_uniform | -8.93 |  -8.71 |  -2.51 |  6.92 |  8.77 |  8.77 |
| beam2 − random_sparse  | -6.7  |  -4.52 |   1.8  | 11.38 | 13.4  | 13.4  |
| beam4 − random_uniform | -8.93 | -11.49 | -10.57 | -2.23 |  5.72 |  7.24 |
| beam4 − random_sparse  | -6.7  |  -7.3  |  -6.15 |  2.4  | 10.4  | 11.76 |
| beam8 − random_uniform | -8.93 | -11.49 | -13.34 | -9.53 | -1.63 |  5.77 |
| beam8 − random_sparse  | -6.7  |  -7.3  |  -8.93 | -5.17 |  3.38 | 11.17 |

At nominal cap 128, beam8's detailed matched-actual differences are:

| Comparison             | ANY_pp                 | SELECTED_pp            |   Mean_shared_actual_routes |
|:-----------------------|:-----------------------|:-----------------------|----------------------------:|
| beam8 − random_uniform | +5.77 [+3.70, +7.95]   | +6.81 [+4.47, +9.20]   |                       91.27 |
| beam8 − random_sparse  | +11.17 [+8.66, +13.78] | +11.55 [+8.82, +14.32] |                       91.27 |

Beam8 loses to both random baselines at 4–32. At 64 its advantage over uniform is not resolved by the interval (−1.63 pp, 95% CI [−4.25,+1.03]); at 128 it beats both in this cohort. This supports a budget-dependent advantage under oracle-q search, not uniformly superior small-budget planning. Nominal-cap differences are also retained separately. All intervals are exploratory and unadjusted for multiple comparisons or interim looks.

**4. PRELIMINARY marginal rescue gains**

Each cell is **ΔANY / ΔSELECTED in percentage points** on the exact same 612 W cohort.

| Algorithm      | 4→8             | 8→16            | 16→32           | 32→64           | 64→128        |
|:---------------|:----------------|:----------------|:----------------|:----------------|:--------------|
| greedy         | +12.58 / +10.95 | +11.44 / +10.29 | +3.76 / +3.92   | +0.00 / +0.00   | +0.00 / +0.00 |
| beam2          | +8.82 / +6.70   | +11.93 / +10.29 | +12.25 / +11.27 | +2.29 / +2.61   | +0.00 / +0.00 |
| beam4          | +6.05 / +3.76   | +7.03 / +5.72   | +12.42 / +11.60 | +10.29 / +8.82  | +1.80 / +1.63 |
| beam8          | +6.05 / +3.76   | +4.25 / +2.94   | +8.33 / +6.21   | +11.44 / +10.78 | +9.80 / +8.50 |
| random_uniform | +8.61 / +5.88   | +6.10 / +5.45   | +4.52 / +3.43   | +4.08 / +3.76   | +3.70 / +3.16 |
| random_sparse  | +6.64 / +4.68   | +5.88 / +4.03   | +4.58 / +3.65   | +3.21 / +2.61   | +3.00 / +2.56 |
| binary_mcts    | +7.35 / +5.17   | +6.54 / +5.01   | +5.50 / +4.19   | +5.07 / +3.54   | +3.49 / +2.83 |

For beam8, 64→128 gives **+9.80pp ANY (95% CI [+7.52,+12.25])** and **+8.50pp SELECTED ([+6.05,+10.95])**. Sixty additional UIDs acquire a correct candidate ; a net 52 additional UIDs are correct under q selection. 411 UIDs receive additional beam8 evaluations, and their conditional discovery gain is 14.60pp. Greedy and beam2 add zero evaluations from 64 to 128; their zero gains reflect algorithm termination. Beam4 adds evaluations on 198 UIDs. Random/MCTS extend on 572 UIDs and retain roughly 3–4 pp discovery gains. These are within-policy budget increments; “matched actual” refers separately to beam-versus-random comparisons.

**5. PRELIMINARY first-rescue-rank distribution**

Ranks count unique complete-route evaluations, including Dense at rank 1. For stochastic methods, the table first computes one mean observed-rescue-seed rank per UID, then reports its distribution; it is explicitly conditional on observed rescue. Censored/no-rescue seeds and each seed's own quantiles are retained separately, rather than assigning them rank 129 or silently treating them as successes.

| algorithm      |   UIDs with observed rescue |   UIDs with no observed rescue |   Median rank |   p75 rank |   p90 rank |
|:---------------|----------------------------:|-------------------------------:|--------------:|-----------:|-----------:|
| greedy         |                         259 |                            353 |          7    |      10.5  |      15.2  |
| beam2          |                         281 |                            331 |         11    |      19    |      28    |
| beam4          |                         295 |                            317 |         17    |      32    |      49    |
| beam8          |                         309 |                            303 |         25    |      52    |      81    |
| random_uniform |                         304 |                            308 |          8.33 |      26    |      56.63 |
| random_sparse  |                         269 |                            343 |          8.33 |      24    |      61.5  |
| binary_mcts    |                         322 |                            290 |          8.33 |      29.58 |      62    |

The [PRELIMINARY unconditional first-rescue CDF](figures/PRELIMINARY_first_rescue_cdf.png) keeps all 612 UIDs in its denominator and averages seeds within UID. Thus it does not hide no-rescue UIDs. [Per-seed rank distributions](first_rescue_distribution_per_seed.csv) and [censoring-aware UID records](first_rescue_per_uid.csv) accompany the conditional summary.

**6. PRELIMINARY ranking versus generation failures at maximum B=128**

Ranking failure means a correct candidate exists but the highest-q candidate is wrong. Generation failure means none of that algorithm's evaluated candidates is correct. These are algorithm-specific measured failures, not proof that no correct READ route exists.

| algorithm      |   Selected correct % |   Ranking failure % |   Generation failure % |   Mean actual routes |   UID-seed runs reaching128 % |
|:---------------|---------------------:|--------------------:|-----------------------:|---------------------:|------------------------------:|
| greedy         |                35.78 |                6.54 |                  57.68 |                15.28 |                          0    |
| beam2          |                38.4  |                7.52 |                  54.08 |                28.56 |                          0    |
| beam4          |                39.05 |                9.15 |                  51.8  |                53.12 |                          0    |
| beam8          |                39.71 |               10.78 |                  49.51 |                91.27 |                         32.35 |
| random_uniform |                34.15 |               12.42 |                  53.43 |               122.65 |                         93.46 |
| random_sparse  |                29.58 |               11.06 |                  59.37 |               120.71 |                         87.75 |
| binary_mcts    |                35.35 |               13.13 |                  51.53 |               122.65 |                         93.46 |

For beam8 these are 243 selected successes, 66 ranking failures and 303 generation failures. The ranking loss is material, but candidate discovery remains the larger unresolved component. Gold-answer q is an analysis oracle; none of this establishes deployable label-free routing.

**7. PRELIMINARY Hamming-2: complete censuses only**

- All completed Hamming-2 censuses: **468 Hamming-1-unrescued W UIDs**, **53,305 exact distance-2 routes**; **70 UIDs rescued (14.96%)**. q-selected correctness across Dense+Hamming-1+Hamming-2 is 37/468 (7.91%).
- Intersection with the primary 612 W cohort: **465 Hamming-1-unrescued W UIDs**, **52,929 exact distance-2 routes**, **70 additional rescued UIDs (15.05%)**, and 37 q-selected successes. Together, Hamming-1 plus completed Hamming-2 discovers corrections for 217/612 (35.46%) in this primary cohort.

The three extra completed Hamming-2 UIDs outside the primary cohort are used only in the explicitly separate Hamming-2 census table; their incomplete full-search histories do not enter any primary curve. Each included census has exactly C(T,2) distinct valid two-OFF routes. No partial census or extrapolated pending-route count enters these rates.

**8. PRELIMINARY necessity and stopping assessment**

**The current curves do not show overall saturation.** They already answer a narrower question: in this selected completed cohort, gold-q-guided multi-layer search can discover many READ-only corrections, and increasing beam8's budget from 64 to 128 still produces substantial gains. They do not establish the final planning category or READ-only ceiling.

- **Finish the remaining B≤128 population** to support the original population-level claim. The 695 pending W UIDs differ in dataset/source, trigger depth, suffix length and known Hamming-1 correctability. Current-cohort flattening or gains cannot answer for them.
- **The planned B=256 audit remains scientifically warranted** to measure the unresolved search tail and budget saturation. This does not mean spending more on naturally terminated greedy/beam2 histories; it concerns searches with unexamined candidates under the frozen extension policy.
- **B=512 is not yet scientifically established as necessary.** Keep it conditional on the frozen 128→256 rule: extend unresolved cases only if the rescue gain is at least 1 percentage point among evaluated B=128-unresolved UIDs. Do not treat this interim report as authorization for unconditional 512 exploration.

Independent research review returned **stable**, ranking completion of the base population followed by the frozen conditional 256 audit above stopping now or automatic 512. Its strongest objection—the completion cohort's selection bias—is retained explicitly. No final P-READ/ceiling category, population rescue forecast, external benchmark claim or controller recommendation is made here. Running jobs were left unchanged.

**9. PRELIMINARY verification and evidence**

An independent direct scan reconstructed all **47,736 UID×algorithm×seed×budget cells**, checking ANY, first-highest-q selection and actual evaluation counts. All snapshot hashes and 468 complete Hamming-2 censuses passed. See [PRELIMINARY_verification.json](PRELIMINARY_verification.json). Raw completed histories are preserved under the snapshot link; full per-UID metrics, all marginal gains, all paired comparisons, and rank distributions accompany this report.
