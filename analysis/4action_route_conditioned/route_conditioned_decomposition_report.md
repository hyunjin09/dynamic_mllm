# Route-Conditioned READ/WRITE Decomposition Report

## Scope and estimand

This experiment decomposes one deterministic, current-runtime-correct binary anchor route per frozen A+ sample. At each anchor-OFF position, every other layer remains fixed to that correcting route while the target layer is evaluated as M00=BOTH_OFF, M10=WRITE_OFF/READ_ONLY, M01=READ_OFF/WRITE_ONLY, and M11=FULL restoration. All continuous effects are within the unified executor.

This is **route-conditioned** evidence. The earlier experiment is a distinct **FULL-context** local intervention in which every non-target layer is FULL. Neither result is global causal attribution, and the two contexts are never conflated below.

## Integrity, pilot, and execution

The full merge contains 1,804 samples, 17,262 anchor-OFF positions, and 69,048 saved action rows. Every sample/layer/action gate and exact-coverage check passed.

The matched all-eight-H100 pilot selected `two_replicas` (2 replica(s)/GPU) at 12.1839 valid new cells/s. The prelaunch estimate was 1.18 wall hours and 9.45 GPU-hours.

## Required final questions

### 1. How many frozen A+ samples had a validated current-runtime anchor?

1,804/1,880 frozen A+ samples had a validated current-runtime correcting anchor: 1,170 GQA and 634 TextVQA. 76 were excluded because no cached correcting route remained correct; no route was invented or searched.

### 2. How many anchor-OFF positions were individually necessary?

7,880/17,262 (45.6%, image-group bootstrap 95% CI 44.3% to 46.9%) were individually necessary: restoring FULL made the correcting anchor wrong. The remaining 9,382/17,262 (54.4%) were redundant in this anchor-route context.

### 3. Which suppression mechanism preserved correction among necessary positions?

| Mechanism | Count | Share among necessary | Image-group bootstrap 95% CI |
|---|---:|---:|---:|
| READ suppression required | 1,619 | 20.5% | 19.4%–21.7% |
| WRITE suppression required | 3,379 | 42.9% | 41.5%–44.4% |
| Either removal sufficient | 783 | 9.9% | 9.2%–10.7% |
| Both READ and WRITE required OFF | 2,099 | 26.6% | 25.3%–28.1% |

Dataset-specific conditional shares are retained with the same image-group bootstrap contract:

| Dataset | Mechanism | Count | Share among necessary | 95% CI |
|---|---|---:|---:|---:|
| gqa | READ suppression required | 1,134 | 20.3% | 19.0%–21.7% |
| gqa | WRITE suppression required | 2,346 | 42.1% | 40.4%–43.8% |
| gqa | Either removal sufficient | 538 | 9.6% | 8.7%–10.6% |
| gqa | Both READ and WRITE required OFF | 1,559 | 28.0% | 26.2%–29.6% |
| textvqa | READ suppression required | 485 | 21.1% | 18.9%–23.1% |
| textvqa | WRITE suppression required | 1,033 | 44.9% | 42.3%–47.5% |
| textvqa | Either removal sufficient | 245 | 10.6% | 9.3%–12.1% |
| textvqa | Both READ and WRITE required OFF | 540 | 23.4% | 21.0%–25.9% |
| joint | READ suppression required | 1,619 | 20.5% | 19.4%–21.7% |
| joint | WRITE suppression required | 3,379 | 42.9% | 41.5%–44.4% |
| joint | Either removal sufficient | 783 | 9.9% | 9.2%–10.7% |
| joint | Both READ and WRITE required OFF | 2,099 | 26.6% | 25.3%–28.1% |

### 4. Are READ- and WRITE-mediated corrections distributed differently across depth?

Mean layer was 17.79 for READ-mediated and 9.97 for WRITE-mediated positions. The READ-minus-WRITE difference was 7.82 (image-group bootstrap 95% CI 7.31 to 8.31). READ-mediated positions occurred later on average than WRITE-mediated positions. Layerwise counts and effects for all 28 layers are in `aggregate/depth_taxonomy.*`, `aggregate/depth_effects.*`, and `figures/`.

### 5. Do long correcting routes contain necessary operations or redundancy?

| Anchor OFF-count stratum | Samples | OFF positions | Necessary | Redundant |
|---|---:|---:|---:|---:|
| 2-4 | 12 | 44 | 43.2% | 56.8% |
| 5-8 | 761 | 5,464 | 29.2% | 70.8% |
| 9-12 | 762 | 7,761 | 47.5% | 52.5% |
| 13-16 | 222 | 3,149 | 62.8% | 37.2% |
| >16 | 47 | 844 | 71.0% | 29.0% |
The longest populated stratum had no more redundancy than the shortest (29.0% versus 56.8%); the full stratification retains category-specific shares and confidence intervals.

### 6. How often does FULL-context local harmfulness agree with route-conditioned necessity?

The discrete FULL-context-local-rescue versus route-necessity classification agreed on 51.8% of 17,262 matched sample/layer positions. FULL-context local rescue recalled 7.3% of route-necessary positions and had 36.3% precision for route necessity. Continuous harmful-effect sign agreement was 62.6% for READ and 58.8% for WRITE; pooled Spearman correlations were 0.3573 and 0.1771, respectively.

### 7. How often did route conditioning reveal operations missed in FULL context?

7,305 route-necessary positions (92.7% of all route-necessary positions) had no discrete W→C rescue in the earlier FULL-context single-layer sweep. This is direct evidence that the final behavioral effect of a layer can depend on the other suppressions in the successful trajectory. Median within-sample FULL-versus-route rank correlation was 0.4424 for READ and 0.2473 for WRITE.

### 8. Does binary routing correct errors by suppressing answer-unaligned READ/WRITE?

Conditionally, yes for 7,880 individually necessary OFF positions: 5,781 (73.4%) permitted at least one READ/WRITE component to be restored while preserving correction, whereas 2,099 required both components suppressed. This supports answer-unaligned READ and/or WRITE suppression as a mechanism inside these selected correcting routes, but does not make the corresponding operation globally harmful or prove that every OFF choice caused the route.

### 9. Does the evidence justify a true four-action trajectory search/router?

The 5,781 component-relaxable necessary positions justify a separately approved bounded joint-refinement pilot, but not an immediate claim that a four-action search/router will improve accuracy or compute. Individual relaxations were tested one at a time; simultaneous relaxations may interact. A suitable next proposal is a 64-sample, route-size-stratified beam refinement from the validated anchors, beam width 4, at most 8 OFF positions, testing at most two partial restorations per expansion (upper bound 4,096 evaluations, approximately 0.09 eight-GPU wall hours or 0.75 GPU-hours at the measured throughput). Do not launch this proposed experiment without a separate decision.

## Continuous route-conditioned factorial effects

The sign convention is restoration minus suppression. Because M00 is the correct anchor, a negative effect means restoring the named operation shifts the fixed answer margin away from the correct answer. No magnitude threshold was imposed.

| Effect | Mean | Median | Fraction negative |
|---|---:|---:|---:|
| read_w0 | -0.1620 | -0.0625 | 60.7% |
| write_r0 | -0.3174 | -0.1250 | 64.4% |
| read_w1 | -0.1428 | -0.0540 | 59.1% |
| write_r1 | -0.2982 | -0.1250 | 63.4% |
| interaction | 0.0193 | 0.0000 | 43.9% |

## Sample structure and evidence inventory

Sample-level mechanism counts: either_only_ambiguous=69, joint_both_suppression=19, mixed_read_write=1,203, multiple_read_mediated=46, multiple_write_mediated=111, no_essential_off=195, one_dominant_operation=161.

Raw per-sample/layer/action outputs, fixed targets, generated answers, evaluator correctness, continuous scores, effects, route metadata, worker provenance, exact mergers, aggregate tables, and figures are retained under `analysis/4action_route_conditioned/` with SHA-256 sidecars.
