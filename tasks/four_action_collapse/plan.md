# Implementation Plan: Four-Action Collapse Isolation

## Contract

Execute `plans/four_action_collapse.md` in its stated order. Canonical reports
live under `analysis/4action_collapse/`; compatibility links under
`analysis/4action_router/` may point to the canonical A0 boundary files.

The fixed A1 population is 96 W2C train samples (32 each from GQA, ChartQA,
and TextVQA), deterministically stratified over boundary depth and valid
boundary actions, plus 24 deterministic C2C preservation samples (8 per
dataset). Sample IDs and selection metadata are frozen in `pilot_subset.json`
before training.

The A1 intervention is only guaranteed exact all-FULL-prefix boundary
exposure. Architecture, branches, structured head, set-valued loss, optimizer,
learning rate, C2C labels, backbone, and executor stay unchanged.

## Prospective A1 outcome gate

Evaluate checkpoints by the following fixed criteria:

- boundary Valid-Action@1 at least 0.95;
- boundary non-FULL recall at least 0.95;
- singleton recall at least 0.80 for every boundary action represented by at
  least five pilot cases;
- at least 0.90 of W2C free rollouts leave all-FULL;
- W2C routed rescue at least 0.25;
- C2C preservation at least 0.90.

Train for a fixed maximum of 50 epochs, validate every five epochs, and save
every validation checkpoint. A2 is authorized only if one checkpoint satisfies
all applicable criteria. Saturation without a passing checkpoint is Outcome A.

## Execution slices

1. Add tested boundary and deterministic pilot-selection helpers.
2. Materialize and audit A0 artifacts.
3. Add tested pilot sampling, boundary metrics, checkpointing, and launch path.
4. Run and interpret A1; conditionally launch matched ten-epoch A2.
5. Build and run matched exact-NLL POLAR B1 with route-empty exclusions.
6. Build and run the matched upfront-vs-online boundary probe.
7. Write the phase decision summary and cross-server handoff evidence.
