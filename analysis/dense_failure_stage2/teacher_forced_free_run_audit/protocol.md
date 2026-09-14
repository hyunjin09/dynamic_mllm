# Teacher-forced versus free-run divergence audit

- Contract: `3771fd971f07160c95be055f2aff4f4b9016e10c4665811663922e240a69be0f`
- Seen view: final full-refit checkpoint on all 569 training-side UIDs.
- Held-out view: exact frozen internal-dev checkpoint on the frozen 115 image-group-disjoint dev UIDs.
- Teacher forcing: score every exact cached route-state occurrence without executing the prediction.
- R0/R1/R2: free from trigger; force before first corrective action; force through first corrective action.
- No training, search, relabeling, threshold change, or external evaluation is part of this phase.
