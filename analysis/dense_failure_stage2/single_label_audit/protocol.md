# Stage-2 single-label distribution audit protocol

Analysis contract: `32261c07880a0b38a96ce5f25c5af1c5147bb7408f085ea197b2819ad53f7bc5`  
Phase-56 source contract: `6489a0b39af3efb22bcb527d3b43c474dc7bcb01cc453445aab45f420eb9905b`  
Source artifact manifest SHA-256: `a26d31e1d118e811ef107152213edcef3cd48ca3ed83ec17b18447ff490ff4d4`

## Frozen scope

- Inputs: exactly 39 train triggered Dense-C preservation samples/routes, 698 train SINGLE_FIXABLE Dense-W samples, 7,628 successful single routes, and their replay-valid routed states.
- Units remain separate: sample, route, and pre-layer routed state.
- Excluded: Corpus C/MCTS, unresolved samples, validation/test supervision, Qwen inference, search, training, rollout, threshold changes, and external evaluation.

## Weighting and simulation

- Route-weighted counts give each retained successful route weight one.
- Sample-weighted counts give each W sample total weight one, divided uniformly across its routes.
- Sampling simulations are exact expected class counts; no state is randomly drawn in this audit.
- S1 retains one corrective state plus up to two available pre-intervention and two available post-intervention FULL states from the stored trigger suffix.
- S2 retains one corrective state plus up to K available FULL suffix states for K=2/4/6.
- S3 first chooses one route uniformly per W sample, then applies S0/S1/S2 in expectation.

## State identity

The semantic pre-layer state ID hashes `(source contract, feature schema, UID, layer, action prefix strictly before layer)`. Because Phase 56 stores the state entering the chosen layer, neither the current action nor future actions belong in this identity. The audit additionally hashes the concatenated BF16 bytes of `text_final`, `text_mean`, and `visual_mean` for every A/B state row and checks shard/index alignment. This quantifies exact feature duplication without deleting or rewriting source tensors.

## Provenance

- Git commit: `6c07e0aa1f0f1469c399b0b21caed9fa7f6f3ef2`; dirty worktree: `true`; status hash: `328300471d3af26b5aecc558d391977fe88fe0ba126f5f7a52bcf60ab6e57a33`.
- Config/plan/script/module hashes: `05d053c5967c99a06ded041231e23a817f11f55023636a09ddf221672ef74d2a`, `c0b0ed9b0423a3a27acda97c4f2cea593ade64e2b707badf5e23038433533159`, `84591d4cde6a6bb4c21ce56a5b9ac36640d2389b268f2185821d38a29606c4fe`, `c7a74182214a4b80c30ab3f420ab75b1dbfdd198c7533879e61664872b66d74f`.
- All 977 Phase-56 artifacts were hash-verified before analysis.

This audit describes discovered supervision only. Observed successful actions are not an exhaustive action-validity oracle.
