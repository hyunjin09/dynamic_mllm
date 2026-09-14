# READ-harm mechanism summary

## Matched evidence

- Exact nuisance matching produced only **26 harmful-vs-beneficial pairs** (52 states), plus 595 harmful-vs-stable-wrong and 176 harmful-vs-stable-correct pairs. All matching fixed dataset, source, layer, trigger-relative depth, and visual/text token-count bins.
- Across the 34 preregistered F1-F7 scalars, **1/34** harmful-vs-beneficial group-bootstrap intervals excludes zero. Hence the apparent differences do not form a robust multi-feature READ signature.
- F1 update magnitude does not support larger harmful updates: the largest F1 effect is READ-update norm, standardized effect `-0.139`, raw mean difference `-3.014`, CI `[-11.197, 3.917]`.
- F2 update direction, F3 token concentration, F4 attention mass/entropy, F5 query-key compatibility, and F6 output/value statistics all have their largest absolute standardized effects at or below `0.150`, with intervals crossing zero. The data therefore do not establish different direction, concentration/diffusion, query-key compatibility, or concentration on a few text tokens.
- F7 visual concentration contains the sole interval-excluding-zero result: harmful flips have lower token-index attention spread, standardized effect `-0.732`, raw difference `-0.01826`, CI `[-0.03153, -0.00404]`. The related spatial-spread and spatial-concentration intervals cross zero, so this is a narrow index-concentration association, not evidence that the model attends to a semantically wrong object.
- Dataset/source sign consistency for that largest effect is `1.0` over only **one supported stratum**, which is insufficient for a robust cross-source claim.

## Separability

- Five-fold image-group-disjoint matched probes are below chance: linear AUROC/AUPRC/precision@top10% is `0.4186/0.4679/0.3333`; the two-layer MLP is `0.3254/0.4333/0.1667`.
- Therefore even the strongest behavioral flips are not separable OOF from these local operation statistics.

## Decision and limits

**R-MECH-B — no robust local signature.** The nuisance-matched evidence does not show a repeatable attention/update mechanism for harmful READ. It supports only one narrow association in measured token-index spread. It does **not** establish causal mechanism, attention to a wrong object, semantic error localization, benchmark benefit, or a deployable READ policy.
