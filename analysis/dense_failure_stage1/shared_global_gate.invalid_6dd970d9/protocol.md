# Shared Stage-1 Global Risk Gate Protocol

- Frozen contract: `6dd970d9a6a76b1ce859143358244bb2b6cc1540c028124802fdb5bc43fa59f4`
- Phase-48 contract: `3cf49a46d0a47a0ae1955f8a518f5a74bef65b7de8736980a6e3d26e170234cd`
- Exact frozen split: 6,399 train / 800 validation / 800 test; zero UID and image-group overlap.
- One train-only normalization is pooled across all train samples and all 28 layers.
- Four configs only: state-only All-28, layer-only All-28, state+layer All-28, state+layer Random-4.
- Architecture: 10,752 -> 256 projection, 32-dimensional layer embedding when used, 256-dimensional GELU head, no dropout.
- Optimization: deterministic FP32 AdamW, lr 5e-4, weight decay 0.01, cosine schedule, 10 epochs; selected by minimum full-28 validation BCE.
- Global thresholds are the most permissive tie-safe validation trajectory thresholds under strict `p > tau` at 99%/98%/95% correct preservation.
- State/layer-only controls remain validation-only. The two state+layer schemes and their windows are frozen before one aggregate test pass.
- The validation-designated primary scheme cannot be switched from test results.
- A trigger is admission only; no treatment or OOD evaluation is executed.
