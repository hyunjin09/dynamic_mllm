# MCTS Label Relocation — 2026-08-19

## Result

The canonical regenerated MCTS label trees were moved into the shared project
dataset tree without rewriting or splitting their contents.

| Contents | Records | Size | Canonical location |
|---|---:|---:|---|
| GQA (4,000), TextVQA (2,000), ChartQA (2,000) | 8,000 | 23 GB | `datasets/mcts_labels/gqa_textvqa_chartqa_v1/` |
| WeMath2.0-Pro hard-cap-400 v2 | 4,544 | 14 GB | `datasets/math_labels/wemath20_pro_mcts_max400_v2/` |

The mixed 8K label tree remains one self-contained artifact so its raw shards,
derived supervision views, split manifests, provenance, and audits cannot drift
apart.

## Compatibility

The former locations remain continuously usable as relative symlinks:

- `outputs/label_regeneration/v1` ->
  `datasets/mcts_labels/gqa_textvqa_chartqa_v1`
- `outputs/label_regeneration/wemath2pro_cap400_v2` ->
  `datasets/math_labels/wemath20_pro_mcts_max400_v2`

The move used a same-filesystem atomic directory/symlink exchange. This avoided
an unavailable-path interval while WeMath greedy-recovery jobs 101708 and
101709 were reading the former path. Both jobs remained `RUNNING` after the
exchange.

## Integrity evidence

- 8K source manifest: 8,000 rows; SHA-256
  `6abad68ad6c3a9ca2b1bfc1f5502ea2c61ca0e81d0e42f841bc9e257de5f236a`.
- 8K final audit: SHA-256
  `d24278687112fafe50d413f8644353ab95b9ebc71df91b5db5c83de8701f8fb1`.
- WeMath2.0-Pro manifest: 4,544 rows; SHA-256
  `f3a3d8d11c48c508451d819c467f5ed3c91ff369e1931196f43cc7d334920946`.
- WeMath cap-400 resume audit: SHA-256
  `fdbd5ad0afa49610e72fb8a82d50911731afbdfe9787e1aad177f0518954b9fd`.
- All four hashes were recomputed after relocation and matched their stored
  sidecars.

## Retained provenance

The incomplete `wemath2pro_cap400_v1` predecessor and the older
`wemath2pro_v1` lineage were intentionally not merged with the canonical v2
cache. They remain in `outputs/label_regeneration/` as historical provenance.
