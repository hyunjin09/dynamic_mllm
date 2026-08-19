# Binary Visual MCTS Reproduction Bundle v1

This folder is a portable snapshot of the Dynamic MLLM binary visual-routing
executor and unrestricted graph-MCTS label search. It is intended to be copied
to another server, combined with a local dataset manifest and the pinned Qwen
snapshot, and run without depending on the rest of the project repository.

Start with [RUN_MCTS.md](RUN_MCTS.md). It defines the input schema, scoring
contract, parity gate, exact search algorithm, single- and multi-GPU launch,
resume behavior, output schema, completion audit, and failure rules.

The bundle deliberately excludes:

- model weights;
- dataset images and annotations;
- generated route caches;
- predictor/router training code;
- POLAR training and segment-constrained search.

Those exclusions are intentional. The route search is the project/DVR-style
unordered graph MCTS over complete 28-bit binary masks, not POLAR's tri-state
search.

Pinned core runtime:

- model: `Qwen/Qwen2.5-VL-7B-Instruct`;
- revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- PyTorch: `2.6.0`;
- Transformers: `5.3.0`;
- precision: BF16;
- attention backend: SDPA;
- generation: deterministic greedy;
- image processing: native Qwen defaults, with no custom visual-token cap.

Verify the copied folder before using it:

```bash
cd binary_visual_mcts_reproduction_v1
sha256sum --check BUNDLE_SHA256SUMS
```

`bundle_manifest.json` contains the same file inventory in machine-readable
form. If the benchmark evaluator or another bundled file is intentionally
changed, regenerate the inventory and freeze a new execution contract; do not
continue using the old checksums.
