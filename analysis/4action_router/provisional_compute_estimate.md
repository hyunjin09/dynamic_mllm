# Online Four-Action Router Provisional Compute Estimate

This estimate is saved before launch and will be replaced by
`calibrated_compute_estimate.json` automatically after the passed eight-GPU
smoke.

## Frozen workload

| Stage | Route-equivalent executions |
|---|---:|
| Ten training epochs (teacher replay) | 61,440 |
| Ten validation passes (teacher replay + online routed execution) | 17,320 |
| External evaluation (online route + live unified FULL) | 29,920 |
| Total | 108,680 |

The teacher routes average about 28.65 decoder-layer calls because WRITE_ONLY
requires a second target-layer call and is sparse in the authoritative labels.
Online routes are unknown before training and may require 28–56 layer calls.

No directly matched online-router timing exists yet. Using a deliberately wide
20–60 seconds per route equivalent per GPU gives:

| Stage | 8-GPU wall-hours | Allocated GPU-hours |
|---|---:|---:|
| Train + per-epoch validation | 54.7–164.1 | 437.6–1,312.7 |
| External evaluation | 20.8–62.3 | 166.2–498.7 |
| Total | 75.5–226.4 | 603.8–1,811.3 |

This range excludes model-load and final CPU-bootstrap overhead and is not an
ETA claim. The smoke records measured body time and converts it to the same
route-equivalent accounting before main training starts.
