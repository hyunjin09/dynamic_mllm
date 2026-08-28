# Route-Conditioned Four-Action Pilot Report

## Scope

The frozen pilot contains 56 samples selected across
GQA/TextVQA and small, medium, and large validated anchor-OFF-count strata.
Every configuration used the identical manifest and all eight H100s.

## Semantic and numerical gates

- Anchor BOTH_OFF generation, evaluator correctness, and fixed-target scores
  reproduce the current validated anchor within the frozen tolerance.
- Every four-action branch starts from the same anchor pre-layer state.
- Every non-target layer retains its exact anchor FULL/IGNORE action.
- READ_ONLY, WRITE_ONLY, FULL, and explicit pilot IGNORE satisfy their READ,
  WRITE, visual-row, two-call target, and heterogeneous-cache contracts.
- Correct and original-FULL-wrong answer targets remain fixed across states.
- Resume/shard coverage is unique and complete; no disqualifying failure or OOM
  remains.

All semantic and numerical gates passed in both concurrency configurations.

## Throughput benchmark

Useful intervention cells per wall second is the primary optimization target;
GPU utilization is diagnostic only.

| Configuration | Replicas/GPU | New cells/s | Samples/s | Ratio | Peak VRAM | Mean GPU util | Failures/OOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| one_replica | 1 | 8.6138 | 0.3074 | 1.000x | 17475 MiB | 11.4% | 0 |
| two_replicas | 2 | 12.1839 | 0.4349 | 1.414x | 34745 MiB | 27.8% | 0 |

Selected configuration: **two_replicas**
(2 replica(s) per GPU), because it has the highest
passing useful intervention throughput.

## Full-launch gate

**PASS.** The full route-conditioned A+ sweep may launch automatically with
the selected concurrency. Completed pilot artifacts remain separate and are
not reused as production cells.
