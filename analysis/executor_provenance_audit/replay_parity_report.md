# Recovered Historical Executor Replay Parity

Date: 2026-08-30 KST

## Purpose and scope

This is the single permitted reproduction check. It replays only the existing
Phase-42 W2C smoke set against the authoritative cached route outputs. It does
not regenerate labels, search new routes, execute W2C repair, or modify the
current executor.

## Recovered runtime

The historical implementation was instantiated in a separate ignored source
tree under `outputs/executor_provenance_audit/historical_checkout/`:

1. export commit `838c8527e0d976c04893df5cb28af5d6376be65c`;
2. reverse only the later `capture_online_four_action_route` addition and its
   export;
3. verify all 16 code hashes and the config hash against the authoritative full
   execution contract;
4. put the recovered tree first on `PYTHONPATH` for the audit process;
5. leave the current checkout and label records unchanged.

Hash verification passed 16/16 source paths plus the YAML config. The recovered
executor hash is
`e8c503618998946b4411fb7beb43c42d1be9f8954527064597b1c34ed2571868`.

## Replay set

The replay selection is exactly the tracked Phase-42 set in
`analysis/w2c_when_repair/smoke/smoke_executions.jsonl`:

| Dataset | Samples | Routes |
|---|---:|---:|
| GQA | 4 | 184 |
| ChartQA | 4 | 26 |
| TextVQA | 4 | 102 |
| **Total** | **12** | **312** |

Every one of the 12 authoritative source-record files matches the SHA-256
stored in `analysis/w2c_when_repair/smoke/smoke_manifest.json`. Each tracked
route key is resolved to the original `evaluation` in that source record. The
comparison fields are:

- exact generated token ID list;
- final decoded answer string;
- evaluator correctness boolean;
- dataset and action-composition mismatch patterns.

## Execution

Audit job 1763 was submitted for eight H100s, eight processes, one process/GPU,
Torch 2.6.0+cu124, Transformers 5.3.0, BF16 SDPA, and CUDA module 12.8. Its
launch set deterministic algorithms, disabled TF32, and used
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The job remained pending behind another user's exclusive eight-H100 job. At
the user's instruction to report without running the GPU check, job 1763 was
canceled on 2026-08-30 at 17:16:48 KST. Slurm records state `CANCELLED`, elapsed
time `00:00:00`, no start time, and no GPU allocation. Therefore no replay
route was executed.

| Requested parity result | Measured value |
|---|---:|
| Exact cached-token match rate | not measured |
| Final-answer match rate | not measured |
| Correctness match rate | not measured |
| Per-action mismatch pattern | not measured |
| Recovered executor restores parity | unresolved |

## Independent review condition

The required read-only review agreed that source reconstruction is exact and
the action semantics are scientifically valid, but required a fail-closed
execution decision:

- choose A only if the recovered-source H100 run matches all 312 cached token
  sequences, answers, and correctness outcomes;
- otherwise choose C because some unrecorded execution component remains;
- B is not supported by the line-level READ/WRITE audit.

## Final decision

**C — the complete historical execution contract could not be exactly verified
end to end.**

This fail-closed result needs two qualifications:

1. The historical **source implementation** is exactly reconstructed: all
   16/16 contract-bound source hashes and the frozen configuration hash match.
2. That source implements the intended READ/WRITE scientific truth table, so
   decision B is ruled out by direct code evidence.

What remains missing is execution-parity evidence, not source bytes or semantic
clarity. The canceled H100 replay is the smallest check that could distinguish
the leading hardware/kernel inference from an additional unrecorded runtime
component. No label regeneration, repair, or current-executor modification was
performed.
