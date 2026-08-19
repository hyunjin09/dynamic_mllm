# Label Regeneration P4 Cache Audit

Date: 2026-08-12

Decision: **PASS**

P4 audited only the original Dynamic MLLM label population:

- GQA: 4,000 records;
- TextVQA: 2,000 records;
- ChartQA: 2,000 records;
- total: 8,000 records.

The separate WeMath2.0-Pro manifest and route cache were explicitly excluded
and were not loaded or searched.

## Integrity result

All strict conditions passed:

| Condition | Result |
|---|---:|
| Source-manifest records | 8,000 |
| Terminal cache records | 8,000 |
| Contract-valid records | 8,000 |
| Missing UIDs | 0 |
| Unexpected UIDs | 0 |
| Duplicate terminal records | 0 |
| Invalid records | 0 |
| Error records | 0 |
| Temporary records | 0 |
| Zero-byte records | 0 |

The audit verified each record's source UID and immutable source fields,
dataset membership, frozen contract SHA-256, pinned model revision, native
image-processing/no-cap settings, greedy generation configuration, complete
MCTS budget, unrestricted layer-action policy, 28-bit binary candidate masks,
route IDs, ALL-ON and ALL-OFF anchors, score/correctness/reward consistency,
successful-route linkage, minimum-ON successful route, token geometry, and
MCTS trace/candidate linkage.

The frozen contract SHA-256 is:

```text
64f525f5d0a4333e1aeae27f41b9055c8da19a9a0fc566ab3c7db270ea37fc7d
```

The source-manifest SHA-256 is:

```text
6abad68ad6c3a9ca2b1bfc1f5502ea2c61ca0e81d0e42f841bc9e257de5f236a
```

The checksum-bound record index contains 8,000 rows and has SHA-256:

```text
f61eb0ac6c40e0498cdfaa53c328b3de34cbb67733a8fbd44c5bd590db051ebe
```

## Execution provenance

The reconciled records came from the two approved stages of the same frozen
scientific contract:

- original four-worker job `99741`: 2,291 records;
- approved eight-worker resume job `99758`: 5,709 records.

Search-budget integrity counts were:

- 200 simulations: 4,045 records;
- 400 simulations: 2,775 records;
- 600 simulations: 1,180 records.

These counts are reported only as P4 execution-contract checks. Current
ALL-ON outcomes, correction-route prevalence, and route diversity remain P5
and P6 analyses and were not interpreted here.

## Artifacts

- `outputs/label_regeneration/v1/post_generation/cache_audit_v1.json`
- `outputs/label_regeneration/v1/post_generation/cache_audit_v1.json.sha256`
- `outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl`
- `outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl.sha256`
- `runs/label_regeneration/p4_cache_audit_v1.log`

The P4 audit ran as CPU-only Slurm job `100342` on node05. It loaded no model,
generated no answer, changed no route label, and required no failed-record
rerun.

## Next boundary

P4 is complete. The next action is P5 per-sample and current ALL-ON/correction-
route summary construction. P5 was not executed in this action; predictor
training remains blocked through P9.
