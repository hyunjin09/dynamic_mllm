# Current-Runtime Dense Replay Parity Report

## Execution status

**Not executed.** No model was loaded and no dense or four-action inference ran.

The deterministic 128-record subset cannot be frozen from the authoritative 8K
population because the complete source manifest is missing. Selecting only
from the transferred 6,917 positive-route derivative would systematically
exclude all 1,083 zero-positive records and would therefore not audit the
specified Stage-1 population. In addition, the transferred derivative has null
stored dense predictions, preventing answer/token parity measurement against
the original A6000 dense run.

All four RTX 6000 Ada GPUs became idle during the audit. Compute availability
was therefore not the stopping reason. The source-validity prerequisite was.

## Required artifacts intentionally not fabricated

```text
replay_subset.json
dense_replay_results.jsonl
```

After the canonical 8K source manifest, dense-label manifest/cache index, and
frozen contract are restored, the same plan can freeze 128 records across all
dataset/correctness cells and run four direct workers with one process per GPU.

