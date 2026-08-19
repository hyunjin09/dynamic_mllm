# We-Math2.0-Pro MCTS Scoring-Stall Repair

## Outcome

The stalled eight-worker run was repaired and resumed without discarding or
overwriting completed labels. The active job is Slurm `100398` on `node06`,
using six GPUs, 72 CPUs, and 180 GB RAM.

## Direct observation

Ranks 3 and 6 of job `99850` stopped publishing after 84 and 73 records. Their
GPUs retained roughly 21–22 GB but remained at 0% utilization for approximately
16.0 and 18.7 hours, while the other six ranks continued. Both Python workers
remained alive and CPU-active. No terminal error was written.

## Diagnosis and repair

The initial diagnosis was **suspected**, not proven: MathRuler calls
`sympy.simplify` without a timeout, and its source notes that SymPy may hang.
The repaired evaluator:

- preserves the unchanged MathRuler result on normal completion;
- bounds each unique decoded prediction's grading at five seconds;
- records a timeout as `scoring_timed_out=true`;
- assigns timeout score `0.0` and incorrect status conservatively;
- caches identical decoded predictions within a sample.

Twelve focused tests passed, including a synthetic nonterminating scorer,
unchanged fast-score behavior, timeout-result caching, and explicit compatible
contract resume behavior.

## Preservation and contracts

The resume audit accepted exactly 1,156 complete predecessor-contract records
and identified 3,388 remaining records. The old job was cancelled only after
this audit and its checksum passed.

- Predecessor contract:
  `96b2c632ebc6e020c607b3d9a0eddd2a29f7aff1912f5219327ae96a507c3a50`
- Amended contract:
  `fc4a1df38925d20816770b861989b87d119bcdbf13b3bdff26a89b7abc90d485`
- Audit:
  `outputs/label_regeneration/wemath2pro_v1/resume_compatibility_audit_v2.json`
- Amended contract artifact:
  `outputs/label_regeneration/wemath2pro_v1/frozen_execution_contract_v2.json`

The six-worker shard layout changes only sample assignment. Each sample retains
its UID-derived seed and identical MCTS/executor contract.

## Restart validation

All six ranks completed a new-contract terminal record with zero errors, while
prior completed records were counted as skipped. The two records associated
with the old stalls, `wemath2pro:591` and `wemath2pro:676`, each completed all
600 requested simulations. Neither recorded a scoring timeout. Consequently,
the exact cause of the old stalls remains **unknown**; the bounded scorer is a
valid nontermination guard but was not demonstrated to be the mechanism that
allowed those two samples to complete.

No predictor training was started.
