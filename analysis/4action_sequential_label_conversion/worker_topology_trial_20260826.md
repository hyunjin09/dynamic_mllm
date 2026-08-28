# Sequential Four-Action Worker-Topology Trial

## Decision

Retain two replicas per GPU (16 workers over eight H100s). Neither the
three-replica nor the repeated one-replica candidate passed the prospective
matched-window throughput gate.

## Comparison

The baseline and candidate used the same frozen 12,278-sample source manifest,
the same descending estimated-cost queue, the same unified executor and exact
sequential branching policy, and the same eight H100 / 64 CPU / 180G Slurm
allocation. WeMath was deferred in both launches so the measured queue was
GQA/TextVQA/ChartQA only.

| Metric at 551 seconds wall time | 16 workers / 2 per GPU (job 1629) | 24 workers / 3 per GPU (job 1631) |
|---|---:|---:|
| Committed samples | 5 | 5 |
| Committed estimated-cost units | 4,192 | 4,151 |
| Cost-throughput ratio vs baseline | 1.000 | 0.990 |
| GPU utilization while active | 98–99% | 98–99% |
| Approximate memory per H100 | 33–34 GiB | 50–50.5 GiB |
| Failure files | 0 | 0 |

The prospective keep threshold was at least 1.10x baseline cost throughput with
zero correctness/runtime failures. The observed 0.990x result is neutral and
does not pass. An earlier 518.7-second snapshot showed 0.248x because four
candidate records committed in a burst immediately afterward; the final
matched 551-second window is used for the decision.

## Outcome and provenance

- Job 1631 was canceled cleanly after 9:11.
- Its five records remain isolated under
  `datasets/mcts_labels_4action/sequential_branching_three_replicas_v1/` and
  are not merged into the active 16-worker output.
- The active 16-worker output retained 262 prior WeMath records plus 33 VQA
  records from job 1629, all checksum-backed and with zero failures.
- VQA-first job 1632 and dependent WeMath-last job 1633 restore the accepted
  16-worker topology.

## One-replica follow-up (stopped before decision gate)

The user subsequently requested an 8-worker / one-replica-per-H100 check. Job
1634 used another isolated output and the same VQA-first descending-cost queue.
All eight replicas loaded cleanly at approximately 16.7 GiB/H100, with sampled
GPU utilization around 24–33% and zero failures.

The user requested fallback to 16 workers before the planned 551-second gate.
At the clean cancellation boundary of 440 seconds, the partial matched result
was:

| Metric at 440 seconds wall time | 16 workers / 2 per GPU (job 1629) | 8 workers / 1 per GPU (job 1634) |
|---|---:|---:|
| Committed samples | 2 | 4 |
| Committed estimated-cost units | 1,682 | 3,382 |
| Cost-throughput ratio vs baseline | 1.000 | 2.011 |

This is promising early evidence for one replica, but it is not a completed
topology verdict because the prespecified 551-second comparison was stopped at
the user's direction. Job 1634's four records remain isolated and are not
merged into the accepted labels.

## One-replica repeat (completed decision gate)

At the user's request, job 1638 repeated the 8-worker candidate from a fresh
isolated output root. It used the same frozen manifest, VQA-first descending-
cost policy, executor, eight-H100 allocation, and 551-second wall-clock window
as the baseline. The prospective rule was to retain 8 workers only at at least
1.10x baseline estimated-cost throughput with zero failures.

| Metric at 551 seconds wall time | 16 workers / 2 per GPU (job 1629) | 8 workers / 1 per GPU (job 1638) |
|---|---:|---:|
| Committed samples | 5 | 5 |
| Committed estimated-cost units | 4,192 | 4,282 |
| Cost-throughput ratio vs baseline | 1.000 | 1.021 |
| Approximate memory per H100 | 33–34 GiB | 16.7 GiB |
| Failure files | 0 | 0 |

The repeat does not pass the 1.10x gate. The first trial's apparent 2.011x
advantage at 440 seconds was therefore not reproduced over the complete
window; varying sample completion times and queue composition materially affect
short snapshots. Job 1638 was canceled after the decision snapshot. Its seven
eventual records remain isolated under
`datasets/mcts_labels_4action/sequential_branching_one_replica_v1/`; the five
records committed inside the matched window define the comparison.

The accepted output had reached 296 checksum-backed records (31 GQA, one
TextVQA, two ChartQA, 187 WeMath2.0 Standard, and 75 WeMath2.0 Pro) with zero
failures before the repeat. VQA-first job 1641 now runs the accepted 16-worker
topology, and dependent WeMath-last/finalization job 1642 will follow it.
