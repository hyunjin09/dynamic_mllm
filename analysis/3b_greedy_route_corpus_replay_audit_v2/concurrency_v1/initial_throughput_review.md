# Three processes per GPU: initial throughput review

Checked 2026-09-15T11:36:42.800603+09:00.
Job2994 remains RUNNING with24workers on8H100 GPUs; report2957 waits afterok:2994.

The32-anchor gate and4 sparse probes across24workers passed. All9,323
pilot route outputs across24 complete ChartQA samples matched the saved current
outputs exactly. Pilot wall time270.1s, aggregate
34.52routes/s. Historical same-sample timing
suggests1.90x throughput, but this
is not a controlled isolation of process count: CPU allocation also rose32→96.
The earlier2-process attempt mentioned by the user was not rerun or reconstructed.

Current production:8,414 new routes,8 completed
samples. Independently checked8 complete production samples for
exact source identities, record hashes, unique coverage and unchanged dense reuse.
No worker failures. Latest durable aggregate:728,986/3,907,717 routes.

Observed expensive-first DocVQA throughput:17.53routes/s over433.8s.
Remaining-record projections: pilot rate25.6h;
DocVQA rate50.4h. Use roughly25–55hours as a preliminary replay
planning range (aroundSep16 afternoon–Sep17 evening KST). Dataset and answer
length mix differ; this is not a confidence interval. Report queue/runtime and
final scientific interpretation are additional.

Leave three processes/GPU running. Active assistant monitoring ends after this
requested brief check; the user will notify us when jobs finish. The coordinator
still stops the job on worker/runtime/parity/hash failures. No geometry or training.
