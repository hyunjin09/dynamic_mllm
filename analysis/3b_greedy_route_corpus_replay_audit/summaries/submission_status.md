Current status (2026-09-14 KST): no active or pending jobs for this audit.

GPU job 2939 started at 21:14:13 and FAILED at 21:15:11 (58 seconds, exit 1).
Its dependent CPU report job 2940 was CANCELLED because the prerequisite failed.
The earlier pending status was a submission-time snapshot.

The fixed gate matched 31/32 current native HF versus binary all-on anchors.
For `textvqa:textvqa_19417`, native HF generated `22` (score 0), while binary
all-on generated `23` (score 1). The packaged source gate recorded `22` for
both implementations. All repeated sparse checks agreed across eight workers.
No anchor raised a runtime exception. The gate deliberately stopped full replay.
The cause of this output mismatch is unknown; the recorded Torch difference
alone does not establish causation. No identical resubmission was made.

Original corpus, frozen contract and failed-attempt evidence are preserved.
CPU census and input verification remain complete. Full replay, filtering and
reporting remain unfinished, pending resolution of the fixed gate failure.

Evidence: `replay_gate/gate_result.json`, `replay_gate/job_2939/`,
`logs/jobs_2939_2940_terminal_accounting.txt`, `summaries/job_2939_failure.json`.
