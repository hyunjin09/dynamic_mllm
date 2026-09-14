# Phase 85 stop report

Stopped at 2026-09-13T21:37:50.635129+09:00 per the explicitly authorized branch-critic plan.

- Running: B128 search and Hamming-2 enumeration on four GPUs. Supervisor snapshot shows no adaptive extension had started.
- Supervisor PID 2138508; workers 2138512, 2138513, 2138514, 2138515; process group 2138508. Direct local execution; no Slurm IDs.
- Sent SIGTERM to the verified supervisor first to prevent new submissions, followed by SIGTERM to verified workers. An already-exited worker caused ProcessLookupError during the first stop script; a second identity-checked pass terminated the remaining worker and verified zero live phase processes. No SIGKILL or unrelated-process termination.
- Last complete B128 UID work records: 776. See original supervisor status and copied complete work records in old_phase_stop_snapshot.
- Preserved all completed and intermediate artifacts, caches and the earlier PRELIMINARY interim report at analysis/read_only_bounded_planning and /mnt/hyemin/qwen_train_eval/outputs/read_only_bounded_planning_v1. No old output was deleted or overwritten.
- Preserved pre/post process lists, pre-stop artifact size/mtime inventory, original status and completed work record copies.
- No live bounded-planning supervisor, worker or finalizer remains. No new bounded-planning jobs will be submitted.

The old supervisor_status.json is preserved unchanged and may say running; this report supersedes its stale liveness field.
