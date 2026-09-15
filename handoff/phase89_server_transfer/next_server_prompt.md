You are taking over Dynamic MLLM on another server. Work in English.
The user stopped Phase89 and requested a Git handoff. Do not implement research
changes, submit jobs, restart a supervisor or launch experiments in this takeover.

1. Read local ACCESS_POLICY.md first. If missing, stop and ask for allowed roots
   before any other inspection. Then read AGENTS.md, infra/gpu_policy.md, README.md.
   Retain destination machine policies and inspect only allowed roots.
2. Inspect Git status, branch, remotes and local-only commits. Preserve all local
   work. Fetch origin; fast-forward only if safe. Never reset/clean/force-push.
3. Read research_handoff.md's current Phase89 section, workspace/workflow_state.md,
   workspace/phase_memory/phase_89_3b_greedy_route_replay_audit.md, then
   handoff/phase89_server_transfer/{README.md,progress_snapshot.json,research_summary.md}
   and plans/3b_greedy_route_corpus_audit_replay_filtering_plan_v2.md.
4. Follow the transfer runbook to restore and verify packaged metadata without
   overwriting different files. Use project-local uv if needed. Inventory missing
   external assets under allowed roots; do not download large payloads yet.
5. Record environment/path compatibility and preserved source hash bindings.
   Source H100 labels, PIDs, job IDs and ETAs are historical on this server.
   Do not interpret old Slurm job numbers as destination jobs. Do not edit frozen
   contracts, disable verification or regenerate populations to bypass mismatch.
6. Write handoff_verification_destination_server.md: Git state; exact stopped
   frontier; established/unsupported findings; available/missing assets; path and
   runtime blockers; remaining plan work and source/destination label risks.
   Stop after verification and wait for the user to authorize any continuation.

Facts to preserve: source2994/2957 cancelled; all10,000 Dense results complete;
734,481routed-stream records;1,858complete and31partial routed samples;0saved
errors; final cohort/RS/readiness reporting incomplete. Phase85 stays stopped.
Phase88 completion on its original source server remains unknown. No geometry,
finetuning, pair rebuilding or automatic replay restart is authorized.
