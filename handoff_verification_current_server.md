Handoff verification — current H100 server, 2026-09-14

Repository synchronization and packaged-metadata restoration succeeded. This checkout is sufficient to recover the research context and frozen metadata, but it is not ready to resume Phase88 execution or perform final raw-result verification. No experiment, supervisor, report-generation pipeline, large download, or external transfer was launched. Phase85 remains stopped. This verification ends the currently authorized task.

Git was initially clean on `main` at `6c07e0aa1f0f1469c399b0b21caed9fa7f6f3ef2`, tracking `origin/main`, with no local-only commits. Origin is `git@github.com:hyunjin09/dynamic_mllm.git`. After `git fetch origin`, the branch was zero commits ahead and one behind. All 624 incoming added paths were checked for existing-file or symlink-parent collisions; none were found. `git pull --ff-only origin main` succeeded. HEAD and origin/main both equal the known transfer commit **`7d2feaae86da6192cf7639e13c74912801a7b00f`**; no newer origin commit was available at fetch time.

The earlier agent's ignored environment, datasets/model links, analysis artifacts, outputs, runs, logs, scheduler files/queue, and setup backups were retained. No reset, clean, stash, replacement checkout, commit, or push was performed. The only additions after synchronization are this report, verification evidence, and five restored metadata files; current-server observations were appended to the dataset inventory and machine-local environment state. No research implementation or frozen artifact was edited.

The required initial files were read in order, then the updated AGENTS.md and README.md were re-read. Local policy content survived the pull byte-for-byte:

| Local file | SHA-256 before and after synchronization |
|---|---|
| `ACCESS_POLICY.md` | `b0a5e531251bab79388f6717262961ff6144872b3a05085cad25b07c0e1e7749` |
| `infra/gpu_policy.md` | `e7dcbe39d62c1063a5abe55495fa16641bd21b3c9d58998996f5ed1c20880018` |

Allowed roots are this checkout, `/data/research/datasets`, and `/data/research/models`. Source-server `/mnt/hyemin/...` paths were read only as metadata strings, never inspected. CPU-only verification ran locally, as required by the local policy's mandatory rule, CPU-work section, final summary, and existing environment-state clarification. GPU execution would require Slurm. No GPU allocation or computation was performed.

The research context was reconstructed from [research_handoff.md](research_handoff.md), the active workflow dashboard, [Phase88 memory](workspace/phase_memory/phase_88_benchmark_fixed_rw_schedule.md), the [transfer runbook](handoff/phase88_server_transfer/README.md), its recommended Phase88 summaries and two Phase87 synthesis reports, and the [active plan](plans/benchmark_calibrated_fixed_read_write_schedule_plan.md). Historical plans and raw research caches were not exhaustively read.

The broad hypothesis is that a frozen Qwen2.5-VL-7B model's visual computation can be selectively suppressed to preserve correct answers and rescue failures. READ permits text/control rows to consume visual K/V; WRITE updates visual-token rows for later layers. FULL is READ1/WRITE1, READ_ONLY is READ1/WRITE0, WRITE_ONLY is READ0/WRITE1, and IGNORE is READ0/WRITE0. Same-layer READ uses the entering visual state; WRITE affects later computation.

Established findings in the transferred summaries are scoped: dense-failure prediction works in-domain but is source-sensitive; corrective interventions exist; tested learned Stage2 policies have not produced a deployment winner. The prior external two-stage policy had 3 rescues and 19 regressions, net −16 across 19,960 rows. READ harm exists but tested local and H≤8 representations did not support a useful selector. WRITE has modest structural persistence and verified delayed text divergence, but its local/H≤8 selectors also failed usefulness thresholds. Its H8−H0 Spearman gain of +0.0561, CI [0.0267, 0.0852], is supported but sub-material. These are reported prior results, not newly reproduced experiments. They do not prove impossibility, intrinsic nonlocality, horizon saturation, or that longer/joint planning will work. Phase85 interim search evidence remains preliminary and incomplete.

Phase88 tests whether small labeled calibration sets identify benchmark-specific fixed READ-OFF and WRITE-OFF layers that generalize within each benchmark. Stage1 is disabled; there is no training, sample-dependent routing, or search. Correctness drives independent bit selection; q is secondary. M1 is READ-calibrated-only, M2 WRITE-calibrated-only, and M3 their actually executed combination. The model revision is `cc594898137f460bfe9f0759e9844b3ce807cfb5`, with 28 decoder layers.

| Family | CAL UIDs / groups | TEST UIDs / groups | Selected READ / WRITE layers, zero-indexed |
|---|---:|---:|---:|
| ChartQA | 256 / 118 | 2,500 / 1,509 | 12 / 25 |
| TextVQA | 255 / 160 | 5,000 / 3,166 | 11 / 24 |
| MMMU-Pro | 256 / 127 | 3,204 / 1,479 | 17 / 14 |
| POPE | 252 / 14 | 8,748 / 486 | 7 / 14 |

There are 1,019 CAL and 19,452 TEST UIDs. MMMU-Pro TEST has 1,602 Standard and 1,602 Vision rows; each POPE variant has 2,916. These are frozen held-out partitions, not the full official MMMU-Pro/POPE populations. The global schedule is READ17/WRITE19. WRITE27 is an evaluated zero-effect control excluded from selection. Twenty random draws per family were frozen prospectively; ChartQA has 19 distinct routes, with the duplicate retaining its statistical weight. All primary selected bits are non-NONE.

The exact source frontier is the historical observation at **2026-09-14 19:42:52 KST**: random controls had completed all schedules for **15,703/19,452 UIDs**, with no worker error recorded. Dense parity was reported complete for 20,471 UIDs, both calibration sweeps for 28,532 branches each, and M1/M2/M3/global for all TEST UIDs. Supervisor PID2361209 and its four workers belong to the source server. Current source completion and execution ownership are unknown; neither elapsed time nor these PIDs establish local/live status. No source status command or duplicate supervisor was run.

The source supervisor is intended to aggregate after random completion and stop at `analysis_ready_for_interpretation`. Final interpretation/reporting was paused by the user and remains pending. After explicit resumption, the remaining work within the existing plan is to establish source status and obtain synchronized evidence; reconcile complete exact common cohorts; finish paired 5,000-draw image/content-group bootstrap intervals, family/macro, interaction, overfitting, global/random and cost analyses; complete ten figures and four summaries; review conclusions; verify model/source/artifact/raw-record hashes; update compact state; and provide exactly one unexecuted recommendation. This list records the existing scope, not authorization to execute it in this takeover turn. No Phase85 restart, top2, router, or follow-on study is authorized here.

Only TextVQA WRITE and POPE READ satisfy both frozen stability predicates. The saved scope review therefore precludes CAL-A/B/C and leaves qualified CAL-D before held-out efficacy is interpreted. This does **not** establish no held-out benefit. CAL gains are only 3–5 net correct UIDs per selected bit; no split-half top1 agrees. POPE CAL represents only 14 independent images. One CAL and four TEST TextVQA rows have invalid q but remain in correctness; affected calibration pools disable q tie-breaking. Actual efficacy, regressions, interaction and control comparisons must still determine the scientific account.

Packaged metadata was restored with the existing project `.venv/bin/python` and the unmodified runbook helper. The precheck validated all 59 payloads and found five absent destinations. Restoration created only those five files. The subsequent `--check-only` verified **59/59 with zero missing**, preserving exact uncompressed bytes. Restored files were `splits/split_registry.jsonl`, `parity/dense_execution_manifest.jsonl`, `parity/accepted_dense_parent_manifest.jsonl`, and the `heldout/methods_execution_manifest.jsonl` and `heldout/global_execution_manifest.jsonl` files, all under the Phase88 analysis directory. The accepted-parent manifest is superseded provenance, not active Dense evidence.

Additional static verification established:

- Canonical contract hash: `b995483d75b300ef0170db9f06eff2a68801ba4f78fcb922e17951e96ee7492e`.
- Split byte hash: `21dc39f7ca4ac02a47fafda2dd583ed6fb37515a059f8eb74b5547fe42caa512`; 20,471 distinct UIDs, zero cross-split stored image-group IDs or identity keys.
- Execution config equals the contract config; the protocol hash and all 21 bound source hashes match, including both installed Transformers files.
- All three schedule-file hashes and both calibration-manifest hashes match the schedule freeze. Dense/R/W/methods/global manifests contain their exact expected UID sets without duplicates, and completion markers bind the active contract.

These checks verify metadata and source identity. They do not independently reproduce image grouping from all pixels, native generation parity, raw outcomes, or freeze-before-launch chronology. Raw records and the methods launch evidence are absent. Random completion/manifest, aggregate completion, final three summaries, artifact manifest and final verification record are absent from this snapshot.

The current environment is an existing project-local uv environment: uv0.11.29, Python3.12.7, Torch2.6.0+cu124, Transformers5.3.0. All 84 installed distributions match their corresponding entries in the 95-package transfer lock; `uv pip check` passes. Eleven snapshot packages are absent: colorama0.4.6, evaluate0.4.6, lmms-eval0.7.3, loguru0.7.3, lxml6.1.2, portalocker4.3.0, pytz2026.3.post1, sacrebleu2.6.0, scipy1.18.1, tabulate0.10.0, and tenacity9.1.4. The stdlib-only metadata helper needed no installations. The prior environment was preserved rather than rebuilt; complete Phase88 environment bootstrap remains outstanding. Future setup should use the transfer lock, local uv cache/Python storage, and preserve existing package versions. Full imports and H100 execution compatibility were not tested.

The dataset link is `datasets -> /data/research/datasets/dynamic_mllm`. The following checks substituted a candidate local prefix for inventory only; no frozen path or file was changed.

| Asset | Current-server evidence and limitation |
|---|---|
| TEST images | Corresponding paths exist for all 19,452 TEST UIDs. One image hash per family was checked and matched; full image hash/decode verification remains pending. |
| MMMU-Pro/POPE CAL images | Corresponding paths exist for all 256/252 CAL UIDs. One hash per family matched. |
| ChartQA CAL | Expected `stage2_scale_sources/chartqa/.../train/png` materialization is absent; all 256 CAL UID image paths are missing there. |
| TextVQA CAL | Required Phase88 `calibration_images` materialization is absent; all 255 CAL UID image paths are missing at the inventoried local work candidate. |
| Evaluation metadata | All three corresponding `samples.jsonl` files exist. Pinned `eval/sources` and `stage2_scale_sources` directories are absent. |
| Alternative dataset caches | Shared ChartQA and TextVQA cache directories exist under the allowed dataset root. Their revision/content suitability for the missing CAL assets is unverified; no substitution or extraction was performed. |
| Phase88 raw work | No analysis `work` link, no project `outputs/benchmark_calibrated_fixed_rw_schedule_v1`, and no derived EXT directory. Dense/R/W/method/global/random raw records, timing/launch evidence, and failed-attempt payloads are unavailable at these expected locations. No exhaustive unrelated-cache search was performed. |
| Exact model revision | Project model link resolves to the allowed `/data/research/models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898...` snapshot. All 14 contract-listed files exist; all nine non-weight files hash-match. Five weight shards total 16,584,414,560 bytes; full weight hashing remains pending. Blob filenames alone were not accepted as verification. |

The frozen split/config/model and execution manifests contain source absolute paths under `/mnt/hyemin`, outside this server's allowed roots. Moreover, the implementation computes `EXT = datasets.resolve().parent / 'outputs/benchmark_calibrated_fixed_rw_schedule_v1'`, which here becomes **`/data/research/datasets/outputs/benchmark_calibrated_fixed_rw_schedule_v1`**. A work link to an arbitrary directory would not fix this. Before any future resume/report verification, a provenance-preserving relocation mechanism is needed under allowed roots, or separately authorized compatible path exposure. No mapping implementation, symlink change, split regeneration, contract refreeze, or check bypass was made. Final source sync must preserve destination work and exclude machine-local policies, environments, queues and runnable PID state.

The remaining inconsistencies and uncertainties are concrete:

- Older handoff paragraphs still call some READ/WRITE work “not yet tested” or recommend earlier actions. Phase87 synthesis and the active Phase88 section supersede those historical recommendations; Phase85 remains stopped and incomplete.
- The plan's original CAL-D heading says “No held-out benefit,” while the frozen protocol and scope review define a qualified residual category. The reporter's automatic D recommendation must be reviewed against actual complete efficacy/control evidence, without changing the frozen rubric.
- The updated Git AGENTS.md removes the local research-control routing/reviewer sections present in the user's supplied instructions. Those explicit session instructions remain applicable. No research-action selection was made during this mechanical verification.
- GPU-policy section16A and older examples still prescribe Slurm for CPU work, conflicting with its explicit local-CPU mandatory rule, section8, final summary and existing environment clarification. Direct local CPU verification followed the latter. Historical direct four-GPU authority does not transfer to this H100 server; any later resource request must follow local Slurm/QOS permissions.
- The tracked dataset inventory contains source-server “current” claims, while the machine-local environment state predates Phase88. Current-server findings were appended without removing historical provenance.
- Source completion, final incremental sync, full raw hashes, full image/model hashes, reporting dependencies and relocation compatibility remain unverified. Existing completion markers establish saved claims, not a live finished experiment.

Detailed evidence is saved under [workspace/handoff_verification_current_server](workspace/handoff_verification_current_server/): `git_sync.json`, the three metadata-check outputs, `contract_inventory.json`, `static_verification.json`, `environment_inventory.json`, `uv_pip_check.txt`, `asset_inventory.json`, `asset_roots.json`, and `model_inventory.json`. No final Phase88 scientific interpretation was produced. Await the user's next instruction.
