# Repository Root Cleanup — 2026-08-19

## Migrations

| Former root path | Organized path |
|---|---|
| `03_experiments/` | `runs/experiments/` |
| `analysis/` | `tools/research_analysis/v2/` |
| `analysis_v3/` | `tools/research_analysis/v3/` |
| `analysis_v4/` | `tools/research_analysis/v4/` |
| `analysis_query_refinement/` | `tools/research_analysis/query_refinement/` |
| `binary_mcts_label_geometry_and_bce_oracle_report.md` | `reports/binary_mcts_label_geometry_and_bce_oracle_report.md` |
| `binary_polar_full10_polar_matched_results.md` | `reports/binary_polar_full10_polar_matched_results.md` |

No experiment contents were deleted. The scheduler default was changed from
`03_experiments/<id>/run.log` to `runs/experiments/<id>/run.log`, preventing the
old root directory from being recreated. Live Python imports and focused tests
now use the versioned `tools.research_analysis` package paths.

## Verification

- All four organized analysis packages import successfully.
- `infra/gpu_scheduler.py` compiles and its CLI loads successfully.
- Focused analysis suite after migration: 60 passed, one failed—the exact same
  pre-existing failure observed before migration.
- Pre-existing failure:
  `test_truncate_dynamic_cache_removes_only_right_padding_rows`, caused by the
  Transformers 5.3.0 `DynamicCache` API no longer exposing `key_cache`.
- The moved full-training report still passes its stored SHA-256 check.
- Active Slurm jobs 101708 and 101709 remained running throughout.

Historical frozen plans were not rewritten merely to update old directory
examples. Operational source, tests, workspace state, and report indexes use
the organized paths.
