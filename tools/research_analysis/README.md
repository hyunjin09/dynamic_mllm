# Research Analysis Layout

Research-analysis source code is grouped here instead of occupying separate
repository-root directories.

| Package | Scope | Former root |
|---|---|---|
| `tools.research_analysis.v2` | v2 Stage B/C and validity analysis | `analysis/` |
| `tools.research_analysis.v3` | v3 four-action and null analysis | `analysis_v3/` |
| `tools.research_analysis.v4` | v4 query-conditional analysis | `analysis_v4/` |
| `tools.research_analysis.query_refinement` | frozen-model query-refinement analysis | `analysis_query_refinement/` |

Run modules from the repository root, for example:

```bash
.venv/bin/python -m tools.research_analysis.v3.reanalyze_stage_b --help
```

Experiment logs created through the scheduler now default to
`runs/experiments/<experiment-id>/run.log`.
