# ZEDProfiler uv Environment

This folder is intentionally independent from the repository root environment.
The pilot process runs Python through:

```bash
uv run --project 3a.nextflow_pilot/environments python
```

The environment pins `zedprofiler==0.1.3`, the current PyPI release.
`0.1.3` carries [WayScience/ZedProfiler#51](https://github.com/WayScience/ZedProfiler/pull/51)'s granularity speedup (previously benchmarked here against a git-branch reference to that PR; see the README's 2026-08-12 notes) -- no git dependency needed anymore.
Build the environment on the execution system before the first real run:

```bash
cd 3a.nextflow_pilot
uv sync --project environments
make doctor
```

If the cluster blocks package resolution from compute nodes, run `uv sync` from an allowed login or build node first so the shared `UV_CACHE_DIR` is populated.
The CURC profile stores uv cache files under `NF1_PILOT_PROJECT` by default.

There is no warehouse catalog dependency (no Iceberg, no SQLite).
Each run's outputs land directly as plain namespaced parquet-dataset directories under its result directory -- see the README's warehouse-layout notes.
