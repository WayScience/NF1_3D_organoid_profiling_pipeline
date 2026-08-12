# ZEDProfiler uv Environment

This folder is intentionally independent from the repository root environment.
The pilot process runs Python through:

```bash
uv run --project 3a.nextflow_pilot/environments python
```

The environment normally pins `zedprofiler==0.1.2` for the Phase A/B pilot,
but is currently pinned to
[WayScience/ZedProfiler#51](https://github.com/WayScience/ZedProfiler/pull/51)
(`zedprofiler @ git+https://github.com/d33bs/ZedProfiler.git@gran-32`) to
benchmark that PR's granularity speedup against the 0.1.2 baseline; see the
README's 2026-08-12 notes. Revert to the PyPI pin once the PR merges and a
release picks up the change. Build the environment on the execution system
before the first real run:

```bash
cd 3a.nextflow_pilot
uv sync --project environments
make doctor
```

If the cluster blocks package resolution from compute nodes, run `uv sync` from
an allowed login or build node first so the shared `UV_CACHE_DIR` is populated.
The CURC profile stores uv cache files under `NF1_PILOT_PROJECT` by default.

The same environment includes `pyiceberg[sql-sqlite,pyarrow]` so each run can
publish a local SQLite-catalog Apache Iceberg warehouse under its result
directory.
