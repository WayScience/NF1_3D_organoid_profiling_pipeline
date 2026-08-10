# ZEDProfiler uv Environment

This folder is intentionally independent from the repository root environment.
The pilot process runs Python through:

```bash
uv run --project 3a.nextflow_pilot/environments python
```

The environment pins `zedprofiler==0.1.2` for the Phase A/B pilot. Build it on
the execution system before the first real run:

```bash
cd 3a.nextflow_pilot
uv sync --project environments
make doctor
```

If the cluster blocks package resolution from compute nodes, run `uv sync` from
an allowed login or build node first so the shared `UV_CACHE_DIR` is populated.
The CURC profile stores uv cache files under `NF1_PILOT_PROJECT` by default.
