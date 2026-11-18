# ZEDProfiler uv Environment

This folder is intentionally independent from the repository root environment.
The z-step workflow runs Python through:

```bash
uv run --project 3c.nextflow_zstep/environments python
```

The environment pins `zedprofiler==0.1.4`, the current PyPI release --
the same pin as `3b.nextflow_production`, deliberately: `0.1.4` carries
the anisotropic-spacing correctness fixes
([WayScience/ZedProfiler#54](https://github.com/WayScience/ZedProfiler/pull/54))
that make physical-space quantities correct for _non-1.0_ z-spacing, which
is exactly the regime this z-step experiment lives in (0.1-1.0 um steps
against 0.1 um xy). Running any z-step older than `0.1.4` would compute
physically wrong volumes/edges for every group but the isotropic 0.1 um
one. It also includes
[#52](https://github.com/WayScience/ZedProfiler/pull/52) (Texture
undersized-object warning) and
[#53](https://github.com/WayScience/ZedProfiler/pull/53) (empty-feature-
frame ID columns) -- benchmarked in `3a.nextflow_pilot/README.md`'s
2026-08-27 notes.
Build the environment on the execution system before the first real run:

```bash
cd 3c.nextflow_zstep
uv sync --project environments
make doctor
```

If the cluster blocks package resolution from compute nodes, run `uv sync` from an allowed login or build node first so the shared `UV_CACHE_DIR` is populated.
The CURC profile stores uv cache files under `NF1_ZSTEP_PROJECT` by default.

There is no warehouse catalog dependency (no Iceberg, no SQLite).
Each run's outputs land directly as plain namespaced parquet-dataset directories under its result directory -- see the README's warehouse-layout notes.
