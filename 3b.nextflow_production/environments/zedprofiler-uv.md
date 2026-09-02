# ZEDProfiler uv Environment

This folder is intentionally independent from the repository root environment.
The production workflow runs Python through:

```bash
uv run --project 3b.nextflow_production/environments python
```

The environment pins `zedprofiler==0.1.4`, the current PyPI release.
`0.1.3` carried [WayScience/ZedProfiler#51](https://github.com/WayScience/ZedProfiler/pull/51)'s granularity speedup -- benchmarked in `3a.nextflow_pilot/README.md`'s 2026-08-12 notes, ~2-3x faster and ~3x less peak RSS than the pre-#51 baseline.
`0.1.4` adds [#52](https://github.com/WayScience/ZedProfiler/pull/52) (Texture undersized-object warning), [#53](https://github.com/WayScience/ZedProfiler/pull/53) (empty-feature-frame ID columns), and, most importantly, [#54](https://github.com/WayScience/ZedProfiler/pull/54) (anisotropic-spacing correctness fixes to Volume/BboxVolume/EquivalentDiameter, `*Edge` intensity features, MassDisplacement, texture, and neighbor adjacency) -- benchmarked in `3a.nextflow_pilot/README.md`'s 2026-08-27 notes, ~13-25% faster *and* numerically different (not just faster) from `0.1.3`. See this file's README's 2026-08-28 notes for why this production environment moved off `0.1.3` and into a fresh results directory rather than overwriting the existing warehouse.
Build the environment on the execution system before the first real run:

```bash
cd 3b.nextflow_production
uv sync --project environments
make doctor
```

If the cluster blocks package resolution from compute nodes, run `uv sync` from an allowed login or build node first so the shared `UV_CACHE_DIR` is populated.
The CURC profile stores uv cache files under `NF1_PROD_PROJECT` by default.

There is no warehouse catalog dependency (no Iceberg, no SQLite).
Each run's outputs land directly as plain namespaced parquet-dataset directories under its result directory -- see the README's warehouse-layout notes.
