# NF0055 B10-1 Pilot

This pilot runs one ZEDProfiler extraction through Nextflow on Alpine.

Data staged on PetaLibrary:

```text
/pl/active/koala/nf1-3d-pilot-workflow-db/data/NF0055_T1/zstack_images/B10-1/
/pl/active/koala/nf1-3d-pilot-workflow-db/data/NF0055_T1/segmentation_masks/B10-1/
```

Manifest:

```text
3a.nextflow_pilot/manifest/nf0055_b10_1_alpine.yaml
```

## Run

From a repo checkout on Alpine:

```bash
cd /pl/active/koala/nf1-3d-pilot-workflow-db/NF1_3D_organoid_profiling_pipeline/3a.nextflow_pilot
make check
UV_CACHE_DIR=/pl/active/koala/nf1-3d-pilot-workflow-db/tools/uv-cache \
  uv sync --project environments
make submit-dry-run RUN_ID=nf0055-b10-1-zp-benchmark ACCOUNT=amc-general
make submit RUN_ID=nf0055-b10-1-zp-benchmark ACCOUNT=amc-general
```

Use `make submit` for benchmark runs instead of a foreground `make run` over
SSH. The coordinator job owns the Nextflow process and survives local SSH
disconnects.

If already inside a durable scheduler session with `nextflow` available:

```bash
NXF_HOME=/pl/active/koala/nf1-3d-pilot-workflow-db/tools/nextflow-home \
NF1_PILOT_PROJECT=/pl/active/koala/nf1-3d-pilot-workflow-db/tools \
UV_CACHE_DIR=/pl/active/koala/nf1-3d-pilot-workflow-db/tools/uv-cache \
  make run RUN_ID=nf0055-b10-1-zp-benchmark PROFILE=curc_alpine ACCOUNT=amc-general
```

## Benchmark

After the run finishes, check:

```text
/pl/active/koala/nf1-3d-pilot-workflow-db/results/nf0055-b10-1-zp-benchmark/run_record.json
/pl/active/koala/nf1-3d-pilot-workflow-db/results/nf0055-b10-1-zp-benchmark/resource_usage.txt
/pl/active/koala/nf1-3d-pilot-workflow-db/results/nf0055-b10-1-zp-benchmark/trace.tsv
```

Use `run_record.json` for total elapsed seconds and per-feature timing.
Use `resource_usage.txt` for `/usr/bin/time -v` wall time and peak RSS.
Use `trace.tsv` for the Nextflow task runtime, status, work directory, and
Slurm native job ID.

## Alpine notes from 2026-08-10

- The local SSH config exposes Alpine as `alpine`; `ssh_alpine` was not a
  configured host alias.
- The remote checkout used for testing is:

```text
/pl/active/koala/nf1-3d-pilot-workflow-db/NF1_3D_organoid_profiling_pipeline
```

- The default Alpine login environment did not have `nextflow` on `PATH`, and
  default Java was OpenJDK 8. A project-local Java 21 runtime and Nextflow
  launcher were installed under:

```text
/pl/active/koala/nf1-3d-pilot-workflow-db/tools/jdk-21/
/pl/active/koala/nf1-3d-pilot-workflow-db/NF1_3D_organoid_profiling_pipeline/3a.nextflow_pilot/.nextflow/nextflow
```

- Alpine reported `$HOME` as 99% full. Keep `NXF_HOME`, `NF1_PILOT_PROJECT`,
  and `UV_CACHE_DIR` under `/pl/active/koala/nf1-3d-pilot-workflow-db/tools`.
- A foreground `make run` over SSH submitted the feature task, but a later SSH
  timeout detached the local Nextflow controller. Prefer `make submit`.
- Early attempts found and fixed Nextflow 26 parser issues from top-level
  `def` declarations, a `baseDir` path issue, strict NaN handling for
  `DNA_Texture`, and duplicate ZEDProfiler metadata columns during profile
  merging.
- The first scheduler-owned full feature attempt reached the merge stage after
  19m23s and used about 21.5 GB peak RSS before failing on duplicate metadata
  columns.
- The merge-fix run completed successfully:

```text
run_id: nf0055-b10-1-zp-benchmark-20260810T2310Z
coordinator_job: 31053374
feature_job: 31053382
nextflow_duration: 23m 10s
feature_walltime: 22m 33s
script_elapsed_seconds: 1320.222
max_rss_resource_usage: 18,746,160 KB
max_rss_slurm_batch: 20,526,348 KB
output_rows: 11
output_columns: 897
validation_status: pass
quality_warning_count: 4
```

The four quality warnings are non-finite texture features in the DNA, ER, AGP,
and Mito texture stages. The merged profile still passed structural validation:
all expected feature families were present, all 11 nuclei objects were present,
metadata matched the manifest, and no feature column was entirely null.
