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

The split fan-out workflow exposes resource knobs for extraction stages:

```bash
make submit RUN_ID=nf0055-b10-1-zp-benchmark ACCOUNT=amc-general \
  GRANULARITY_MEMORY="64 GB" \
  NONGRANULARITY_MEMORY="24 GB"
```

By default, `make submit` also creates or reuses a shared worker virtualenv at:

```text
/pl/active/koala/nf1-3d-pilot-workflow-db/tools/zedprofiler-uv-env
```

The coordinator runs `uv sync` once before launching Nextflow. Worker tasks then
execute the shared environment's `bin/python` directly instead of running
`uv run`, avoiding concurrent mutation of the same virtualenv. Override the path
with `UV_PROJECT_ENVIRONMENT=/path/to/env` if needed.

Use `make submit` for benchmark runs instead of a foreground `make run` over
SSH. The coordinator job owns the Nextflow process and survives local SSH
disconnects.

Before reusing an experiment `RUN_ID`, clear only that run's output directory:

```bash
make clear-output-dry-run RUN_ID=nf0055-b10-1-all-compartments-20260811
make clear-output RUN_ID=nf0055-b10-1-all-compartments-20260811
```

This removes `${RESULTS_ROOT}/${RUN_ID}` and refuses to run without an explicit
`RUN_ID`.

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
/pl/active/koala/nf1-3d-pilot-workflow-db/results/nf0055-b10-1-zp-benchmark/warehouse/warehouse_manifest.json
/pl/active/koala/nf1-3d-pilot-workflow-db/results/nf0055-b10-1-zp-benchmark/resource_usage.txt
/pl/active/koala/nf1-3d-pilot-workflow-db/results/nf0055-b10-1-zp-benchmark/trace.tsv
```

Use `run_record.json` for total elapsed seconds and per-feature timing.
Use `resource_usage.txt` for `/usr/bin/time -v` wall time and peak RSS.
Use `trace.tsv` for the Nextflow task runtime, status, work directory, and
Slurm native job ID.

The full image-set manifest now includes all four compartments. A full run
writes one profile parquet per compartment:

```text
nuclei_profiles.parquet
cell_profiles.parquet
cytoplasm_profiles.parquet
organoid_profiles.parquet
```

The analysis output is also published as a local Apache Iceberg warehouse:

```text
warehouse/
  catalog.db
  warehouse_manifest.json
  images/
    image_assets/
  profiles/
    nuclei_profiles/
    cell_profiles/
    cytoplasm_profiles/
    organoid_profiles/
```

`warehouse_manifest.json` records namespace-qualified table names, roles,
formats, join keys, columns, row counts, table locations, and current Iceberg
metadata locations. A compatibility copy is written at the run root as
`warehouse_manifest.json`, but the canonical manifest lives under `warehouse/`.

The workflow writes the warehouse directly into the final results directory.
Iceberg metadata contains absolute table, manifest, and data-file paths, so the
warehouse should not be created in a Nextflow scratch directory and copied into
place afterward.

## Channel and Compartment Roles

The pilot follows the repository channel mapping in `config/channel_mapping.toml`.
Raw image filenames use wavelength tokens, while ZEDProfiler feature columns use
the biological channel names:

| Filename token | Channel name | Segmentation role                                                                                   |
| -------------- | ------------ | --------------------------------------------------------------------------------------------------- |
| `405`          | `DNA`        | Primary source for the `Nuclei` mask. Also seeds `Cell` segmentation.                               |
| `488`          | `ER`         | Measured inside each compartment mask.                                                              |
| `555`          | `AGP`        | Primary source for `Cell` and `Organoid` segmentation. `Cytoplasm` is derived from `Cell - Nuclei`. |
| `640`          | `Mito`       | Measured inside each compartment mask.                                                              |
| `TRANS`        | `BF`         | Present in source data, but not used by the current ZEDProfiler pilot.                              |

The manifest records these relationships with `channel_codes`,
`compartment_primary_channels`, `compartment_primary_channel_codes`,
`compartment_seed_channels`, `compartment_seed_channel_codes`, and
`compartment_segmentation_methods`. Each output profile row also includes
`Metadata_Segmentation_PrimaryChannel`,
`Metadata_Segmentation_PrimaryChannelCode`,
`Metadata_Segmentation_SeedChannel`,
`Metadata_Segmentation_SeedChannelCode`, and
`Metadata_Segmentation_Method`.

## Alignment Contract

ZEDProfiler does not register, resample, or otherwise align channels to masks.
The source code loads arrays into `ImageSetLoader.image_set_dict`, and
`ObjectLoader` / `TwoObjectLoader` retrieve the image and label arrays directly
by key. Pixel/voxel correspondence is therefore assumed.

For this pilot, alignment means all channel z-stacks and all compartment masks
must already be in the same `(z, y, x)` coordinate space. The runner records
`alignment_validation.json` and fails before feature extraction if any shape
differs.

For `NF0055_T1 / B10-1`, metadata-only TIFF inspection on Alpine showed all
arrays have the same shape:

```text
DNA       [105, 1527, 1528]
ER        [105, 1527, 1528]
AGP       [105, 1527, 1528]
Mito      [105, 1527, 1528]
Nuclei    [105, 1527, 1528]
Cell      [105, 1527, 1528]
Cytoplasm [105, 1527, 1528]
Organoid  [105, 1527, 1528]
```

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

## Alpine notes from 2026-08-11

The all-compartment single image-set run completed successfully:

```text
run_id: nf0055-b10-1-all-compartments-iceberg-20260811T141040Z
repo_commit: a45e9db
coordinator_job: 31102701
feature_job: 31102710
workflow_slurm_jobs: 2
nextflow_duration: 1h 29m 20s
feature_walltime_trace: 1h 29m 3s
script_elapsed_seconds: 5307.541
resource_usage_walltime: 1:29:02
resource_usage_cpu: 119%
max_rss_resource_usage: 19,803,960 KB
max_rss_slurm_batch: 24,190,828 KB
validation_status: pass
quality_warning_count: 4
```

The workflow job count above includes the scheduler-owned Nextflow coordinator
job plus the one Slurm task submitted by Nextflow for `FEATURIZE_IMAGE_SET`.
It does not include the short `srun` process snapshot used during manual
monitoring.

Profile outputs:

```text
profiles.nuclei_profiles: 11 rows x 903 columns
profiles.cell_profiles: 9 rows x 903 columns
profiles.cytoplasm_profiles: 9 rows x 903 columns
profiles.organoid_profiles: 2 rows x 903 columns
```

Alignment validation passed for all four channels and all four masks at shape
`[105, 1527, 1528]`.

Warehouse validation inside the run passed and produced `images` and `profiles`
namespaces. A follow-up check found that this commit's published warehouse was
not directly loadable from the results directory because the PyIceberg catalog
metadata still pointed at Nextflow scratch paths under `/tmp/nxf...`. The
workflow now writes directly to `params.outdir` to avoid relocating an Iceberg
warehouse after metadata creation.

The 2026-08-11 run output was repaired in place after the benchmark: the
scratch-copied warehouse was archived as `warehouse_scratch_copy_broken`, and a
new loadable warehouse was rebuilt at:

```text
/pl/active/koala/nf1-3d-pilot-workflow-db/results/nf0055-b10-1-all-compartments-iceberg-20260811T141040Z/warehouse
```

PyIceberg scans against the repaired catalog returned matching row counts for
`images.image_assets`, `profiles.nuclei_profiles`, `profiles.cell_profiles`,
`profiles.cytoplasm_profiles`, and `profiles.organoid_profiles`.

### Per-compartment fan-out benchmark

The workflow was also modeled as one Slurm task per compartment followed by one
single-writer warehouse task. This avoids concurrent writes to the same Iceberg
SQLite catalog while letting the ZEDProfiler feature work run in parallel.

Successful run:

```text
run_id: nf0055-b10-1-fanout-compartments-iceberg-20260811T1635Z
mode: per_compartment_fanout
coordinator_job: 31105893
feature_jobs: 31105895, 31105896, 31105897, 31105898
warehouse_job: 31107020
workflow_slurm_jobs: 6
nextflow_exit_status: 0
coordinator_walltime: 26m 20s
feature_critical_path: 25m 26s
warehouse_duration_trace: 34.9s
warehouse_walltime_resource_usage: 0:09.07
validation_status: pass
quality_warning_count: 4
```

Trace durations:

```text
FEATURIZE_COMPARTMENT (2): 22m 27s, 17.6 GB peak RSS
FEATURIZE_COMPARTMENT (3): 22m 30s, 17.6 GB peak RSS
FEATURIZE_COMPARTMENT (1): 24m 29s, 17.6 GB peak RSS
FEATURIZE_COMPARTMENT (4): 25m 26s, 17.6 GB peak RSS
BUILD_WAREHOUSE: 34.9s, 326.7 MB peak RSS
```

Slurm accounting reported the coordinator at `00:26:20`, feature jobs from
`00:21:44` to `00:24:43`, and the warehouse job at `00:00:10`. The run used
more Slurm jobs than the sequential workflow (`6` vs. `2`) but reduced
end-to-end wall time from `01:29:32` to `00:26:20` for the same single image
set, a roughly 3.4x improvement.

The final warehouse loaded successfully through PyIceberg:

```text
images.image_assets: 8 rows
profiles.nuclei_profiles: 11 rows
profiles.cell_profiles: 9 rows
profiles.cytoplasm_profiles: 9 rows
profiles.organoid_profiles: 2 rows
```

### Granularity-channel fan-out benchmark

The workflow now fans out the slow granularity work by `compartment x channel`.
Each compartment also gets one non-granularity task for volume, intensity,
texture, colocalization, and neighbors. A final single-writer task merges the
partials and builds the Iceberg warehouse.

An initial run with `GRANULARITY_MEMORY="24 GB"` failed: one granularity task
was OOM-killed and another raised a NumPy allocation error while allocating a
`105 x 1527 x 1528` float64 intermediate. The default granularity memory was
raised to `64 GB`.

Successful run:

```text
run_id: nf0055-b10-1-granularity-channel-fanout-iceberg-20260811T190813Z
mode: granularity_channel_fanout
coordinator_job: 31115409
worker_jobs: 21
workflow_slurm_jobs: 22
nextflow_exit_status: 0
nextflow_duration: 13m 16s
coordinator_walltime: 00:13:21
cpu_hours: 3.2
validation_status: pass
```

Trace summary:

```text
FEATURIZE_NONGRANULARITY: 4 tasks, 2m 7s to 3m 11s, max 2.3 GB peak RSS
FEATURIZE_GRANULARITY: 16 tasks, 4m 39s to 12m 13s, max 16.3 GB peak RSS
BUILD_WAREHOUSE: 49.7s trace duration, 18.48s /usr/bin/time wall, 1.1 GB peak RSS
```

Slurm accounting reported the granularity jobs with `ReqMem=64G`; most finished
in about 4.5-6 minutes, while the slowest task took `00:11:04` Slurm elapsed.
Compared with the per-compartment fan-out run, this reduced coordinator wall
time from `00:26:20` to `00:13:21` for the same single image set, at the cost of
using 22 Slurm jobs instead of 6.

The final warehouse loaded successfully through PyIceberg with catalog name
`nf1_pilot`:

```text
images.image_assets: 8 rows
profiles.nuclei_profiles: 11 rows
profiles.cell_profiles: 9 rows
profiles.cytoplasm_profiles: 9 rows
profiles.organoid_profiles: 2 rows
```

### Shared uv environment experiment

A first shared-env attempt used the same virtualenv path but still launched
workers through `uv run`. That is unsafe for this fan-out pattern on Alpine: some
workers observed different `/usr/bin/python3.12` patch versions and `uv` removed
and recreated the shared environment while other workers were importing from it.
Observed failures included stale file handles and partially imported
NumPy/SciPy modules.

The worker launcher now uses `${UV_PROJECT_ENVIRONMENT}/bin/python` directly
when a pre-synced environment exists, and only falls back to `uv run` for
task-local environments that do not exist yet. This keeps `uv` out of concurrent
worker execution while preserving the local fallback behavior.
