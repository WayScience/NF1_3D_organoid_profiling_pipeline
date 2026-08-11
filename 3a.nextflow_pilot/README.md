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
