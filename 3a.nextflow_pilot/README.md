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
writes one profile parquet **dataset directory** per compartment (one
parquet file per image set inside, named by that image set's
`Metadata_Imaging_ImageID`; a single-image-set run's directory just has
one file in it). Nothing concatenates these files together, so reading the
whole compartment table is `pd.read_parquet(directory)` or any
multi-file-aware parquet reader:

```text
nuclei_profiles/
  NF0055_T1__NF0055_T1__B10__F1.parquet
  NF0014_T1__NF0014_T1__C4__F2.parquet
cell_profiles/
cytoplasm_profiles/
organoid_profiles/
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

## Alpine notes from 2026-08-12

### ZedProfiler PR #51 (granularity upsampling speedup)

`environments/pyproject.toml` was switched from PyPI `zedprofiler==0.1.2` to
[WayScience/ZedProfiler#51](https://github.com/WayScience/ZedProfiler/pull/51)
via a git dependency on the PR's source branch:

```text
zedprofiler @ git+https://github.com/d33bs/ZedProfiler.git@gran-32
```

This installed commit `1ca0c7d2` (`v0.0.post1.dev48`). The PR replaces
`compute_granularity`'s per-scale full-image upsample with an upsample
restricted to the voxels that actually belong to a labeled object, since
`scipy.ndimage.mean(image, masked_labels, label_range)` discards everything
else anyway. The PR claims ~2.8x speedup on its own benchmark (5.15s to
1.85s) and byte-identical output.

The shared worker virtualenv was resynced from the login node (`uv lock` +
`uv sync` against the updated `pyproject.toml`) before submitting, so compute
nodes never needed GitHub access. A fresh `granularity_channel_fanout` run
against the same `NF0055_T1/B10-1` image set, same resource directives
(`GRANULARITY_MEMORY="64 GB"`, etc.) as the 2026-08-11 baseline, completed
successfully:

```text
run_id: nf0055-b10-1-granularity-channel-fanout-pr51-20260812T122331Z
mode: granularity_channel_fanout
coordinator_job: 31153849
worker_jobs: 21
zedprofiler_version: 0.0.post1.dev48 (git+d33bs/ZedProfiler@1ca0c7d2, PR #51)
nextflow_exit_status: 0
coordinator_walltime: 00:04:26
validation_status: pass
quality_warning_count: 4
```

Comparison against the 2026-08-11 baseline run
(`nf0055-b10-1-granularity-channel-fanout-iceberg-20260811T190813Z`, same
image set, same resource directives, `zedprofiler==0.1.2` from PyPI):

| Metric                                             | Baseline (0.1.2)  | PR #51            | Change            |
| --------------------------------------------------- | ------------------ | ------------------ | ------------------ |
| Coordinator wall time                                | 00:13:21            | 00:04:26            | 3.0x faster         |
| Granularity task duration, mean of 16 tasks          | 317.1s              | 48.5s               | 6.5x faster         |
| Granularity task duration, slowest task               | 642.6s              | 86.3s               | 7.4x faster         |
| Granularity task peak RSS (Nextflow trace)           | 16.3 GB             | 5.2 GB              | ~3.1x less memory   |
| Granularity task peak RSS (Slurm `MaxRSS`, max)      | ~19.8 GB            | ~6.6 GB             | ~3.0x less memory   |
| Total allocated CPU-time (`AllocCPUS x Elapsed`, all Slurm tasks) | 27.55 cpu-hours     | 5.56 cpu-hours      | ~5.0x less          |

Both `AllocCPUS x Elapsed` totals were recomputed directly from each run's
`slurm.tsv` with the same method, since `AllocCPUS` on this partition tracks
requested memory (64 GB implies 18 allocated CPUs per granularity task,
regardless of the `granularity-cpus` directive) rather than actual thread
usage.

Output correctness was verified, not just claimed: the `nuclei_profiles`
table's 64 Granularity feature columns across all 11 objects were
byte-identical between the baseline and PR #51 run (max absolute difference
`0.0`), and every other numeric column matched too. Row counts (11/9/9/2),
column counts (903), quality warning counts (4, all in `Nuclei`), and
validation status (`pass`) were unchanged.

Granularity's peak RSS dropping from ~19.8 GB to ~6.6 GB means a future run
could likely lower `GRANULARITY_MEMORY` well below the current `64 GB`
default (with headroom), which would also reduce the CURC-side allocated-CPU
cost per task since this partition scales `AllocCPUS` to requested memory.
Not changed in this experiment, to keep the resource directives identical
between the two compared runs.

**Update, later the same day:** PR #51 merged and shipped as ZedProfiler
`v0.1.3` on PyPI ("Speed up granularity ~2x by only upsampling labeled
voxels (@d33bs via #51)", confirmed via `gh release view v0.1.3 --repo
WayScience/ZedProfiler`). `environments/pyproject.toml` was switched back
from the PR branch git dependency to a normal PyPI pin,
`zedprofiler==0.1.3`, for all runs from this point forward.

## Alpine notes from 2026-08-12, continued

### Multi-image-set fan-out (two patients, one shared warehouse)

Added an outer fan-out layer over multiple image sets in one Nextflow run,
on top of the existing per-compartment x per-channel fan-out, converging
into a single shared Iceberg warehouse instead of one warehouse per image
set. Backward compatible: a plain `--manifest PATH` invocation (no
`--image-sets`) runs through an untouched code path with byte-identical
output layout to before this change (verified locally: the legacy CLI path
produces the exact same `run_record.json` key set as a pre-change run of the
same fixture).

New `--image-sets PATH,PATH,...` / `IMAGE_SETS` (comma-separated manifest
list) flag on `bin/nf1-nextflow-pilot run|submit` and `make run|submit`,
mutually exclusive with `--manifest`/`MANIFEST`. Each image set's
per-compartment output nests under
`${outdir}/image_sets/<manifest-filename-stem>/compartments/...`; a single
`BUILD_WAREHOUSE` task at the end reads every image set's compartment
outputs and unions them into one warehouse, one row set per compartment
table, keyed by each image set's own (already-unique)
`Metadata_Imaging_ImageID`. `scripts/build_warehouse_from_compartments.py`
gained a repeatable `--image-set MANIFEST COMPARTMENT_ROOT` pair (append),
kept `--manifest`/`--compartment-root` for the single-image-set case, and
now raises a clear error instead of a downstream PyArrow failure if image
sets disagree on compartments, columns, or column dtypes.

**Second image set:** `NF0014_T1/C4-2`, staged from
`~/mnt/bandicoot/NF1_organoid_data/data/NF0014_T1/{zstack_images,segmentation_masks}/C4-2/`
to
`/pl/active/koala/nf1-3d-pilot-workflow-db/data/NF0014_T1/{zstack_images,segmentation_masks}/C4-2/`
(the koala `GFF_Data` mirror used by the main pipeline does not carry a
plain combined `segmentation_masks` folder per patient, only bandicoot
does). All 5 channel TIFFs and all 4 compartment masks confirmed present and
shape-aligned at `(33, 1537, 1540)` before submitting — a notably smaller
z-stack than `NF0055_T1/B10-1`'s `(105, 1527, 1528)`. Manifest built with
the existing, unmodified `scripts/build_manifest.py`.

**A first submission failed** at the final `BUILD_WAREHOUSE` step (all 40
worker tasks across both image sets succeeded) with a PyArrow error:
`Metadata_Imaging_FieldID` was an unquoted int (`1`) in the checked-in
`NF0055_T1/B10-1` manifest but a quoted string (`'2'`) in `NF0014_T1/C4-2`'s
freshly-generated one (`scripts/build_manifest.py`'s current code always
writes a string; the NF0055 manifest predates that convention or was hand
edited). Concatenating the two image sets' profile frames silently upcast
the column to mixed-type `object`, which PyArrow refused to write to
parquet. Fixed the root cause (requoted `Metadata_Imaging_FieldID: '1'` in
`manifest/nf0055_b10_1_alpine.yaml`) and hardened
`build_warehouse_from_compartments.py` to normalize that field to string at
manifest-load time regardless, plus added an explicit dtype-mismatch check
across image sets before any concatenation so a future recurrence fails
with a clear message naming the column and per-image-set dtypes instead of
a bare PyArrow traceback.

Resubmitted run, using the now-0.1.3 environment and `GRANULARITY_MEMORY="24
GB"` (informed by the same-day finding that granularity's real peak RSS is
~5-6.6 GB, not the `64 GB` default) — completed successfully:

```text
run_id: nf0055-nf0014-multi-image-set-20260812T211627Z
mode: granularity_channel_fanout
coordinator_job: 31170628
workflow_slurm_jobs: 42
nextflow_exit_status: 0
coordinator_walltime: 00:05:45
zedprofiler_version: 0.1.3
validation_status: pass
quality_warning_count: 8
cpu_hours: 5.07
max_MaxRSS_across_all_jobs: 6.23 GB
```

Compared with the same-day single-image-set PR #51 run (one image set,
`GRANULARITY_MEMORY="64 GB"`, `coordinator_walltime: 00:04:26`,
`cpu_hours: 5.56`): running **two** image sets together took only ~30% more
coordinator wall time and used *fewer* total CPU-hours, because lowering
`GRANULARITY_MEMORY` to `24 GB` roughly halved `AllocCPUS` per granularity
task on this partition's memory-scaled core allocation (~3.5 GB/core),
letting more of the 32 granularity tasks run concurrently within
`executor.queueSize = 20` and more than offsetting the doubled task count.

The warehouse union was verified by loading it directly through PyIceberg
(not just this pilot's own scripts):

```text
profiles.nuclei_profiles:    56 rows (NF0014: 45, NF0055: 11)
profiles.cell_profiles:      51 rows (NF0014: 42, NF0055: 9)
profiles.cytoplasm_profiles: 51 rows (NF0014: 42, NF0055: 9)
profiles.organoid_profiles:   3 rows (NF0014: 1,  NF0055: 2)
images.image_assets:         16 rows (8 per image set: 4 channels + 4 masks)
```

NF0055's counts (11/9/9/2) exactly match its previously documented
single-image-set baseline, confirming no cross-contamination between image
sets in the union. NF0014 is a distinct organoid with substantially more
detected objects despite its smaller z-stack. All rows carry the correct,
distinct `Metadata_Imaging_ImageID` for their source image set; validation
status is `pass` for every table.

Granularity peak RSS across both image sets topped out at 6.23 GB (Slurm
`MaxRSS`), comfortably inside the `24 GB` request used here — a real
data point (not just NF0055 alone) to inform lowering the persisted
`GRANULARITY_MEMORY` default in a future change.

### Multi-image-set warehouse writer no longer concatenates in memory

The initial multi-image-set implementation above joined image sets by
`pd.concat`-ing every image set's compartment profile frame into one
combined `DataFrame` before writing it out as a single parquet file and a
single Iceberg `table.append()` call. That's fine at 2 image sets, but it
means peak memory in the single `BUILD_WAREHOUSE` process scales with the
*total* number of image sets in a run — the wrong shape for the roadmap's
eventual production-scale run.

Revised so no cross-image-set concatenation happens anywhere. Each image
set's compartment frame (already merged from its own nongranularity +
granularity outputs, a per-image-set step that was already memory-bounded
and unchanged) is now written to its own parquet file and appended to the
Iceberg table as its own data file, in a loop over image sets. `write_table`
in `scripts/iceberg_warehouse.py` takes `frames: list[pd.DataFrame]` instead
of one `pd.DataFrame` and calls `table.append()` once per frame; row counts
are summed across the list rather than read off one concatenated frame.
Peak memory in `BUILD_WAREHOUSE` is now bounded by one image set's data at a
time, regardless of how many image sets are in the run.

The top-level convenience output changed shape to match: `nuclei_profiles`
(etc.) is now a **directory** containing one parquet file per image set,
named by that image set's `Metadata_Imaging_ImageID`
(`nuclei_profiles/NF0055_T1__NF0055_T1__B10__F1.parquet`, ...), not a single
`nuclei_profiles.parquet` file — a "parquet dataset" in the normal sense,
readable as one logical table by `pd.read_parquet(directory)` or any
multi-file-aware reader. This is now the standard shape for every run,
single-image-set or multi: the single/multi distinction that used to gate a
copy-shortcut for the old single-file output was removed along with it, so
`build_warehouse_from_compartments.py`'s parquet-writing path is the same
code regardless of image-set count. (The `run_record.json`/`validation.json`
flat-vs-nested-by-`ImageID` shape decision is unrelated and unchanged.)

Verified locally against the same synthetic smoke fixture used earlier
(`results/split-granularity-aggregate-smoke/compartments`, two throwaway
manifests): both the flat dataset directory and the Iceberg table's own
`data/` directory contained one file per image set, `pd.read_parquet` on the
directory and `catalog.load_table(...).scan()` both returned the correctly
unioned 2-row result, and the legacy single-manifest path still produced the
same `run_record.json` key set as before (still one file, just now inside a
one-entry directory).

Re-ran the full `NF0055_T1/B10-1` + `NF0014_T1/C4-2` experiment end-to-end
after clearing the prior run's output, same `IMAGE_SETS`/`GRANULARITY_MEMORY="24
GB"` arguments:

```text
run_id: nf0055-nf0014-multi-image-set-20260812T215903Z
coordinator_job: 31173131
nextflow_exit_status: 0
coordinator_walltime: 00:05:47
```

Identical to the concatenated run in every observable way except file
count: every profile table and `images.image_assets` now has exactly 2
Iceberg data files (`len(table.scan().plan_files()) == 2`, one per image
set), and the same row counts and per-`ImageID` breakdown as before
(`nuclei_profiles`: 56 total, 45 NF0014 + 11 NF0055; `images.image_assets`:
16 total, 8 + 8) — confirmed by loading the warehouse through PyIceberg
directly, and by reading the flat `nuclei_profiles/` directory through
plain `pd.read_parquet`.

### Production-scale (4200 image-set) time estimate

This is a **projection**, not a measured run — the pilot has only ever run
1-2 image sets. Grounded in two real inputs measured today: per-task
duration from the 2-image-set run's `trace.tsv`, and Alpine's actual live
queue/account limits (queried directly, not assumed).

**Task count is deterministic, not estimated.** Each image set is 4
compartments x (4 granularity channels + 1 nongranularity task) = 20 worker
tasks. At `N = 4200`: `16 x 4200 = 67,200` granularity tasks, `4 x 4200 =
16,800` nongranularity tasks, **84,000 worker tasks**, plus one
`BUILD_WAREHOUSE` and one coordinator = **84,002 total Slurm jobs** for a
single run.

**Per-task duration**, from today's real `trace.tsv` (`realtime` column,
excludes queue wait):

```text
granularity avg:    52.0s  (32 tasks, range 22.1s-96s)
nongranularity avg: 124.9s (8 tasks, range 53.5s-222s)
weighted avg:        66.6s across all worker tasks
```

At `N=4200` and this weighted average, total compute is
`84,000 x 66.6s x 7 AllocCPUS = ~10,878 cpu-hours` (matches a linear
scale-up of today's measured `5.07 cpu-hours / 40 tasks` to `84,000` tasks
within about 2%, a useful cross-check on the model).

**Alpine's real constraints** (queried live today via `sacctmgr`/`sinfo`,
not assumed from documentation):

```text
acpu partition:        420 nodes, 26,976 total CPUs (64/node)
                        23,442 allocated / 2,846 idle at query time (~87% busy)
QOS cpu-normal:         MaxSubmitJobsPerUser = 1000, MaxTRESPerUser = 128 nodes
Association (amc-general, this user): MaxJobs = 200
```

`MaxJobs = 200` is the binding constraint, not partition size: 200
concurrent jobs at 7 `AllocCPUS` each is only 1,400 cores, well inside the
26,976-core partition. **This 200-job ceiling is shared across every QOS
this account uses** (`cpu-normal`, `cpu-long`, `mem-normal`, `interactive`,
the `gpu-*` QOS's, etc.) — any other Alpine work running under this account
at the same time eats into the same budget.

Wall-clock time is throughput-bound: `total_tasks / min(concurrency-limited
throughput, submission-rate-limit)`, where concurrency-limited throughput is
`concurrency_slots x 60 / weighted_avg_duration` tasks/min.

| Scenario | Concurrency | Rate limit | Throughput | Wall time |
| --- | --- | --- | --- | --- |
| A: today's pilot config (`queueSize=20`, `submitRateLimit='20/min'`) | 20 | 20/min | 18.0 tasks/min | **~78 hours (3.2 days)** |
| B: right-sized (`queueSize≈180`, rate raised so it doesn't bind) | 180 | not binding | 162 tasks/min | **~8.6 hours** |
| C: maxed at the account's hard ceiling, zero contention | 200 | not binding | 180 tasks/min | **~7.8 hours** |

**The current pilot config (`conf/curc_alpine.config`: `executor.queueSize =
20`, `submitRateLimit = '20 / 1 min'`) is the dominant bottleneck at this
scale, not compute efficiency or Alpine's real limits.** Both were sized
for a 2-image-set pilot; at `N=4200` they cap throughput at ~18 tasks/min,
while the account's actual `MaxJobs=200` ceiling would support ~180
tasks/min — 10x more headroom than today's settings allow. Getting close to
Scenario B/C requires raising both values before a production-scale run,
not just requesting more resources per task.

**What this estimate does not capture, and why the honest range is wider
than "~8 hours":**

- **Shared-cluster contention.** Scenario B/C assume slots are available the
  instant they're requested. Today's live snapshot showed `acpu` at ~87%
  allocated *before* this workload starts. Getting a sustained 180-200
  concurrent slots on a cluster shared with every other Alpine cpu-normal
  user is not guaranteed at any given moment; Slurm fair-share can leave
  jobs `PENDING` for a queued account well below its `MaxJobs` ceiling. This
  is the single largest source of uncertainty and can't be modeled from one
  point-in-time snapshot — real wall time could plausibly run 1.5-3x past
  the compute-bound estimate depending on when the run lands.
- **`BUILD_WAREHOUSE` doesn't parallelize, and its N=4200 scaling is
  unmeasured.** It's a deliberate single-writer step (see the SQLite-catalog
  discussion above), so it runs after every worker task finishes, entirely
  serially, reading every image set's compartment outputs one at a time
  (including a `tifffile` metadata read per channel/mask for alignment
  validation — `8 x 4200 = 33,600` file opens). Today's only two data points
  (34.9s and 27.8s for `N=1` and `N=2`) are too close together to tell
  whether this is dominated by fixed per-invocation overhead or genuinely
  scales with `N` — a naive linear extrapolation (`27.8s / 2 x 4200 ≈ 16
  hours`) is plausible but not trustworthy from two points. **This needs its
  own measurement at an intermediate scale (e.g. 50-100 image sets) before
  trusting any number here**, since if it does scale linearly it could rival
  the worker-task wall time above.
- **84,002 individual Slurm jobs in one run is a lot of jobs**, independent
  of whether the account's limits technically allow it — this is exactly
  the situation `PLAN.md`'s still-deferred "job arrays for homogeneous
  shards" item exists for. Job arrays reduce Slurm controller overhead per
  task and are the more considerate way to submit this many near-identical
  jobs on a shared cluster, separate from the throughput math above.

**Recommendation:** before committing to a full 4200 run, do a staged
capacity test — e.g. 50 image sets with `queueSize`/`submitRateLimit` raised
toward Scenario B — to get a real `BUILD_WAREHOUSE`-scaling data point and a
real achieved-throughput number under actual same-day cluster contention,
rather than trusting this projection's ~8-hour compute-bound figure as a
commitment.

#### Fair-share priority impact of a run this size

Queried live (`sshare -u $USER -A amc-general -l`, `sacctmgr show
associations`, `scontrol show config`) rather than assumed, since a
4200-image-set run's ~10,878 cpu-hours is enough compute to meaningfully
move this account's fair-share standing, and the real worry is whether that
makes the account unable to get *anything* scheduled afterward.

**Current standing** (`amc-general`, this user):

```text
RawShares:     1        (this user's allocated share within amc-general)
NormShares:    0.001416 (normalized share of the account)
RawUsage:      574,047  (decayed usage so far, all recent -- see below)
EffectvUsage:  0.000126 (this user's share of amc-general's decayed usage)
FairShare:     0.192688 (composite score, 0-1, folds in amc-general's own
                          standing under its parent "amc" account too)
LevelFS:       11.27    (at just this level: using ~1/11th of allocated
                          share -- currently well *under* fair share)
```

`GrpTRESMins`/`MaxTRESMins` are blank for this association — **there is no
hard usage-time cap**. Only `MaxJobs=200` (concurrency, discussed above) is
a hard limit; everything about usage volume is a *soft priority* effect, not
an access restriction.

**How much would this workload move that.** `RawUsage=574,047` is small
enough (~160 cpu-hour-equivalent if the units are raw core-seconds, which
the ratio check above is consistent with) that it looks like mostly recent
activity, not a multi-year accumulation — expected, since `PriorityDecayHalfLife
= 14-00:00:00` (14 days) continuously decays old usage; anything older than
a few months has decayed to statistical noise. Today's four real pilot runs
alone (~20.8 cpu-hours) don't fully explain the current `RawUsage`, so other
Alpine work is also contributing to it recently — this is a working
account, not one starting from zero.

A 4200-image-set run's ~10,878 cpu-hours is **roughly 68x** today's
`RawUsage` (order-of-magnitude, not exact — depends on whether Alpine bills
fair-share usage in plain core-seconds or applies `TRESBillingWeights` I
haven't confirmed). That would swing this user from currently sitting
*under* fair share (`LevelFS=11.27`) to sitting *substantially over it* —
a real, meaningful priority hit, not a rounding error. Worth saying plainly
rather than downplaying it.

**Why that doesn't mean "locked out," concretely:**

- **The hit decays with a 14-day half-life**, continuously, with no periodic
  reset (`PriorityUsageResetPeriod = NONE`) — this *is* the recovery
  mechanism, and it's automatic:

  | Days after the run | Fair-share impact remaining |
  | --- | --- |
  | 0 | 100% |
  | 7 | 71% |
  | 14 | 50% |
  | 28 | 25% |
  | 42 | 12.5% |
  | 56 | 6.2% |

  Effectively back to baseline within 6-8 weeks; already half-gone in 2.

- **Fair-share isn't the only factor, and it's not even the largest one.**
  Alpine's multifactor priority weights (`scontrol show config`):
  `PriorityWeightJobSize=40320`, `PriorityWeightQOS=30240`,
  `PriorityWeightAge=20160`, `PriorityWeightFairShare=20160`,
  `PriorityWeightPartition=0`, `PriorityWeightAssoc=0`. **Age and FairShare
  are weighted exactly equally**, and QOS and JobSize both outweigh
  FairShare on their own.
- **The anti-starvation guarantee is explicit and load-bearing here**:
  `PriorityMaxAge = 14-00:00:00`. A job's age-based priority contribution
  grows the longer it waits and **saturates at 14 days**, at which point it
  contributes the same as a perfect `FairShare=1.0` job would. Combined with
  FairShare and Age being equally weighted, this is a hard guarantee that no
  job queues forever because of this run's usage — worst case, a
  low-priority job waits toward that ceiling once, not indefinitely, and
  only during periods where the partition is actually contended (right now:
  ~87% allocated, so this is a real possibility, not theoretical).

**One implication for the earlier wall-time estimate, not previously
modeled**: `RawUsage` accrues continuously as each task completes, not just
at the end of the run. On an 8+ hour run, the fair-share hit from the
*first* few thousand completed tasks is already partially in effect by the
time the *last* few thousand submit — meaning, under real contention, a run
this size could plausibly see its own back half scheduled slightly slower
than its front half, self-throttling somewhat as it progresses. This
compounds with the shared-cluster-contention uncertainty already flagged
above rather than replacing it.

**Bottom line for "could I end up unable to use anything":** no — there's
no hard cap that blocks submission or execution, and the 14-day max-age
guarantee means no individual job can be starved indefinitely regardless of
how low fair-share drops. What's real is a multi-week window (roughly
matching the 14-day half-life, mostly faded by 4-6 weeks) where *other*,
unrelated Alpine jobs from this account would compete less favorably against
other users' jobs during periods of partition contention, and would likely
see longer queue waits than usual during that window — an inconvenience,
not a lockout.

#### Open questions for CURC, checked against formascute first

[formascute](https://github.com/d33bs/formascute) (this pilot's sibling
characterization project) already has a live CURC contact (Gregory Way) and
a substantial answered/still-open question history from `2026-08-06`/`07` —
see its `docs/alpine-findings.md` and `.agents/skills/alpine.md`. Checked
that history before adding anything here, specifically to avoid re-asking
what's already answered.

**Already answered by CURC — do not re-ask:** the `MaxJobs=200` cap is
real and `queueSize=200` with no rate limit is explicitly fine for a
many-short-task workload; Apptainer/Singularity is CURC's preferred runtime
over `uv`/conda (though this pilot has used `uv` successfully throughout);
`Persistence1`'s per-user cgroup caps are quantified exactly (~1.6 GB RAM,
80% of 8 CPUs); `cpu-long` is for walltime >24h only, not a priority
shortcut; GPU work needs a separate partition/QOS sharing the same 200-job
budget; institution-level fairshare methodology is confirmed (see below).

**Gap this pilot's own analysis (above) missed, that formascute already
flagged:** fairshare has a *second*, institution-level number
(`levelfs $USER` reports both `LevelFS_User` and `LevelFS_Inst`) that
this session never checked — only user- and `amc-general`-account-level
standing were queried. formascute's `2026-08-07` reading had
`LevelFS_Inst≈1.01` for institution `amc` — parity, not headroom, and not
something this project's own usage controls. Re-check `levelfs $USER`
(both numbers) before a real production run, not just `sshare`.

**New questions, specific to this pilot's actual architecture, that
formascute's own estimate doesn't cover:**

1. **Task-count shape is well outside what CURC signed off on.** Formascute's
   4200-image-set estimate (and CURC's "`queueSize=200` is fine" answer)
   assumed **one task per image set** (4200 total). This pilot's actual
   implementation fans out per compartment x channel: **20 tasks per image
   set, 84,000 total at N=4200** — 20x formascute's baseline, beyond even
   the "5-10x higher" range formascute's own doc flagged as needing
   rescaling before trusting its estimate. Worth asking CURC directly
   whether `queueSize=200`/no-rate-limit guidance still holds at this task
   count, or whether they'd want batching (fewer, coarser tasks) first —
   this is the single most consequential unanswered question here.
2. **Repeated-read amplification against shared storage.** Confirmed today
   (via `scripts/run_zedprofiler_image_set.py`) that this pilot's
   fine-grained split reads each channel TIFF independently per task: ~8x
   per channel per image set (once per compartment at the nongranularity
   stage, again once per compartment at the granularity stage), all against
   the same PetaLibrary-mounted source files. At N=4200 that's tens of
   thousands of redundant multi-hundred-MB reads against shared storage.
   This is exactly what formascute's own architecture notes recommended
   avoiding ("one task can load an image-set once and compute multiple
   compatible feature families" instead of "one Slurm job per feature
   family, channel, and compartment") — worth asking CURC or PetaLibrary's
   storage admins whether this read pattern is a real concern at production
   scale, independent of the Slurm-side questions above.
3. **`BUILD_WAREHOUSE`'s serial, single-writer scaling has no formascute
   precedent.** Formascute's estimate only covers the feature-extraction
   fan-out; it never modeled a downstream aggregation step. This pilot's
   Iceberg-warehouse merge is deliberately single-threaded (see the
   SQLite-catalog discussion above) and reads every image set's outputs
   serially, including a `tifffile` metadata open per channel/mask (`8 x
   4200 = 33,600` file opens at N=4200). Worth asking whether CURC has
   guidance for a single long-running, I/O-bound, single-threaded process
   following a large fan-out, separate from the many-small-jobs questions
   above.
4. **`TRESBillingWeights` on `acpu`, unconfirmed.** The fair-share magnitude
   estimate above assumed Slurm bills fair-share usage in plain
   core-seconds; `scontrol show partition acpu` doesn't surface this
   directly, and it changes how much a real run would move fair-share
   standing. Cheap to ask CURC directly rather than guess.
5. **Timing, if a real run lands September-November.** CURC already told
   formascute the current `cpu-normal`/`cpu-long` QOS split is new and they
   "cannot guarantee" it holds up during named peak season. Worth a status
   check with CURC if a production run is likely to land in that window,
   since every queue-wait number in both projects was measured outside it.

**One loop closed, worth reporting back to formascute:** its open question
"why does granularity run ~7x slower than upstream-reported benchmarks" is
now answered by *this* project's own work today — see the ZedProfiler PR #51
section above (upsampling the whole image per scale instead of just labeled
voxels; fixed, merged, shipped as `zedprofiler 0.1.3`). Worth a note back
in formascute's docs since it was flagged there as open and unresolved.

## Alpine notes from 2026-08-13

### Optimizing BUILD_WAREHOUSE's serial reduce phase

Four changes, aimed at the risks flagged in the production-scale estimate
above (`BUILD_WAREHOUSE`'s serial, single-writer scaling had no measurement
and no formascute precedent):

1. **New parallel `MERGE_COMPARTMENT` stage.** The per-compartment merge
   (joining nongranularity + 4 granularity outputs, validating, checking
   alignment) used to happen inside `BUILD_WAREHOUSE`'s single-threaded loop
   over every image set. It's genuinely parallel work — one Nextflow process
   per (image set, compartment), same tier as `FEATURIZE_NONGRANULARITY`,
   synced to fire once its own nongranularity task and all 4 granularity
   tasks complete (Nextflow `groupTuple`/`join` on a `(image_set_slug,
   compartment)` key). `BUILD_WAREHOUSE` now just reads already-merged,
   already-validated files.
2. **Stopped re-reading masks to revalidate what's already known.** Each
   source task's own `validate_profiles` call already reads the mask once
   and records `mask_object_count`. The merge step used to read the mask
   *again* (`tifffile.imread`, full pixel array, not metadata) just to
   recompute that count. New `validate_merged_profile` trusts the recorded
   count instead — it re-checks that the merge itself didn't drop or
   duplicate objects (row count matches, no duplicate IDs), not the
   mask-vs-profile identity match a second time.
3. **Alignment checking moved into the per-compartment merge, split by
   compartment.** Each `MERGE_COMPARTMENT` task checks its own one mask
   against all 4 channels (metadata-only TIFF reads). Four compartments'
   worth of sub-checks collectively cover the same files a single
   whole-image-set check would (`merge_alignment_results` combines them
   with zero extra file I/O), just parallelized instead of one process
   opening all 8 files serially.
4. **Parallelized what's left in `BUILD_WAREHOUSE`.** After 1-3, the
   per-image-set loop is mostly small-file reads (pre-merged parquet +
   JSON). Runs through a `ThreadPoolExecutor` (`--max-workers`, default 8)
   instead of a plain loop — safe since each image set only touches its own
   files.

**A real bug surfaced during this work, unrelated to the logic above:**
Alpine's `nextflow` launcher silently self-updated to `26.04.6` between
yesterday's runs and today's. Its parser is stricter than `25.04.6` (what
local testing used first) — it rejects top-level executable statements
(the `if (!params.manifest...)` guards, direct variable assignments) mixed
with `process`/`workflow` declarations in the same file, where the older
version allowed it. First submission failed in 16 seconds on a compile
error. Fixed by moving all shared logic (compartment/channel parsing, the
image-set list, manifest-path-by-slug lookup) into top-level `def`
functions — legal at file scope on both versions — called fresh from
inside each process's `script:` block instead of relying on closure capture
of `workflow{}`-block-local variables. Updated the local Nextflow install to
`26.04.6` to match and re-verified before resubmitting.

**Verified correctness before trusting any of this**, in order:
- `merge_compartment.py` run standalone against a real completed
  `NF0055_T1/B10-1` compartment directory (copied off an earlier real run):
  produced 11 rows / 903 columns (exact match), `mask_object_count: 11`
  (correctly reused, not re-derived), `alignment.valid: true` with the
  correct `(105, 1527, 1528)` shapes — and the actual feature *values*
  compared byte-identical against the original run's output
  (`np.allclose(..., equal_nan=True)` on every numeric column).
- The full `build_warehouse_from_compartments.py` fallback path (all 4
  compartments routed through the new `merge_compartment`, no pre-merged
  files present) reproduced the exact known `NF0055_T1/B10-1` baseline:
  11/9/9/2 rows, 903 columns, `quality_warning_count: 4`, `validation_status:
  pass`.
- The Nextflow channel wiring itself (the riskiest part — the
  `groupTuple`/`join` synchronization and the top-level-function scoping
  fix) was verified with a stubbed local run (process bodies replaced with
  `echo` of their resolved arguments): exactly 4 `MERGE_COMPARTMENT`
  invocations for 2 image sets x 2 compartments, each with the correct
  manifest path, and exactly 1 `BUILD_WAREHOUSE` firing with correctly
  built `--image-set` arguments for both.

**Real run, full `NF0055_T1/B10-1` + `NF0014_T1/C4-2`, same
`GRANULARITY_MEMORY="24 GB"` as the last comparable run:**

```text
run_id: nf0055-nf0014-merge-optimized-20260813T180124Z
workflow_slurm_jobs: 50   (was 42 -- +8 for the new MERGE_COMPARTMENT stage,
                            job-count formula fixed to match)
nextflow_exit_status: 0
coordinator_walltime: 00:07:56
validation_status: pass
quality_warning_count: 8
```

Warehouse output identical to the pre-optimization run in every respect:
`nuclei_profiles` 56 rows (45 NF0014 + 11 NF0055), `cell_profiles`/
`cytoplasm_profiles` 51 rows each (42+9), `organoid_profiles` 3 rows (1+2),
`images.image_assets` 16 rows (8+8), 2 Iceberg data files per table, both
image sets' alignment valid. Confirmed by loading the warehouse through
PyIceberg directly, same as every other check today.

**MERGE_COMPARTMENT tasks are genuinely cheap**: 7.3-31.4s realtime, peak
RSS 128-176 MB across all 8 (trace.tsv) — far lighter than either
`FEATURIZE_NONGRANULARITY` (0.9-4.3 GB) or `FEATURIZE_GRANULARITY` (up to
5.2 GB). `BUILD_WAREHOUSE` itself got cheaper too: peak RSS dropped from
1.1 GB to 280.7 MB (~4x), realtime from 23.4s to 16.9s, comparing this run
against the last equivalent 2-image-set run before this change.

**Coordinator wall time went up, not down, at this scale** (`00:05:47` to
`00:07:56`, +~2m9s) — worth stating plainly rather than only reporting the
flattering numbers. `MERGE_COMPARTMENT` can't start until *every* upstream
`FEATURIZE_*` task for its image set finishes, so it's a full extra
sequential wave of Slurm dispatch/queue latency on top of each task's own
(small) runtime, and at only 2 image sets there's nowhere near enough total
serial reduce-phase cost yet for the parallelization to pay for that
overhead. That trade only inverts at scale: the whole point was bounding
`BUILD_WAREHOUSE`'s cost to roughly one image set's worth of work
regardless of `N`, instead of it scaling linearly with the full run's image
count the way it used to. Confirming that inversion actually happens
requires a real run at a scale large enough for it to show up (the earlier
"moderate rehearsal, tens to ~100 image sets" recommendation) — not yet
done here.

### Write-once architecture: no Iceberg, no registration step

Simplified further: dropped Apache Iceberg (the SQLite catalog, `write_table`,
`publish_iceberg_warehouse`, the `pyiceberg` dependency) entirely in favor of
plain namespaced parquet-dataset directories that *are* the warehouse. The
motivating question was "are we joining the parquet data at the end, and do
we need to" — the answer was no, and going further, the per-task write
pattern was actually writing every profile **twice**: once to a private
per-image-set location, again as Iceberg's own `table.append()` data file.
Now each task that produces a table's row for one image set writes it
**once**, directly to its final location:

```text
${outdir}/profiles/nuclei_profiles/<image_id>.parquet      (+ .validation.json, .run_record.json sidecars)
${outdir}/profiles/cell_profiles/<image_id>.parquet
${outdir}/profiles/cytoplasm_profiles/<image_id>.parquet
${outdir}/profiles/organoid_profiles/<image_id>.parquet
${outdir}/images/image_assets/<image_id>.parquet            (+ .validation.json sidecar)
```

Reading a table across all image sets is `pd.read_parquet(<table_dir>)` — a
normal multi-file parquet dataset, no catalog needed.

Three changes made this possible:

1. **`MERGE_COMPARTMENT` writes directly to the joint directory** instead of
   a private `image_sets/<slug>/compartments/<compartment>/` location that
   something else later collected and rewrote. Its read side (merging the
   split nongranularity/granularity outputs) is unchanged.
2. **New `BUILD_IMAGE_ASSETS` process, one per image set**, independent of
   the compartment/channel fan-out — it only needs the manifest, not any
   feature-extraction output, so it runs immediately alongside everything
   else instead of waiting on it. It also now owns the whole-image-set
   channel/mask alignment check (moved back from the per-compartment split
   introduced in the previous change): it already opens all 4 channels and
   all 4 masks to build `images.image_assets`, so checking alignment there
   is free, and it means alignment isn't split across 4 tasks' sidecars
   anymore.
3. **`BUILD_WAREHOUSE` is no longer a writer.** With nothing left to
   collect or register, its only job is confirming every expected file
   landed and cross-checking schemas across image sets (`pyarrow.parquet.
   read_schema`, metadata only — no data materialized) before writing one
   run-level summary. This is a much smaller, cheaper thing than either the
   old Iceberg-writing version or yesterday's `ThreadPoolExecutor` version.

**Verified before trusting on Alpine**, in order: standalone tests of
`merge_compartment.py`'s and `build_image_assets.py`'s new write targets
against real merge/alignment logic and real (small synthetic) TIFFs, confirming
each writes its file exactly once to the correct joint-dir path; a full
`build_warehouse_from_compartments.py` scan-and-summarize pass against that
output, including deliberately removing a file to confirm the missing-artifact
check fires and passes again once restored; and a stubbed local Nextflow run
(process bodies replaced with `echo` of resolved arguments, Nextflow
26.04.6, matching Alpine) confirming the new `BUILD_IMAGE_ASSETS` process and
the now-two-input `BUILD_WAREHOUSE` wire up correctly for a 2-image-set fan-out
(19/19 tasks correct).

**Real run, full `NF0055_T1/B10-1` + `NF0014_T1/C4-2`, same
`GRANULARITY_MEMORY="24 GB"`:**

```text
run_id: nf0055-nf0014-writeonce-20260813T231032Z
workflow_slurm_jobs: 52   (was 50 -- +2 for the new BUILD_IMAGE_ASSETS process)
nextflow_exit_status: 0
coordinator_walltime: 00:08:27
validation_status: pass
quality_warning_count: 8
```

Output identical to every prior baseline: `nuclei_profiles` 56 rows (45+11),
`cell_profiles`/`cytoplasm_profiles` 51 rows each (42+9), `organoid_profiles`
3 rows (1+2), 903 columns, `images.image_assets` 16 rows (8+8), both image
sets' alignment valid, no `warehouse/` directory or `catalog.db` anywhere on
disk (confirmed by `find`).

**Per-task cost dropped again, on top of yesterday's numbers** (`trace.tsv`,
realtime / peak RSS):

| Process | n | realtime | peak RSS |
| --- | --- | --- | --- |
| `FEATURIZE_NONGRANULARITY` | 8 | 57.7-237.0s (avg 116.9s) | 0.89-4.4 GB |
| `FEATURIZE_GRANULARITY` | 32 | 20.4-79.0s (avg 50.1s) | 1.9-5.3 GB |
| `MERGE_COMPARTMENT` | 8 | **0.8-5.2s** (was 7.3-31.4s) | 3.1-123.5 MB |
| `BUILD_IMAGE_ASSETS` | 2 | 6.3s | ~109 MB |
| `BUILD_WAREHOUSE` | 1 | **0.99s** (was 16.9s) | **3.1 MB** (was 280.7 MB) |

`MERGE_COMPARTMENT` got faster because it no longer does its own alignment
sub-check (moved to `BUILD_IMAGE_ASSETS`, see above). `BUILD_WAREHOUSE`
dropped ~17x in time and ~90x in memory because it no longer reads or writes
any table data at all -- just JSON sidecars and parquet schema headers.

**Coordinator wall time went up again, not down** (`00:07:56` to
`00:08:27`, +31s) -- same story as yesterday, one level further. Every
individual task got cheaper, but total job *count* went up (52 vs 50) with
the new `BUILD_IMAGE_ASSETS` wave, and at only 2 image sets there's nowhere
near enough serial-reduce-phase savings to pay for one more wave of Slurm
dispatch/queue latency. This is the same pattern as yesterday's
`MERGE_COMPARTMENT` addition, and it's the direct motivation for the
job-count-reduction proposal below: at pilot scale (N=2), *fewer, chunkier
tasks beat more, cheaper tasks* every time, because dispatch/queue overhead
is paid per task regardless of how little work that task does.

#### Proposal: reduce jobs per image set for the N=4200 case

Today's architecture is 25 Slurm tasks per image set (4
`FEATURIZE_NONGRANULARITY` + 16 `FEATURIZE_GRANULARITY` + 4
`MERGE_COMPARTMENT` + 1 `BUILD_IMAGE_ASSETS`), plus `BUILD_WAREHOUSE` and the
coordinator. At `N=4200` that's **105,002 total Slurm jobs** -- more than
the original 84,002 estimate from the production-scale write-up above, and
today's own measurement shows the cost of each additional per-image-set task
type: a full extra sequential dispatch/queue-wait wave, multiplied by every
image set in the run.

The codebase already has an unused code path that undoes this for free:
`extract_compartment_profile()` in `run_zedprofiler_image_set.py`, reachable
today via `--compartment X` with no `--feature-mode` flag. It's the
*original*, pre-split, whole-compartment function -- one task computes
nongranularity **and** all 4 granularity channels for one compartment,
loading each channel's image once and reusing it for both (today's split
design loads every channel's image twice: once in its
`FEATURIZE_NONGRANULARITY` task, again in its own `FEATURIZE_GRANULARITY`
task). Wiring this into the joint-dir write pattern established above would
eliminate the channel-level fan-out *and* the separate merge job in one
move -- no new extraction logic needed, just a new write target matching
what `merge_compartment.py`/`build_image_assets.py` already do.

| Option | Per-image-set tasks | N=4200 total jobs | vs. today |
| --- | --- | --- | --- |
| Today | 25 (4+16+4+1) | 105,002 | -- |
| **A -- one task per compartment** (nongran+gran+merge combined, reuses `extract_compartment_profile`) | 5 (4 compartment + 1 image-assets) | 21,002 | **5x fewer** |
| A+ -- also fold image-assets into one compartment task | 4 | 16,802 | 6.25x fewer |
| B -- one task per image set (all 4 compartments + image-assets, already exists as the default `--feature-mode all` path) | 1-2 | 4,202-8,402 | 12.5-25x fewer |

**Recommendation: implement A first.** Low risk (reuses already-tested
extraction code, only the output write target changes), directly reverses
the coordinator-walltime regression measured twice now, and lands well under
even the original 84,002-job estimate while keeping this week's other wins
(write-once, no Iceberg, bounded `BUILD_WAREHOUSE`). Task duration becomes
roughly 5.5 min per compartment (125s nongranularity + 4x52s granularity),
trivial against any realistic time limit. Peak memory likely stays close to
today's `granularity_memory` ceiling (64 GB) rather than adding across
channels, since channels are still processed one at a time within the
process either way -- worth confirming empirically once implemented.

**Option B is the fallback** if 21,002 jobs is still too many: it gives up
all intra-image-set task parallelism (~20-22 min per image set) for the
largest possible job-count cut. Since total CPU-work is roughly conserved
either way (repackaging, not less compute), wall-clock time at saturated
concurrency should land near the existing ~8-hour compute-bound estimate --
the win is fewer Slurm submissions and less controller/accounting overhead,
not a faster run.

**Orthogonal, complementary lever:** Nextflow's Slurm executor array support
(`executor.array`) can batch homogeneous tasks into fewer controller-visible
job-array entries without changing task granularity at all -- stackable on
top of A or B, and it's `PLAN.md`'s already-deferred "job arrays for
homogeneous shards" item.

#### Option A implemented and benchmarked

`FEATURIZE_NONGRANULARITY`, `FEATURIZE_GRANULARITY`, and `MERGE_COMPARTMENT`
are gone, replaced by one `FEATURIZE_COMPARTMENT` process per (image set,
compartment) that calls `run_zedprofiler_image_set.py --compartment X` (no
`--feature-mode` anymore -- there's only one mode now) and writes directly to
`${outdir}/profiles/<compartment>_profiles/<image_id>.parquet`, its one and
only write. `extract_granularity_profile()`,
`extract_compartment_nongranularity_profile()`, `validate_merged_profile()`,
and `scripts/merge_compartment.py` are deleted -- dead code once nothing
called them. `conf/base.config` lost the `granularity_cpu`/
`nongranularity_cpu`/`merge_cpu` labels and their params; `FEATURIZE_COMPARTMENT`
reuses the `zedprofiler_cpu` label (`feature_cpus`/`feature_memory`/
`feature_time`) that already existed, unused, from before the split was ever
introduced. `bin/nf1-nextflow-pilot` and `Makefile` lost the matching
`--granularity-*`/`--nongranularity-*` flags -- there's only `--feature-*`
now. A nice side effect: `FEATURIZE_COMPARTMENT` writes straight to the
joint dir, so the private `image_sets/<slug>/compartments/...` scratch tree
this pilot has had since day one no longer exists at all.

**Verified before trusting on Alpine**, in order: a real end-to-end run of
`run_zedprofiler_image_set.py --compartment Nuclei` against synthetic
16x16x4 TIFFs (via `tifffile`) with `zedprofiler` actually installed and
executing locally, confirming the parquet lands at the correct joint-dir
path with 903 columns, both feature families (granularity and
nongranularity-only were checked separately before; this confirms they
coexist correctly in one profile), and row count exactly matching mask
object count (the only validation failure was expected null texture
features from an object too small for GLCM at this synthetic scale, not a
code defect); a stubbed local Nextflow run confirming the collapsed
workflow wires up correctly (7 tasks for 2 image sets x 2 compartments: 4
`FEATURIZE_COMPARTMENT` + 2 `BUILD_IMAGE_ASSETS` + 1 `BUILD_WAREHOUSE`,
correct `--manifest` args reaching `BUILD_WAREHOUSE`).

**Real run, full `NF0055_T1/B10-1` + `NF0014_T1/C4-2`, `FEATURE_MEMORY="32 GB"`:**

```text
run_id: nf0055-nf0014-optionA-20260813T233613Z
workflow_slurm_jobs: 12   (was 52 -- 4.3x fewer at N=2; formula predicts 5x at N=4200)
nextflow_exit_status: 0
coordinator_walltime: 00:08:53
validation_status: pass
quality_warning_count: 8
```

Output identical to every prior baseline: 56/51/51/3 rows, 903 columns, 16
`images.image_assets` rows, no `image_sets/` directory anywhere on disk.

```text
Process                  n   realtime            peak RSS
FEATURIZE_COMPARTMENT    8   132.0-415.0s (288.4s avg)   2.46-6.76 GB
BUILD_IMAGE_ASSETS       2   1.2s                        27.6-121.5 MB
BUILD_WAREHOUSE          1   6.1s                        117.6 MB
```

**Honest finding: coordinator wall time went up again at this scale**
(`00:08:27` to `00:08:53`, +26s), continuing the same pattern as both prior
changes -- not what the job-count table above might suggest at a glance.
The reason is specific to Option A and worth stating plainly: folding
nongranularity and all 4 granularity channels into one task removes the
*channel-level* parallelism those channels used to get as separate Slurm
tasks, not just the separate-merge-job overhead. `FEATURIZE_COMPARTMENT`'s
288.4s average is close to the sum of what used to be several
concurrently-running tasks (old `FEATURIZE_NONGRANULARITY` avg 116.9s +
`FEATURIZE_GRANULARITY` avg 50.1s x4 channels done serially inside one
process). At `N=2`, Alpine has far more free concurrency than 8 tasks need,
so that lost parallelism shows up directly as wall time; the fewer-waves
saving (no separate merge dispatch) is real but smaller than the
now-serialized channel work costs.

**This doesn't undermine the N=4200 case, and here's why.** Total wall time
at production scale is throughput-bound
(`total_tasks / concurrency_slots x task_duration`), not latency-bound the
way a 2-image-set pilot is. Plugging in today's *real* `FEATURIZE_COMPARTMENT`
average (288.4s, not the earlier hand-estimated ~330s) at the same
200-concurrent-slot scenario used in the original estimate:
`16,800 FEATURIZE_COMPARTMENT tasks / (200 x 60 / 288.4s) ~= 6.7 hours` --
in the same range as the original ~7.8-8.6 hour compute-bound estimate, not
worse, because total CPU-time is conserved (repackaged into fewer, longer
tasks) and concurrency is what dominates at that scale, not per-task
latency. The pilot-scale wall-time regression is a real, now-measured
artifact of having *more spare concurrency than work* at `N=2`, not a
preview of what happens at `N=4200`.

#### Option B implemented and benchmarked: one task per image set

Went one level further: `FEATURIZE_COMPARTMENT` is gone too, replaced by
`FEATURIZE_IMAGE_SET`, one task per image set with no `--compartment` flag
at all -- `run_zedprofiler_image_set.py`'s `for compartment in compartments`
loop (already written generically for this from the start) now processes
every compartment the manifest lists in one task, loading each of the 4
channels once and reusing it across all 4 compartments instead of reloading
per compartment. `params.compartments`/`parseCompartments()` are gone from
the workflow -- nothing needs to enumerate compartments at the Nextflow
level anymore, the manifest alone drives it.

**Verified before trusting on Alpine**, same rigor as before: a stubbed
local Nextflow run confirmed the collapsed wiring (5 tasks for 2 image sets:
2 `FEATURIZE_IMAGE_SET` + 2 `BUILD_IMAGE_ASSETS` + 1 `BUILD_WAREHOUSE`,
correct manifest resolution); a real local run of
`run_zedprofiler_image_set.py` with no `--compartment` against synthetic
TIFFs confirmed all 4 compartments land correctly in one invocation, each
with row count exactly matching its mask's object count and all 6 feature
families present.

**Real run, full `NF0055_T1/B10-1` + `NF0014_T1/C4-2`, `FEATURE_MEMORY="32 GB"`:**

```text
run_id: nf0055-nf0014-optionB-20260814T013546Z
workflow_slurm_jobs: 6   (was 12 for Option A, 52 originally -- 2.5x fewer than A,
                           12.5x fewer than where this started)
nextflow_exit_status: 0
coordinator_walltime: 00:22:41
validation_status: pass
quality_warning_count: 8
```

Output identical to every prior baseline again: 56/51/51/3 rows, 903
columns, 16 `images.image_assets` rows.

```text
Process                n   realtime                     peak RSS
FEATURIZE_IMAGE_SET    2   1122.0-1299.0s (1210.5s avg)  5.32-8.19 GB
BUILD_IMAGE_ASSETS     2   8.1-8.2s                      105.2-112.9 MB
BUILD_WAREHOUSE        1   2.7s                          118.2 MB
```

**The wall-time regression from Option A continues, and much more sharply
this time**: coordinator walltime jumped from `00:08:53` to `00:22:41`
(+13m48s). This is the predicted trade-off, now measured: `FEATURIZE_IMAGE_SET`
gives up *all* intra-image-set parallelism, not just channel-level
parallelism. Its 1210.5s average is close to the sum of what Option A's 4
separate compartment tasks would take run one after another (4 x 288.4s =
1153.6s -- within 5% of the observed average, a nice independent
confirmation that this really is the same total work, just packaged
differently), but Option A's 4 tasks ran *concurrently* at this small scale
while `FEATURIZE_IMAGE_SET` runs them serially inside one process. Peak
memory also rose as expected (5.32-8.19 GB vs Option A's 2.46-6.76 GB, from
holding all 4 compartments' masks at once instead of one) but stays well
under the 32 GB ceiling -- no memory risk found.

**Extrapolating with today's real number, not the earlier estimate:** at
`N=4200`, throughput-bound wall time is
`4,200 FEATURIZE_IMAGE_SET tasks / (200 x 60 / 1210.5s) ~= 7.1 hours` --
essentially the same as Option A's ~6.7 hour estimate and the original
~7.8-8.6 hour figure. The pattern holds: consolidating task granularity
doesn't change production-scale wall time (throughput-bound, total compute
conserved either way), it only changes job count and pilot-scale iteration
latency.

**The real trade-off between A and B is not production wall time -- it's
job count vs. iteration/debugging cost.** At `N=4200`: Option A is 21,002
jobs at ~5 min/task; Option B is 8,402 jobs at ~20 min/task. Production wall
time is a wash either way. What differs: Option B is 2.5x more considerate
of Slurm's job controller and this account's submission budget, but a
single failed task now costs ~20 min to retry instead of ~5, and there's no
per-compartment task boundary left to isolate *which* compartment failed
without reading the task's own log -- worth weighing against how often
individual tasks are expected to fail or need debugging at production
scale, which this pilot hasn't yet measured.

**One operational note discovered during this run, unrelated to the
architecture**: the background `squeue`-polling check used to detect this
run's completion reported "done" once, prematurely, on an apparently
transient empty `squeue` result while the coordinator and both
`FEATURIZE_IMAGE_SET` tasks were still genuinely running (confirmed via
`sacct` immediately after -- `RUNNING`, 15+ minutes elapsed). Re-checking
against the actual `completion_status.txt` file's existence instead of
queue emptiness caught this and avoided reporting fabricated results.
Prefer checking for a run's own completion artifact over polling `squeue`
emptiness for future long-running submissions.

#### Decision: Option B is the main pilot architecture going forward

Both A and B are fully measured now, not projections, and production-scale
wall time is a wash between them (~6.7h vs ~7.1h at `N=4200`, both
throughput-bound). Chose **Option B** -- one `FEATURIZE_IMAGE_SET` task per
image set, no compartment-level fan-out at all -- as this pilot's main path
going forward: 8,402 total jobs at `N=4200` (vs Option A's 21,002, vs
105,002 where this thread started) is a meaningfully more considerate
submission profile for a shared cluster, and today's per-task duration
(~20 min) is nowhere near any Slurm time-limit concern. `workflows/
featurize_image_set.nf`, `scripts/run_zedprofiler_image_set.py`, `conf/
base.config`, `bin/nf1-nextflow-pilot`, and `Makefile` all already reflect
this -- there's no separate Option A code path left to choose between,
Option B *is* what `git status` on this repo shows today. The sections
above stay as the record of how this was decided and what the real
alternative would have cost, not as a currently-live fork to pick between.

The retry-cost trade-off flagged above (a failed task now costs ~20 min to
retry, no per-compartment log boundary) is worth watching once this runs at
real scale, but isn't a blocker: `conf/base.config`'s existing
`errorStrategy = { task.attempt <= 2 ? 'retry' : 'finish' }` already retries
a failed `FEATURIZE_IMAGE_SET` task up to twice before giving up, and the
compartment loop in `run_zedprofiler_image_set.py` keeps going through every
compartment even when one fails *validation* (writing each one's sidecar
regardless, `all_valid` only gates the final exit code) -- so a data-quality
failure still identifies which compartment via its own sidecar. A hard
crash partway through (a corrupt TIFF, an unhandled exception) still loses
that per-compartment breakdown and requires reading the task's own log, same
as before.

## Alpine notes from 2026-08-14

### Warehouse directory: parquet-only, portable, with lightweight DuckDB views

Two follow-up requests once Option B was decided: keep `validation.json`/
`run_record.json`/every other JSON artifact out of the warehouse's data
directories entirely, and add a lightweight SQL-queryable layer over the
parquet without copying data or reintroducing catalog machinery.

**Layout, now with a dedicated `warehouse/` directory holding data only:**

```text
${outdir}/
├── warehouse/                          <- data only, nothing else
│   ├── profiles/
│   │   ├── nuclei_profiles/<image_id>.parquet
│   │   ├── cell_profiles/<image_id>.parquet
│   │   ├── cytoplasm_profiles/<image_id>.parquet
│   │   └── organoid_profiles/<image_id>.parquet
│   ├── images/
│   │   └── image_assets/<image_id>.parquet
│   └── warehouse.duckdb                <- views only, no copied data
└── metadata/                           <- every JSON sidecar, mirroring the same paths
    ├── profiles/<table>/<image_id>.validation.json, .run_record.json
    ├── images/image_assets/<image_id>.validation.json
    ├── run_record.json                 <- BUILD_WAREHOUSE's run-level summary
    └── validation.json
```

`FEATURIZE_IMAGE_SET` and `BUILD_IMAGE_ASSETS` write straight into
`warehouse/profiles/...`/`warehouse/images/...` from the start -- same
"write once, no move" principle as before, just one directory level deeper.
Nothing needs shuffling once every task lands; `warehouse/` is already the
complete, final artifact.

**The DuckDB layer** (`scripts/build_duckdb_views.py`): one `CREATE VIEW
namespace.table AS SELECT * FROM read_parquet(glob)` per discovered
`profiles/<table>/` or `images/<table>/` directory, mapped onto the same
`profiles.nuclei_profiles` / `images.image_assets` naming already used
everywhere else in this pilot. A view is a stored query, not a copy, so
`warehouse.duckdb` stays a few KB regardless of data volume, and rerunning
the script picks up newly landed image sets for free -- confirmed locally
by adding a second image set's parquet file to an existing table directory
and rerunning: the view immediately returned both, no `DROP`/recreate
needed.

**Portability was the point, not just convenience.** View definitions use
paths *relative* to `warehouse/`, never the absolute PetaLibrary path --
verified by copying a `warehouse/` directory to `/tmp` under a completely
different path and querying it successfully with zero knowledge of where it
came from. The tradeoff: DuckDB resolves relative `read_parquet` globs
against the *querying process's* working directory, not the `.duckdb`
file's own location, so both view creation and later querying need `cwd`
set to `warehouse/` (`build_views()` handles this internally via a
temporary `os.chdir`; anyone querying later needs `cd warehouse/ &&
duckdb warehouse.duckdb`, documented in the script's own docstring).

`build_warehouse_from_compartments.py` calls `build_views()` right after
validation, at zero extra Slurm-job cost -- it already has every table's
path in hand from the scan it just did. The script is also fully usable
standalone against any warehouse directory, finished or still landing.

**Verified before trusting on Alpine**: local smoke test confirming the
exact layout above; a real local run through `run_zedprofiler_image_set.py`
→ `build_image_assets.py` → `build_warehouse_from_compartments.py` against
synthetic TIFFs, including a cross-table `JOIN` between `profiles.
nuclei_profiles` and `images.image_assets` through the views; the same
portability check (copy `warehouse/` elsewhere, query with no absolute-path
knowledge) repeated against real Alpine output.

**Real run, full `NF0055_T1/B10-1` + `NF0014_T1/C4-2`:**

```text
run_id: nf0055-nf0014-warehouse-dir-retry-20260814T030514Z
coordinator_walltime: 00:20:33
validation_status: pass
quality_warning_count: 8
```

Output identical to every prior baseline (56/51/51/3 rows, 903 columns, 16
`images.image_assets` rows). `warehouse/` confirmed to hold only `.parquet`
files plus `warehouse.duckdb`; `metadata/` confirmed to hold every JSON
artifact, nothing else. Queried `profiles.nuclei_profiles` and its `JOIN`
against `images.image_assets` directly through the views on this real data
-- both returned correct row counts (56 total nuclei rows across 2 image
sets; 448 = 56 x 8 from the join, as expected for an unfiltered join against
8 asset rows per image set).

### A real instance of the flagged retry-cost risk, caught live

The first submission attempt for this change failed outright --
`FEATURIZE_IMAGE_SET` for one image set hit `numpy._core._exceptions.
ArrayMemoryError: Unable to allocate 467. MiB` inside `tifffile.imread`,
retried twice per `conf/base.config`'s `errorStrategy`, failed identically
each time, and Nextflow gave up (`Execution cancelled -- Finishing pending
tasks before exit`), taking the whole run's exit status to 1. `sacct`
showed all three attempts landed on the *same* node (`c3cpu-c13-u1-2`) with
`MaxRSS` of only 6.4-9.2 GB against a 32 GB request -- not this job
exceeding its own limit, but that node's actual free memory being consumed
by something else sharing it, exactly the "shared-cluster contention" risk
flagged (as a theoretical uncertainty) in the production-scale estimate
above. Confirmed this wasn't a code regression: the traceback is inside the
very first `tifffile.imread` call, before any of this session's write-path
changes run at all, and the same manifests had already succeeded earlier
the same day.

**What actually happened before drawing conclusions -- verified, not
assumed:** the write-once design meant nothing needed cleanup after the
failure. `warehouse/` for the *other*, successful image set was already
complete and correctly laid out (all 4 compartments + `images.
image_assets`); `BUILD_WAREHOUSE` correctly never fired (still waiting on
the failed image set's `.collect()`); a plain resubmission with identical
arguments landed both `FEATURIZE_IMAGE_SET` tasks on two different,
uncontended nodes and completed cleanly in 20m33s. This is the retry-cost
trade-off from the Option A/B decision section above, now a real measured
incident rather than a hypothetical: one bad node cost one lost ~20-minute
task and a full pipeline restart, not data loss or a corrupted warehouse.

### Example notebook: querying the warehouse through its DuckDB views

`notebooks/query_warehouse_views.ipynb` -- connects to a run's
`warehouse.duckdb`, lists the views, shows `head()` on each profile table
and `images.image_assets`, a cross-table `JOIN` example, a `UNION
ALL`/`GROUP BY` aggregate (object counts per compartment per image set),
and both `joined.*` composite views (see below). Executed against the real
`nf0055-nf0014-warehouse-dir-retry-20260814T030514Z` output, so opening it
shows real data without rerunning anything; edit the `WAREHOUSE_DIR`
variable in the first cell to point at a different run.

### Two composite `joined.*` views, image-asset columns first

`build_duckdb_views.py` also creates a `joined` schema with two views built
on top of the per-table ones, both requested to put image-asset columns
first:

- `joined.images_nuclei_cell_cytoplasm` -- Nuclei, Cell, and Cytoplasm
  joined together on `(Metadata_Imaging_ImageID, Metadata_Object_ObjectID)`
  before joining to `images.image_assets` on `Metadata_Imaging_ImageID`
  alone. The object-ID join is valid because Cell and Cytoplasm are seeded
  from Nuclei's own segmentation (`compartment_seed_channels` in
  `run_zedprofiler_image_set.py`) -- confirmed empirically before writing
  the query, not assumed: for `NF0014_T1/C4-2`, `nuclei_profiles` has 45
  rows but `cell_profiles`/`cytoplasm_profiles` have 42 each, and joining
  all three on that key gives exactly 42 -- Cell/Cytoplasm object IDs are a
  strict subset of Nuclei's, one row per successfully segmented cell.
- `joined.images_organoid` -- Organoid joined directly to `images.
  image_assets`. Organoid is segmented independently
  (`segmented_from_agp`, no seed channel) with its own, much smaller
  object-ID space (1-2 per image set), so it isn't part of the first view.

Both views fan out against `image_assets` the same way the notebook's plain
`JOIN` example does (each object row repeated once per image asset, 8 per
image set) -- consistent with the pattern already established, not a new
convention.

**Column collisions handled explicitly, not left ambiguous.** Six columns
carry different values per compartment (`Metadata_Compartment`,
`Metadata_Segmentation_Method`, and the four `PrimaryChannel`/
`SeedChannel` fields) -- confirmed by comparing real values for the same
image set across `nuclei_profiles` and `cell_profiles` before deciding this
mattered (naive testing without pinning to one image set gave false
"differs" results for other columns due to comparing unrelated rows; fixed
by filtering to one `Metadata_Imaging_ImageID` before comparing). These get
a per-compartment prefix via DuckDB's `* RENAME (...)`
(`Metadata_Nuclei_Compartment`, `Metadata_Cell_Compartment`, etc.) rather
than colliding. Seven more columns (patient/plate/well/field/image IDs)
are identical across every table for the same image set and are kept
exactly once, from `images.image_assets`, via `* EXCLUDE (...)` on every
profile table joined in. Verified: `duplicate column names: False` on both
views against real data, and row counts land exactly on the expected
compartment counts x 8 (`NF0014_T1/C4-2`: 42 x 8 = 336 for the single-cell
view, 1 x 8 = 8 for organoid; `NF0055_T1/B10-1`: 9 x 8 = 72 and 2 x 8 = 16;
408 and 24 total).
Open with `uv run --project environments --with jupyter jupyter lab
notebooks/query_warehouse_views.ipynb` -- `jupyter` isn't a tracked
dependency of the pilot's own environment, so it's pulled in transiently
rather than added to `environments/pyproject.toml`.
