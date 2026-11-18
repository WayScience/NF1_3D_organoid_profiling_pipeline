# NF1 ZEDProfiler Z-Step Workflow

Runs ZEDProfiler feature extraction through Nextflow on Alpine for the
z-step-size technical-analysis experiment: how features change as the
z-slice step grows from 0.1 um to 1 um, and whether the segmentation the
features are extracted on (its own z-step's masks vs the 0.1 um masks,
subsampled) matters as much as the images themselves.

This folder does not fork production's logic. `scripts/`, `workflows/`,
and the manifest/warehouse shapes are the same code as
`3b.nextflow_production`, copied and renamed for a separate PetaLibrary
root (`nf1-3d-zstep-workflow-db`) and a separate `NF1_ZSTEP_*`
environment-variable namespace so this run can never collide with pilot
or production output. Where this folder's defaults differ, it's because
the z-step experiment adds two dimensions production didn't have -- read
[Differences from production](#differences-from-production-3bnextflow_production)
below, and `3a.nextflow_pilot/README.md` + `3b.nextflow_production/README.md`
for the full benchmark history and channel/compartment role tables behind
every default here.

## The experiment, in one table

Each row is one image set = one (patient, well_fov, mask_group) triple =
one FEATURIZE_IMAGE_SET Slurm task:

```text
NF0037_T1-Z-{0.1,0.2,0.5,1}   FOVs F4-1, F4-2, F4-3
NF0055_T1-Z-{0.1,0.2,0.5,1}   FOVs E4-1-0, E4-3-0, E4-5-0, G5-1-0, G5-2-0, G5-5-0
mask_group native      segmentation 1: segmented on this z-step's own z-stack
mask_group from_0_1um  segmentation 2: segmented on the 0.1 um z-stack, then
                      subsampled to this z-step's slice count (not present
                      for the 0.1 um groups -- they ARE the 0.1 um z-stack)
```

Counts: NF0037 3 FOVs x (1 + 2 + 2 + 2) = 21, NF0055 6 FOVs x 7 = 42,
**63 image sets total** (~61 was the estimate; F4-1 was included after
confirming all complete FOVs should run).

Downstream analysis reads `Metadata_Experiment_ZStep` (0.1/0.2/0.5/1.0)
and `Metadata_Segmentation_MaskGroup` (`native`/`from_0_1um`) off every
profile row -- these two columns are the whole point of the run, so
apples-to-apples comparisons (e.g. 1 um images under 0.1 um masks vs
under native masks) are one filter away.

## Why per-task resource sizing matters here

Image-set sizes differ by a factor of 10 across this batch (the user's
own framing, and the real numbers agree -- NF0037 F4-2 measured via
tifffile on koala):

```text
z-step   slices per stack   voxels vs production (63-slice) baseline
0.1 um   527                 ~8.4x
0.2 um   264                 ~4.2x
0.5 um   106                 ~1.7x
1.0 um    53                 ~0.84x
```

Production's flat `feature_memory = 32 GB` per task would strangle the
0.1 um sets and waste the 1.0 um ones. Instead, `PLAN_IMAGE_SETS` emits
each image set's Slurm sizing on its plan line
(`slug<TAB>status<TAB>mem=...<TAB>cpus=...<TAB>time=...`), sourced from
`scripts/plan_image_sets.py`'s single `Z_STEP_RESOURCES` table, and
`FEATURIZE_IMAGE_SET` consumes it via Nextflow dynamic directives
(closures over the task's input tuple, which take precedence over the
`withLabel` defaults in `conf/base.config`):

```text
z-step   memory   cpus   time
0.1 um   224 GB     8    24h
0.2 um   128 GB     8    24h
0.5 um    64 GB     4    18h
1.0 um    48 GB     2    12h
```

These are starting points anchored on the production run's observed
14.5 GB peak RSS for a 63-slice set (roughly linear voxel scaling), not
validated numbers -- after the first tasks complete, check
`sacct -j <job> --format=JobID,MaxRSS,Elapsed` and tighten the table
before rerunning anything. 24h is acpu's `MaxTime` ceiling; 224 GB fits
the 245 GB 64-cpu nodes.

## Architecture

Unchanged from production -- PLAN_IMAGE_SETS (1 task) checks the
warehouse for already-complete image sets and emits the plan,
FEATURIZE_IMAGE_SET (1 task per pending image set, sized by z-step)
loads all four channels and masks once, checks alignment, extracts
every compartment's profile and writes each directly into the shared
warehouse, then BUILD_WAREHOUSE (1 task) validates every landed parquet
and builds the DuckDB views. See `3b.nextflow_production/README.md`'s
Architecture section for the DAG and compartment-relationship diagrams;
the only structural difference is the plan line carrying resource fields
through the pending channel into per-task directives.

Data staged on PetaLibrary:

```text
/pl/active/koala/nf1-3d-zstep-workflow-db/data/<patient_group>/zstack_images/<well_fov>/
/pl/active/koala/nf1-3d-zstep-workflow-db/data/<patient_group>/segmentation_masks/<well_fov>/
/pl/active/koala/nf1-3d-zstep-workflow-db/data/<patient_group>/segmentation_masks_from_0_1um_adjusted/<well_fov>/
```

bandicoot is not reachable from Alpine (checked Nov 2026: no `~/mnt` on
login, compute, or persistence1 nodes), so data is staged from the lab
Mac over both mounts -- see [`staging/README.md`](staging/README.md) for
the bandicoot -> Mac -> koala hops and the completeness check. Results
are copied back to bandicoot after the run (also documented there).

## Run

From a repo checkout on Alpine, once staging has landed data and the
shared `uv` environment exists (see `environments/zedprofiler-uv.md`):

```bash
cd <checkout>/3c.nextflow_zstep
make check
UV_CACHE_DIR=/pl/active/koala/nf1-3d-zstep-workflow-db/tools/uv-cache \
  uv sync --project environments
make build-image-sets-index SOURCE_ROOT=/pl/active/koala/nf1-3d-zstep-workflow-db
# expect "Wrote 63 image sets" -- fewer means incomplete staging
make submit-dry-run RUN_ID=nf1-zstep-run-1 ACCOUNT=amc-general
```

Before the full batch, run one small task end-to-end (a 1 um set, ~48 GB,
minutes not hours) to validate the whole chain against real staged data:

```bash
make run IMAGE_SETS_INDEX= MANIFEST=manifest/generated_manifest.yaml \
  PATIENT=NF0037_T1-Z-1 WELL_FOV=F4-2 MASK_GROUP=native
```

Note a `--manifest`/`make run` path bypasses the plan's resource fields
(flat fallback sizing from `conf/base.config`), so use it for wiring
checks on small sets, not for the 0.1 um monsters. Then the full batch:

```bash
make submit RUN_ID=nf1-zstep-run-1 ACCOUNT=amc-general
```

`make submit` defaults to the index-driven path
(`IMAGE_SETS_INDEX=manifest/image_sets_index.csv`). Every other Make
target, flag, and environment variable works exactly as documented in
`3a.nextflow_pilot/README.md`'s "Run" section, with `NF1_ZSTEP_*` in
place of `NF1_PILOT_*`. `make help` prints the full list from this
folder's own `bin/nf1-nextflow-zstep`.

## Differences from production (`3b.nextflow_production`)

- **PetaLibrary root**: `nf1-3d-zstep-workflow-db`, not
  `nf1-3d-production-workflow-db`. Separate `data/`, `results/`, and
  `tools/` trees; nothing here reads or writes under the pilot or
  production folders.
- **Environment variable prefix**: `NF1_ZSTEP_*`, not `NF1_PROD_*`.
- **One image set = one (patient, well_fov, mask_group) triple.** The
  index CSV gains `mask_group` and `z_step` columns; manifests derive
  both from the patient directory name; `Metadata_Imaging_ImageID` gains
  a `maskgroup_{group}` suffix so the two runs of one FOV never
  overwrite each other's parquet.
- **z_spacing is derived, not hardcoded.** The manifest reads the
  patient directory's `_T1-Z-{step}` suffix (0.1/0.2/0.5/1.0) instead of
  hardcoding 1.0 -- z_spacing feeds ZEDProfiler's anisotropy-aware
  features and `compute_neighbors`' `anisotropy_factor`, so it is a real
  computational parameter here. `xy_spacing` stays 0.1 um (the 0.1 um
  groups' anisotropy factor is 1, isotropic, per the experiment's
  README).
- **Per-z-step task sizing** via the plan channel (see above), replacing
  production's flat 32 GB/2 cpu/2h defaults. Fallbacks remain in
  `conf/base.config` for `--manifest` runs.
- **`executor.queueSize = 24`** (vs production's 180): 63 tasks, very
  unequal memory requests, no MaxJobs pressure -- a modest queue keeps
  Slurm's backfill effective instead of stacking many big tasks.
- **Staging runs on the lab Mac**, not Alpine: bandicoot is not
  reachable from Alpine at all, so the staging script runs on the Mac
  with both shares mounted. Production's staging docs assumed bandicoot
  was mountable wherever the script ran; that assumption no longer
  holds. Results are rsynced back to bandicoot after the run.
- **`schema_version` is `0.1.0-zstep`** in manifests this folder
  generates.
- **Three new profile columns**: `Metadata_Experiment_ZStep`,
  `Metadata_Segmentation_MaskGroup`,
  `Metadata_Segmentation_MaskGroupLabel`, stamped on every profile row
  and image_assets row by the runner, deduped to the image_assets side
  in joined DuckDB views (see `scripts/build_duckdb_views.py`).

## Before the first full run

- The 0.1 um tasks (224 GB request) have never run -- the sizing table is
  an estimate. Run one 1 um and one 0.1 um image set first, check `sacct`
  MaxRSS/Elapsed, and adjust `Z_STEP_RESOURCES` before the full batch.
- Check a completed task's
  `<outdir>/metadata/profiles/<table>/<image_id>.run_record.json` for
  per-family timings -- Granularity dominates on deep stacks and is where
  the 24h walltime estimate is most likely to be wrong.
- Re-query live Alpine state (`sinfo`, `sshare -u $USER -A amc-general
-l`) immediately before submitting.
