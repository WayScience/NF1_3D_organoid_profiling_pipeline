# 3b.nextflow_production -- production rollout plan

## Scope

Expand `3a.nextflow_pilot`'s validated Nextflow/Slurm fan-out engine (proven
on `NF0055_T1/B10-1` and `NF0014_T1/C4-2`, see that folder's README) to an
initial production batch: **12 patients, every complete well/FOV**, staged
from bandicoot to a new, isolated PetaLibrary root. This is
`docs/source/future_processing_plan.md`'s implementation-sequence step 7,
"Production workflow rollout" -- it follows step 6's pre-operational
validation of the pilot, not a replacement for it.

Explicitly **not** in this plan:

- The full ~4200-image-set dataset in one run. This batch (~3,449 well/FOVs
  across 12 patients) is itself the next scale step up from the pilot's 1-2
  image sets, and per the pilot's own PLAN.md, `BUILD_WAREHOUSE`'s scaling
  behavior needs a real intermediate-scale data point before a full-dataset
  run is trustworthy. This batch is that data point.
- Any change to ZEDProfiler feature logic, compartment/channel roles, or the
  warehouse table shape. Those are unchanged from the pilot.
- GPU / deep-learning features (SAMMed3D, morphem). Same reserved,
  unimplemented seam the pilot left in `conf/curc_alpine.config`.
- QC policy. Out of scope per `future_processing_plan.md`.

## Data source and staging

Source of truth is bandicoot, same layout the pilot used:

```text
~/mnt/bandicoot/NF1_organoid_data/data/{patient}/zstack_images/{well_fov}/
~/mnt/bandicoot/NF1_organoid_data/data/{patient}/segmentation_masks/{well_fov}/
```

**Patients for this batch** (`staging/patients.txt`):
`NF0014_T1`, `NF0014_T2`, `NF0016_T1`, `NF0018_T6`, `NF0021_T1`,
`NF0030_T1`, `NF0035_T1`, `NF0037_T1`, `NF0040_T1`, `NF0055_T1`,
`SARCO219_T2`, `SARCO361_T1` -- the last three added in a follow-up staging
pass after the first nine were verified.

**Working copy: a new PetaLibrary root, not bandicoot directly and not the
pilot's root.** `nf1-3d-production-workflow-db` was created alongside the
existing `nf1-3d-pilot-workflow-db`, matching its `data/` + `results/` +
`tools/` layout, so both share the same operational pattern on Alpine
without either one touching the other's files.

**Staging is a manual/scripted prep step, before any Nextflow run** -- see
`staging/README.md` for the transfer script (`stage_from_bandicoot.sh`,
`rsync -a --partial`, resumable) and how to verify completeness with
`scripts/build_image_sets_index.py` once transferred. All 12 patients are
staged and verified: 3,443 of 3,449 well/FOVs complete. The 6-well
shortfall (4 in `NF0030_T1`, 1 in `NF0035_T1`, 1 in `NF0055_T1`) is missing
masks on bandicoot itself, not a transfer issue -- see `staging/README.md`
for the specific paths.

## Reused, unchanged from the pilot

Everything that isn't listed under "Changed for production" below is a
straight copy of `3a.nextflow_pilot`'s already-validated code, not a
reimplementation:

- `workflows/featurize_image_set.nf` -- the `PLAN_IMAGE_SETS` ->
  `FEATURIZE_IMAGE_SET` (per compartment x channel granularity fan-out) ->
  `BUILD_WAREHOUSE` DAG, including the skip-already-completed-image-sets
  logic and the shared-`uv`-environment worker launch pattern.
- `scripts/build_manifest.py`, `build_image_sets_index.py`,
  `plan_image_sets.py`, `manifest_io.py`, `run_zedprofiler_image_set.py`,
  `build_warehouse_from_compartments.py`, `build_duckdb_views.py`,
  `smoke_synthetic.py` -- all already `source_root`/`patient`/`well_fov`
  parameterized in the pilot; production only changes the values passed in,
  not the code.
- The manifest schema, `Metadata_*` identifier rules, warehouse directory
  shape (`warehouse/images/image_assets/`,
  `warehouse/profiles/<compartment>_profiles/`, `metadata/` sidecars), and
  compartment segmentation relationships (Nuclei from DNA; Cell/Cytoplasm
  from AGP seeded/subtracted by Nuclei; Organoid from AGP independently).
- ZEDProfiler pin (`0.1.3`) and the `uv`-based isolated environment pattern.

## Changed for production

- **PetaLibrary root, environment-variable prefix (`NF1_PROD_*`), binary
  name (`bin/nf1-nextflow-production`)** -- renamed so a production run can
  never be pointed at pilot paths by an unset/default environment variable.
- **Default run mode is index-driven** (`IMAGE_SETS_INDEX`), not a single
  checked-in manifest. The pilot supported both from early on precisely so
  large batches wouldn't need one YAML file per image set; production is
  the batch this was built for.
- **`conf/curc_alpine.config`: `executor.queueSize` 20 -> 180, no
  `submitRateLimit`.** See `README.md`'s "Before the first full run" section
  for the reasoning, carried directly from the pilot's own production-scale
  throughput analysis.
- **`schema_version: 0.1.0-production`** in generated manifests, distinct
  from the pilot's `0.1.0-pilot`, so a stray manifest is identifiable by
  origin.

## Execution sequence

1. **Stage data** (`staging/stage_from_bandicoot.sh`) for all 12 patients.
   Verify with `build_image_sets_index.py` against the expected well/FOV
   counts in `staging/README.md`.
2. **One-time Alpine setup**: repo checkout under
   `/pl/active/koala/nf1-3d-production-workflow-db/`, `uv sync --project
   environments`, confirm `nextflow`/JDK 21 available (mirror the pilot
   README's "Alpine notes from 2026-08-10" setup, same tooling).
3. **Build the image-sets index** (`make build-image-sets-index
   SOURCE_ROOT=...`) once staging is verified complete.
4. **Staged capacity test**: run a deliberately small slice of the index
   first (e.g. `--image-sets` with a handful of manifests, or a hand-trimmed
   index CSV) to get a real, current `BUILD_WAREHOUSE` timing point and a
   real achieved-throughput number under actual same-day cluster
   contention -- exactly the test the pilot's PLAN.md recommended before
   trusting any wall-time projection. Do not skip this to save time; it's
   what makes the full-batch estimate trustworthy rather than guessed.
5. **Full batch run** (`make submit`) once the capacity test's numbers look
   sane, with `queueSize`/resource directives adjusted based on what step 4
   actually measured, not just this plan's starting defaults.
6. **Validate**: `scripts/build_duckdb_views.py` against the run's
   warehouse; spot-check row counts per patient against the staged well/FOV
   counts; confirm no all-null feature columns and that `Metadata_*`
   identifiers match the manifest/index rows exactly (same checks the pilot
   used).

## Open risks (carried over from the pilot's own analysis, not re-verified)

- `BUILD_WAREHOUSE` scaling past N=2 is unmeasured; step 4 above exists to
  close that gap before committing to the full batch.
- Alpine cluster contention at submission time is unpredictable from any
  point-in-time snapshot; real wall time could run well past a
  compute-bound estimate depending on when the run lands.
- A run at this scale will visibly move this account's fair-share standing
  for several weeks (14-day decay half-life); not a lockout risk, but worth
  timing around other queued work if possible.
- The workflow this folder runs (`workflows/featurize_image_set.nf`) is the
  single-task-per-image-set version -- one `FEATURIZE_IMAGE_SET` Slurm job
  per pending image set, not the compartment x channel fan-out the pilot's
  own README documents benchmarking separately. Total jobs for a fresh
  `submit` run of the current 3,443-image-set index is `P + 3` (3,446: one
  `FEATURIZE_IMAGE_SET` per image set, `PLAN_IMAGE_SETS`, `BUILD_WAREHOUSE`,
  and the coordinator), not the pilot's per-compartment-fan-out job counts.
  Still a lot of individual Slurm jobs independent of whether account
  limits allow it. Job arrays for homogeneous shards remain deferred, same
  as in the pilot.

## Success criteria

- Staging verified complete for all 12 patients (index row count matches
  expected well/FOV counts, allowing for genuinely incomplete source data).
- Staged capacity test completes with a valid warehouse and a real
  `BUILD_WAREHOUSE` timing point at the tested scale.
- Full batch run completes, exit `0`, with `PLAN_IMAGE_SETS`/
  `BUILD_WAREHOUSE` reporting `validation_status: pass`.
- DuckDB validation view confirms expected joins, row counts, and no
  all-null feature columns across every patient in the batch.
- Run record, `warehouse_manifest`-equivalent artifacts, and full Nextflow/
  Slurm observability (`trace.tsv`, `timeline.html`, `report.html`,
  `dag.html`, `sacct` accounting) present and reviewed.

## Explicitly deferred

- The remaining patients beyond this 12-patient batch, and the full
  ~4200-image-set dataset.
- Job arrays for homogeneous shards.
- Any GPU/deep-learning execution.
- Isilon publishing as the durable warehouse location (this batch continues
  to use the PetaLibrary result directory as the warehouse root, same as
  the pilot; see `future_processing_plan.md` for when Isilon becomes the
  target).
