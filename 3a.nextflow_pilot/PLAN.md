# 3a.nextflow_pilot — ZedProfiler CPU fan-out pilot plan

## Scope

Prove that ZEDProfiler's six handcrafted feature classes (VolumeSizeShape,
Intensity, Texture, Colocalization, Neighbors, Granularity) can run as a
Nextflow-orchestrated Slurm task against **one real NF1 image-set**, on
Alpine, with output shaped for the target warehouse schema in
[`docs/source/future_processing_plan.md`](../docs/source/future_processing_plan.md).

Explicitly **not** in this plan, but architecturally reserved:

- SAMMed3D / morphem deep-learning features (GPU, `aa100`). These stay on
  their current bespoke path per the roadmap. This pilot reserves a named,
  unimplemented seam (a second Nextflow profile) so GPU work slots in later
  without restructuring anything built here.
- Multi-well/multi-patient pilot subset — that is the roadmap's own step 2
  ("Pilot subset selection"). Premature until single-image-set mechanics are
  proven.
- The roadmap's production warehouse — Isilon publishing, the full namespaced
  table set, DuckDB validation, production rollout (roadmap steps 4-8). This
  pilot borrows the roadmap's naming/identifier conventions and writes a
  minimal single-table manifest stub (see Phase B) so the shape is proven
  early, but it does not stand up the warehouse itself.

All pilot output stays entirely inside this directory
(`3a.nextflow_pilot/results/`) — nothing is written under the repository's
`data/` tree. This keeps the pilot fully reversible and impossible to confuse
with production Stage 3/4 output.

## Relationship to formascute's findings

[formascute](https://github.com/d33bs/formascute) already validated, on
**synthetic and ZedProfiler's own bundled tutorial data**:

- Nextflow submits/tracks real Slurm jobs on Alpine via `SUBMIT_HOST=Persistence1`,
  `acpu`, `cpu-normal`, `nextflow/25.10.2`.
- A project-owned `uv` env (Python 3.12) runs real ZedProfiler end-to-end
  through Nextflow with explicit `memory '4 GB'` / `cpus 2` / `time 30.m`.
- On that tutorial data (`100×258×258` volumes, 5 objects), all 6 extractors
  took **~23.0s/image-set, ~1.0GB peak RSS**.

**That last number does not transfer directly to this pilot.** The tutorial
volumes are far smaller than real NF1 acquisitions (per the pipeline
README: ~30-50 z-slices, ~1500×1500 XY pixels). Treat formascute's
23s/1GB figures as a *floor*, not a planning number, until this pilot
measures real NF1-scale timing and memory directly. This is the single most
important correction to carry forward — resource directives sized off the
tutorial-scale number risk OOM or silent truncation on real data.

## Data source

Source of truth for the raw data, per patient-data mapping supplied for
ZedProfiler dev work:

```text
raw images:         ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/zstack_images/{well_fov}/
segmentation masks: ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/segmentation_masks/{well_fov}/
```

Available patients (`patient_IDs.txt`): `NF0014_T1`, `NF0014_T2`,
`NF0016_T1`, `NF0018_T6`, `NF0021_T1`, `NF0030_T1`.

**Patient for this pilot: `NF0014_T1`** (arbitrary choice, confirmed fine —
no need to hunt for a "representative" patient at n=1).

**Working copy: PetaLibrary, not bandicoot directly.** bandicoot is not
assumed reachable from Alpine compute nodes, so it is not used directly by
either pilot phase. Both Phase A and Phase B read from a PetaLibrary copy,
so the data source stays identical across both phases instead of switching
mid-pilot.

**Well/FOV for this pilot:** exactly one, chosen during the manual prep step
below — not the first of several candidates, since only one is being copied
to PetaLibrary at this stage. No need to specifically seek out a DMSO/control
well for a mechanics-only smoke test; treatment identity only matters once
profile *values* are being interpreted, not while proving the fan-out
plumbing works. (The shared platemap — `WellRow, WellCol, WellPosition,
Treatment, Dose, Unit` — is available for the annotation join described
below, once we get there.)

## Manual prep step (human, before Phase A)

**Action item, not automatable from here:** copy exactly **one** well_fov's
data for `NF0014_T1` from bandicoot to PetaLibrary before Phase A starts —
its complete `zstack_images/{well_fov}/` channel set and matching
`segmentation_masks/{well_fov}/` masks. Even though this patient has many
well_fovs available, the pilot starts with one and only extends to more once
the single-image-set path is proven end to end (see Success criteria).

- Which well_fov: pick any one with a complete channel set and a complete
  matching mask set — arbitrary is fine, same as the patient choice above.
- Exact PetaLibrary destination path is TBD — confirm the project's
  PetaLibrary allocation/group path and record it here once known (commonly
  mounted on Alpine under `/pl/active/<group>/...`, but do not assume that
  without confirming for this project's allocation).
- Once this copy exists, treat PetaLibrary (not bandicoot) as this pilot's
  data source for both phases below. Extending to more well_fovs or patients
  later means repeating this same manual bandicoot → PetaLibrary prep step
  for the additional image-sets, not changing how Phase A/B consume data.

## ZedProfiler version

Pin to **0.1.2** (current release, ships a functional CLI). This is a step
up from formascute's validated `0.1.1` — re-validate the `uv` environment
build against `0.1.2` rather than assuming the prior install recipe carries
over unchanged.

Because 0.1.2 has a real CLI, **prefer invoking it directly as the Nextflow
task payload** over writing a custom Python glue script
(`3.cellprofiling/scripts/zp_feature_call.py`-style). Fewer lines of
pipeline-owned code, and it stays in sync with upstream automatically.
Confirm during Phase A whether the CLI's supported inputs/outputs cover what
this pilot needs (per-compartment mask input, multi-channel intensity input,
all 6 feature classes, Parquet output); fall back to the existing Python
wrapper only if the CLI has a real gap.

## Phase A — real data, no orchestration

Goal: confirm ZedProfiler 0.1.2 runs correctly and measure real cost, before
any Nextflow/Slurm variable is introduced. Mirrors formascute's own
"real ZedProfiler feature calls on real data" experiment, but against actual
NF1 data instead of ZedProfiler's tutorial dataset.

1. Build/refresh the `uv` ZedProfiler `0.1.2` environment.
2. Enumerate `NF0014_T1/zstack_images/` and `segmentation_masks/`, pick the
   first well_fov with a complete Nuclei mask and full channel set.
3. Run the ZedProfiler CLI directly (no Nextflow) against that one
   image-set's Nuclei compartment, DNA (405) channel as the primary mask
   channel, all channels the CLI needs for Colocalization.
4. Record wall-clock time and peak RSS for each of the 6 feature classes
   individually and combined (`/usr/bin/time -v` or equivalent).
5. Record output shape: row count (should equal Nuclei object count in the
   mask), column count, any NaN/inf columns.

This produces the real per-task resource numbers Phase B's `process.time`
and `process.memory` directives should be based on — not formascute's
tutorial-scale figures.

## Phase B — one Nextflow process, one Slurm task, on Alpine

Only after Phase A produces a working CLI invocation and real timing/memory
numbers.

```
3a.nextflow_pilot/
├── PLAN.md                        # this document
├── workflows/
│   └── featurize_image_set.nf     # one process, wraps the ZedProfiler 0.1.2 CLI call proven in Phase A
├── conf/
│   ├── base.config                 # generic SLURM-compatible resources, no CURC specifics
│   └── curc_alpine.config          # Persistence1 / acpu / cpu-normal / module names — the named CURC profile
├── environments/
│   └── zedprofiler-uv.md           # uv env build recipe, pinned to ZedProfiler 0.1.2
├── manifest/
│   └── smoke_test_image_set.yaml   # one hardcoded row, filled in from Phase A's chosen well_fov
└── results/                        # gitignored; trace.tsv, timeline.html, report.html, dag.html, run_record.json, warehouse_manifest.json
```

Manifest row shape (`Metadata_*` naming per the roadmap, used from the start
since retrofitting it later is expensive):

| field | value |
|---|---|
| `Metadata_Biology_PatientTumor` | `NF0014_T1` |
| `Metadata_Experiment_WellID` | *(from Phase A directory listing)* |
| `Metadata_Imaging_FieldID` | *(from Phase A directory listing)* |
| `Metadata_Imaging_ImageID` | deterministic build from the three fields above |
| channel paths | all channels under the chosen `zstack_images/{well_fov}/` |
| mask path | `segmentation_masks/{well_fov}/Nuclei_mask.tif` |
| feature families | `[VolumeSizeShape, Intensity, Texture, Colocalization, Neighbors, Granularity]` |
| resources | set from Phase A's measured numbers, with headroom — not formascute's tutorial-scale figures |

Execution: `curc_alpine.config` profile, `Persistence1` submission, `acpu`,
`cpu-normal`, `nextflow/25.10.2` loaded in the coordinator job, `uv`
ZedProfiler `0.1.2` env activated for the task.

Output columns: `Metadata_*` identifiers above, plus ZEDProfiler's own
feature-naming-convention columns
(`{Compartment}_{Channel}_{FeatureType}_{Measurement}`).

Run record (`results/<run_id>/run_record.json`): command line, git commit of
this repo, ZedProfiler version + `uv` env hash, manifest row, output row/column
counts, elapsed time, exit status, pass/fail.

**Minimal manifest stub (`results/<run_id>/warehouse_manifest.json`):** not
the roadmap's production manifest — a single-table stub, scoped to this
pilot's one output, written to prove the manifest *shape* early rather than
retrofit it later (same reasoning as adopting `Metadata_*` naming now).
Points at this pilot's own `results/` directory, not Isilon:

```json
{
  "warehouse_root": "3a.nextflow_pilot/results/<run_id>/",
  "tables": [
    {
      "name": "profiles.nuclei_profiles",
      "path": "results/<run_id>/nuclei_profiles.parquet",
      "schema_version": "0.1.0-pilot",
      "join_keys": [
        "Metadata_Biology_PatientTumor",
        "Metadata_Imaging_ImageID",
        "Metadata_Object_ObjectID"
      ],
      "source_image_root": "<PetaLibrary path from the manual prep step>",
      "run_id": "<run_id>",
      "git_commit": "<sha>",
      "row_count": null,
      "validation_status": "pending"
    }
  ]
}
```

`row_count` and `validation_status` are filled in once the validation checks
below run. This stub is deliberately small: one table, one source asset, no
attempt at the roadmap's full namespace (`images.image_assets`,
`profiles.plate_map_annotations`, etc.) or at Isilon publishing — those stay
deferred to the roadmap's real step-2 pilot.

Observability: collect Nextflow's `trace.tsv`, `timeline.html`,
`report.html`, `dag.html`, `.nextflow.log`, plus `sacct` output for the one
Slurm task.

**Optional, non-blocking:** join the manifest row against the shared
platemap (`WellRow, WellCol, WellPosition, Treatment, Dose, Unit`) to attach
treatment metadata to the output. Nice to have for interpretability, not
required to prove the fan-out mechanics — do this only if Phase B is
otherwise done early.

## Validation

- Output file(s) exist and are non-empty.
- Row count matches distinct object labels in the Nuclei mask.
- All six feature-family column groups are present.
- No all-null feature columns.
- `Metadata_*` identifiers match the manifest row exactly.

## Reserved seam for GPU / deep-learning work (not implemented now)

`conf/curc_alpine.config` reserves a second, **unused** profile name (e.g.
`alpine_gpu`, `aa100` / `gpu-normal`) with a comment pointing at this
section — not filled in, not tested. SAMMed3D and morphem stay on their
current bespoke path. When GPU work starts, it becomes its own Nextflow
process(es) under this separate profile, never mixed into the CPU `acpu`
executor (formascute: don't mix CPU and GPU work in one profile). No code,
stub `.nf` file, or directory for this yet.

## Success criteria

Gates moving to the roadmap's actual step-2 pilot subset
(`NF0014_T1` + `NF0055_T1`, multiple wells/FOVs):

- Phase A: ZedProfiler 0.1.2 CLI produces valid output on one real NF1
  image-set, with real timing/memory numbers recorded.
- Phase B: task completes end-to-end through Nextflow + Slurm on Alpine,
  exit `0`, using resource directives derived from Phase A (not formascute's
  tutorial-scale numbers).
- Output passes all validation checks above.
- Run record + minimal manifest stub + full observability artifact set
  present and reviewed; manifest stub's `row_count`/`validation_status`
  filled in and correct.
- Output columns pass a manual spot check against the roadmap's identifier
  and feature-naming tables.

## Explicitly deferred

- Manifest generation/discovery tooling.
- Multi-compartment (Cell, Cytoplasm, Organoid), multi-channel fan-out.
- Multi-well/multi-patient pilot subset.
- The roadmap's full production `warehouse_manifest.json` (namespaced
  `images.image_assets` / `profiles.*` tables) and Isilon publishing — this
  pilot writes only a single-table stub manifest inside its own `results/`
  directory (see Phase B), not the production warehouse.
- DuckDB validation views.
- Job arrays for homogeneous shards.
- Any GPU/deep-learning execution.
- Production-scale (4200 image-set) run.

## Open decisions to resolve during execution, not planning

- Whether `~/mnt/bandicoot/...` is reachable from Alpine compute nodes or
  needs staging (see Data source above).
- Whether the ZedProfiler 0.1.2 CLI alone is sufficient, or whether
  Colocalization/Granularity need extra parameters not exposed by the CLI
  yet (resolve in Phase A).
- Exact well_fov choice (resolve by directory listing in Phase A).
