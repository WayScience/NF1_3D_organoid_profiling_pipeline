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

Pilot data and run artifacts stay under
`/pl/active/koala/nf1-3d-pilot-workflow-db/`. This keeps the pilot fully
reversible and impossible to confuse with production Stage 3/4 output while
making inputs and benchmark outputs visible from Alpine.

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
23s/1GB figures as a _floor_, not a planning number, until this pilot
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

Available patients include `NF0055_T1`, which is the current single-image-set
pilot target.

**Patient for this pilot: `NF0055_T1`**.

**Working copy: PetaLibrary, not bandicoot directly.** The selected image set
has been staged under `/pl/active/koala/nf1-3d-pilot-workflow-db`, so both
Phase A and Phase B read from Alpine-visible paths instead of the local
bandicoot mount.

**Well/FOV for this pilot:** exactly one, chosen during the manual prep step
below — not the first of several candidates, since only one is being copied
to PetaLibrary at this stage. No need to specifically seek out a DMSO/control
well for a mechanics-only smoke test; treatment identity only matters once
profile _values_ are being interpreted, not while proving the fan-out
plumbing works. (The shared platemap — `WellRow, WellCol, WellPosition,
Treatment, Dose, Unit` — is available for the annotation join described
below, once we get there.)

## Manual prep step (human, before Phase A)

**Completed for the current pilot:** `NF0055_T1/B10-1` was copied from
bandicoot into PetaLibrary:

```text
/pl/active/koala/nf1-3d-pilot-workflow-db/data/NF0055_T1/zstack_images/B10-1/
/pl/active/koala/nf1-3d-pilot-workflow-db/data/NF0055_T1/segmentation_masks/B10-1/
```

This staged folder contains five z-stack TIFFs (`405`, `488`, `555`, `640`,
`TRANS`) and four masks (`nuclei`, `cell`, `cytoplasm`, `organoid`). The
checked-in Nextflow pilot manifest uses the four fluorescence channels plus
`nuclei_mask.tiff`.

## ZedProfiler version

Pin to **0.1.2** (current release, ships a functional CLI). This is a step
up from formascute's validated `0.1.1` — re-validate the `uv` environment
build against `0.1.2` rather than assuming the prior install recipe carries
over unchanged.

Because 0.1.2 is expected to have a functional CLI, **prefer invoking it
directly once the CLI's exact contract is confirmed on the copied image-set**.
The executable pilot added here uses a thin Python API adapter
(`scripts/run_zedprofiler_image_set.py`) for the first real run because this
repository already has a known ZEDProfiler API integration pattern and the data
is not yet available to verify the 0.1.2 CLI arguments. Treat that adapter as a
compatibility bridge, not a new feature library: keep it small, keep output
validation around it, and replace its task payload with the upstream CLI as soon
as Phase A proves the CLI covers per-compartment mask input, multi-channel
intensity input, all 6 feature classes, and Parquet output.

## Phase A — real data, no orchestration

Goal: confirm ZedProfiler 0.1.2 runs correctly and measure real cost, before
any Nextflow/Slurm variable is introduced. Mirrors formascute's own
"real ZedProfiler feature calls on real data" experiment, but against actual
NF1 data instead of ZedProfiler's tutorial dataset.

1. Build/refresh the `uv` ZedProfiler `0.1.2` environment.
2. Use the staged `NF0055_T1/B10-1` manifest under
   `manifest/nf0055_b10_1_alpine.yaml`.
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
├── Makefile                       # isolated pilot entrypoint, patterned after formascute
├── bin/
│   └── nf1-nextflow-pilot         # check/doctor/preflight/run/submit wrapper
├── nextflow.config                # local and CURC profile selection + observability
├── workflows/
│   └── featurize_image_set.nf     # one process, wraps the ZedProfiler 0.1.2 CLI call proven in Phase A
├── conf/
│   ├── base.config                 # generic SLURM-compatible resources, no CURC specifics
│   └── curc_alpine.config          # Persistence1 / acpu / cpu-normal / module names — the named CURC profile
├── environments/
│   ├── pyproject.toml              # isolated uv project, pinned to ZEDProfiler 0.1.2
│   └── zedprofiler-uv.md           # uv env build recipe, pinned to ZedProfiler 0.1.2
├── manifest/
│   ├── nf0055_b10_1_alpine.yaml    # default single-image-set manifest with /pl/active paths
│   └── smoke_test_image_set.yaml   # legacy placeholder manifest
├── scripts/
│   ├── build_manifest.py           # scan PetaLibrary copy and write one manifest row
│   ├── manifest_io.py              # minimal manifest reader/writer
│   ├── run_zedprofiler_image_set.py # temporary API adapter + run record + validation
│   └── smoke_synthetic.py          # data-free artifact smoke check
└── results/                        # local smoke-only output; real benchmark outputs go to /pl/active/koala/nf1-3d-pilot-workflow-db/results/
```

Manifest row shape (`Metadata_*` naming per the roadmap, used from the start
since retrofitting it later is expensive):

| field                           | value                                                                                        |
| ------------------------------- | -------------------------------------------------------------------------------------------- |
| `Metadata_Biology_PatientTumor` | `NF0055_T1`                                                                                  |
| `Metadata_Experiment_WellID`    | `B10`                                                                                        |
| `Metadata_Imaging_FieldID`      | `1`                                                                                          |
| `Metadata_Imaging_ImageID`      | deterministic build from the three fields above                                              |
| channel paths                   | all channels under the chosen `zstack_images/{well_fov}/`                                    |
| mask path                       | `segmentation_masks/{well_fov}/Nuclei_mask.tif`                                              |
| feature families                | `[VolumeSizeShape, Intensity, Texture, Colocalization, Neighbors, Granularity]`              |
| resources                       | set from Phase A's measured numbers, with headroom — not formascute's tutorial-scale figures |

Execution: use the folder-local `Makefile`. The intended sequence is:

```bash
cd 3a.nextflow_pilot
make check
uv sync --project environments
make validate-manifest
make submit-dry-run ACCOUNT=<allocation> SUBMIT_HOST=Persistence1
make submit ACCOUNT=<allocation> SUBMIT_HOST=Persistence1
```

The `curc_alpine` profile submits the one feature task to Slurm on `acpu` /
`cpu-normal`, while the generated coordinator job loads `nextflow/25.10.2` on
`Persistence1`. The task runs Python through the isolated `uv` project in
`3a.nextflow_pilot/environments`.

Output columns: `Metadata_*` identifiers above, plus ZEDProfiler's own
feature-naming-convention columns
(`{Compartment}_{Channel}_{FeatureType}_{Measurement}`).

Run record (`/pl/active/koala/nf1-3d-pilot-workflow-db/results/<run_id>/run_record.json`):
command line, git commit of this repo, ZedProfiler version + `uv` env hash,
manifest row, output row/column counts, elapsed time, exit status, pass/fail.

**Minimal manifest stub (`/pl/active/koala/nf1-3d-pilot-workflow-db/results/<run_id>/warehouse_manifest.json`):**
not the roadmap's production manifest — a single-table stub, scoped to this
pilot's one output, written to prove the manifest _shape_ early rather than
retrofit it later (same reasoning as adopting `Metadata_*` naming now).
Points at this pilot's own benchmark results directory, not Isilon:

```json
{
  "warehouse_root": "/pl/active/koala/nf1-3d-pilot-workflow-db/results/<run_id>/",
  "tables": [
    {
      "name": "profiles.nuclei_profiles",
      "path": "/pl/active/koala/nf1-3d-pilot-workflow-db/results/<run_id>/nuclei_profiles.parquet",
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

- Generalized multi-row manifest generation/discovery tooling.
- Multi-compartment (Cell, Cytoplasm, Organoid), multi-channel fan-out.
- Multi-well/multi-patient pilot subset.
- The roadmap's full production `warehouse_manifest.json` (namespaced
  `images.image_assets` / `profiles.*` tables) and Isilon publishing — this
  pilot writes only a single-table stub manifest inside its own benchmark
  results directory (see Phase B), not the production warehouse.
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
- Whether to replace the temporary Python API adapter with a direct
  ZEDProfiler 0.1.2 CLI invocation after the first benchmark run.
