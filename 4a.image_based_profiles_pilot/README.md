# IBP pilot: stage 4 (organoid-cell relationships) on a ZEDProfiler warehouse

## What this is

A pilot checking whether `4.processing_image_based_profiles`'s downstream
steps -- specifically step 3, `3.organoid_cell_relationship.py` (organoid-cell
assignment + spatial shell/distance features) -- can run directly against a
ZEDProfiler warehouse (`3a.nextflow_pilot` / `3b.nextflow_production`
output), instead of the older CellProfiler-style per-feature-file pipeline
stage 4 was originally built for.

Two reference image sets, matching the ones used throughout
`3a.nextflow_pilot`: `NF0055_T1/B10-1` and `NF0014_T1/C4-2`
(`manifest/reference_image_sets.yaml`).

## Why steps 00/0a/1/2 are skipped

Those steps convert old per-feature parquet files (101 per image set) into a
merged per-well_fov DuckDB, then merge that into
`sc_profiles_{well_fov}.parquet`/`organoid_profiles_{well_fov}.parquet`/`nucleocentric_profiles_{well_fov}.parquet`
-- exactly what a ZEDProfiler warehouse already holds natively via
`warehouse.duckdb`'s `joined.images_nuclei_cell_cytoplasm` view (an inner
join across Nuclei/Cell/Cytoplasm on `Metadata_Object_ObjectID`, the same
object-intersection step 2 computes) and `profiles.organoid_profiles`.
Reimplementing steps 00/0a/1/2 against this data would just reproduce work
the warehouse already did. `scripts/build_ibp_inputs_from_warehouse.py`
reads those views for one image set and writes the three files step 3
expects, bridging two real differences along the way:

- **Column names**: step 3 finds centroid/bbox columns by substring-matching
  `"area"` (CellProfiler-era `*_AreaSizeShape_*` naming). ZEDProfiler's own
  naming convention (same `format_morphology_feature_name()` helper,
  different feature-type string) produces `*_VolumeSizeShape_*` instead --
  `"volumesizeshape"` contains no `"area"` substring, so the match originally
  missed silently. `VolumeSizeShape` is this project's preferred naming
  (ZEDProfiler lets us be opinionated about our own feature names rather
  than carrying CellProfiler-era conventions forward), so rather than
  disguise ZEDProfiler's columns as the older convention, step 3's own
  matching was widened to accept `"volumesizeshape"` directly (the four
  `if "area" in x.lower() ...` checks in
  `4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`,
  mirrored in the paired notebook, `.ipynb` being the source of truth --
  see that folder's own conversion convention). ZEDProfiler's column names
  flow through completely unchanged, start to finish -- no renaming
  anywhere in this pilot. (An earlier version of this pilot instead
  temporarily renamed columns to satisfy step 3's original matching and
  renamed them back afterward; per review feedback that round-trip was
  removed in favor of widening step 3's matching, since it's a smaller,
  more direct fix than disguising and undisguising column names around an
  unmodified script.)
- **Identifiers**: step 3 originally expected `object_id` and `image_set`
  columns (the older CellProfiler-era pipeline's own naming). Rather than
  have this pilot rename `Metadata_Object_ObjectID` to `object_id` and
  invent an `image_set` column before handoff, step 3 was changed to accept
  `Metadata_Object_ObjectID` directly (normalized to `object_id` -- step
  3's own internal working name, and the name the *rest* of stage 4
  downstream of step 3 already expects in these particular files, e.g.
  `6.annotation.py`/`7b.single_cell_qc.py` -- right after loading) and to
  derive `image_set` itself from its own `well_fov` argument, which it
  already has, rather than requiring it as an input column. So this pilot's
  adapter writes ZEDProfiler's own `Metadata_Object_ObjectID` straight
  through with no renaming; only step 3's persisted *output* uses
  `object_id`/`image_set`, matching the existing convention every other
  stage-4 step downstream of step 3 already relies on.
- **No Nucleocentric data**: ZEDProfiler doesn't produce deep-learning
  nucleocentric features. Step 3 still hard-requires a
  `nucleocentric_profiles_{well_fov}.parquet` to exist -- it strictly
  resolves that path and crashes immediately if missing -- so the adapter
  writes an empty placeholder *only if nothing is already there*, never
  overwriting real Nucleocentric data from an actual run of IBP steps
  00/0a/1/2 for the same well_fov. Step 3's resulting (always-empty)
  `nucleocentric_profiles_*_related.parquet` output is not persisted into
  `warehouse/ibp/` -- there's nothing in it worth keeping.

`3.organoid_cell_relationship.py`'s own logic (organoid-cell assignment,
shell/distance calculations) is otherwise unchanged -- the only edit is
widening those four column-matching checks to recognize ZEDProfiler's
naming alongside the original CellProfiler-era one.

## Environment

`4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`
imports from `image_analysis_3D` (`utils/`), a local editable package that's
part of the repo's _root_ uv environment. That root/utils environment also
declares heavy GPU dependencies (torch, napari, cellpose, medim) that step 3
never actually touches -- tracing its real imports
(`feature_writing_utils.py`, `neighbors_utils.py`, `loading_classes.py`,
`arg_parsing_utils.py`, `notebook_init_utils.py`) shows only pandas, numpy,
matplotlib, scikit-image, and tqdm are needed. So this pilot uses its own
small isolated environment (`environments/pyproject.toml`, matching 3a/3b's
pattern) with just those, plus `PYTHONPATH` pointed at `utils/src` so
`import image_analysis_3D...` resolves without an actual package install --
no torch/napari/cellpose required.

```bash
cd 4a.image_based_profiles_pilot
uv sync --project environments --locked
uv run --project environments python scripts/run_ibp_pilot.py \
  --warehouse-dir /path/to/a/3a-or-3b/results/<run_id>/warehouse
```

## Warehouse layout addition

Step 3's outputs land back in the **source warehouse's own directory**,
under a new `ibp/` folder alongside `profiles/`/`images/` -- same
one-file-per-image-set convention as `profiles/<compartment>_profiles/`, so
it's immediately queryable the same way and clearly separated from
ZEDProfiler's own output. Additive only -- never touches `profiles/` or
`images/`.

```text
warehouse/
  profiles/...                                  <- unchanged, ZEDProfiler's own output
  images/...                                     <- unchanged
  warehouse.duckdb                               <- unchanged base views; gains 2 new ibp.* views (see below)
  ibp/                                            <- new, this pilot's output
    sc_profiles_related/<image_id>.parquet        <- Nuclei+Cell+Cytoplasm + ParentOrganoid + shell/distance features
    organoid_profiles_related/<image_id>.parquet   <- Organoid + OrganoidSingleCellCount
```

(No `nucleocentric_profiles_related/` -- ZEDProfiler has no Nucleocentric
features, so step 3's output for it is always empty and isn't persisted.)

```python
import pandas as pd

pd.read_parquet(
    "warehouse/ibp/sc_profiles_related/NF0055_T1__NF0055_T1__B10__F1.parquet"
)
```

`run_ibp_pilot.py` also (re)creates two convenience DuckDB views in the
warehouse's existing `warehouse.duckdb`, under a new `ibp` schema --
`ibp.sc_profiles_related`, `ibp.organoid_profiles_related` -- matching the
same `CREATE OR REPLACE VIEW ... read_parquet(relative_glob)` pattern
`build_duckdb_views.py` uses for `profiles.*`/`images.*` (a stored query,
no data copy). Relative paths resolve against the current working
directory at query time, so `cd` into the warehouse directory first:

```bash
cd <warehouse_dir> && duckdb warehouse.duckdb
D SELECT * FROM ibp.sc_profiles_related LIMIT 5;
```

## Findings

**Verified working end-to-end** against a real 3a.nextflow_pilot warehouse
(`nf0055-nf0014-post-revert-20260821T150143Z`) for both reference image
sets. Both ran through unmodified step 3 with sane, stable output:

| Image set         | Cells (sc_profiles rows) | Assigned to an organoid | Organoids          | Max single-cell count on one organoid |
| ----------------- | ------------------------ | ----------------------- | ------------------ | ------------------------------------- |
| `NF0055_T1/B10-1` | 9                        | 9 (0 unassigned)        | 2 (1 with 0 cells) | 9                                     |
| `NF0014_T1/C4-2`  | 42                       | 42 (0 unassigned)       | 1                  | 42                                    |

Shell/distance features (`Nuclei_NoChannel_Neighbors_*`) are populated with
plausible, non-degenerate values -- e.g. `ShellsUsed=3` for all 9 cells in
`B10-1` (the script's own small-sample-size fallback: "9 cells with 4
shells = 2.2 cells/shell, reducing to 3 shells"), `NeighborsCountAdjacent`
ranging 0-2. Both image sets triggered the script's built-in small-N
fallbacks (Euclidean instead of Mahalanobis distance for `B10-1`'s 9 cells;
regularized covariance for `C4-2`'s 42) -- expected, graceful behavior
already present in step 3, not something this pilot needed to handle.

**One real bug found and fixed**: the first version of
`build_ibp_inputs_from_warehouse.py` read from
`joined.images_nuclei_cell_cytoplasm`, which -- as documented in
`build_duckdb_views.py` -- joins through `images.image_assets` and repeats
each object row once per image asset (channel/mask). For `B10-1` that
inflated 9 real cells into 72 duplicate rows before step 3 even ran, and
step 3's own shell-classification merge multiplied that further to 576.
Fixed by joining `profiles.nuclei_profiles`/`cell_profiles`/
`cytoplasm_profiles` directly (no `image_assets` join), matching what step 2
(the code this replaces) actually produces. Row counts were stable and
correct (9 and 42, matching mask object counts) after the fix -- this is
the version reflected in the table above.

**Nucleocentric**: as expected, both image sets produced an empty
nucleocentric table (ZEDProfiler has no deep-learning nucleocentric
features) -- step 3 handled this without incident. Per review feedback,
this output is no longer persisted into `warehouse/ibp/` at all (see the
2026-09-09 update below) -- there's nothing in it worth keeping, and always
writing an empty *input* placeholder for step 3 risked silently clobbering
real Nucleocentric data if this same well_fov/subparent_name is ever also
processed by an actual run of IBP steps 00/0a/1/2.

**Data quality, checked directly rather than assumed**: for both image
sets, all 2,665 feature columns had zero all-null columns; centroid
coordinates used for organoid assignment are real, physically plausible
pixel values (not zero/NaN); and -- most importantly -- every cell's
`ParentOrganoid` assignment was independently re-verified by checking that
the cell's own centroid actually falls inside the assigned organoid's
bounding box on every axis (`True` for both organoids with cells), not just
inferred from a 100% assignment rate.

**One real, expected difference from what the notebooks capture**: step 2's
own docstring states object IDs are reassigned to a sequential `1..N` range,
discarding the original segmentation mask IDs. This pilot skips step 2, so
`object_id` is the _original_ mask ID with gaps where the Nuclei/Cell/
Cytoplasm intersection dropped an object (e.g. `B10-1`'s IDs are
`[1,2,3,4,7,8,9,10,11]` -- 5 and 6 didn't survive the intersection). This
doesn't affect step 3's logic (it only needs unique, stable IDs, not a
contiguous range), but it is a real behavioral difference worth knowing
about if anything downstream ever assumes sequential IDs.

**Reproducibility, checked independently**: re-ran the full pilot a second
time from a different machine entirely (a local workstation, not Alpine),
executing the code locally and reading/writing the same warehouse directly
over PetaLibrary via an sshfs mount (`~/mnt/alpine/active/koala/...`), no
Slurm/SSH-to-Alpine involved. Produced identical results (same row counts,
same assignments) confirming the pilot doesn't depend on anything
Alpine-specific.

Not yet checked: behavior at higher object counts (both reference image
sets are small).

**Update (review feedback, 2026-09-09):** an earlier version of this pilot
persisted the `AreaSizeShape` rename into `warehouse/ibp/`'s final output.
Per review feedback, `VolumeSizeShape` is this project's preferred naming
end to end -- the rename is now applied only to the scratch files fed into
step 3, and reversed on step 3's own output before anything is written into
`warehouse/ibp/`. Re-verified against the real warehouse: 0 `AreaSizeShape`
columns remain in persisted output, organoid assignment still correct (9/9
and 42/42 cells assigned).

**Update (review feedback, 2026-09-09), metadata dedup:** the shared/
per-compartment metadata dedup in `build_ibp_inputs_from_warehouse.py`
(deciding which of Nuclei/Cell/Cytoplasm's copy of a metadata column to
keep when joining them) previously hardcoded a fixed list of exact column
names. Per review feedback, this is now driven by the metadata category
prefix instead (per `docs/RFC-2119-Feature-Naming-Convention.md` section
2.2): any `Metadata_Biology_*`/`Metadata_Experiment_*`/`Metadata_Imaging_*`
column is treated as shared across compartments (sample/experiment/imaging-
session metadata doesn't change because you're looking at a different
compartment's segmentation), and any `Metadata_Compartment*`/
`Metadata_Segmentation_*` column as per-compartment. This is both an answer
to the review question (yes, Biology/Experiment/Imaging metadata is shared)
and a real robustness fix: a newly added field under those categories is
now picked up automatically instead of silently leaking through as an
unexcluded, potentially name-colliding column. Re-verified: same output
shape (9/42 rows, 2682 columns, 7 shared metadata columns, zero duplicate
column names) as before this change.

**Update (review feedback, 2026-09-09), nucleocentric:** per review, this
pilot no longer persists a `nucleocentric_profiles_related` table into
`warehouse/ibp/` at all -- ZEDProfiler produces no real Nucleocentric
features, so step 3's output for it was always empty and not worth
keeping. Step 3 (unmodified) still hard-requires a
`nucleocentric_profiles_{well_fov}.parquet` *input* to exist or it crashes
immediately on a strict path resolve, so `build_ibp_inputs_from_warehouse.py`
still writes an empty placeholder for that -- but now only if nothing is
already there, so it can never clobber real Nucleocentric data from an
actual run of IBP steps 00/0a/1/2 for the same well_fov/subparent_name.
Verified the overwrite guard directly: seeded a fake "real" nucleocentric
input file, re-ran the pilot, confirmed the file was left untouched. Also
re-verified the full pilot end to end afterward: same correct results (9/9
and 42/42 cells assigned) as before this change.

**Update (review feedback, 2026-09-09), two more fixes:**
- The shared-metadata dedup's `shared` set was computed from Nuclei's
  columns only, then unconditionally excluded from Cell's and Cytoplasm's
  own `*` projections too. If either table were ever missing one of those
  columns (schema drift -- currently doesn't happen, since all three tables
  come from the same `add_metadata()` call, but this dedup logic exists
  specifically to tolerate that changing), `EXCLUDE` on a column that
  doesn't exist is a hard DuckDB error, not a no-op. `shared` is now
  intersected with each table's own live columns before being excluded
  from that table.
- The nucleocentric input placeholder's `if not nucleocentric_path.exists()`
  check and its write were not atomic -- a real file (from an actual IBP
  steps 00/0a/1/2 run for the same well_fov) could appear in the gap
  between them, and the write itself would still clobber it. Now written to
  a private temp file first, then published via `os.link()`, which raises
  `FileExistsError` instead of silently overwriting if the destination
  exists by the time of the actual publish. Verified directly: simulated a
  file appearing in exactly that gap (write temp file, then create the
  "real" destination file, then attempt the link) and confirmed `os.link()`
  raises, the real file survives untouched, and the temp file is cleaned up.

Re-verified the full pilot end to end after both fixes: same correct
results (9/9 and 42/42 cells assigned, zero duplicate columns) as before.

**Update (review feedback, 2026-09-10), removed the rename round-trip
entirely:** a reviewer raised a concern about the VolumeSizeShape <->
AreaSizeShape rename round-trip described above -- confirmed by direct
before/after comparison against real data that it only ever changed column
*names* (never values) and only in gitignored scratch files, fully reversed
before anything was persisted, but the round-trip itself was still an
unnecessary extra moving part. Per follow-up feedback, removed it entirely:
`4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`'s
four `"area" in x.lower()` column-matching checks (and the paired notebook,
the actual source of truth) now also accept `"volumesizeshape"` directly, so
ZEDProfiler's own column names flow through unchanged from warehouse to
warehouse -- no renaming anywhere in this pilot. Both
`rename_volumesizeshape_to_areasizeshape()` (`build_ibp_inputs_from_warehouse.py`)
and `rename_areasizeshape_to_volumesizeshape()` (`run_ibp_pilot.py`) are
deleted. This is the only change in this PR to a file outside `4a/` and
`4.processing_image_based_profiles`'s own `3.organoid_cell_relationship.py`/
`.ipynb` -- step 3's organoid-assignment and shell/distance logic is
otherwise byte-for-byte unchanged; only the column-name matching was
widened. Re-verified end to end against the real warehouse: same correct
results (9/9 and 42/42 cells assigned), and confirmed the bbox-column sort
order the code depends on (`[MaxX, MaxY, MaxZ, MinX, MinY, MinZ]`) holds
identically for `VolumeSizeShape` columns as it did for `AreaSizeShape`,
since sort order is determined only by the trailing Max/Min + axis letter,
not the family-name prefix.

**Update (review feedback, 2026-09-10), removed the identifier renaming
too:** a further reviewer comment raised the same concern about
`build_ibp_inputs_from_warehouse.py` renaming `Metadata_Object_ObjectID` to
`object_id` -- "if we need object_id for downstream code, we should change
the downstream code." Same fix pattern as the VolumeSizeShape round-trip
above: rather than rename ZEDProfiler's identifier column before handoff,
`3.organoid_cell_relationship.py` (and the paired notebook) was changed to
accept `Metadata_Object_ObjectID` directly, normalizing it to `object_id`
right after loading -- a no-op wherever `object_id` is already present, so
the older CellProfiler-era pipeline (steps 00/0a/1/2) is unaffected. The
adapter no longer renames anything at all: it writes ZEDProfiler's native
`Metadata_Object_ObjectID` straight through.

`image_set` needed a different fix, since ZEDProfiler has no equivalent
column at all to rename -- it's a synthetic per-well-FOV run label the
older pipeline's raw feature files happened to carry alongside `object_id`.
Rather than have the adapter invent this column, step 3 now derives it
directly from its own `well_fov` argument (which it already has) and sets
it unconditionally on the two dataframes that need it for an internal
merge (`sc_profile_df`, `nucleocentric_df` -- `organoid_profile_df` never
actually used this column; confirmed by grep, so the adapter's earlier
`organoid_df["image_set"] = ...` was dead weight and is also removed, not
just moved). Verified this is semantically correct, not just convenient:
`6.annotation.py` (a later stage-4 step) parses `image_set` as a literal
`"{well}-{fov}"` string (`.str.split("-").str[0]`) -- exactly the format
`--well-fov` already uses (e.g. `"B10-1"`), confirming `image_set == well_fov`
is the pipeline's own existing invariant, not a new assumption introduced
here.

The nucleocentric placeholder's columns changed from `["object_id",
"image_set"]` to `["Metadata_Object_ObjectID"]` for the same reason --
`image_set` is no longer supplied by any input file, and the identifier
column matches ZEDProfiler's own native naming like everything else this
adapter now writes.

Whether `object_id`/`image_set` persist as such in step 3's own *output*
(`warehouse/ibp/`) was a separate question from whether the adapter should
rename them on the way in. Confirmed via grep that `object_id` is the
existing, long-standing convention every stage-4 step *downstream* of step
3 already expects in these particular files (`6.annotation.py`,
`7b.single_cell_qc.py`) -- unlike VolumeSizeShape, which was purely a
naming-convention artifact with no other consumers, `object_id`/`image_set`
in `*_related.parquet` are the pipeline's own genuine schema for these
files, not something invented by this pilot. So no un-rename is applied
before persisting; `warehouse/ibp/`'s output columns are unchanged from
before this fix.

Re-verified end to end against the real warehouse: same correct results
(9/9 and 42/42 cells assigned, zero duplicate columns) as before this
change, and confirmed directly that the scratch input file the adapter
writes now carries `Metadata_Object_ObjectID` (not `object_id`) with no
`image_set` column at all, while step 3's internal working copy correctly
ends up with both (`Nucleocentric profile shape: (0, 2)` in this run's own
log output, matching the empty placeholder plus the derived `image_set`
column).

## New IBP workflow diagram

```mermaid
flowchart TD
    A1[cellpainting images and segmentations]

    A1 -->|featurization| B[ZEDProfiler single cell features ]
    A1 -->|featurization| C[ZEDProfiler organoid features ]
    A1 -->|featurization| D[Masked SAM-Med3D single cell features ]
    A1 -->|featurization| E[Masked SAM-Med3D organoid features ]
    A1 -->|featurization| F[Nucleocentric SAM-Med3D features ]
    A1 -->|featurization| G[Nucleocentric MorphEM features ]


    D --> |merging| G3[single-cell Masked SAM-Med3D features ]
    E --> |merging| G4[organoid Masked SAM-Med3D features ]
    F --> |merging| G5[nucleocentric SAM-Med3D features ]
    G --> |merging| G6[nucleocentric MorphEM features ]

    B --> H1[relate objects to organoids]
    C --> H2[relate objects to organoids]
    G3 --> H3[relate objects to organoids]
    G4 --> H4[relate objects to organoids]
    G5 --> H5[relate objects to organoids]
    G6 --> H6[relate objects to organoids]
    H1 --> |ZEDProfiler single cell features| I[Annotation]
    H2 --> |ZEDProfiler organoid features| I
    H3 --> |Masked SAM-Med3D single cell features| I
    H4 --> |Masked SAM-Med3D organoid features| I
    H5 --> |Nucleocentric SAM-Med3D features| I
    H6 --> |Nucleocentric MorphEM features| I
    I --> |ZEDProfiler single cell features| J[Normalized features]
    I --> |ZEDProfiler organoid features| J
    I --> |Masked SAM-Med3D single cell features| J
    I --> |Masked SAM-Med3D organoid features| J
    I --> |Nucleocentric SAM-Med3D features| J
    I --> |Nucleocentric MorphEM features| J
```
