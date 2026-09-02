# IBP pilot: stage 4 (organoid-cell relationships) on a ZedProfiler warehouse

## What this is

A pilot checking whether `4.processing_image_based_profiles`'s downstream
steps -- specifically step 3, `3.organoid_cell_relationship.py` (organoid-cell
assignment + spatial shell/distance features) -- can run directly against a
ZedProfiler warehouse (`3a.nextflow_pilot` / `3b.nextflow_production`
output), instead of the older CellProfiler-style per-feature-file pipeline
stage 4 was originally built for.

Two reference image sets, matching the ones used throughout
`3a.nextflow_pilot`: `NF0055_T1/B10-1` and `NF0014_T1/C4-2`
(`manifest/reference_image_sets.yaml`).

## Why steps 00/0a/1/2 are skipped

Those steps convert old per-feature parquet files (101 per image set) into a
merged per-well_fov DuckDB, then merge that into
`sc_profiles_{well_fov}.parquet` / `organoid_profiles_{well_fov}.parquet` /
`nucleocentric_profiles_{well_fov}.parquet` -- exactly what a ZedProfiler
warehouse already holds natively via `warehouse.duckdb`'s
`joined.images_nuclei_cell_cytoplasm` view (an inner join across Nuclei/
Cell/Cytoplasm on `Metadata_Object_ObjectID`, the same object-intersection
step 2 computes) and `profiles.organoid_profiles`. Reimplementing steps
00/0a/1/2 against this data would just reproduce work the warehouse already
did. `scripts/build_ibp_inputs_from_warehouse.py` reads those views for one
image set and writes the three files step 3 expects, bridging two real
differences along the way -- without touching step 3's own code:

- **Column names**: step 3 finds centroid/bbox columns by substring-matching
  `"area"` (CellProfiler-era `*_AreaSizeShape_*` naming). ZedProfiler's own
  naming convention (same `format_morphology_feature_name()` helper,
  different feature-type string) produces `*_VolumeSizeShape_*` instead --
  `"volumesizeshape"` contains no `"area"` substring, so the match misses
  silently. Every `VolumeSizeShape` column is renamed to its `AreaSizeShape`
  equivalent.
- **Identifiers**: step 3 expects `object_id` (ours: `Metadata_Object_ObjectID`)
  and `image_set` (ours: derived from `--well-fov` directly).
- **No Nucleocentric data**: ZedProfiler doesn't produce deep-learning
  nucleocentric features. The adapter writes an empty (0-row) nucleocentric
  parquet -- step 3 already has empty-dataframe handling for this case.

`3.organoid_cell_relationship.py` itself is invoked completely unmodified.

## Environment

`4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`
imports from `image_analysis_3D` (`utils/`), a local editable package that's
part of the repo's *root* uv environment. That root/utils environment also
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
uv sync --project environments
uv run --project environments python scripts/run_ibp_pilot.py \
  --warehouse-dir /path/to/a/3a-or-3b/results/<run_id>/warehouse
```

## Warehouse layout addition

Step 3's outputs land back in the **source warehouse's own directory**,
under a new `ibp/` folder alongside `profiles/`/`images/` -- same
one-file-per-image-set convention as `profiles/<compartment>_profiles/`, so
it's immediately queryable the same way and clearly separated from
ZedProfiler's own output. Additive only -- never touches `profiles/` or
`images/`.

```text
warehouse/
  profiles/...                                  <- unchanged, ZedProfiler's own output
  images/...                                     <- unchanged
  warehouse.duckdb                               <- unchanged
  ibp/                                            <- new, this pilot's output
    sc_profiles_related/<image_id>.parquet        <- Nuclei+Cell+Cytoplasm + ParentOrganoid + shell/distance features
    organoid_profiles_related/<image_id>.parquet   <- Organoid + OrganoidSingleCellCount
    nucleocentric_profiles_related/<image_id>.parquet  <- empty (ZedProfiler has no nucleocentric features)
```

```python
import pandas as pd
pd.read_parquet("warehouse/ibp/sc_profiles_related/NF0055_T1__NF0055_T1__B10__F1.parquet")
```

## Findings

_To fill in after the first real run: row counts in/out per image set,
whether organoid-cell assignment looks sane (assigned vs. unassigned cell
counts), spot-checked shell/distance feature values, and anything about the
missing Nucleocentric family worth flagging back to whoever owns real IBP
data._
