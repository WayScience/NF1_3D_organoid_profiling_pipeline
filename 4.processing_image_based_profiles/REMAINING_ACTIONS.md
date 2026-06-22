# Remaining Actions — Profile Review Fixes

Tracks actions that are needed outside `4.processing_image_based_profiles/` or
require pipeline re-execution after in-folder fixes are committed.

---

## Issue 1 — CHAMMI features completely eliminated at feature selection

**Fix implemented:** Row-level NaN filter (`drop_high_na_rows`, cutoff >20%) added
to `8.normalization.py` before `normalize()` is called. Drops ~948 CHAMMI rows
with no model output, preventing them from entering Stage 5 files.

**Remaining actions:**

- [ ] **Upstream — re-run CHAMMI model** on the ~40% of cells with missing output.
  The row filter is a safeguard; the real fix is complete model coverage. Identify
  whether the missing cells were excluded due to FOV boundary conditions, a partial
  batch run, or a pipeline error.
- [ ] **Re-run pipeline stages 5–8** for all patients (starting from
  `8.normalization.py`) to produce clean normalized outputs reflecting the new filter.

---

## Issue 2 — `Cell_NoChannel_AreaSizeShape_SurfaceArea` has 24.5% missing values

**Root cause:** Bug in `utils/src/image_analysis_3D/featurization_utils/area_size_shape_utils.py`.
`calculate_surface_area()` receives the full multi-cell label image instead of the
single-cell masked image. `volume_truths = volume > 0` marks neighboring cells' voxels
as foreground, so for small cells surrounded by neighbors the bounding box contains no
background voxels — causing `ValueError: Surface level must be within volume data range`
in `skimage.measure.marching_cubes`. The failure is silently caught and stored as NaN.

**Fix implemented:** *(pending — see decision point below)*

**Decision point:**
- [ ] **Should `Cell_NoChannel_AreaSizeShape_SurfaceArea` be added to the feature
  selection blocklist?** It has 24.5% NaN and would be dropped at Stage 6 regardless,
  but the underlying bug (see remaining actions below) should be fixed first so the
  feature can be re-evaluated once NaN rate drops. Discuss before implementing.

**Remaining actions — upstream bugs in `utils/area_size_shape_utils.py`:**

- [ ] **Bug 1 (High) — Wrong image passed to `calculate_surface_area`** (line 132):
  `label_object` (full image) is passed instead of `subset_lab_object` (single-cell
  mask). Fix: change `label_object=label_object` → `label_object=subset_lab_object`
  in the `calculate_surface_area` call. Re-run featurization and re-evaluate NaN rate.
  **Regression test added** (`test_surface_area_not_nan_for_surrounded_cell` in
  `utils/tests/test_featurization_utils.py`) — currently failing, will pass once fixed.

- [ ] **Bug 2 (Medium) — Inconsistent units**: `Volume` is stored in voxels (raw
  `regionprops area` count), while `SurfaceArea` is computed with physical spacing
  applied and is in µm². Downstream volume/SA ratios or joint analyses are comparing
  incompatible units. Fix: either convert `Volume` to µm³ by multiplying by
  `spacing[0] * spacing[1] * spacing[2]`, or document the unit mismatch explicitly.

- [ ] **Bug 3 (Medium) — `MaxZ`/`MaxY`/`MaxX` are exclusive end coordinates** (lines
  122–123): `regionprops_table` bbox max values follow Python slice convention
  (exclusive), so stored `MaxZ` = last occupied z-slice + 1. The value is misleading
  as a spatial coordinate (off-by-one). Span computations (`MaxZ - MinZ`) happen to
  be correct, but any use of `MaxZ` as an absolute position is wrong. Fix: subtract 1
  when storing, or document clearly.

- [ ] **Bug 4 (Low) — Silent exception swallows all errors** (lines 137–138):
  `except (RuntimeError, ValueError) as e` catches expected edge cases but also any
  unexpected programming errors without logging. The caught `e` is never used. Fix:
  add at minimum a `warnings.warn` or logging call with the object label and error
  message so failures are observable.

- [ ] **Bug 5 (Low) — Full image copied per object** (line 105): `label_object.copy()`
  is called once per cell, then zeroed everywhere except the target label. For large
  3D images with many cells this is O(N×M) in memory and time. Fix: replace with a
  boolean mask (`label_object == label`) without a full array copy.

**Remaining actions — full-image copy anti-pattern in other NF1 utils:**

The `label_object.copy()` per-object anti-pattern (performance, not correctness)
also appears in:
- `utils/intensity_utils.py` — copies both `label_object` and `image_object` per cell
- `utils/neighbors_utils.py` — copies `label_object` per cell

`texture_utils.py` is already correct: it precomputes all bboxes once with a single
`regionprops_table` call, then crops per object. This is the pattern to follow.
`colocalization_utils.py` and `granularity_utils.py` are clean (different design).

**Remaining actions — ZedProfiler status:**

ZedProfiler (`ZedProfiler/src/zedprofiler/featurization/`) was checked against all
5 area/size/shape bugs and the copy anti-pattern:

| Module | `label_object` bug (Bug 1) | Full-image copy per object |
|---|---|---|
| `volumesizeshape.py` | ❌ present (line 173) | ❌ present (line 141) |
| `intensity.py` | n/a | ❌ present (lines 68–69) |
| `neighbors.py` | n/a | ❌ present (line 123) |
| `texture.py` | n/a | ✅ fixed (precomputes bboxes, crops per object) |
| `granularity.py` | n/a | ✅ n/a (image-wide design) |
| `colocalization.py` | n/a | ✅ clean |

ZedProfiler's `texture.py` already implements the correct pattern. The same fix
should be applied to `volumesizeshape.py`, `intensity.py`, and `neighbors.py` in
ZedProfiler, and then mirrored back into the NF1 pipeline `utils/`.

---

## Issue 3 — Well C4 / DMSO: Nuclei intensity + colocalization dropout for 55 cells

**Clarification:** This is not a complete Nuclei feature dropout. Nuclei shape/
morphology features are intact. The 120 null features are exclusively Nuclei
**Intensity** (72) and **Colocalization** (48) across all 4 channels (AGP, DNA,
ER, Mito). This points to missing or failed fluorescence channel images for well
C4, not a segmentation failure.

**Fix implemented:** *(no in-scope fix — upstream issue)*

**Decision point — check bandicoot for raw images:**

- [ ] **Do the raw z-stack channel images exist for well C4 on bandicoot?**
  - **If yes → recompute**: re-run intensity and colocalization featurization for
    C4 (`3.cellprofiling/scripts/intensity.py`, `colocalization.py`) and
    re-propagate through stages 4–8.
  - **If no → drop cells**: add well C4 to an explicit exclusion list in
    `7b.single_cell_qc.py` (or `8.normalization.py`) so these 55 cells are
    removed before normalization. They currently contribute to the DMSO reference
    population with no channel intensity data, which biases the normalization fit.

---

## Issue 4 — Small/aberrant nuclei producing near-complete NaN rows

**Finding:** Exactly 4 cells have >20% NaN features (all ~26%, exclusively Nuclei
intensity features). 3/4 are flagged by `cqc_small_nuclei_outlier`. 1 cell (well
G2) has severe Nuclei feature dropout but a nucleus not extreme enough in volume
to trigger the z-score threshold — it slips through cosmicqc but is caught by the
row filter.

**Fix implemented:** Fully resolved by Issue 1 row filter — all 4 cells are
dropped before normalization.

**Remaining actions:**

- [ ] *(Low priority)* The 1 unflagged cell in well G2 illustrates a cosmicqc
  coverage gap: Nuclei intensity failure does not always co-occur with extreme
  nucleus volume. No action needed in `4.processing_image_based_profiles/` — the
  row filter is the appropriate safety net here.

---

## Issue 5 — `sammed_organoid` has 100% NaN rows for some organoids

**Finding:** Exactly 1 organoid (well D4, FOV D4-1, DMSO 1%) has 100% of its
3072 SAMMed3D features null. All other 241 organoids are intact. The SAM-Med
model produced zero output for this specific organoid — same root cause as the
CHAMMI gap in Issue 1 (model not run or failed silently).

**Fix implemented:** Fully resolved by Issue 1 row filter — the 1 affected
organoid is dropped before normalization.

**Remaining actions:**

- [ ] **Upstream — check SAM-Med run logs for well D4, FOV D4-1**: determine
  whether the model was never submitted for this organoid, failed silently, or
  produced an empty crop. Re-run SAM-Med on this organoid if recoverable.

---

## Issue 6 — No cosmicqc columns on deep learning profile types

**Fix implemented:** New script `scripts/7c.propagate_cqc_to_dl_profiles.py` added.
Propagates `Metadata_cqc_*` columns from handcrafted profiles onto DL profiles via
a left-join on `(Metadata_Experiment_WellFOV, Metadata_Object_ObjectID)`. Includes
duplicate-key assertions, row-count validation, and a `validate="1:1"` merge guard.

SC propagation is clean (2,351/2,351 rows matched for sammed_sc, nucleocentric_sammed,
nucleocentric_morphem). Organoid propagation is partial — see below.

**Remaining actions:**

- [ ] **Create companion notebook `7c.propagate_cqc_to_dl_profiles.ipynb`** from the
  script (consistent with pipeline convention of notebook-first development).
- [ ] **Add `7c` to the pipeline execution order** in the folder README — it must run
  after `7b.single_cell_qc` and before `8.normalization`.
- [ ] **Re-run stages 7c and 8** for all patients to produce DL profiles with CQC flags.

**Decision point — 38 unmatched sammed_organoid rows:**

At Stage 5, `sammed_organoid` has 242 rows and handcrafted `organoid` has 204 rows.
Investigation of the pipeline code shows the gap most likely reflects a **pipeline
reproducibility issue**: the two profiles were generated by different runs at different
times. `sammed_organoid_anno.parquet` appears to have been regenerated directly from
raw segmentation masks (via `3.cellprofiling/scripts/dl_features.py`), which processes
all non-zero label IDs from the mask. The handcrafted `organoid_anno.parquet` reflects
the AreaSizeShape-anchored left join in `1.merge_feature_parquets.py`, which only
retains organoids that were successfully measured by all handcrafted feature types.
Organoids that failed handcrafted measurement (e.g. due to the surface area bug in
Issue 2, missing channel images, or other silent featurization errors) would be absent
from the handcrafted profile but present in the SAMMed profile.

The 38 unmatched organoids are **not** filtered by any QC step — `7a.organoid_qc.py`
only flags, never removes rows, and `normalize()` cannot drop rows. The mismatch
originates upstream of Stage 3.

**Additionally: 3 of the 38 have `Metadata_Object_ObjectID = 0`** (wells D10-1,
D2-2, E2-2). Root cause identified and fixed: these FOVs had no real organoids, and
`3.organoid_cell_relationship.ipynb` was inserting a zero-filled fallback row
(including `object_id=0`) when the organoid DataFrame was empty. The fix was to remove
the fallback entirely — parquet preserves schema with zero rows, so DuckDB
`union_by_name` handles empty files correctly. The fake row has been replaced with a
`pass` statement. Cells in organoid-empty FOVs are correctly flagged with
`Metadata_cqc_missing_parent_organoid=True` in 7b regardless of this change.

- [ ] **Re-run the full pipeline from Stage 1** (`1.merge_feature_parquets.py` through
  `8.normalization.py`) for all patients to ensure handcrafted and SAMMed organoid
  profiles are derived from the same pipeline run and are in sync.
- [ ] **Verify ObjectID=0 fix on bandicoot**: confirm that DuckDB `union_by_name`
  handles empty organoid parquets correctly in practice and that the 3 ObjectID=0 rows
  no longer appear after re-run.
- [ ] **After re-run**: re-evaluate how many unmatched organoids remain and whether
  they correspond to organoids with failed handcrafted measurements (see Issue 2 for
  the surface area bug connection). If a consistent gap remains, flag those organoids
  with `Metadata_cqc_no_handcrafted_match = True`.

---

## Issue 7 — Large metadata column footprint in handcrafted profiles at Stage 5

**Resolution: non-issue.** Feature selection operates only on feature columns —
`Metadata_*` columns are never touched. `Metadata_Location_*` coordinate columns
persist through all stages by design and are excluded from modeling by virtue of
being metadata. No fix needed, no remaining actions.

---

## Issue 8 — Texture features have astronomical values (array reset bug in texture_utils)

**Root cause:** Critical bug in
`utils/src/image_analysis_3D/featurization_utils/texture_utils.py` (line 154) and
identically in `ZedProfiler/src/zedprofiler/featurization/texture.py` (line 170).

`features = numpy.empty((n_directions, 13, max(labels)))` is called **inside** the
per-object loop. Every iteration creates a brand new uninitialized array, discarding
all previously computed objects' values. Only the final object's Haralick values
survive into the output loop. Every other object receives uninitialized memory from
`numpy.empty`, which can contain any value — typically very large floats or inf.

The fix is to move `features = numpy.empty(...)` to **before** the loop, so the
array is allocated once and all objects accumulate into it.

**Why tests pass:** The existing tests (`test_featurization_utils.py`) only check
structural properties — that output dict keys exist and lengths are consistent.
No test verifies that a specific object's texture values match what Haralick
actually computed for that object's pixels. The single-object test
(`test_measure_3d_texture_uniform_full_object`) incidentally works correctly
because with one object there is only one loop iteration and no overwrite.
Tests have been updated to catch this regression (see `test_featurization_utils.py`).

**Impact:** All objects except the last in each field of view receive garbage
texture values. 1,686 / 2,028 Texture columns (83%) show overflow std at Stage 5;
1,264 survive feature selection into Stage 6. The overflow is not caused by
normalization (MAD/epsilon) — the raw values are corrupted before normalization.

**Fix implemented:** *(pending — see decision point below)*

**Decision point — drop or recalculate?**

- [ ] **Option A — Recalculate**: Move `features = numpy.empty(...)` to before the
  per-object loop in both `texture_utils.py` (NF1) and ZedProfiler's `texture.py`,
  re-run texture featurization for all patients, and re-propagate through stages 4–8.
  This is the correct long-term fix and recovers all texture features.

- [ ] **Option B — Drop**: Add all Texture features to the feature selection blocklist
  in `9.feature_selection.py` until the upstream bug is fixed. Safe short-term option
  if recalculation is not immediately feasible.

**Remaining actions — upstream bugs:**

- [x] Fix array reset bug in `utils/texture_utils.py` (NF1 pipeline) — moved
  `features = numpy.empty(...)` outside the per-object loop. Regression test
  added and passing.
- [x] Fix same bug in `ZedProfiler/src/zedprofiler/featurization/texture.py` —
  same one-line fix applied. Regression test added and passing.
- [ ] After fix: re-run texture featurization for all patients and re-propagate
  through stages 4–8.

---

## Issue 9 — `MinIntensityEdge` is always zero (boundary masking bug in intensity_utils)

**Root cause:** Bug in
`utils/src/image_analysis_3D/featurization_utils/intensity_utils.py`.

`skimage.segmentation.find_boundaries` defaults to `mode='thick'`, which returns
both the inner boundary (voxels inside the object touching background) and the
outer boundary (background voxels touching the object). Since `selected_image_object`
has all non-cell pixels zeroed before the outline is computed, the outer boundary
voxels always have intensity 0. `numpy.min(selected_image_object[mask_outlines > 0])`
therefore always picks up one of these zeros, producing a constant 0 for all cells.

`MaxIntensityEdge`, `StdIntensityEdge`, and `IntegratedIntensityEdge` are unaffected
because max/std/sum are not pulled down to zero by the outer boundary pixels.

**Impact:** 12 columns are constant zero at Stage 5 — `MinIntensityEdge` across
all 3 compartments (Cell, Cytoplasm, Nuclei) × 4 channels (AGP, DNA, ER, Mito).
All 12 are correctly removed by `variance_threshold` at Stage 6. No data survives
to downstream stages.

**Fix implemented:** Changed `find_boundaries(mask[z])` to
`find_boundaries(mask[z], mode='inner')` in `get_outline` in
`utils/src/image_analysis_3D/featurization_utils/intensity_utils.py`.
`mode='inner'` returns only pixels inside the object boundary, so outer
(background) pixels are never included in the edge mask.

**Regression test added:** `test_min_intensity_edge_not_zero_for_bright_cell`
in `utils/tests/test_featurization_utils.py` — a uniform-intensity cell must
have `MinIntensityEdge` equal to its intensity, not 0. Now passing.

**Remaining actions:**

- [ ] Apply same fix to `ZedProfiler/src/zedprofiler/featurization/intensity.py`
  (identical bug at line 34 of `get_outline`).
- [ ] Re-run intensity featurization for all patients and re-propagate through
  stages 4–8 to recover `MinIntensityEdge` as a usable feature.

---

## Issue 10 — Feature column prefix is `Cell_` not `Cells_` (naming convention)

**No fix needed in the pipeline** — the normalization and feature selection scripts
correctly identify features by metadata exclusion, not prefix matching.

**Remaining actions:**

- [ ] **Document the `Cell_` naming convention** in the folder README so that
  downstream users calling pycytominer's `infer_cp_features()` know to pass
  `compartments=["Cell", "Nuclei", "Cytoplasm", "Organoid", "Nucleocentric"]`
  instead of the default.

---

## Issue 11 — CQC flags cannot be propagated to deep learning profiles without caveats

**Resolution: fully addressed by Issue 6 fix.**

`7c.propagate_cqc_to_dl_profiles.ipynb` implements CQC propagation to all four DL
profile types via a validated left-join on `(Metadata_Experiment_WellFOV,
Metadata_Object_ObjectID)`. SC and nucleocentric joins are 1:1 and complete.
Organoid join is partial — see Issue 6 decision point for the 38 unmatched
`sammed_organoid` rows.

The key caveat is documented in the notebook: **NaN CQC ≠ "passed QC"** for the
38 unmatched organoids. Downstream users must treat NaN CQC flags as unknown, not
as a passing grade.

**No further remaining actions specific to Issue 11.** See Issue 6 for the
unmatched organoid decision point.
