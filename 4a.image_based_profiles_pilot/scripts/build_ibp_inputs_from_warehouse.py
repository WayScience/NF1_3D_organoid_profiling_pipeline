#!/usr/bin/env python3
"""Build IBP stage 4 step-3 inputs directly from a ZEDProfiler warehouse.

`4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`
expects two required parquet files per well-FOV under
`data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/`:
`sc_profiles_{well_fov}.parquet`, `organoid_profiles_{well_fov}.parquet`
(a third, `nucleocentric_profiles_{well_fov}.parquet`, is optional -- see
below). Those are normally produced by IBP steps 00/0a/1/2, which convert
old CellProfiler-style per-feature parquet files into a merged per-well_fov
DuckDB and then merge that into the files above.

A ZEDProfiler warehouse (3a.nextflow_pilot / 3b.nextflow_production) already
holds the same information in a different shape: one parquet per compartment
per image set, joined via warehouse.duckdb's `joined.images_nuclei_cell_cytoplasm`
(inner join across Nuclei/Cell/Cytoplasm on Metadata_Object_ObjectID -- the
same object-intersection step 2 computes) and `profiles.organoid_profiles`.
This script reads those views for one image set and writes the files step 3
expects. Real differences between ZEDProfiler's native shape and what step 3
was originally written for are bridged inside step 3 itself
(4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py,
same changes mirrored in the paired notebook), not here -- this script does
no column renaming at all, and writes ZEDProfiler's own column names through
unchanged:

- Column names: step 3 finds centroid/bbox columns by substring-matching
  "area" (CellProfiler-era `*_AreaSizeShape_*` naming). ZEDProfiler's own
  convention produces `*_VolumeSizeShape_*` instead, which the substring
  match originally missed entirely. Step 3's own matching was widened to
  accept "volumesizeshape" directly.
- Identifiers: step 3 originally expected `object_id` and `image_set`
  columns (the older CellProfiler-era pipeline's own naming). Step 3 now
  accepts ZEDProfiler's native `Metadata_Object_ObjectID` directly
  (normalized to `object_id` internally, right after loading) and derives
  `image_set` itself from its own `well_fov` argument, so neither needs to
  be supplied here.
- Nucleocentric data: ZEDProfiler does not produce deep-learning
  Nucleocentric features. Step 3 originally hard-required a
  nucleocentric_profiles_{well_fov}.parquet input to exist, so this script
  used to write an empty placeholder to satisfy that. Step 3 was changed to
  treat that input as optional (falls back to an empty dataframe if the
  file doesn't exist), so this script no longer touches any nucleocentric
  file at all -- neither reading nor writing one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd


# Per the project's metadata naming convention
# (docs/RFC-2119-Feature-Naming-Convention.md section 2.2), a metadata
# column's category -- the `<featurecategory>` in
# `Metadata_<featurecategory>_<featurename>` -- says whether it describes
# the sample/experiment/imaging session as a whole (shared, identical
# regardless of which compartment table it's read from: a patient, plate,
# well, field, or image doesn't change because you're looking at Nuclei
# instead of Cell) or the specific compartment/segmentation that produced
# this row (Metadata_Compartment itself, and the Segmentation_* fields
# describing how *that* compartment was segmented -- genuinely different
# per compartment). Matched by category prefix rather than a hardcoded list
# of exact column names, so a newly added Biology/Experiment/Imaging field
# is picked up automatically instead of silently leaking through
# unexcluded and colliding with Nuclei's own copy of the same name.
_SHARED_METADATA_PREFIXES = (
    "Metadata_Biology_",
    "Metadata_Experiment_",
    "Metadata_Imaging_",
)
_PER_COMPARTMENT_METADATA_PREFIXES = (
    "Metadata_Compartment",
    "Metadata_Segmentation_",
)


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Actual column names of a table, queried live rather than assumed --
    see build_duckdb_views.py's own _table_columns() for the same
    reasoning: schemas can vary."""
    return list(con.execute(f"SELECT * FROM {table} LIMIT 0").df().columns)


def _exclude_clause(columns: list[str]) -> str:
    """DuckDB's `* EXCLUDE (...)` needs a non-empty, literal column list --
    build one from whichever of `columns` actually needs excluding, or
    return "" (meaning: no EXCLUDE at all) if none do."""
    return f" EXCLUDE ({', '.join(columns)})" if columns else ""


def load_from_warehouse(
    warehouse_dir: Path, patient: str, well: str, field: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query the warehouse's per-compartment tables for one image set.

    Deliberately does NOT use warehouse.duckdb's joined.images_nuclei_cell_cytoplasm
    view here: that view joins through images.image_assets (documented in
    build_duckdb_views.py as "each object row is repeated once per image
    asset"), which is exactly the wrong shape for a single-cell profile
    table -- it fans out one row per object into one row per (object,
    channel/mask asset). Joining Nuclei/Cell/Cytoplasm directly, without
    the image_assets join, gives the same one-row-per-object shape step 2
    (the code being bypassed here) originally produced.

    DuckDB resolves the views' relative parquet globs against the process's
    current working directory (see build_duckdb_views.py), so this
    temporarily chdirs into warehouse_dir for the query.
    """
    previous_cwd = Path.cwd()
    os.chdir(warehouse_dir)
    try:
        with duckdb.connect("warehouse.duckdb", read_only=True) as con:
            nuclei_columns = _table_columns(con, "profiles.nuclei_profiles")
            cell_columns = _table_columns(con, "profiles.cell_profiles")
            cytoplasm_columns = _table_columns(con, "profiles.cytoplasm_profiles")

            shared = {
                column
                for column in nuclei_columns
                if column.startswith(_SHARED_METADATA_PREFIXES)
            }
            nuclei_exclude = [
                column
                for column in nuclei_columns
                if column.startswith(_PER_COMPARTMENT_METADATA_PREFIXES)
            ]
            # Intersect `shared` with each table's own columns before
            # excluding: EXCLUDE on a column name that doesn't exist in
            # that table is a hard DuckDB error, not a no-op (same
            # constraint noted in build_duckdb_views.py). All three tables
            # currently carry identical Biology/Experiment/Imaging fields
            # (same add_metadata() call in run_zedprofiler_image_set.py),
            # but this makes that an assumption this code tolerates being
            # wrong for, rather than one it silently requires.
            cell_exclude = sorted(
                (shared & set(cell_columns))
                | {"Metadata_Object_ObjectID"}
                | {
                    column
                    for column in cell_columns
                    if column.startswith(_PER_COMPARTMENT_METADATA_PREFIXES)
                }
            )
            cytoplasm_exclude = sorted(
                (shared & set(cytoplasm_columns))
                | {"Metadata_Object_ObjectID"}
                | {
                    column
                    for column in cytoplasm_columns
                    if column.startswith(_PER_COMPARTMENT_METADATA_PREFIXES)
                }
            )

            sc_df = con.execute(
                f"""
                SELECT
                    n.*{_exclude_clause(nuclei_exclude)},
                    c.*{_exclude_clause(cell_exclude)},
                    cy.*{_exclude_clause(cytoplasm_exclude)}
                FROM profiles.nuclei_profiles n
                JOIN profiles.cell_profiles c
                    ON c.Metadata_Imaging_ImageID = n.Metadata_Imaging_ImageID
                    AND c.Metadata_Object_ObjectID = n.Metadata_Object_ObjectID
                JOIN profiles.cytoplasm_profiles cy
                    ON cy.Metadata_Imaging_ImageID = n.Metadata_Imaging_ImageID
                    AND cy.Metadata_Object_ObjectID = n.Metadata_Object_ObjectID
                WHERE n.Metadata_Biology_PatientTumor = ?
                  AND n.Metadata_Experiment_WellID = ?
                  AND n.Metadata_Imaging_FieldID = ?
                """,
                [patient, well, field],
            ).df()
            organoid_df = con.execute(
                """
                SELECT * FROM profiles.organoid_profiles
                WHERE Metadata_Biology_PatientTumor = ?
                  AND Metadata_Experiment_WellID = ?
                  AND Metadata_Imaging_FieldID = ?
                """,
                [patient, well, field],
            ).df()
    finally:
        os.chdir(previous_cwd)
    return sc_df, organoid_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--well-fov", required=True, help="e.g. 'B10-1'")
    parser.add_argument("--image-based-profiles-subparent-name", required=True)
    args = parser.parse_args()

    well, field = args.well_fov.rsplit("-", 1)
    sc_df, organoid_df = load_from_warehouse(
        args.warehouse_dir.resolve(strict=True), args.patient, well, field
    )
    if sc_df.empty:
        raise SystemExit(
            f"No data found in {args.warehouse_dir} for "
            f"{args.patient}/{args.well_fov} -- check the warehouse actually "
            "contains this image set (profiles/*_profiles/<image_id>.parquet)."
        )

    outdir = (
        args.repo_root
        / "data"
        / args.patient
        / args.image_based_profiles_subparent_name
        / "0.converted_profiles"
        / args.well_fov
    )
    outdir.mkdir(parents=True, exist_ok=True)

    sc_df.to_parquet(outdir / f"sc_profiles_{args.well_fov}.parquet", index=False)
    organoid_df.to_parquet(
        outdir / f"organoid_profiles_{args.well_fov}.parquet", index=False
    )
    # No nucleocentric_profiles_{well_fov}.parquet is written here at all --
    # ZEDProfiler has no real Nucleocentric (deep-learning) data to put in
    # it, and step 3 now treats that input as optional (falls back to an
    # empty dataframe if the file doesn't exist), so there's nothing for
    # this script to do here. A real Nucleocentric input from an actual run
    # of IBP steps 00/0a/1/2 for the same well_fov/subparent_name, if one
    # exists, is picked up by step 3 directly and is never touched by this
    # script either way.

    print(
        "NF1_IBP_PILOT_INPUTS_OK "
        f"patient={args.patient} well_fov={args.well_fov} "
        f"sc_rows={sc_df.shape[0]} sc_cols={sc_df.shape[1]} "
        f"organoid_rows={organoid_df.shape[0]} outdir={outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
