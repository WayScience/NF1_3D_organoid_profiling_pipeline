#!/usr/bin/env python3
"""Build IBP stage 4 step-3 inputs directly from a ZedProfiler warehouse.

`4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`
expects three parquet files per well-FOV under
`data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/`:
`sc_profiles_{well_fov}.parquet`, `organoid_profiles_{well_fov}.parquet`,
`nucleocentric_profiles_{well_fov}.parquet`. Those are normally produced by
IBP steps 00/0a/1/2, which convert old CellProfiler-style per-feature
parquet files into a merged per-well_fov DuckDB and then merge that into the
three files above.

A ZedProfiler warehouse (3a.nextflow_pilot / 3b.nextflow_production) already
holds the same information in a different shape: one parquet per compartment
per image set, joined via warehouse.duckdb's `joined.images_nuclei_cell_cytoplasm`
(inner join across Nuclei/Cell/Cytoplasm on Metadata_Object_ObjectID -- the
same object-intersection step 2 computes) and `profiles.organoid_profiles`.
This script reads those views for one image set and writes the three files
step 3 expects, bridging two real differences instead of touching step 3's
own code:

- Column names: step 3 finds centroid/bbox columns by substring-matching
  "area" (CellProfiler-era `*_AreaSizeShape_*` naming). ZedProfiler's own
  convention produces `*_VolumeSizeShape_*` instead, which the substring
  match misses entirely. Every VolumeSizeShape column is renamed to its
  AreaSizeShape equivalent here -- but only in the scratch files written to
  `0.converted_profiles/` that feed step 3's own unmodified matching logic.
  ZedProfiler's VolumeSizeShape naming is preferred everywhere else,
  including the *_related.parquet files this pilot actually persists into
  warehouse/ibp/ -- run_ibp_pilot.py renames those columns back to
  VolumeSizeShape immediately after step 3 produces them, before they're
  written anywhere durable. This file's rename is a one-way, step-3-local
  compatibility shim, not a change to this pilot's preferred naming.
- Identifiers: step 3 expects `object_id` (ours: Metadata_Object_ObjectID)
  and `image_set` (ours: implicit, derived from patient/well_fov here).

ZedProfiler does not produce deep-learning Nucleocentric features. Step 3
(unmodified) still hard-requires a nucleocentric_profiles_{well_fov}.parquet
to exist -- it strictly resolves that path and crashes immediately if it's
missing -- so an empty (0 rows, `object_id`/`image_set` columns only)
placeholder is written *only if nothing is already there*, never
overwriting real Nucleocentric data from an actual run of IBP steps
00/0a/1/2 for the same well_fov/subparent_name. Its downstream
`*_related.parquet` output is not persisted into warehouse/ibp/ -- see
run_ibp_pilot.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd


def rename_volumesizeshape_to_areasizeshape(df: pd.DataFrame) -> pd.DataFrame:
    """Rename every `*VolumeSizeShape*` column to its `*AreaSizeShape*` form.

    Step-3-input-only compatibility shim: lets step 3's own unmodified
    "area" substring matching find ZedProfiler's centroid/bbox columns.
    ZedProfiler's VolumeSizeShape naming is preferred for anything this
    pilot actually persists -- see run_ibp_pilot.py's
    rename_areasizeshape_to_volumesizeshape(), which undoes this on step 3's
    output before it's written into warehouse/ibp/.
    """
    return df.rename(
        columns={
            column: column.replace("VolumeSizeShape", "AreaSizeShape")
            for column in df.columns
            if "VolumeSizeShape" in column
        }
    )


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
            cell_exclude = sorted(
                shared
                | {"Metadata_Object_ObjectID"}
                | {
                    column
                    for column in cell_columns
                    if column.startswith(_PER_COMPARTMENT_METADATA_PREFIXES)
                }
            )
            cytoplasm_exclude = sorted(
                shared
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

    sc_df = rename_volumesizeshape_to_areasizeshape(sc_df)
    organoid_df = rename_volumesizeshape_to_areasizeshape(organoid_df)

    sc_df = sc_df.rename(columns={"Metadata_Object_ObjectID": "object_id"})
    organoid_df = organoid_df.rename(columns={"Metadata_Object_ObjectID": "object_id"})
    sc_df["image_set"] = args.well_fov
    organoid_df["image_set"] = args.well_fov

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

    # Step 3 (unmodified) does a strict path resolve on this file and
    # crashes immediately if it's missing, so something has to exist here
    # for step 3 to run at all -- ZedProfiler has no real Nucleocentric
    # (deep-learning) features to put in it, hence empty. Only write it if
    # nothing is there yet: if this same well_fov/subparent_name has real
    # Nucleocentric data from an actual run of IBP steps 00/0a/1/2, this
    # placeholder must never clobber it.
    nucleocentric_path = outdir / f"nucleocentric_profiles_{args.well_fov}.parquet"
    if not nucleocentric_path.exists():
        pd.DataFrame(columns=["object_id", "image_set"]).to_parquet(
            nucleocentric_path, index=False
        )

    print(
        "NF1_IBP_PILOT_INPUTS_OK "
        f"patient={args.patient} well_fov={args.well_fov} "
        f"sc_rows={sc_df.shape[0]} sc_cols={sc_df.shape[1]} "
        f"organoid_rows={organoid_df.shape[0]} outdir={outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
