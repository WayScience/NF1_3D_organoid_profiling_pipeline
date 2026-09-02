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
  AreaSizeShape equivalent here.
- Identifiers: step 3 expects `object_id` (ours: Metadata_Object_ObjectID)
  and `image_set` (ours: implicit, derived from patient/well_fov here).

ZedProfiler does not produce deep-learning Nucleocentric features, so the
nucleocentric output is written empty (0 rows, `object_id`/`image_set`
columns only) -- step 3 already has empty-dataframe handling for exactly
this case.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd


def rename_volumesizeshape_to_areasizeshape(df: pd.DataFrame) -> pd.DataFrame:
    """Rename every `*VolumeSizeShape*` column to its `*AreaSizeShape*` form."""
    return df.rename(
        columns={
            column: column.replace("VolumeSizeShape", "AreaSizeShape")
            for column in df.columns
            if "VolumeSizeShape" in column
        }
    )


def load_from_warehouse(
    warehouse_dir: Path, patient: str, well: str, field: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query the warehouse's joined views for one image set.

    DuckDB resolves the views' relative parquet globs against the process's
    current working directory (see build_duckdb_views.py), so this
    temporarily chdirs into warehouse_dir for the query.
    """
    previous_cwd = Path.cwd()
    os.chdir(warehouse_dir)
    try:
        with duckdb.connect("warehouse.duckdb", read_only=True) as con:
            sc_df = con.execute(
                """
                SELECT * FROM joined.images_nuclei_cell_cytoplasm
                WHERE Metadata_Biology_PatientTumor = ?
                  AND Metadata_Experiment_WellID = ?
                  AND Metadata_Imaging_FieldID = ?
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

    nucleocentric_df = pd.DataFrame(columns=["object_id", "image_set"])

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
    nucleocentric_df.to_parquet(
        outdir / f"nucleocentric_profiles_{args.well_fov}.parquet", index=False
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
