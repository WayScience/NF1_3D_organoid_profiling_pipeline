#!/usr/bin/env python3
"""Build IBP stage 4 step-3 inputs directly from a ZEDProfiler warehouse.

`4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`
expects two required parquet files per well-FOV under
`data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/`:
`sc_profiles_{well_fov}.parquet`, `organoid_profiles_{well_fov}.parquet`
(a third, `nucleocentric_profiles_{well_fov}.parquet`, is optional and not
produced here -- ZEDProfiler has no Nucleocentric data, and step 3 treats
that input as optional). Those two files are normally produced by IBP
steps 00/0a/1/2, which convert old CellProfiler-style per-feature parquet
files into a merged per-well_fov DuckDB and then merge that into the files
above.

This is the production-scale sibling of
`4a.image_based_profiles_pilot/scripts/build_ibp_inputs_from_warehouse.py`
-- same code, adapted for one real difference at 4,000+ image-set scale:
the pilot's version queried `warehouse.duckdb`'s `profiles.*` views (each
one `read_parquet(glob)` over *every* file in that table) filtered down to
one image set per call. At pilot scale (2 files/table) that's free; at
production scale (4,134 files/table) a single such query was measured at
35GB RAM and still not finished after 9+ minutes -- a query-planning cost
problem, not a data-volume one. This script instead reads each image
set's own parquet files directly by their known path (confirmed at 0.8s
per image set for all 4 compartment files combined), which also removes
the need for any patient/well/field SQL filtering at all: one file already
*is* one image set. No `warehouse.duckdb` connection is opened by this
script at all.

`image_id()` below reproduces the exact naming convention
`3b.nextflow_production/scripts/build_manifest.py`'s own `image_id()`
already uses for this dataset (plate == patient): confirmed against real
filenames in the warehouse, e.g. `NF0014_T1__NF0014_T1__C10__F1.parquet`.
Reimplemented here rather than imported, matching this repo's own
copy-not-share convention between pilot/production sibling folders (see
`3b.nextflow_production/README.md`'s own framing of its relationship to
`3a.nextflow_pilot`).

The metadata dedup logic (deciding which of Nuclei/Cell/Cytoplasm's copy
of a shared or per-compartment metadata column to keep when merging them)
is unchanged from the pilot, translated from SQL `EXCLUDE` clauses to
pandas `.drop()` -- see `_SHARED_METADATA_PREFIXES`/
`_PER_COMPARTMENT_METADATA_PREFIXES` below, identical to the pilot's own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

# Both are join keys for the Nuclei/Cell/Cytoplasm merge below, so they're
# kept out of each table's own drop list even where the dedup logic above
# would otherwise exclude them (Metadata_Imaging_ImageID falls under the
# shared-metadata prefix; Metadata_Object_ObjectID is excluded from
# Cell/Cytoplasm explicitly, same as the pilot's own SQL EXCLUDE) --
# pandas' merge(on=...) needs a column present on both sides to join on
# it, then naturally collapses the two sides to one copy, equivalent to
# the SQL joining on columns excluded from its own SELECT list.
_JOIN_KEYS = ["Metadata_Imaging_ImageID", "Metadata_Object_ObjectID"]


def image_id(patient: str, well: str, field: str) -> str:
    """Same convention as build_manifest.py's own image_id(patient, plate,
    well, field), with plate == patient for this dataset."""
    return "__".join([patient, patient, well, f"F{field}"])


def load_from_warehouse(
    warehouse_dir: Path, patient: str, well: str, field: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one image set's compartment parquet files directly by path and
    merge Nuclei/Cell/Cytoplasm into one single-cell profile table, the
    same shape step 2 (the code being bypassed here) originally produced.
    """
    iid = image_id(patient, well, field)
    profiles_dir = warehouse_dir / "profiles"
    nuclei = pd.read_parquet(profiles_dir / "nuclei_profiles" / f"{iid}.parquet")
    cell = pd.read_parquet(profiles_dir / "cell_profiles" / f"{iid}.parquet")
    cytoplasm = pd.read_parquet(profiles_dir / "cytoplasm_profiles" / f"{iid}.parquet")
    organoid_df = pd.read_parquet(profiles_dir / "organoid_profiles" / f"{iid}.parquet")

    shared = {
        column
        for column in nuclei.columns
        if column.startswith(_SHARED_METADATA_PREFIXES)
    }
    nuclei_kept = nuclei.drop(
        columns=[
            column
            for column in nuclei.columns
            if column.startswith(_PER_COMPARTMENT_METADATA_PREFIXES)
        ]
    )
    cell_exclude = (
        (shared & set(cell.columns))
        | {"Metadata_Object_ObjectID"}
        | {
            column
            for column in cell.columns
            if column.startswith(_PER_COMPARTMENT_METADATA_PREFIXES)
        }
    )
    cytoplasm_exclude = (
        (shared & set(cytoplasm.columns))
        | {"Metadata_Object_ObjectID"}
        | {
            column
            for column in cytoplasm.columns
            if column.startswith(_PER_COMPARTMENT_METADATA_PREFIXES)
        }
    )
    cell_kept = cell.drop(
        columns=[column for column in cell_exclude if column not in _JOIN_KEYS]
    )
    cytoplasm_kept = cytoplasm.drop(
        columns=[column for column in cytoplasm_exclude if column not in _JOIN_KEYS]
    )

    sc_df = nuclei_kept.merge(cell_kept, on=_JOIN_KEYS, how="inner")
    sc_df = sc_df.merge(cytoplasm_kept, on=_JOIN_KEYS, how="inner")
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
    # An empty sc_df is not an error: some image sets have real, detected
    # Nuclei but zero Cell/Cytoplasm objects (segmentation genuinely found
    # nothing there), so the inner merge above legitimately produces zero
    # rows even though organoid_df can still hold real organoids. Step 3
    # (unmodified) already has explicit, tested fallback handling for both
    # an empty sc_profile_df and an empty organoid_profile_df -- see its
    # own "Empty dataframe fallbacks" section -- so this is passed through
    # rather than treated as a hard failure, matching the same "let step 3
    # handle it" precedent already used for the nucleocentric input. A
    # genuinely missing image set (no files at all) still fails loudly via
    # FileNotFoundError inside load_from_warehouse() above, before this
    # point is ever reached.

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
    # it, and step 3 treats that input as optional (falls back to an empty
    # dataframe if the file doesn't exist).

    print(
        "NF1_IBP_PRODUCTION_INPUTS_OK "
        f"patient={args.patient} well_fov={args.well_fov} "
        f"sc_rows={sc_df.shape[0]} sc_cols={sc_df.shape[1]} "
        f"organoid_rows={organoid_df.shape[0]} outdir={outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
