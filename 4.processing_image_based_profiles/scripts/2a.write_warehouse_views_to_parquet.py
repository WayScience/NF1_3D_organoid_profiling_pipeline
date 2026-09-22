#!/usr/bin/env python
# coding: utf-8

# This notebook should take the converted (merged) warehouse view and write out the image-based profiles to the correct parquet file.
# This enables the downstream analysis notebooks/files to be run without needing to refactor the import code.
# The production run takes the place of `4.processing_image_based_profiles/notebooks/1.merge_feature_parquets.ipynb` and `4.processing_image_based_profiles/notebooks/2.merge_sc.ipynb`.
# After converting, this notebook's written files will feed into the `4.processing_image_based_profiles/notebooks/3.organoid_cell_relationship.ipynb` module.

# # Write warehouse views to parquet (single image set)
#
# ## Purpose
# Convert one image set (`patient`, `well_fov`) from a ZEDProfiler warehouse
# into the per-well_fov parquet files that step 3
# (`3.organoid_cell_relationship`) expects, in place of steps 1 and 2.
#
# 1. Read the image set's Nuclei/Cell/Cytoplasm/Organoid compartment parquet
#    files directly by path from `warehouse_dir/profiles/`.
# 2. Merge Nuclei/Cell/Cytoplasm into one single-cell table on
#    `Metadata_Imaging_ImageID` + `Metadata_Object_ObjectID`, dropping
#    duplicated shared and per-compartment metadata.
# 3. Write `sc_profiles_<well_fov>.parquet` and
#    `organoid_profiles_<well_fov>.parquet`. No nucleocentric file is written;
#    step 3 treats it as optional.
#
# This runs one image set per invocation (called per well_fov from
# `merge_features_local.sh`). For full-index processing (concurrency,
# resumability, run record), use
# `4b.image_based_profiles_production/scripts/run_ibp_production.py`.
#
# ## Inputs
# - `warehouse_dir` -- a ZEDProfiler warehouse directory (contains `profiles/`).
# - `patient`, `well_fov` -- the image set to convert.
# - `image_based_profiles_subparent_name` -- output subparent directory name.
#
# ## Outputs
# Written to
# `<NF1_organoid_data>/data/<patient>/<image_based_profiles_subparent_name>/0.converted_profiles/<well_fov>/`:
# - `sc_profiles_<well_fov>.parquet`
# - `organoid_profiles_<well_fov>.parquet`
#

# In[1]:


import argparse
import re
from pathlib import Path

import pandas as pd
from nas_path_package.core import init_notebook, nas_path_check

root_dir, in_notebook = init_notebook()

PROD_ROOT = root_dir / "4b.image_based_profiles_production"
SCRIPTS_DIR = PROD_ROOT / "scripts"
image_base_dir = nas_path_check(root_dir, "bandicoot")
image_base_dir = image_base_dir / "NF1_organoid_data"
IBP_SUBPARENT_NAME = "image_based_profiles"


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser(
        description="Run IBP stage 4 step 3 against a ZEDProfiler warehouse."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse_dir", required=True, type=Path)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--well_fov", required=True, help="e.g. 'B10-1'")
    parser.add_argument("--image_based_profiles_subparent_name", required=True)
    args = parser.parse_args()
    warehouse_dir = args.warehouse_dir
    patient = args.patient
    well_fov = args.well_fov
    image_based_profiles_subparent_name = args.image_based_profiles_subparent_name

else:
    # Bandicoot mirror of the production ZEDProfiler warehouse (same
    # relative layout as koala's own
    # nf1-3d-production-workflow-db/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse,
    # per sync_warehouse_to_bandicoot.sh) -- used when koala/Alpine isn't
    # mounted locally. Swap for a direct koala mount
    # (~/mnt/alpine/active/koala/...) if that's mounted instead.
    warehouse_dir = (
        image_base_dir
        / "data"
        / "results-zedprofiler-0.1.4"
        / "nf1-production-full-run-2"
        / "warehouse"
    ).resolve(strict=True)
    patient = "NF0014_T1"
    well_fov = "C11-2"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


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


def parse_well_fov(well_fov: str) -> tuple[str, str]:
    """Same convention as build_manifest.py's own parse_well_fov():
    match a leading "letter + 1-2 digits" well name, treating anything
    after an optional separator as the field, defaulting to field "1" if
    the whole string doesn't match that shape.

    Reimplemented here rather than imported, matching this repo's
    copy-not-share convention between sibling folders. Not just naive
    `well_fov.rsplit("-", 1)`: at least one real well in this dataset is
    itself named with a hyphen (`NF0018_T6`'s well "E-3", landed as
    `NF0018_T6__NF0018_T6__E-3__F1.parquet`), which rsplit misparses as
    well "E" field "3" -- a combination that doesn't exist -- while this
    regex correctly falls through to (well_fov, "1") for that one case,
    since "E-3" has no digit directly after its leading letter. Verified
    against every row of manifest/image_sets_index.csv: this parses
    identically to rsplit("-", 1) for all 4,135 other entries, and only
    differs (correctly) for that one.
    """
    match = re.match(r"^([A-Ha-h][0-9]{1,2})[-_]?(.+)$", well_fov)
    if not match:
        return well_fov, "1"
    return match.group(1).upper(), str(match.group(2))


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


# In[4]:


well, field = parse_well_fov(well_fov)
sc_df, organoid_df = load_from_warehouse(
    warehouse_dir.resolve(strict=True), patient, well, field
)

outdir = (
    image_base_dir
    / "data"
    / patient
    / image_based_profiles_subparent_name
    / "0.converted_profiles"
    / well_fov
)
outdir.mkdir(parents=True, exist_ok=True)

sc_df.to_parquet(outdir / f"sc_profiles_{well_fov}.parquet", index=False)
organoid_df.to_parquet(outdir / f"organoid_profiles_{well_fov}.parquet", index=False)
# No nucleocentric_profiles_{well_fov}.parquet is written here at all --
# ZEDProfiler has no real Nucleocentric (deep-learning) data to put in
# it, and step 3 treats that input as optional (falls back to an empty
# dataframe if the file doesn't exist).

print(
    "NF1_IBP_PRODUCTION_INPUTS_OK "
    f"patient={patient} well_fov={well_fov} "
    f"sc_rows={sc_df.shape[0]} sc_cols={sc_df.shape[1]} "
    f"organoid_rows={organoid_df.shape[0]} outdir={outdir}"
)
