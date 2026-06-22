#!/usr/bin/env python
# coding: utf-8

# # 7c. Propagate CQC Flags to Deep-Learning Profiles
#
# ## Purpose
# Propagate cosmicqc (`Metadata_cqc_*`) flags from handcrafted profiles onto the
# corresponding deep-learning profiles, which have no native QC step.
#
# This is **step 7c of Stage 4 (image-based profiling)**. It runs once per patient
# and must follow `7b.single_cell_qc.ipynb` (which produces the CQC-flagged
# handcrafted profiles used as the source here).
#
# ## Inputs
#
# | File | Role |
# |---|---|
# | `4.qc_profiles/sc_flagged_outliers.parquet` | Source of CQC flags for SC cells |
# | `4.qc_profiles/organoid_flagged_outliers.parquet` | Source of CQC flags for organoids |
# | `3.annotated_profiles/sammed_sc_anno.parquet` | DL SC profile to annotate |
# | `3.annotated_profiles/sammed_organoid_anno.parquet` | DL organoid profile to annotate |
# | `3.annotated_profiles/nucleocentric_sammed_anno.parquet` | DL nucleocentric (SAMMed3D) |
# | `3.annotated_profiles/nucleocentric_morphem_anno.parquet` | DL nucleocentric (morphem) |
#
# ## Outputs
#
# Four CQC-annotated parquets in `4.qc_profiles/`:
#
# | File | Notes |
# |---|---|
# | `sammed_sc_flagged_outliers.parquet` | Full 1:1 join — all SC CQC flags propagated |
# | `sammed_organoid_flagged_outliers.parquet` | Partial join — 38 unmatched organoids retain NaN CQC (see Notes) |
# | `nucleocentric_sammed_flagged_outliers.parquet` | Full 1:1 join — SC CQC flags propagated |
# | `nucleocentric_morphem_flagged_outliers.parquet` | Full 1:1 join — SC CQC flags propagated |
#
# ## Join key
# `(Metadata_Experiment_WellFOV, Metadata_Object_ObjectID)` uniquely identifies
# each object within a patient. This key is present in all profiles.
#
# ## Notes on unmatched sammed_organoid rows
# 38 organoids exist in `sammed_organoid` but not in `organoid`. These organoids
# were detected and embedded by SAM-Med but were filtered out of the handcrafted
# organoid profile by CellProfiler QC (likely size or shape thresholds applied
# during segmentation post-processing). Because they have no corresponding row in
# the handcrafted profile, no CQC flags can be propagated to them. They are retained
# in the output with all CQC columns set to NaN — downstream users should be aware
# that NaN CQC ≠ "passed QC" for these 38 organoids.
#
# Resolving this mismatch requires upstream investigation:
# - Identify which CellProfiler QC step removes these 38 organoids
# - Determine whether equivalent QC criteria can be applied to SAM-Med outputs
#   directly (e.g., organoid volume/shape thresholds from the DL embeddings)

import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
profile_base_dir = root_dir


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# ## Paths

base = pathlib.Path(profile_base_dir) / "data" / patient / image_based_profiles_subparent_name
qc_dir = base / "4.qc_profiles"
anno_dir = base / "3.annotated_profiles"
qc_dir.mkdir(parents=True, exist_ok=True)

# Source CQC profiles (handcrafted, already flagged)
sc_cqc_path = (qc_dir / "sc_flagged_outliers.parquet").resolve(strict=True)
organoid_cqc_path = (qc_dir / "organoid_flagged_outliers.parquet").resolve(strict=True)

# DL profiles to annotate
sammed_sc_path = (anno_dir / "sammed_sc_anno.parquet").resolve(strict=True)
sammed_organoid_path = (anno_dir / "sammed_organoid_anno.parquet").resolve(strict=True)
nucleocentric_sammed_path = (anno_dir / "nucleocentric_sammed_anno.parquet").resolve(strict=True)
nucleocentric_morphem_path = (anno_dir / "nucleocentric_morphem_anno.parquet").resolve(strict=True)

# Outputs
sammed_sc_output_path = (qc_dir / "sammed_sc_flagged_outliers.parquet").resolve()
sammed_organoid_output_path = (qc_dir / "sammed_organoid_flagged_outliers.parquet").resolve()
nucleocentric_sammed_output_path = (qc_dir / "nucleocentric_sammed_flagged_outliers.parquet").resolve()
nucleocentric_morphem_output_path = (qc_dir / "nucleocentric_morphem_flagged_outliers.parquet").resolve()


# ## Load profiles

sc_cqc_df = pd.read_parquet(sc_cqc_path)
organoid_cqc_df = pd.read_parquet(organoid_cqc_path)

sammed_sc_df = pd.read_parquet(sammed_sc_path)
sammed_organoid_df = pd.read_parquet(sammed_organoid_path)
nucleocentric_sammed_df = pd.read_parquet(nucleocentric_sammed_path)
nucleocentric_morphem_df = pd.read_parquet(nucleocentric_morphem_path)

print(f"SC CQC source:          {sc_cqc_df.shape}")
print(f"Organoid CQC source:    {organoid_cqc_df.shape}")
print(f"SAMMed SC:              {sammed_sc_df.shape}")
print(f"SAMMed organoid:        {sammed_organoid_df.shape}")
print(f"Nucleocentric SAMMed:   {nucleocentric_sammed_df.shape}")
print(f"Nucleocentric morphem:  {nucleocentric_morphem_df.shape}")


# ## Extract CQC columns

JOIN_KEY = ["Metadata_Experiment_WellFOV", "Metadata_Object_ObjectID"]
CQC_PREFIX = "Metadata_cqc_"

sc_cqc_cols = [c for c in sc_cqc_df.columns if c.startswith(CQC_PREFIX)]
organoid_cqc_cols = [c for c in organoid_cqc_df.columns if c.startswith(CQC_PREFIX)]

print(f"\nSC CQC columns to propagate: {sc_cqc_cols}")
print(f"Organoid CQC columns to propagate: {organoid_cqc_cols}")

# Validate join key is present in all profiles
for name, df in [
    ("sc_cqc", sc_cqc_df),
    ("organoid_cqc", organoid_cqc_df),
    ("sammed_sc", sammed_sc_df),
    ("sammed_organoid", sammed_organoid_df),
    ("nucleocentric_sammed", nucleocentric_sammed_df),
    ("nucleocentric_morphem", nucleocentric_morphem_df),
]:
    for col in JOIN_KEY:
        assert col in df.columns, (
            f"Join key column '{col}' missing from {name}. "
            f"Cannot propagate CQC flags without a reliable join key."
        )

print("\nJoin key present in all profiles. ✓")


# ## Helper: propagate CQC flags via join with validation

def propagate_cqc(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cqc_cols: list[str],
    source_name: str,
    target_name: str,
    expect_full_match: bool = True,
) -> pd.DataFrame:
    """Join CQC flag columns from source onto target using the composite join key.

    Parameters
    ----------
    source_df : pd.DataFrame
        Handcrafted profile with Metadata_cqc_* columns.
    target_df : pd.DataFrame
        DL profile to annotate.
    cqc_cols : list[str]
        CQC column names to propagate.
    source_name : str
        Human-readable name for error messages.
    target_name : str
        Human-readable name for error messages.
    expect_full_match : bool
        If True, assert that every target row has a matching source row.
        Set to False for sammed_organoid where unmatched rows are expected.
    """
    source_key_df = source_df[JOIN_KEY + cqc_cols].copy()

    # Assert join key is unique in source (no duplicate object IDs)
    dupes = source_key_df.duplicated(subset=JOIN_KEY)
    assert not dupes.any(), (
        f"{source_name}: join key is not unique — {dupes.sum()} duplicate rows found. "
        f"CQC propagation requires a 1:1 key."
    )

    # Assert join key is unique in target
    dupes_target = target_df.duplicated(subset=JOIN_KEY)
    assert not dupes_target.any(), (
        f"{target_name}: join key is not unique — {dupes_target.sum()} duplicate rows found."
    )

    merged = target_df.merge(
        source_key_df,
        on=JOIN_KEY,
        how="left",
        validate="1:1",
    )

    # Check for rows in target with no matching source row
    unmatched = merged[cqc_cols[0]].isna().sum() if cqc_cols else 0
    n_target = len(target_df)

    if expect_full_match:
        assert unmatched == 0, (
            f"{target_name}: {unmatched}/{n_target} rows have no matching {source_name} row. "
            f"Expected a full 1:1 match. Check that both profiles were generated "
            f"from the same annotated input."
        )
        print(f"{target_name}: all {n_target} rows matched. ✓")
    else:
        matched = n_target - unmatched
        print(
            f"{target_name}: {matched}/{n_target} rows matched to {source_name}. "
            f"{unmatched} unmatched rows will have NaN CQC flags — see script header for details."
        )

    assert len(merged) == len(target_df), (
        f"Row count changed after merge: {len(target_df)} → {len(merged)}. "
        f"This indicates a join key issue (e.g. many:1 match). "
        f"Check for duplicates in {source_name}."
    )

    return merged


# ## Propagate SC CQC flags to DL SC profiles

print("\n=== SC-level CQC propagation ===")

sammed_sc_flagged = propagate_cqc(
    source_df=sc_cqc_df,
    target_df=sammed_sc_df,
    cqc_cols=sc_cqc_cols,
    source_name="sc_cqc",
    target_name="sammed_sc",
    expect_full_match=True,
)

nucleocentric_sammed_flagged = propagate_cqc(
    source_df=sc_cqc_df,
    target_df=nucleocentric_sammed_df,
    cqc_cols=sc_cqc_cols,
    source_name="sc_cqc",
    target_name="nucleocentric_sammed",
    expect_full_match=True,
)

nucleocentric_morphem_flagged = propagate_cqc(
    source_df=sc_cqc_df,
    target_df=nucleocentric_morphem_df,
    cqc_cols=sc_cqc_cols,
    source_name="sc_cqc",
    target_name="nucleocentric_morphem",
    expect_full_match=True,
)


# ## Propagate organoid CQC flags to sammed_organoid
#
# 38 sammed_organoid rows have no matching handcrafted organoid — these organoids
# were filtered by CellProfiler QC upstream but are retained in the SAM-Med
# embeddings. They will have NaN CQC flags in the output.

print("\n=== Organoid-level CQC propagation ===")

sammed_organoid_flagged = propagate_cqc(
    source_df=organoid_cqc_df,
    target_df=sammed_organoid_df,
    cqc_cols=organoid_cqc_cols,
    source_name="organoid_cqc",
    target_name="sammed_organoid",
    expect_full_match=False,
)

# Report the unmatched organoids so they can be investigated upstream
unmatched_mask = sammed_organoid_flagged[organoid_cqc_cols[0]].isna()
if unmatched_mask.any():
    print("\nUnmatched sammed_organoid rows (no handcrafted counterpart):")
    print(
        sammed_organoid_flagged.loc[
            unmatched_mask, ["Metadata_Experiment_WellFOV", "Metadata_Object_ObjectID"]
        ].to_string(index=False)
    )


# ## Write outputs

sammed_sc_flagged.to_parquet(sammed_sc_output_path, index=False)
sammed_organoid_flagged.to_parquet(sammed_organoid_output_path, index=False)
nucleocentric_sammed_flagged.to_parquet(nucleocentric_sammed_output_path, index=False)
nucleocentric_morphem_flagged.to_parquet(nucleocentric_morphem_output_path, index=False)

print(f"\nOutputs written:")
print(f"  {sammed_sc_output_path}")
print(f"  {sammed_organoid_output_path}")
print(f"  {nucleocentric_sammed_output_path}")
print(f"  {nucleocentric_morphem_output_path}")
