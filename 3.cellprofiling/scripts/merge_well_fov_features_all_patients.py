#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Merge & Validate Well-FOV Feature Parquets — All Patients
#
# ## Purpose
# Same QC as `merge_well_fov_features_single_patient.ipynb`, but across **every patient**
# listed in `data/patient_IDs.txt` instead of one hardcoded patient. For every well-FOV of
# every patient, this attempts to merge the per-compartment x channel x feature-type
# parquet files into a single feature space per compartment, and flags every place that
# could silently go wrong: missing/unreadable files, missing merge keys, duplicate
# `object_id`s, merge blow-ups, an Organoid merge that fails outright due to an empty
# input file, object-ID misalignment across single-cell compartments, source segmentation
# masks that no longer match the extracted features, and profiles containing any
# `object_id` greater than 255.
#
# ## Why this is structured differently from the single-patient version
# Across all 13 patients this dataset has roughly **680,000** feature parquet files, some
# with 700+ feature columns (e.g. SAMMed3D). Reading every column of every file the way
# the single-patient notebook does would take hours. This notebook only needs `object_id`
# and `image_set` to do every check below (including the Organoid and source-mask checks
# it shares with the single-patient notebook), so it:
# 1. Checks each file's **schema** first (no data read) to catch missing merge keys cheaply.
# 2. Reads only the `object_id`/`image_set` columns (parquet column projection) instead of
#    the full file.
# 3. Reads files **in parallel** with a thread pool, since this is I/O-bound (many small
#    files) rather than CPU-bound.
#
# This still takes on the order of **20-30 minutes** for the full dataset. To test on a
# subset first, edit `PATIENTS` in the constants cell below to a shorter list.
#
# ## Inputs
# - `data/patient_IDs.txt` — the list of patients to process
# - `data/{patient}/extracted_features/{well_fov}/*.parquet` for each patient
#   - Expected filename format: `{Compartment}_{Channel}_{FeatureType}_{Processor}_features.parquet`
#   - Each file is expected to contain `object_id` and `image_set` columns
# - `data/{patient}/segmentation_masks/{well_fov}/*_mask.tiff` for each patient
#   - Read directly (not through the extracted features), and only for well-FOV/compartment
#     combinations implicated in a merge-related issue, to check whether an object-ID
#     discrepancy traces back to the segmentation mask itself
#
# ## Outputs (written to `3.cellprofiling/logs/`, one row per patient+well-FOV/issue)
# - `well_fov_feature_merge_summary_all_patients.csv`
# - `well_fov_feature_merge_issues_all_patients.csv`
# - `well_fov_feature_merge_report_all_patients.md`
#
# ## What counts as an issue here
# - A file that fails to read, or is missing the `object_id`/`image_set` merge keys
# - A well-FOV with **zero** parquet files (not yet processed at all) — tracked separately
#   from a partial file-count mismatch, since there's nothing to merge
# - A well-FOV whose file count differs from *that patient's own* expected count (each
#   patient can have a different channel/feature-type combination, so the expected count
#   is computed per patient as the mode of its well-FOVs' file counts, not hardcoded)
# - Duplicate `object_id` values within a single feature file
# - A within-compartment merge whose row count exceeds the union of `object_id` values
#   across that compartment's input files (a "blow-up" from a duplicated merge key)
# - An Organoid merge that fails outright because one of its input files is an empty
#   (0-row) profile — a known dtype-clash symptom, reclassified from a generic
#   `merge_error` — this is still a real bug and counts as an issue
# - **Within a compartment, any `object_id` across all of that compartment's input files
#   is greater than 255** — small sequential IDs are the expected "sequential" scheme, so
#   any value above 255 suggests that well-FOV/compartment is using the "z_slice_global"
#   ID scheme (IDs that encode a z-slice offset) instead
# - Object IDs that disagree across the single-cell compartments (`Nuclei`, `Cell`,
#   `Cytoplasm`, `Nucleocentric`) for a well-FOV as a whole. `Organoid` is excluded from
#   this comparison — it's a distinct feature space (one row per whole organoid, not per
#   cell) and is never expected to align with the single-cell compartments
# - A segmentation mask TIFF, read directly, whose unique object IDs don't match the
#   object IDs present in the extracted feature files for that compartment — checked for
#   every well-FOV/compartment implicated in one of the merge-related issues above, to
#   distinguish a **stale/corrupted source mask** (re-segmented/overwritten after
#   featurization last ran) from a bug in the merge/extraction code itself
#
# **Not** counted as an issue: a well-FOV with no readable Organoid feature files, or an
# Organoid merge that produces zero rows. Some well-FOVs legitimately have no organoids,
# so an empty Organoid profile is tracked and reported on its own (see
# `organoid_profiles_empty` in the summary and the Organoid section of the report) but
# does **not** count toward `n_issues` and does **not** trigger the source-mask
# consistency check on its own.
#


# In[2]:


import json
import os
import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
# Segmentation masks are only ever present on the NAS mount, not synced locally, so
# the NAS-resolved path is kept separately before profile_base_dir is overridden below.
segmentation_masks_base_dir = profile_base_dir
profile_base_dir = root_dir  # default to root_dir instead of NAS


# In[3]:


output_features_subparent_name = "extracted_features"
mask_subparent_name = "segmentation_masks"
logs_dir = pathlib.Path(f"{root_dir}/3.cellprofiling/logs").resolve(strict=True)

MERGE_KEYS = ["object_id", "image_set"]
COMPARTMENTS = ["Organoid", "Nuclei", "Cell", "Cytoplasm", "Nucleocentric"]
SINGLE_CELL_COMPARTMENTS = ["Nuclei", "Cell", "Cytoplasm", "Nucleocentric"]
# Nucleocentric features are extracted against the Nuclei mask (see
# scripts/nucleo_centric_featurization.py), so it shares Nuclei\'s source mask file.
COMPARTMENT_MASK_FILENAME = {
    "Organoid": "organoid_mask.tiff",
    "Nuclei": "nuclei_mask.tiff",
    "Cell": "cell_mask.tiff",
    "Cytoplasm": "cytoplasm_mask.tiff",
    "Nucleocentric": "nuclei_mask.tiff",
}
# A profile\'s object_id > 255 is the signature of the "z_slice_global" ID scheme
# (ids encode a z-slice offset), vs. small sequential ids (1, 2, 3, ...) from the
# "sequential" scheme.
OBJECT_ID_SCHEME_THRESHOLD = 255
# Issue types that describe a legitimately-empty Organoid profile rather than a real
# merge/extraction bug — logged for visibility, but excluded from the n_issues/error
# counts and from triggering the source-mask consistency check.
NON_ERROR_ISSUE_TYPES = {"empty_organoid_profiles", "organoid_object_ids_empty"}
N_READ_WORKERS = 16

PATIENTS = pd.read_csv(
    pathlib.Path(f"{profile_base_dir}/data/patient_IDs.txt").resolve(strict=True),
    header=None,
    names=["patient_id"],
).patient_id.tolist()
# For a quick test run, uncomment and shorten:
# PATIENTS = PATIENTS[:1]
PATIENTS


# ## Discover well-FOV directories for every patient
# Every subdirectory of a patient's `extracted_features` is a well-FOV except
# `run_stats`, which holds per-run diagnostic parquets rather than merged feature files.
# The corresponding `segmentation_masks` directory (on the NAS mount) is recorded per
# patient too, for the source-mask consistency check later on.


# In[4]:


well_fov_dirs_by_patient = {}
segmentation_masks_dir_by_patient = {}
for patient in PATIENTS:
    patient_dir = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/{output_features_subparent_name}"
    )
    if not patient_dir.is_dir():
        print(f"Skipping {patient}: no {output_features_subparent_name} directory")
        continue
    well_fov_dirs_by_patient[patient] = sorted(
        d for d in patient_dir.iterdir() if d.is_dir() and d.name != "run_stats"
    )
    segmentation_masks_dir_by_patient[patient] = pathlib.Path(
        f"{segmentation_masks_base_dir}/data/{patient}/{mask_subparent_name}"
    )

n_well_fovs_total = sum(len(v) for v in well_fov_dirs_by_patient.values())
print(f"{len(well_fov_dirs_by_patient)} patients, {n_well_fovs_total} well-FOVs total")


# ## Build the full file list, then read `object_id`/`image_set` in parallel
#
# Each task reads a file's schema first (cheap) to catch missing merge keys without a
# data read, then projects to just the two key columns. Errors (corrupt/zero-byte files,
# missing keys) are captured per file rather than raising.


# In[5]:


def parse_feature_filename(path: pathlib.Path) -> dict:
    """Expected format: {Compartment}_{Channel}_{FeatureType}_{Processor}_features.parquet"""
    parts = path.stem.split("_")
    return {
        "compartment": parts[0],
        "channel": parts[1] if len(parts) > 1 else None,
        "feature_type": parts[2] if len(parts) > 2 else None,
        "processor": parts[-2] if len(parts) > 1 else None,
    }


# In[6]:


def read_key_columns(task: tuple) -> dict:
    """Read just object_id/image_set (+ mtime) for one file, capturing any error."""
    patient, well_fov, path = task
    result = {
        "patient": patient,
        "well_fov": well_fov,
        "file_name": path.name,
        "df": None,
        "mtime": None,
        "error_kind": None,
        "error_detail": None,
    }
    try:
        schema_cols = pq.ParquetFile(path).schema.names
    except Exception as e:
        result["error_kind"] = "read_error"
        result["error_detail"] = str(e)
        return result
    missing_keys = [k for k in MERGE_KEYS if k not in schema_cols]
    if missing_keys:
        result["error_kind"] = "missing_merge_keys"
        result["error_detail"] = f"missing {missing_keys}"
        return result
    try:
        result["df"] = pd.read_parquet(path, columns=MERGE_KEYS)
        result["mtime"] = path.stat().st_mtime
    except Exception as e:
        result["error_kind"] = "read_error"
        result["error_detail"] = str(e)
        result["df"] = None
    return result


# ## Read a segmentation mask's object IDs directly
# Used only for the source-mask consistency check below, and only for the specific
# well-FOV/compartment combinations implicated in a merge-related issue — reading every
# mask TIFF for every well-FOV upfront would be far more I/O than this notebook needs.


# In[7]:


def get_mask_object_ids(mask_path: pathlib.Path) -> set | None:
    """Unique non-background object IDs read directly from a segmentation mask TIFF.

    Returns None if the mask file does not exist, so callers can distinguish a
    missing mask from a mask with zero detected objects.
    """
    if not mask_path.exists():
        return None
    mask = tifffile.imread(mask_path)
    return set(np.unique(mask).tolist()) - {0}


# In[8]:


all_tasks = [
    (patient, well_fov_dir.name, f)
    for patient, well_fov_dirs in well_fov_dirs_by_patient.items()
    for well_fov_dir in well_fov_dirs
    for f in well_fov_dir.glob("*.parquet")
]
print(f"{len(all_tasks)} files to read across {len(well_fov_dirs_by_patient)} patients")

with ThreadPoolExecutor(max_workers=N_READ_WORKERS) as pool:
    file_results = list(
        tqdm.tqdm(
            pool.map(read_key_columns, all_tasks),
            total=len(all_tasks),
            desc="Reading object_id/image_set columns",
        )
    )


# ## Group per-file results by (patient, well-FOV, compartment)
# Also compute, per patient, the *expected* file count as the mode of that patient's
# well-FOVs' file counts (excluding well-FOVs with zero files) — each patient can have a
# different channel/feature-type combination, so a single hardcoded expected count
# (as used in the single-patient notebook) doesn't generalize.


# In[9]:


files_by_well_fov = defaultdict(list)
for r in file_results:
    files_by_well_fov[(r["patient"], r["well_fov"])].append(r)

file_counts_by_patient = defaultdict(list)
for (patient, well_fov), results in files_by_well_fov.items():
    file_counts_by_patient[patient].append(len(results))

expected_n_files_by_patient = {
    patient: (
        pd.Series([c for c in counts if c > 0]).mode().iloc[0]
        if any(c > 0 for c in counts)
        else 0
    )
    for patient, counts in file_counts_by_patient.items()
}
pd.Series(expected_n_files_by_patient, name="expected_n_files").sort_index()


# ## Per-well-FOV merge attempt
#
# For each (patient, well-FOV):
# 1. Route each file's cached read result by compartment (files with a read/schema error
#    are logged and excluded from merging).
# 2. Within each compartment, outer-merge all its files on `object_id` + `image_set`
#    (outer, not left, so a merge error can never silently drop rows without being counted).
# 3. Flag a merge "blow-up" if the merged row count exceeds the union of `object_id`
#    values across that compartment's input files. For `Organoid` specifically, also
#    flag when the merge itself fails because an input file is an empty (0-row) profile
#    (a known dtype-clash symptom, reclassified from a generic `merge_error`). A well-FOV
#    with no Organoid files at all, or an Organoid merge that produces zero rows, is
#    tracked separately and does **not** count as an issue — some well-FOVs legitimately
#    have no organoids.
# 4. Within each compartment, flag when **any** `object_id` across all of that
#    compartment's input files is greater than 255 — small sequential IDs are the
#    expected scheme, so any value above 255 suggests the "z_slice_global" ID scheme is
#    in play for that well-FOV/compartment instead.
# 5. Compare object-ID sets across the single-cell compartments to catch alignment drift.
# 6. For every compartment implicated in a merge-related issue above (a blow-up, a
#    duplicate `object_id`, a failed Organoid merge, an object ID over 255, or a
#    cross-compartment misalignment), read that compartment's segmentation mask TIFF
#    directly and compare its actual unique object IDs against the object IDs present in
#    the extracted feature files. A mismatch traces the discrepancy back to the
#    segmentation mask on disk rather than to a bug in the merge/extraction code.


# In[10]:


def process_well_fov(
    patient: str,
    well_fov: str,
    expected_n_files: int,
    segmentation_masks_dir: pathlib.Path,
) -> tuple[dict, list[dict]]:
    """Attempt to merge one well-FOV's cached feature-file reads per compartment."""
    results = files_by_well_fov[(patient, well_fov)]
    issues = []
    implicated_compartments = set()

    def log_issue(kind, detail, compartment=None):
        issues.append(
            {
                "patient": patient,
                "well_fov": well_fov,
                "issue_type": kind,
                "detail": detail,
            }
        )
        if compartment is not None:
            implicated_compartments.add(compartment)

    if len(results) == 0:
        log_issue("no_files_found", "0 parquet files found for this well-FOV")
        return (
            {
                "patient": patient,
                "well_fov": well_fov,
                "n_files_found": 0,
                "n_files_expected": expected_n_files,
                "file_count_mismatch": False,
                "no_files_found": True,
                "n_issues": 0,
                "n_read_errors": 0,
                "n_merge_errors": 0,
                "n_merge_blowups": 0,
                "n_duplicate_object_id_files": 0,
                "n_object_id_exceeds_threshold": 0,
                "object_ids_aligned_across_compartments": None,
                "misalignment_root_cause": None,
                "organoid_profiles_empty": None,
                "organoid_object_ids_empty": None,
                "n_organoid_empty_profile_merge_errors": 0,
                "n_source_mask_mismatches": 0,
                "n_source_mask_missing": 0,
                "compartments_present": [],
                "compartment_row_counts": {},
            },
            issues,
        )

    if len(results) != expected_n_files:
        log_issue(
            "file_count_mismatch",
            f"found {len(results)} files, expected {expected_n_files} "
            f"(this patient's mode)",
        )

    per_compartment_dfs = {c: [] for c in COMPARTMENTS}
    for r in results:
        if r["error_kind"] is not None:
            log_issue(r["error_kind"], f"{r['file_name']}: {r['error_detail']}")
            continue
        meta = parse_feature_filename(pathlib.Path(r["file_name"]))
        compartment = meta["compartment"]
        if compartment not in COMPARTMENTS:
            log_issue(
                "unknown_compartment", f"{r['file_name']}: compartment '{compartment}'"
            )
            continue
        df = r["df"]
        if df["object_id"].duplicated().any():
            n_dupes = int(df["object_id"].duplicated().sum())
            log_issue(
                "duplicate_object_id_in_file",
                f"{r['file_name']}: {n_dupes} duplicate object_id rows",
                compartment=compartment,
            )
        per_compartment_dfs[compartment].append((r["file_name"], df, r["mtime"]))

    # Organoid profiles are expected for every well-FOV (one row per organoid, unlike
    # the per-cell compartments), but some well-FOVs legitimately have none. Track it
    # for visibility without a `compartment=` kwarg — this is deliberately NOT counted
    # as an error and does NOT trigger the source-mask consistency check on its own.
    if not per_compartment_dfs["Organoid"]:
        log_issue(
            "empty_organoid_profiles",
            f"no readable Organoid feature files for {well_fov}; Organoid merge skipped",
        )

    compartment_shapes = {}
    for compartment, items in per_compartment_dfs.items():
        if not items:
            continue
        names = [n for n, _, _ in items]
        dfs = [d for _, d, _ in items]
        union_ids = set().union(*(set(d["object_id"]) for d in dfs))
        union_n_objects = len(union_ids)

        if union_ids and max(union_ids) > OBJECT_ID_SCHEME_THRESHOLD:
            log_issue(
                "object_id_exceeds_threshold",
                f"{compartment}: object_id values across {len(dfs)} input file(s) "
                f"include a maximum of {max(union_ids)}, greater than "
                f"{OBJECT_ID_SCHEME_THRESHOLD} — likely the 'z_slice_global' ID "
                f"scheme rather than small sequential IDs for this profile",
                compartment=compartment,
            )

        try:
            merged = reduce(
                lambda left, right: pd.merge(left, right, on=MERGE_KEYS, how="outer"),
                dfs,
            )
        except Exception as e:
            # A merge failure on the Organoid compartment that complains about a
            # dtype clash on the merge key is a known symptom of one of the input
            # files being an empty (0-row) profile: pandas infers float64 for an
            # empty object_id column, which then conflicts with the object/int
            # dtype of the populated Organoid files during the outer join. Detect
            # that root cause directly (rather than string-matching the pandas
            # error) and reclassify it as an Organoid-specific empty-profile issue
            # instead of a generic merge_error.
            empty_names = (
                [n for n, d, _ in items if len(d) == 0]
                if compartment == "Organoid"
                else []
            )
            if compartment == "Organoid" and empty_names:
                log_issue(
                    "organoid_empty_profile_merge_error",
                    f"Organoid merge failed because {empty_names} contain 0 rows "
                    f"(an empty object_id column typed as float64 conflicts with "
                    f"the object/int dtype in the populated Organoid files during "
                    f"merge). Original error: {e}",
                    compartment="Organoid",
                )
                continue
            log_issue("merge_error", f"{compartment} ({names}): {e}")
            continue
        compartment_shapes[compartment] = merged.shape
        if merged.shape[0] > union_n_objects:
            log_issue(
                "merge_blowup",
                f"{compartment}: merged to {merged.shape[0]} rows, but the union of "
                f"object_id values across its {len(dfs)} input files is only "
                f"{union_n_objects} — a merge key was duplicated somewhere",
                compartment=compartment,
            )
        # Organoid files can exist and read successfully but still carry zero
        # object_id rows (e.g. an upstream step wrote an empty table). Distinct
        # from the "no Organoid files at all" case checked above — this one only
        # shows up after merging, since union_n_objects is also 0 in this case so
        # it would not otherwise trip the merge-blowup check. Also not counted as
        # an error, for the same reason as the "no Organoid files" case.
        if compartment == "Organoid" and merged.shape[0] == 0:
            log_issue(
                "organoid_object_ids_empty",
                f"Organoid files present ({names}) but merged frame has 0 rows — "
                f"no object_id values found in any Organoid feature file for "
                f"{well_fov}",
            )

    id_sets = {
        c: set(pd.concat([d for _, d, _ in per_compartment_dfs[c]])["object_id"])
        for c in SINGLE_CELL_COMPARTMENTS
        if per_compartment_dfs[c]
    }
    aligned = None
    misaligned_compartments = set()
    if len(id_sets) > 1:
        reference_compartment, reference_ids = next(iter(id_sets.items()))
        aligned = True
        for compartment, ids in id_sets.items():
            if ids != reference_ids:
                aligned = False
                log_issue(
                    "object_id_misalignment",
                    f"{compartment} vs {reference_compartment}: "
                    f"only-in-{compartment}={sorted(ids - reference_ids)[:10]}, "
                    f"only-in-{reference_compartment}={sorted(reference_ids - ids)[:10]}",
                    compartment=compartment,
                )
                implicated_compartments.add(reference_compartment)
                misaligned_compartments.add(compartment)
                misaligned_compartments.add(reference_compartment)

    # For every compartment implicated in a merge-related issue above, read its
    # segmentation mask TIFF directly once (cached in mask_ids_by_compartment so the
    # image-level root-cause check below doesn't re-read the same file), then check
    # whether the mask's object IDs match the extracted feature object IDs. A
    # mismatch here means the mask itself is stale/corrupted relative to the
    # features (or vice versa) — not a bug in this merge/QC logic.
    mask_ids_by_compartment = {}
    for compartment in sorted(implicated_compartments):
        mask_path = (
            segmentation_masks_dir / well_fov / COMPARTMENT_MASK_FILENAME[compartment]
        )
        mask_ids = get_mask_object_ids(mask_path)
        if mask_ids is None:
            log_issue(
                "source_mask_missing",
                f"{compartment}: expected mask at {mask_path}",
            )
            continue
        mask_ids_by_compartment[compartment] = mask_ids
        feature_ids = (
            set().union(
                *(set(d["object_id"]) for _, d, _ in per_compartment_dfs[compartment])
            )
            if per_compartment_dfs[compartment]
            else set()
        )
        only_in_mask = mask_ids - feature_ids
        only_in_features = feature_ids - mask_ids
        if only_in_mask or only_in_features:
            log_issue(
                "source_mask_object_id_mismatch",
                f"{compartment}: mask has {len(mask_ids)} object IDs, extracted "
                f"features have {len(feature_ids)} — only-in-mask="
                f"{sorted(only_in_mask)[:10]}, only-in-features="
                f"{sorted(only_in_features)[:10]} — the segmentation mask on disk "
                f"does not match the extracted features, so the source mask is the "
                f"likely cause (re-segmented/overwritten after features were last "
                f"extracted) rather than a merge/extraction bug",
            )

    # Root-cause the cross-compartment object_id_misalignment above: is it already
    # present in the segmentation masks themselves (image-level — e.g. the Nuclei
    # and Cell masks were built from different object-ID assignments for the same
    # image), or do the masks actually agree with each other and the discrepancy
    # only shows up in the extracted feature tables (feature-extraction-level — a
    # bug in how object_id was written out per compartment during featurization)?
    # Every misaligned compartment's mask was already read into
    # mask_ids_by_compartment by the source-mask check above, since misaligned
    # compartments are always added to implicated_compartments.
    misalignment_root_cause = None
    if misaligned_compartments:
        available_mask_ids = {
            c: mask_ids_by_compartment[c]
            for c in misaligned_compartments
            if c in mask_ids_by_compartment
        }
        if len(available_mask_ids) < len(misaligned_compartments):
            misalignment_root_cause = "unknown_mask_missing"
        else:
            ref_compartment, ref_mask_ids = next(iter(available_mask_ids.items()))
            masks_agree = all(
                ids == ref_mask_ids for ids in available_mask_ids.values()
            )
            if not masks_agree:
                misalignment_root_cause = "image_level"
                for compartment, ids in available_mask_ids.items():
                    if ids != ref_mask_ids:
                        log_issue(
                            "object_id_misalignment_traced_to_image",
                            f"{compartment} vs {ref_compartment}: the segmentation "
                            f"masks themselves disagree on object IDs — "
                            f"only-in-{compartment}-mask="
                            f"{sorted(ids - ref_mask_ids)[:10]}, "
                            f"only-in-{ref_compartment}-mask="
                            f"{sorted(ref_mask_ids - ids)[:10]} — this well-FOV's "
                            f"object_id misalignment originates in the source "
                            f"masks/images, not in feature extraction or merging",
                            compartment=compartment,
                        )
            else:
                misalignment_root_cause = "feature_extraction_level"
                log_issue(
                    "object_id_misalignment_traced_to_features",
                    f"segmentation masks for {sorted(available_mask_ids)} agree "
                    f"with each other on object IDs, but the extracted feature "
                    f"tables for these compartments disagree — the misalignment "
                    f"was introduced during feature extraction or merging, not in "
                    f"the source masks/images",
                )

    summary = {
        "patient": patient,
        "well_fov": well_fov,
        "n_files_found": len(results),
        "n_files_expected": expected_n_files,
        "file_count_mismatch": len(results) != expected_n_files,
        "no_files_found": False,
        "n_issues": sum(
            1 for i in issues if i["issue_type"] not in NON_ERROR_ISSUE_TYPES
        ),
        "n_read_errors": sum(i["issue_type"] == "read_error" for i in issues),
        "n_merge_errors": sum(i["issue_type"] == "merge_error" for i in issues),
        "n_merge_blowups": sum(i["issue_type"] == "merge_blowup" for i in issues),
        "n_duplicate_object_id_files": sum(
            i["issue_type"] == "duplicate_object_id_in_file" for i in issues
        ),
        "n_object_id_exceeds_threshold": sum(
            i["issue_type"] == "object_id_exceeds_threshold" for i in issues
        ),
        "object_ids_aligned_across_compartments": aligned,
        "misalignment_root_cause": misalignment_root_cause,
        "organoid_profiles_empty": any(
            i["issue_type"]
            in (
                "empty_organoid_profiles",
                "organoid_object_ids_empty",
                "organoid_empty_profile_merge_error",
            )
            for i in issues
        ),
        "organoid_object_ids_empty": any(
            i["issue_type"] == "organoid_object_ids_empty" for i in issues
        ),
        "n_organoid_empty_profile_merge_errors": sum(
            i["issue_type"] == "organoid_empty_profile_merge_error" for i in issues
        ),
        "n_source_mask_mismatches": sum(
            i["issue_type"] == "source_mask_object_id_mismatch" for i in issues
        ),
        "n_source_mask_missing": sum(
            i["issue_type"] == "source_mask_missing" for i in issues
        ),
        "compartments_present": sorted(compartment_shapes.keys()),
        "compartment_row_counts": {c: s[0] for c, s in compartment_shapes.items()},
    }
    return summary, issues


# In[11]:


summaries = []
all_issues = []
well_fov_keys = [
    (patient, well_fov_dir.name)
    for patient, well_fov_dirs in well_fov_dirs_by_patient.items()
    for well_fov_dir in well_fov_dirs
]
for patient, well_fov in tqdm.tqdm(
    well_fov_keys, desc="Merging all patients' well-FOVs"
):
    summary, issues = process_well_fov(
        patient,
        well_fov,
        expected_n_files_by_patient[patient],
        segmentation_masks_dir_by_patient[patient],
    )
    summaries.append(summary)
    all_issues.extend(issues)

summary_df = pd.DataFrame(summaries)
issues_df = pd.DataFrame(
    all_issues, columns=["patient", "well_fov", "issue_type", "detail"]
)
print(
    f"{len(summary_df)} well-FOVs processed across {len(PATIENTS)} patients, "
    f"{len(issues_df)} issues logged"
)


# ## Save the flattened per-well-FOV summary and the long-format issue log


# In[12]:


summary_out_path = logs_dir / "well_fov_feature_merge_summary_all_patients.csv"
issues_out_path = logs_dir / "well_fov_feature_merge_issues_all_patients.csv"

csv_summary_df = summary_df.copy()
for col in ["compartments_present", "compartment_row_counts"]:
    csv_summary_df[col] = csv_summary_df[col].apply(json.dumps)
csv_summary_df.to_csv(summary_out_path, index=False)
issues_df.to_csv(issues_out_path, index=False)

print(f"Wrote {summary_out_path}")
print(f"Wrote {issues_out_path}")


# ## Summarize findings, overall and per patient


# In[13]:


n_well_fovs = len(summary_df)
n_no_files = int(summary_df["no_files_found"].sum())
n_file_count_mismatch = int(summary_df["file_count_mismatch"].sum())
n_with_issues = int((summary_df["n_issues"] > 0).sum())
n_misaligned = int(
    (summary_df["object_ids_aligned_across_compartments"] == False).sum()
)
n_misaligned_image_level = int(
    (summary_df["misalignment_root_cause"] == "image_level").sum()
)
n_misaligned_feature_level = int(
    (summary_df["misalignment_root_cause"] == "feature_extraction_level").sum()
)
n_misaligned_unknown = int(
    (summary_df["misalignment_root_cause"] == "unknown_mask_missing").sum()
)
n_read_errors = int(summary_df["n_read_errors"].sum())
n_merge_errors = int(summary_df["n_merge_errors"].sum())
n_merge_blowups = int(summary_df["n_merge_blowups"].sum())
n_organoid_profiles_empty = int(summary_df["organoid_profiles_empty"].sum())
n_organoid_object_ids_empty = int(summary_df["organoid_object_ids_empty"].sum())
n_organoid_empty_profile_merge_errors = int(
    summary_df["n_organoid_empty_profile_merge_errors"].sum()
)
n_source_mask_mismatches = int(summary_df["n_source_mask_mismatches"].sum())
n_source_mask_missing = int(summary_df["n_source_mask_missing"].sum())
n_well_fovs_with_source_mask_mismatch = int(
    (summary_df["n_source_mask_mismatches"] > 0).sum()
)
n_object_id_exceeds_threshold = int(summary_df["n_object_id_exceeds_threshold"].sum())
n_well_fovs_with_object_id_exceeds_threshold = int(
    (summary_df["n_object_id_exceeds_threshold"] > 0).sum()
)

issue_type_counts = (
    issues_df["issue_type"].value_counts() if len(issues_df) else pd.Series(dtype=int)
)

print(f"Well-FOVs processed:                {n_well_fovs}")
print(f"Well-FOVs with zero files found:    {n_no_files}")
print(f"Well-FOVs with >=1 issue:           {n_with_issues}")
print(f"Well-FOVs with file count mismatch: {n_file_count_mismatch}")
print(f"Well-FOVs with object-ID misalignment across compartments: {n_misaligned}")
print(f"  - traced to the source masks/images themselves: {n_misaligned_image_level}")
print(
    "  - traced to feature extraction/merging (masks agree, features don't): "
    f"{n_misaligned_feature_level}"
)
print(f"  - root cause unknown (a mask was missing): {n_misaligned_unknown}")
print(
    "Well-FOVs with empty Organoid profiles (informational only, NOT counted as an "
    f"issue): {n_organoid_profiles_empty}"
)
print(
    f"  - of which Organoid files present but zero object IDs: {n_organoid_object_ids_empty}"
)
print(
    "Organoid merges that failed due to an empty (0-row) profile file (a real merge "
    f"bug, reclassified from a generic merge error and still counted as an issue): "
    f"{n_organoid_empty_profile_merge_errors}"
)
print(
    "Well-FOVs where the segmentation mask's object IDs don't match the extracted "
    f"features (source likely stale/corrupted): {n_well_fovs_with_source_mask_mismatch}"
)
print(f"Total source-mask object-ID mismatches: {n_source_mask_mismatches}")
print(f"Total missing source mask files:   {n_source_mask_missing}")
print(
    "Well-FOVs with a profile whose object_id exceeds "
    f"{OBJECT_ID_SCHEME_THRESHOLD}: {n_well_fovs_with_object_id_exceeds_threshold}"
)
print(f"Total object_id-exceeds-threshold flags: {n_object_id_exceeds_threshold}")
print(f"Total read errors:  {n_read_errors}")
print(f"Total merge errors: {n_merge_errors}")
print(f"Total merge blow-ups: {n_merge_blowups}")
print()
print("Issue counts by type:")
issue_type_counts


# In[14]:


per_patient_summary = (
    summary_df.groupby("patient")
    .agg(
        n_well_fovs=("well_fov", "count"),
        n_no_files=("no_files_found", "sum"),
        n_file_count_mismatch=("file_count_mismatch", "sum"),
        n_with_issues=("n_issues", lambda s: (s > 0).sum()),
        n_organoid_profiles_empty=("organoid_profiles_empty", "sum"),
        n_organoid_empty_profile_merge_errors=(
            "n_organoid_empty_profile_merge_errors",
            "sum",
        ),
        n_source_mask_mismatches=("n_source_mask_mismatches", "sum"),
        n_source_mask_missing=("n_source_mask_missing", "sum"),
        n_object_id_exceeds_threshold=("n_object_id_exceeds_threshold", "sum"),
    )
    .reset_index()
)
per_patient_summary["expected_n_files"] = per_patient_summary["patient"].map(
    expected_n_files_by_patient
)
per_patient_summary


# In[15]:


summary_df.loc[
    summary_df["object_ids_aligned_across_compartments"] == False,
    [
        "patient",
        "well_fov",
        "misalignment_root_cause",
        "compartments_present",
        "compartment_row_counts",
    ],
]


# ## Root-causing the object-ID misalignment: image vs. feature extraction
#
# For every well-FOV flagged with `object_id_misalignment` above, the Nuclei/Cell/
# Cytoplasm/Nucleocentric segmentation mask TIFFs implicated in that misalignment were
# read directly and compared to **each other** (not just to their own extracted
# features, as in the source-mask consistency check below). This distinguishes two
# root causes that look identical from the merged feature tables alone:
# - `image_level`: the masks themselves already disagree on object IDs — the
#   misalignment originates upstream, in segmentation, not in this pipeline's
#   feature extraction or merge code.
# - `feature_extraction_level`: the masks agree with each other, but the extracted
#   feature tables for those compartments don't — the misalignment was introduced
#   while extracting or writing out `object_id` per compartment.
# - `unknown_mask_missing`: at least one implicated compartment's mask file was
#   missing on disk, so the comparison couldn't be made.


# In[16]:


issues_df.loc[
    issues_df["issue_type"].isin(
        [
            "object_id_misalignment_traced_to_image",
            "object_id_misalignment_traced_to_features",
        ]
    )
].sort_values(["patient", "well_fov"]).reset_index(drop=True)


# In[17]:


misalignment_root_cause_counts = (
    summary_df.loc[
        summary_df["object_ids_aligned_across_compartments"] == False,
        "misalignment_root_cause",
    ]
    .value_counts(dropna=False)
    .rename_axis("misalignment_root_cause")
    .reset_index(name="n_well_fovs")
)
misalignment_root_cause_counts


# ## Well-FOVs/compartments with an object_id exceeding 255
#
# Small sequential IDs (1, 2, 3, ...) are the expected "sequential" scheme. A compartment
# containing any `object_id` greater than 255 is likely using the "z_slice_global" scheme
# instead — worth a closer look, and one of the triggers for the source-mask consistency
# check below.


# In[18]:


issues_df.loc[issues_df["issue_type"] == "object_id_exceeds_threshold"].sort_values(
    ["patient", "well_fov"]
).reset_index(drop=True)


# ## Organoid profile checks (informational, not counted as issues)
#
# Well-FOVs with no readable Organoid feature files at all, or an Organoid merge that
# produced zero rows. Some well-FOVs legitimately have no organoids, so these are tracked
# for visibility only and do not count toward `n_issues` or trigger the source-mask
# consistency check on their own. Also shown: Organoid merges that failed outright
# because one of their input files was an empty (0-row) profile — a real merge bug
# (dtype-clash symptom, reclassified from a generic `merge_error`), which **is** still
# counted as an issue.


# In[19]:


issues_df.loc[
    issues_df["issue_type"].isin(
        [
            "empty_organoid_profiles",
            "organoid_object_ids_empty",
            "organoid_empty_profile_merge_error",
        ]
    )
].sort_values(["patient", "well_fov"]).reset_index(drop=True)


# ## Source mask consistency check
#
# For every well-FOV/compartment implicated in a merge-related issue (blow-up,
# duplicate `object_id`, a failed Organoid merge, an object ID over 255, or
# cross-compartment misalignment), the segmentation mask TIFF was read directly and its
# unique object IDs compared against the object IDs present in the extracted feature
# files. A mismatch here means the mask on disk itself doesn't match what was extracted
# — most likely it was re-segmented or overwritten after featurization last ran — as
# opposed to a bug in the merge/extraction code.


# In[20]:


issues_df.loc[
    issues_df["issue_type"].isin(
        ["source_mask_object_id_mismatch", "source_mask_missing"]
    )
].sort_values(["patient", "well_fov"]).reset_index(drop=True)


# ## Generate the markdown report


# In[21]:


worst_offenders = summary_df.sort_values("n_issues", ascending=False).loc[
    summary_df["n_issues"] > 0,
    [
        "patient",
        "well_fov",
        "n_issues",
        "file_count_mismatch",
        "object_ids_aligned_across_compartments",
        "misalignment_root_cause",
        "organoid_profiles_empty",
        "n_source_mask_mismatches",
        "n_object_id_exceeds_threshold",
    ],
]

misalignment_root_cause_issues_df = issues_df.loc[
    issues_df["issue_type"].isin(
        [
            "object_id_misalignment_traced_to_image",
            "object_id_misalignment_traced_to_features",
        ]
    )
].sort_values(["patient", "well_fov"])
misalignment_root_cause_section_lines = (
    misalignment_root_cause_issues_df.to_markdown(index=False)
    if len(misalignment_root_cause_issues_df)
    else "_None._"
)
misalignment_root_cause_counts_lines = (
    misalignment_root_cause_counts.to_markdown(index=False)
    if len(misalignment_root_cause_counts)
    else "_None._"
)

organoid_issues_df = issues_df.loc[
    issues_df["issue_type"].isin(
        [
            "empty_organoid_profiles",
            "organoid_object_ids_empty",
            "organoid_empty_profile_merge_error",
        ]
    )
].sort_values(["patient", "well_fov"])
organoid_section_lines = (
    organoid_issues_df.to_markdown(index=False)
    if len(organoid_issues_df)
    else "_None._"
)

source_mask_issues_df = issues_df.loc[
    issues_df["issue_type"].isin(
        ["source_mask_object_id_mismatch", "source_mask_missing"]
    )
].sort_values(["patient", "well_fov"])
source_mask_section_lines = (
    source_mask_issues_df.to_markdown(index=False)
    if len(source_mask_issues_df)
    else "_None._"
)

object_id_threshold_issues_df = issues_df.loc[
    issues_df["issue_type"] == "object_id_exceeds_threshold"
].sort_values(["patient", "well_fov"])
object_id_threshold_section_lines = (
    object_id_threshold_issues_df.to_markdown(index=False)
    if len(object_id_threshold_issues_df)
    else "_None._"
)

report_lines = [
    "# Feature Merge QC Report — All Patients",
    "",
    f"Generated across {len(PATIENTS)} patients from "
    f"`{profile_base_dir}/data/{{patient}}/{output_features_subparent_name}`.",
    "",
    "## Summary",
    "",
    f"- Well-FOVs processed: **{n_well_fovs}**",
    f"- Well-FOVs with zero files found: **{n_no_files}**",
    f"- Well-FOVs with at least one issue: **{n_with_issues}**",
    f"- Well-FOVs with a file-count mismatch (vs. that patient's own expected count): "
    f"**{n_file_count_mismatch}**",
    f"- Well-FOVs with object-ID misalignment across compartments: **{n_misaligned}**",
    f"  - traced to the source masks/images themselves: **{n_misaligned_image_level}**",
    "  - traced to feature extraction/merging (masks agree, features don't): "
    f"**{n_misaligned_feature_level}**",
    f"  - root cause unknown (a mask was missing): **{n_misaligned_unknown}**",
    f"- Well-FOVs with empty Organoid profiles (*informational, not counted as an issue*): **{n_organoid_profiles_empty}**",
    f"- Well-FOVs with Organoid files present but zero object IDs (*informational, not counted as an issue*): **{n_organoid_object_ids_empty}**",
    f"- Organoid merges that failed due to an empty (0-row) profile file (a real merge bug, reclassified from a generic merge error — still counted as an issue): **{n_organoid_empty_profile_merge_errors}**",
    f"- Well-FOVs where the source segmentation mask's object IDs don't match the "
    f"extracted features: **{n_well_fovs_with_source_mask_mismatch}**",
    f"- Total source-mask object-ID mismatches: **{n_source_mask_mismatches}**",
    f"- Total missing source mask files: **{n_source_mask_missing}**",
    f"- Well-FOVs with a profile whose object_id exceeds {OBJECT_ID_SCHEME_THRESHOLD}: **{n_well_fovs_with_object_id_exceeds_threshold}**",
    f"- Total object_id-exceeds-threshold flags: **{n_object_id_exceeds_threshold}**",
    f"- Total read errors: **{n_read_errors}**",
    f"- Total merge errors: **{n_merge_errors}**",
    f"- Total merge blow-ups: **{n_merge_blowups}**",
    "",
    "## Per-patient breakdown",
    per_patient_summary.to_markdown(index=False),
    "",
    "## Issue counts by type",
    "",
    issue_type_counts.to_frame("count").to_markdown()
    if len(issue_type_counts)
    else "_No issues found._",
    "",
    "## Well-FOVs with zero files found",
    "",
    summary_df.loc[summary_df["no_files_found"], ["patient", "well_fov"]].to_markdown(
        index=False
    )
    if n_no_files
    else "_None._",
    "",
    "## Organoid profile checks (informational, not counted as issues)",
    "",
    "Well-FOVs with no readable Organoid feature files at all, or an Organoid merge ",
    "that produced zero rows (tracked for visibility only, not counted toward ",
    "n_issues). Also shown: Organoid merges that failed outright because one input ",
    "file was an empty (0-row) profile — a real merge bug, and still counted as an ",
    "issue.",
    "",
    organoid_section_lines,
    "",
    "## Object IDs exceeding {}".format(OBJECT_ID_SCHEME_THRESHOLD),
    "",
    "Compartments containing any object_id greater than "
    f"{OBJECT_ID_SCHEME_THRESHOLD} — likely the 'z_slice_global' ID scheme rather ",
    "than small sequential IDs for that well-FOV/compartment.",
    "",
    object_id_threshold_section_lines,
    "",
    "## Root-causing the object-ID misalignment: image vs. feature extraction",
    "",
    "For every well-FOV flagged with `object_id_misalignment`, the Nuclei/Cell/",
    "Cytoplasm/Nucleocentric masks implicated in that misalignment were read directly ",
    "and compared to **each other** — not just to their own extracted features (that's ",
    "the source-mask consistency check below). `image_level` means the masks ",
    "themselves already disagree, so the root cause is upstream in segmentation, not ",
    "in this pipeline. `feature_extraction_level` means the masks agree with each ",
    "other but the extracted feature tables don't, so the root cause is in how ",
    "`object_id` was extracted/written per compartment. `unknown_mask_missing` means a ",
    "mask file needed for the comparison wasn't found on disk.",
    "",
    misalignment_root_cause_counts_lines,
    "",
    misalignment_root_cause_section_lines,
    "",
    "## Source mask consistency check",
    "",
    "For every well-FOV/compartment implicated in a merge-related issue (blow-up, ",
    "duplicate object_id, a failed Organoid merge, an object ID over ",
    f"{OBJECT_ID_SCHEME_THRESHOLD}, or cross-compartment misalignment), the ",
    "segmentation mask TIFF was read directly and its unique object IDs compared ",
    "against the object IDs in the extracted features. A mismatch points to the mask ",
    "on disk (likely re-segmented/overwritten after featurization last ran) rather ",
    "than a merge bug.",
    "",
    source_mask_section_lines,
    "",
    "## Well-FOVs with the most issues (top 30)",
    "",
    worst_offenders.head(30).to_markdown(index=False)
    if len(worst_offenders)
    else "_None._",
    "",
    "## Notes",
    "",
    "- Full per-issue detail is in `well_fov_feature_merge_issues_all_patients.csv`.",
    "- An empty Organoid profile (no files, or a merge with zero rows) is tracked",
    "  separately and does NOT count as an issue — some well-FOVs legitimately have",
    "  no organoids. An Organoid merge that fails outright because one input file is",
    "  an empty (0-row) profile IS still counted, since that reflects an actual",
    "  merge bug rather than a legitimately-empty well-FOV.",
    "- The source mask consistency check reads segmentation mask TIFFs directly (not ",
    "  through the extracted features) to determine whether an object-ID discrepancy ",
    "  originates from the mask itself vs. from feature extraction/merging.",
    "- The root-cause check goes one step further for `object_id_misalignment` cases",
    "  specifically: it compares the implicated compartments' masks directly against",
    "  EACH OTHER (not just each mask against its own features), which is the only way",
    "  to tell an image-level misalignment (masks disagree with each other) apart from",
    "  a feature-extraction-level one (masks agree, but the feature tables don't).",
    "- `Organoid` is excluded from the cross-compartment object-ID alignment check; it's",
    "  a distinct feature space (one row per whole organoid) with its own object-ID",
    "  space that isn't expected to match the single-cell compartments.",
    "- Each patient's expected file count is that patient's own mode file count across",
    "  its well-FOVs (excluding zero-file well-FOVs) — patients can have different",
    "  channel/feature-type combinations, so a single global expected count doesn't",
    "  generalize across patients.",
]

report_text = "\n".join(report_lines)
report_out_path = logs_dir / "well_fov_feature_merge_report_all_patients.md"
report_out_path.write_text(report_text)
print(f"Wrote {report_out_path}")


# In[22]:


from IPython.display import Markdown, display

display(Markdown(report_text))
