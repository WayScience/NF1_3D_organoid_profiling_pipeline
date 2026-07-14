#!/usr/bin/env python
# coding: utf-8

# # Merge & Validate Well-FOV Feature Parquets
#
# ## Purpose
# For every well-FOV extracted for a patient, this notebook attempts to merge the
# per-compartment x channel x feature-type parquet files (101 files per well-FOV for
# `NF0014_T1`) into a single feature space per compartment, exactly as Stage 4's
# `1.merge_feature_parquets.py` does. Unlike that production script, this notebook does
# **not** write merged output — its purpose is quality control: surface every place the
# merge could silently go wrong (missing files, unreadable files, missing merge keys,
# duplicate object IDs, merge blow-ups, object-ID misalignment across compartments, and
# object IDs that exceed the expected small-sequential-ID range) and summarize the
# findings in a report.
#
# ## Inputs
# - `data/{patient}/extracted_features/{well_fov}/*.parquet`
#   - One parquet per compartment x channel x feature-type combination
#   - Expected filename format: `{Compartment}_{Channel}_{FeatureType}_{Processor}_features.parquet`
#   - Each file is expected to contain `object_id`, `image_set`, and feature columns
# - `data/{patient}/segmentation_masks/{well_fov}/*_mask.tiff`
#   - Read directly (not through the extracted features) to check whether an
#     object-ID discrepancy traces back to the segmentation mask itself
#
# ## Outputs (written to `3.cellprofiling/logs/`)
# - `well_fov_feature_merge_summary.csv` — one row per well-FOV, scalar QC flags/counts
# - `well_fov_feature_merge_issues.csv` — long-format log, one row per individual issue found
# - `well_fov_feature_merge_report.md` — human-readable summary report
#
# ## What counts as a merge error here
# - A file that fails to read, or is missing the `object_id`/`image_set` merge keys
# - A well-FOV whose file count differs from the expected count (101 for `NF0014_T1`)
# - Duplicate `object_id` values within a single feature file (would silently fan out an outer join)
# - A within-compartment merge whose row count exceeds the union of `object_id` values
#   seen across that compartment's input files (a "blow-up", meaning a merge key was
#   duplicated and fanned out the outer join — a real bug, unlike the expected case where
#   different feature types/channels simply detect slightly different object sets)
# - An Organoid merge that fails outright because one of its input files is an empty
#   (0-row) profile — a dtype-clash symptom, reclassified from a generic merge error
# - A compartment's profile containing any `object_id` greater than 255 — small
#   sequential IDs (1, 2, 3, ...) are the expected scheme, so any ID above 255 suggests
#   that profile is using the "z_slice_global" scheme (IDs that encode a z-slice offset)
#   instead, which is worth a closer look
# - Object IDs that disagree across the single-cell compartments (`Nuclei`, `Cell`,
#   `Cytoplasm`, `Nucleocentric`) for the same well-FOV as a whole — a coarser, known issue
#   tracked separately in `3.cellprofiling/logs/well_fov_object_id_mismatches.csv`.
#
# **Not** counted as an error: a well-FOV with no readable Organoid feature files, or an
# Organoid merge that produces zero rows. Some well-FOVs legitimately have no organoids,
# so an empty Organoid profile is tracked and reported on its own (see
# `organoid_profiles_empty` in the summary and the Organoid section of the report) but
# does **not** count toward `n_issues` or trigger the source-mask consistency check below.
#
# ## Source mask consistency check
# For every compartment implicated in one of the merge errors above, this notebook reads
# that compartment's segmentation mask TIFF directly and compares its unique object IDs
# against the object IDs present in the extracted feature files. This distinguishes a
# **corrupted/stale source mask** (mask object IDs don't match the features — the mask
# was likely re-segmented or overwritten after featurization last ran) from a bug in the
# merge/extraction code itself (mask and features agree, but the merge still misbehaves).
# `Nucleocentric` shares `Nuclei`'s mask file, since nucleocentric features are extracted
# against the nuclei mask (see `scripts/nucleo_centric_featurization.py`).
#
# `Organoid` is **not** included in the cross-compartment object-ID alignment check: it is
# a distinct feature space (one row per whole organoid) with its own, much smaller object-ID
# space, and is never expected to line up with the per-cell compartments — it is merged
# internally, but not checked against Nuclei/Cell/Cytoplasm/Nucleocentric.
#

# In[1]:


import json
import os
import pathlib
from functools import reduce

import numpy as np
import pandas as pd
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


# In[2]:


patient = "NF0014_T1"
output_features_subparent_name = "extracted_features"
mask_subparent_name = "segmentation_masks"

extracted_features_dir = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{output_features_subparent_name}"
).resolve(strict=True)

segmentation_masks_dir = pathlib.Path(
    f"{segmentation_masks_base_dir}/data/{patient}/{mask_subparent_name}"
).resolve(strict=True)

logs_dir = pathlib.Path(f"{root_dir}/3.cellprofiling/logs").resolve(strict=True)

MERGE_KEYS = ["object_id", "image_set"]
COMPARTMENTS = ["Organoid", "Nuclei", "Cell", "Cytoplasm", "Nucleocentric"]
SINGLE_CELL_COMPARTMENTS = ["Nuclei", "Cell", "Cytoplasm", "Nucleocentric"]
EXPECTED_N_FILES = 101
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

extracted_features_dir


# ## Discover well-FOV directories
# Every subdirectory of `extracted_features` is a well-FOV except `run_stats`, which holds
# per-run diagnostic parquets rather than merged feature files.

# In[3]:


well_fov_dirs = sorted(
    d for d in extracted_features_dir.iterdir() if d.is_dir() and d.name != "run_stats"
)
print(f"Found {len(well_fov_dirs)} well-FOV directories for {patient}")


# ## Per-well-FOV merge attempt
#
# For each well-FOV:
# 1. Parse every parquet filename into `compartment` / `channel` / `feature_type` / `processor`.
# 2. Read each file, checking for read errors, missing merge keys, and duplicate `object_id`s.
# 3. Within each compartment, outer-merge all its files on `object_id` + `image_set`
#    (outer, not left, so a merge error can never silently drop rows without being counted).
# 4. Flag a merge "blow-up" if the merged row count exceeds the union of `object_id`
#    values seen across that compartment's input files.
# 5. Flag a profile (compartment) whose `object_id` values include any value greater
#    than 255 — small sequential IDs are the expected scheme, so any value above 255
#    suggests the "z_slice_global" ID scheme is in play for that well-FOV/compartment.
# 6. Compare object-ID sets across the single-cell compartments (`Nuclei`, `Cell`,
#    `Cytoplasm`, `Nucleocentric`) to catch alignment drift. `Organoid` is deliberately
#    excluded from this comparison — it is a separate feature space (one row per whole
#    organoid, not per cell) and its object-ID space is not expected to match the
#    single-cell compartments.
# 7. For every compartment implicated in a merge-related issue above (a blow-up, a
#    duplicate `object_id` within one of its files, an object ID over 255, or a
#    cross-compartment misalignment), read that compartment's segmentation mask TIFF
#    directly and compare its actual unique object IDs against the object IDs present in
#    the extracted feature files. If the mask's own object-ID set doesn't match what was
#    extracted, the discrepancy traces back to the segmentation mask on disk (e.g. it
#    was re-segmented/overwritten after features were last extracted) rather than a
#    bug in the merge/extraction code — this tells you whether to rerun featurization
#    or re-segmentation.
#
# A well-FOV with no readable Organoid feature files, or an Organoid merge that produces
# zero rows, is tracked but **not** counted as an issue and does **not** trigger the
# source-mask check on its own — some well-FOVs legitimately have no organoids.

# In[4]:


def parse_feature_filename(path: pathlib.Path) -> dict:
    """Expected format: {Compartment}_{Channel}_{FeatureType}_{Processor}_features.parquet"""
    parts = path.stem.split("_")
    return {
        "compartment": parts[0],
        "channel": parts[1] if len(parts) > 1 else None,
        "feature_type": parts[2] if len(parts) > 2 else None,
        "processor": parts[-2] if len(parts) > 1 else None,
    }


# In[5]:


def get_mask_object_ids(mask_path: pathlib.Path) -> set | None:
    """Unique non-background object IDs read directly from a segmentation mask TIFF.

    Returns None if the mask file does not exist, so callers can distinguish a
    missing mask from a mask with zero detected objects.
    """
    if not mask_path.exists():
        return None
    mask = tifffile.imread(mask_path)
    return set(np.unique(mask).tolist()) - {0}


# In[6]:


def process_well_fov(well_fov_dir: pathlib.Path) -> tuple[dict, list[dict]]:
    """Attempt to merge one well-FOV's feature parquets per compartment.

    Returns a scalar summary row and a list of individual issue records (empty
    where nothing went wrong).
    """
    well_fov = well_fov_dir.name
    files = sorted(well_fov_dir.glob("*.parquet"))
    issues = []
    implicated_compartments = set()

    def log_issue(kind, detail, compartment=None):
        issues.append({"well_fov": well_fov, "issue_type": kind, "detail": detail})
        if compartment is not None:
            implicated_compartments.add(compartment)

    if len(files) != EXPECTED_N_FILES:
        log_issue(
            "file_count_mismatch",
            f"found {len(files)} files, expected {EXPECTED_N_FILES}",
        )

    per_compartment_dfs = {c: [] for c in COMPARTMENTS}
    for f in files:
        meta = parse_feature_filename(f)
        compartment = meta["compartment"]
        if compartment not in COMPARTMENTS:
            log_issue("unknown_compartment", f"{f.name}: compartment '{compartment}'")
            continue
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            log_issue("read_error", f"{f.name}: {e}")
            continue
        missing_keys = [k for k in MERGE_KEYS if k not in df.columns]
        if missing_keys:
            log_issue("missing_merge_keys", f"{f.name}: missing {missing_keys}")
            continue
        if df["object_id"].duplicated().any():
            n_dupes = int(df["object_id"].duplicated().sum())
            log_issue(
                "duplicate_object_id_in_file",
                f"{f.name}: {n_dupes} duplicate object_id rows",
                compartment=compartment,
            )
        per_compartment_dfs[compartment].append((f.name, df))

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
        names = [n for n, _ in items]
        dfs = [d for _, d in items]
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
                [n for n, d in items if len(d) == 0]
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
        c: set(pd.concat([d for _, d in per_compartment_dfs[c]])["object_id"])
        for c in SINGLE_CELL_COMPARTMENTS
        if per_compartment_dfs[c]
    }
    aligned = None
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

    # For every compartment implicated in a merge-related issue above, check whether
    # the segmentation mask on disk actually matches the extracted feature object IDs.
    # A mismatch here means the mask itself is stale/corrupted relative to the
    # features (or vice versa) — not a bug in this merge/QC logic.
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
        feature_ids = (
            set().union(
                *(set(d["object_id"]) for _, d in per_compartment_dfs[compartment])
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

    summary = {
        "well_fov": well_fov,
        "n_files_found": len(files),
        "n_files_expected": EXPECTED_N_FILES,
        "file_count_mismatch": len(files) != EXPECTED_N_FILES,
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
        "compartment_col_counts": {c: s[1] for c, s in compartment_shapes.items()},
    }
    return summary, issues


# In[7]:


summaries = []
all_issues = []
for well_fov_dir in tqdm.tqdm(well_fov_dirs, desc=f"Merging {patient} well-FOVs"):
    summary, issues = process_well_fov(well_fov_dir)
    summaries.append(summary)
    all_issues.extend(issues)

summary_df = pd.DataFrame(summaries)
issues_df = pd.DataFrame(all_issues, columns=["well_fov", "issue_type", "detail"])
print(f"{len(summary_df)} well-FOVs processed, {len(issues_df)} issues logged")


# In[8]:


summary_df[summary_df["n_merge_errors"] > 0]


# ## Save the flattened per-well-FOV summary and the long-format issue log

# In[9]:


summary_out_path = logs_dir / "well_fov_feature_merge_summary.csv"
issues_out_path = logs_dir / "well_fov_feature_merge_issues.csv"

csv_summary_df = summary_df.copy()
for col in ["compartments_present", "compartment_row_counts", "compartment_col_counts"]:
    csv_summary_df[col] = csv_summary_df[col].apply(json.dumps)
csv_summary_df.to_csv(summary_out_path, index=False)
issues_df.to_csv(issues_out_path, index=False)

print(f"Wrote {summary_out_path}")
print(f"Wrote {issues_out_path}")


# ## Summarize findings

# In[10]:


n_well_fovs = len(summary_df)
n_file_count_mismatch = int(summary_df["file_count_mismatch"].sum())
n_with_issues = int((summary_df["n_issues"] > 0).sum())
n_misaligned = int(
    (summary_df["object_ids_aligned_across_compartments"] == False).sum()
)
n_read_errors = int(summary_df["n_read_errors"].sum())
n_merge_errors = int(summary_df["n_merge_errors"].sum())
n_merge_blowups = int(summary_df["n_merge_blowups"].sum())
n_source_mask_mismatches = int(summary_df["n_source_mask_mismatches"].sum())
n_source_mask_missing = int(summary_df["n_source_mask_missing"].sum())
n_well_fovs_with_source_mask_mismatch = int(
    (summary_df["n_source_mask_mismatches"] > 0).sum()
)
n_organoid_profiles_empty = int(summary_df["organoid_profiles_empty"].sum())
n_organoid_object_ids_empty = int(summary_df["organoid_object_ids_empty"].sum())
n_organoid_empty_profile_merge_errors = int(
    summary_df["n_organoid_empty_profile_merge_errors"].sum()
)
n_object_id_exceeds_threshold = int(summary_df["n_object_id_exceeds_threshold"].sum())
n_well_fovs_with_object_id_exceeds_threshold = int(
    (summary_df["n_object_id_exceeds_threshold"] > 0).sum()
)

issue_type_counts = (
    issues_df["issue_type"].value_counts() if len(issues_df) else pd.Series(dtype=int)
)

print(f"Well-FOVs processed:               {n_well_fovs}")
print(f"Well-FOVs with >=1 issue:          {n_with_issues}")
print(f"Well-FOVs with file count mismatch:{n_file_count_mismatch}")
print(f"Well-FOVs with object-ID misalignment across compartments: {n_misaligned}")
print(f"Total read errors:                 {n_read_errors}")
print(f"Total merge errors:                {n_merge_errors}")
print(f"Total merge blow-ups:              {n_merge_blowups}")
print(
    "Well-FOVs where the segmentation mask's object IDs don't match the extracted "
    f"features (source likely stale/corrupted): {n_well_fovs_with_source_mask_mismatch}"
)
print(f"Total source-mask object-ID mismatches: {n_source_mask_mismatches}")
print(f"Total missing source mask files:   {n_source_mask_missing}")
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
    f"Well-FOVs with a profile whose object_id exceeds {OBJECT_ID_SCHEME_THRESHOLD}: "
    f"{n_well_fovs_with_object_id_exceeds_threshold}"
)
print(f"Total object_id-exceeds-threshold flags: {n_object_id_exceeds_threshold}")
print()
print("Issue counts by type:")
issue_type_counts


# In[11]:


summary_df[summary_df["n_issues"] > 0]


# In[12]:


summary_df.loc[
    summary_df["file_count_mismatch"], ["well_fov", "n_files_found", "n_files_expected"]
]


# ## Object IDs exceeding 255
#
# Profiles (compartments) whose `object_id` values include anything greater than 255 —
# small sequential IDs are the expected scheme, so this likely means the
# "z_slice_global" ID scheme is in play for that well-FOV/compartment instead.

# In[13]:


issues_df.loc[issues_df["issue_type"] == "object_id_exceeds_threshold"].sort_values(
    ["well_fov"]
).reset_index(drop=True)


# ## Source mask consistency check
#
# For every well-FOV/compartment implicated in a merge-related issue (blow-up,
# duplicate `object_id`, or cross-compartment misalignment), the segmentation mask
# TIFF was read directly and its unique object IDs compared against the object IDs
# present in the extracted feature files for that compartment. A mismatch here means
# the mask on disk itself doesn't match what was extracted — most likely it was
# re-segmented or overwritten after featurization last ran — as opposed to a bug in
# this merge/QC logic. `source_mask_missing` means the expected mask TIFF wasn't found
# at all.

# In[14]:


issues_df.loc[
    issues_df["issue_type"].isin(
        ["source_mask_object_id_mismatch", "source_mask_missing"]
    )
].sort_values(["well_fov"]).reset_index(drop=True)


# In[15]:


summary_df.loc[
    summary_df["object_ids_aligned_across_compartments"] == False,
    ["well_fov", "compartments_present", "compartment_row_counts"],
]


# ## Generate the markdown report

# In[16]:


worst_offenders = summary_df.sort_values("n_issues", ascending=False).loc[
    summary_df["n_issues"] > 0,
    [
        "well_fov",
        "n_issues",
        "file_count_mismatch",
        "object_ids_aligned_across_compartments",
        "n_source_mask_mismatches",
        "n_object_id_exceeds_threshold",
    ],
]

source_mask_issues_df = issues_df.loc[
    issues_df["issue_type"].isin(
        ["source_mask_object_id_mismatch", "source_mask_missing"]
    )
].sort_values(["well_fov"])
source_mask_section_lines = (
    source_mask_issues_df.to_markdown(index=False)
    if len(source_mask_issues_df)
    else "_None._"
)

object_id_threshold_issues_df = issues_df.loc[
    issues_df["issue_type"] == "object_id_exceeds_threshold"
].sort_values(["well_fov"])
object_id_threshold_section_lines = (
    object_id_threshold_issues_df.to_markdown(index=False)
    if len(object_id_threshold_issues_df)
    else "_None._"
)

organoid_issues_df = issues_df.loc[
    issues_df["issue_type"].isin(
        ["empty_organoid_profiles", "organoid_object_ids_empty"]
    )
].sort_values(["well_fov"])
organoid_section_lines = (
    organoid_issues_df.to_markdown(index=False)
    if len(organoid_issues_df)
    else "_None._"
)

whole_compartment_section_lines = []
misaligned_well_fovs = summary_df.loc[
    summary_df["object_ids_aligned_across_compartments"] == False, "well_fov"
].tolist()
if misaligned_well_fovs:
    whole_compartment_section_lines.append(
        "These well-FOVs have at least one single-cell compartment (`Nuclei`, `Cell`, "
        "`Cytoplasm`, `Nucleocentric`) whose object-ID set as a whole does not match the "
        "others. See the source mask consistency section above and "
        "`well_fov_object_id_mismatches.csv` for whether the disagreement traces back to "
        "the segmentation mask itself or to feature extraction."
    )
    whole_compartment_section_lines.append("")
    whole_compartment_section_lines.append(
        summary_df.loc[
            summary_df["object_ids_aligned_across_compartments"] == False,
            ["well_fov", "compartments_present", "compartment_row_counts"],
        ].to_markdown(index=False)
    )
else:
    whole_compartment_section_lines.append("_None._")

report_lines = (
    [
        f"# Feature Merge QC Report — {patient}",
        "",
        f"Generated from `{extracted_features_dir}`.",
        "",
        "## Summary",
        "",
        f"- Well-FOVs processed: **{n_well_fovs}**",
        f"- Well-FOVs with at least one issue: **{n_with_issues}**",
        f"- Well-FOVs with a file-count mismatch (expected {EXPECTED_N_FILES}): **{n_file_count_mismatch}**",
        f"- Well-FOVs with object-ID misalignment across compartments: **{n_misaligned}**",
        f"- Well-FOVs where the source segmentation mask's object IDs don't match the "
        f"extracted features: **{n_well_fovs_with_source_mask_mismatch}**",
        f"- Total source-mask object-ID mismatches: **{n_source_mask_mismatches}**",
        f"- Total missing source mask files: **{n_source_mask_missing}**",
        f"- Well-FOVs with a profile whose object_id exceeds {OBJECT_ID_SCHEME_THRESHOLD}: **{n_well_fovs_with_object_id_exceeds_threshold}**",
        f"- Total object_id-exceeds-threshold flags: **{n_object_id_exceeds_threshold}**",
        f"- Well-FOVs with empty Organoid profiles (*informational, not counted as an issue*): **{n_organoid_profiles_empty}**",
        f"- Well-FOVs with Organoid files present but zero object IDs (*informational, not counted as an issue*): **{n_organoid_object_ids_empty}**",
        f"- Organoid merges that failed due to an empty (0-row) profile file (a real merge bug, reclassified from a generic merge error — still counted as an issue): **{n_organoid_empty_profile_merge_errors}**",
        f"- Total read errors: **{n_read_errors}**",
        f"- Total merge errors: **{n_merge_errors}**",
        f"- Total merge blow-ups: **{n_merge_blowups}**",
        "",
        "## Issue counts by type",
        "",
        issue_type_counts.to_frame("count").to_markdown()
        if len(issue_type_counts)
        else "_No issues found._",
        "",
        "## File-count mismatches",
        "",
        summary_df.loc[
            summary_df["file_count_mismatch"],
            ["well_fov", "n_files_found", "n_files_expected"],
        ].to_markdown(index=False)
        if n_file_count_mismatch
        else "_None._",
        "",
        "## Organoid profile checks (informational, not counted as issues)",
        "",
        "Well-FOVs with no readable Organoid feature files at all, or an Organoid merge ",
        "that produced zero rows. Some well-FOVs legitimately have no organoids, so ",
        "these are tracked for visibility only and do not count toward `n_issues` or ",
        "trigger the source-mask consistency check on their own.",
        "",
        organoid_section_lines,
        "",
        "## Object IDs exceeding {}".format(OBJECT_ID_SCHEME_THRESHOLD),
        "",
        "Profiles (compartments) whose object_id values include anything greater than "
        f"{OBJECT_ID_SCHEME_THRESHOLD} — likely the 'z_slice_global' ID scheme rather ",
        "than small sequential IDs for that well-FOV/compartment.",
        "",
        object_id_threshold_section_lines,
        "",
        "## Source mask consistency check",
        "",
        "For every well-FOV/compartment implicated in a merge-related issue (blow-up, ",
        "duplicate object_id, an object ID over {}, or cross-compartment ".format(
            OBJECT_ID_SCHEME_THRESHOLD
        ),
        "misalignment), the segmentation mask TIFF was read directly and its unique ",
        "object IDs compared against the object IDs in the extracted features. A ",
        "mismatch points to the mask on disk (likely re-segmented/overwritten after ",
        "featurization last ran) rather than a merge bug.",
        "",
        source_mask_section_lines,
        "",
        "## Object-ID misalignment across whole single-cell compartments",
        "",
    ]
    + whole_compartment_section_lines
    + [
        "",
        "## Well-FOVs with the most issues",
        "",
        worst_offenders.head(20).to_markdown(index=False)
        if len(worst_offenders)
        else "_None._",
        "",
        "## Notes",
        "",
        '- A "merge blow-up" means the outer join produced more rows than the union of',
        "  object_id values across that compartment's input files — a merge key was",
        "  duplicated and fanned out the join. This is distinct from (and rarer than) two",
        "  feature types/channels simply detecting slightly different object sets, which is",
        "  expected and only grows the merged row count up to the union size.",
        "- An empty Organoid profile (no files, or a merge with zero rows) is tracked",
        "  separately and does NOT count as an issue — some well-FOVs legitimately have",
        "  no organoids. An Organoid merge that fails outright because one input file is",
        "  an empty (0-row) profile IS still counted, since that reflects an actual",
        "  merge bug rather than a legitimately-empty well-FOV.",
        "- The source mask consistency check reads segmentation mask TIFFs directly (not ",
        "  through the extracted features) to determine whether an object-ID discrepancy ",
        "  originates from the mask itself vs. from feature extraction/merging.",
        "- Object-ID misalignment across whole compartments is a coarser rollup of the same",
        "  condition tracked row-by-row in `well_fov_object_id_mismatches.csv`; see that file",
        "  for the exact compartment pairs and object-ID scheme involved.",
        "- Full detail for every individual issue is in `well_fov_feature_merge_issues.csv`.",
    ]
)

report_text = "\n".join(report_lines)
report_out_path = logs_dir / "well_fov_feature_merge_report.md"
report_out_path.write_text(report_text)
print(f"Wrote {report_out_path}")


# In[17]:


from IPython.display import Markdown, display

display(Markdown(report_text))


# In[21]:


well_fov = "G9-1"
# investigate the X well fov
files = list(
    pathlib.Path(
        f"/home/lippincm/Documents/NF1_3D_organoid_profiling_pipeline/data/NF0014_T1/extracted_features/{well_fov}"
    ).glob("*.parquet")
)
files = [f for f in files if not "Organoid" in f.name]
new_df = pd.DataFrame()
for f in files:
    df = pd.read_parquet(f)
    new_df = (
        pd.merge(new_df, df, on=MERGE_KEYS, how="outer") if not new_df.empty else df
    )
new_df


# In[19]:


# get the nan columns now
tmp_df = (
    new_df.loc[new_df["object_id"] == 2]
    .isnull()
    .sum()
    .sort_values(ascending=False)
    .to_frame("n_missing")
    .query("n_missing > 0")
    .sort_values("n_missing", ascending=False)
    .reset_index()
    .rename(columns={"index": "column_name"})
)
tmp_df
tmp_df["compartment_channel_feature"] = (
    tmp_df["column_name"].str.split("_").str[:3].str.join("_")
)
[print(f"rm {col}*") for col in tmp_df["compartment_channel_feature"].unique()]


# In[20]:


cell_unique = get_mask_object_ids(segmentation_masks_dir / well_fov / "cell_mask.tiff")
nuclei_unique = get_mask_object_ids(
    segmentation_masks_dir / well_fov / "nuclei_mask.tiff"
)
organoid_unique = get_mask_object_ids(
    segmentation_masks_dir / well_fov / "organoid_mask.tiff"
)
print(cell_unique - nuclei_unique)
if len(cell_unique - nuclei_unique) > 0:
    print(
        f"Warning: {len(cell_unique - nuclei_unique)} cell IDs are not present in nuclei mask"
    )
print(len(nuclei_unique), len(cell_unique))
print(cell_unique)
print(organoid_unique)


# In[ ]:
