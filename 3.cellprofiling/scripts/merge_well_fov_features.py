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
# duplicate object IDs, merge blow-ups, object-ID misalignment across compartments) and
# summarize the findings in a report.
#
# ## Inputs
# - `data/{patient}/extracted_features/{well_fov}/*.parquet`
#   - One parquet per compartment x channel x feature-type combination
#   - Expected filename format: `{Compartment}_{Channel}_{FeatureType}_{Processor}_features.parquet`
#   - Each file is expected to contain `object_id`, `image_set`, and feature columns
#
# ## Outputs (written to `3.cellprofiling/logs/`)
# - `well_fov_feature_merge_summary.csv` — one row per well-FOV, scalar QC flags/counts
# - `well_fov_feature_merge_issues.csv` — long-format log, one row per individual issue found
# - `well_fov_files_to_rerun.csv` — **one row per specific parquet file** that is stale
#   relative to the most-recently-regenerated object-ID set within its own compartment —
#   this is the actionable "which feature extraction script do I need to rerun" list
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
# - **Within a compartment, one or more individual files whose `object_id` set is stale
#   relative to the most-recently-regenerated file group** (a "file-level group split") —
#   this pinpoints exactly which feature files (e.g. a specific `{Channel}_{FeatureType}`
#   file) were extracted against an older mask than the rest and need to be rerun. The
#   reference group is picked by file **recency**, not file count, since re-segmenting
#   masks and then only rerunning some feature types leaves the stale feature type as
#   the majority by count — see the file-mtime discussion below.
# - Object IDs that disagree across the single-cell compartments (`Nuclei`, `Cell`,
#   `Cytoplasm`, `Nucleocentric`) for the same well-FOV as a whole — a coarser, known issue
#   tracked separately in `3.cellprofiling/logs/well_fov_object_id_mismatches.csv`. Because
#   this compares whole compartments (not individual files), it points to re-running an
#   entire compartment's segmentation/featurization step rather than one file.
#
# `Organoid` is **not** included in the cross-compartment object-ID alignment check: it is
# a distinct feature space (one row per whole organoid) with its own, much smaller object-ID
# space, and is never expected to line up with the per-cell compartments — it is merged,
# and checked for file-level group splits, internally, but not against Nuclei/Cell/
# Cytoplasm/Nucleocentric.

# In[1]:


import json
import os
import pathlib
from collections import defaultdict
from datetime import datetime
from functools import reduce

import pandas as pd
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
profile_base_dir = root_dir  # default to root_dir instead of NAS


# In[2]:


patient = "NF0014_T1"
output_features_subparent_name = "extracted_features"

extracted_features_dir = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{output_features_subparent_name}"
).resolve(strict=True)

logs_dir = pathlib.Path(f"{root_dir}/3.cellprofiling/logs").resolve(strict=True)

MERGE_KEYS = ["object_id", "image_set"]
COMPARTMENTS = ["Organoid", "Nuclei", "Cell", "Cytoplasm", "Nucleocentric"]
SINGLE_CELL_COMPARTMENTS = ["Nuclei", "Cell", "Cytoplasm", "Nucleocentric"]
EXPECTED_N_FILES = 101
# object_id > 256 is the signature of the "z_slice_global" ID scheme (ids encode a
# z-slice offset), vs. small sequential ids (1, 2, 3, ...) from the "sequential" scheme.
# The two schemes are individually valid but must not be mixed within one compartment.
OBJECT_ID_SCHEME_THRESHOLD = 256

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
# 5. Within each compartment (including `Organoid`), group its files by their exact
#    `object_id` set. If more than one distinct set exists, the group whose files were
#    most **recently** written (max file mtime) is treated as the reference — not the
#    group with the most files — and every file in another group is flagged as needing a
#    rerun. File count is also checked as a secondary signal; when the two disagree
#    (`recency_vs_vote_conflict`), that's the clearest sign masks were re-segmented but
#    only some feature types were rerun against the new masks.
# 6. Compare object-ID sets across the single-cell compartments (`Nuclei`, `Cell`,
#    `Cytoplasm`, `Nucleocentric`) to catch alignment drift. `Organoid` is deliberately
#    excluded from this comparison — it is a separate feature space (one row per whole
#    organoid, not per cell) and its object-ID space is not expected to match the
#    single-cell compartments.

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


def process_well_fov(well_fov_dir: pathlib.Path) -> tuple[dict, list[dict], list[dict]]:
    """Attempt to merge one well-FOV's feature parquets per compartment.

    Returns a scalar summary row, a list of individual issue records, and a list of
    per-file "needs rerun" records (all empty/None where nothing went wrong).
    """
    well_fov = well_fov_dir.name
    files = sorted(well_fov_dir.glob("*.parquet"))
    issues = []
    rerun_rows = []

    def log_issue(kind, detail):
        issues.append({"well_fov": well_fov, "issue_type": kind, "detail": detail})

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
            )
        per_compartment_dfs[compartment].append((f.name, df, f.stat().st_mtime))

    compartment_shapes = {}
    for compartment, items in per_compartment_dfs.items():
        if not items:
            continue
        names = [n for n, _, _ in items]
        dfs = [d for _, d, _ in items]
        union_n_objects = len(set().union(*(set(d["object_id"]) for d in dfs)))
        try:
            merged = reduce(
                lambda left, right: pd.merge(left, right, on=MERGE_KEYS, how="outer"),
                dfs,
            )
        except Exception as e:
            log_issue("merge_error", f"{compartment} ({names}): {e}")
            continue
        compartment_shapes[compartment] = merged.shape
        if merged.shape[0] > union_n_objects:
            log_issue(
                "merge_blowup",
                f"{compartment}: merged to {merged.shape[0]} rows, but the union of "
                f"object_id values across its {len(dfs)} input files is only "
                f"{union_n_objects} — a merge key was duplicated somewhere",
            )

        # File-level check: within this compartment, do all files agree on the exact
        # set of object_ids? Group files by their object_id set. The group that was
        # *most recently regenerated* (max file mtime) is treated as authoritative —
        # not the group with the most files — because a mask re-segmentation followed
        # by a partial rerun leaves the newly-detected objects unfeaturized in whichever
        # feature files weren't rerun, and those stale files are very often still the
        # majority by count. Every file outside the most-recent group is flagged.
        id_groups = defaultdict(list)
        for name, df, mtime in items:
            id_groups[frozenset(df["object_id"])].append((name, mtime))
        groups = [
            {
                "ids": ids,
                "files": [n for n, _ in entries],
                "n_files": len(entries),
                "n_objects": len(ids),
                "max_mtime": max(m for _, m in entries),
            }
            for ids, entries in id_groups.items()
        ]
        if len(groups) > 1:
            groups_by_recency = sorted(groups, key=lambda g: -g["max_mtime"])
            reference = groups_by_recency[0]
            vote_majority = max(groups, key=lambda g: g["n_files"])
            heuristic_conflict = reference["files"] != vote_majority["files"]
            if heuristic_conflict:
                log_issue(
                    "recency_vs_vote_conflict",
                    f"{compartment}: the most-recently-regenerated file group "
                    f"({reference['n_files']} file(s), {reference['n_objects']} objects, "
                    f"last modified {datetime.fromtimestamp(reference['max_mtime'])}) is "
                    f"NOT the group with the most files ({vote_majority['n_files']} "
                    f"file(s), {vote_majority['n_objects']} objects, last modified "
                    f"{datetime.fromtimestamp(vote_majority['max_mtime'])}) — this usually "
                    f"means masks were re-segmented and only some feature types were "
                    f"rerun against the new masks",
                )
            for group in groups_by_recency[1:]:
                log_issue(
                    "file_object_id_group_split",
                    f"{compartment}: {sorted(group['files'])} have a "
                    f"{group['n_objects']}-object_id set that is stale relative to the "
                    f"most-recently-regenerated group of {reference['n_files']} file(s) "
                    f"with {reference['n_objects']} object_ids",
                )
                for name in group["files"]:
                    min_object_id = int(min(group["ids"]))
                    rerun_rows.append(
                        {
                            "well_fov": well_fov,
                            "compartment": compartment,
                            "file_name": name,
                            "group_n_files": group["n_files"],
                            "group_n_objects": group["n_objects"],
                            "group_last_modified": datetime.fromtimestamp(
                                group["max_mtime"]
                            ).isoformat(),
                            "reference_n_files": reference["n_files"],
                            "reference_n_objects": reference["n_objects"],
                            "reference_last_modified": datetime.fromtimestamp(
                                reference["max_mtime"]
                            ).isoformat(),
                            "vote_majority_would_pick_this_group": (
                                group["files"] == vote_majority["files"]
                            ),
                            "min_object_id": min_object_id,
                            "id_scheme_suspected": (
                                "z_slice_global"
                                if min_object_id > OBJECT_ID_SCHEME_THRESHOLD
                                else "sequential"
                            ),
                        }
                    )

    id_sets = {
        c: set(pd.concat([d for _, d, _ in per_compartment_dfs[c]])["object_id"])
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
                )

    summary = {
        "well_fov": well_fov,
        "n_files_found": len(files),
        "n_files_expected": EXPECTED_N_FILES,
        "file_count_mismatch": len(files) != EXPECTED_N_FILES,
        "n_issues": len(issues),
        "n_read_errors": sum(i["issue_type"] == "read_error" for i in issues),
        "n_merge_errors": sum(i["issue_type"] == "merge_error" for i in issues),
        "n_merge_blowups": sum(i["issue_type"] == "merge_blowup" for i in issues),
        "n_duplicate_object_id_files": sum(
            i["issue_type"] == "duplicate_object_id_in_file" for i in issues
        ),
        "n_files_to_rerun": len(rerun_rows),
        "files_to_rerun": sorted(r["file_name"] for r in rerun_rows),
        "n_recency_vote_conflicts": sum(
            i["issue_type"] == "recency_vs_vote_conflict" for i in issues
        ),
        "object_ids_aligned_across_compartments": aligned,
        "compartments_present": sorted(compartment_shapes.keys()),
        "compartment_row_counts": {c: s[0] for c, s in compartment_shapes.items()},
        "compartment_col_counts": {c: s[1] for c, s in compartment_shapes.items()},
    }
    return summary, issues, rerun_rows


# In[6]:


summaries = []
all_issues = []
all_rerun_rows = []
for well_fov_dir in tqdm.tqdm(well_fov_dirs, desc=f"Merging {patient} well-FOVs"):
    summary, issues, rerun_rows = process_well_fov(well_fov_dir)
    summaries.append(summary)
    all_issues.extend(issues)
    all_rerun_rows.extend(rerun_rows)

summary_df = pd.DataFrame(summaries)
issues_df = pd.DataFrame(all_issues, columns=["well_fov", "issue_type", "detail"])
rerun_df = pd.DataFrame(
    all_rerun_rows,
    columns=[
        "well_fov",
        "compartment",
        "file_name",
        "group_n_files",
        "group_n_objects",
        "group_last_modified",
        "reference_n_files",
        "reference_n_objects",
        "reference_last_modified",
        "vote_majority_would_pick_this_group",
        "min_object_id",
        "id_scheme_suspected",
    ],
)
print(
    f"{len(summary_df)} well-FOVs processed, {len(issues_df)} issues logged, "
    f"{len(rerun_df)} files flagged to rerun"
)


# ## Save the flattened per-well-FOV summary, the long-format issue log, and the rerun list

# In[7]:


summary_out_path = logs_dir / "well_fov_feature_merge_summary.csv"
issues_out_path = logs_dir / "well_fov_feature_merge_issues.csv"
rerun_out_path = logs_dir / "well_fov_files_to_rerun.csv"

csv_summary_df = summary_df.copy()
for col in [
    "compartments_present",
    "compartment_row_counts",
    "compartment_col_counts",
    "files_to_rerun",
]:
    csv_summary_df[col] = csv_summary_df[col].apply(json.dumps)
csv_summary_df.to_csv(summary_out_path, index=False)
issues_df.to_csv(issues_out_path, index=False)
rerun_df.to_csv(rerun_out_path, index=False)

print(f"Wrote {summary_out_path}")
print(f"Wrote {issues_out_path}")
print(f"Wrote {rerun_out_path}")


# ## Summarize findings

# In[8]:


n_well_fovs = len(summary_df)
n_file_count_mismatch = int(summary_df["file_count_mismatch"].sum())
n_with_issues = int((summary_df["n_issues"] > 0).sum())
n_misaligned = int(
    (summary_df["object_ids_aligned_across_compartments"] == False).sum()
)
n_read_errors = int(summary_df["n_read_errors"].sum())
n_merge_errors = int(summary_df["n_merge_errors"].sum())
n_merge_blowups = int(summary_df["n_merge_blowups"].sum())
n_well_fovs_with_files_to_rerun = int((summary_df["n_files_to_rerun"] > 0).sum())
n_well_fovs_with_recency_vote_conflicts = int(
    (summary_df["n_recency_vote_conflicts"] > 0).sum()
)

issue_type_counts = (
    issues_df["issue_type"].value_counts() if len(issues_df) else pd.Series(dtype=int)
)

print(f"Well-FOVs processed:               {n_well_fovs}")
print(f"Well-FOVs with >=1 issue:          {n_with_issues}")
print(f"Well-FOVs with file count mismatch:{n_file_count_mismatch}")
print(f"Well-FOVs with object-ID misalignment across compartments: {n_misaligned}")
print(
    f"Well-FOVs with specific files flagged to rerun: {n_well_fovs_with_files_to_rerun}"
)
print(f"Total individual files flagged to rerun: {len(rerun_df)}")
print(
    "Well-FOVs where file-count majority-vote would have picked the stale group: "
    f"{n_well_fovs_with_recency_vote_conflicts}"
)
print(f"Total read errors:                 {n_read_errors}")
print(f"Total merge errors:                {n_merge_errors}")
print(f"Total merge blow-ups:              {n_merge_blowups}")
print()
print("Issue counts by type:")
issue_type_counts


# In[9]:


summary_df.loc[
    summary_df["file_count_mismatch"], ["well_fov", "n_files_found", "n_files_expected"]
]


# ## Which specific files are out of alignment, per well-FOV
#
# For every well-FOV with a file-level object-ID group split, this lists the exact files
# that are stale relative to the most-recently-regenerated file group in their
# compartment — these are the files to rerun. The reference group is chosen by
# **recency (file mtime), not by file count**: a mask re-segmentation is often followed
# by only some feature types being rerun against the new masks, and the stale
# (not-yet-rerun) feature types are frequently still the majority by file count — using
# majority vote alone would silently keep the stale, partially-featurized group instead
# of flagging it.

# In[10]:


rerun_df.sort_values(["well_fov", "compartment", "file_name"]).reset_index(drop=True)


# ### Rerun-flagged files whose `object_id` minimum is greater than 256
#
# `object_id > 256` is the signature of the `z_slice_global` ID scheme; small sequential
# ids (starting at 1) are the `sequential` scheme. A file landing in the *minority* group
# here while using the `z_slice_global` scheme means it was extracted with a different
# (and in this compartment, non-consensus) ID scheme than the rest of that compartment's
# files for the same well-FOV.

# In[11]:


rerun_df.loc[
    rerun_df["min_object_id"] > OBJECT_ID_SCHEME_THRESHOLD,
    [
        "well_fov",
        "compartment",
        "file_name",
        "min_object_id",
        "group_n_objects",
        "reference_n_objects",
    ],
].sort_values(["well_fov", "compartment", "file_name"]).reset_index(drop=True)


# ### Files where the file-count heuristic would have picked wrong
#
# These are cases where the group with the most files is *not* the most recently
# regenerated group — i.e. picking a "reference" by majority vote would silently keep
# stale, partially-featurized results. `vote_majority_would_pick_this_group = True` means
# this specific flagged (stale) file's group is the one a naive majority-vote check would
# have kept as "correct".

# In[12]:


rerun_df.loc[
    rerun_df["vote_majority_would_pick_this_group"],
    [
        "well_fov",
        "compartment",
        "file_name",
        "group_n_files",
        "group_last_modified",
        "reference_n_files",
        "reference_last_modified",
    ],
].sort_values(["well_fov", "compartment", "file_name"]).reset_index(drop=True)


# In[13]:


summary_df.loc[
    summary_df["object_ids_aligned_across_compartments"] == False,
    ["well_fov", "compartments_present", "compartment_row_counts"],
]


# ## Generate the markdown report

# In[14]:


worst_offenders = summary_df.sort_values("n_issues", ascending=False).loc[
    summary_df["n_issues"] > 0,
    [
        "well_fov",
        "n_issues",
        "file_count_mismatch",
        "object_ids_aligned_across_compartments",
        "n_files_to_rerun",
    ],
]

z_slice_global_rerun_df = rerun_df.loc[
    rerun_df["min_object_id"] > OBJECT_ID_SCHEME_THRESHOLD
].sort_values(["well_fov", "compartment", "file_name"])

z_slice_global_section_lines = (
    z_slice_global_rerun_df[
        [
            "well_fov",
            "compartment",
            "file_name",
            "min_object_id",
            "group_n_objects",
            "reference_n_objects",
        ]
    ].to_markdown(index=False)
    if len(z_slice_global_rerun_df)
    else "_None._"
)

conflict_rerun_df = rerun_df.loc[
    rerun_df["vote_majority_would_pick_this_group"]
].sort_values(["well_fov", "compartment", "file_name"])
conflict_section_lines = (
    conflict_rerun_df[
        [
            "well_fov",
            "compartment",
            "file_name",
            "group_n_files",
            "group_last_modified",
            "reference_n_files",
            "reference_last_modified",
        ]
    ].to_markdown(index=False)
    if len(conflict_rerun_df)
    else "_None._"
)

rerun_section_lines = []
if len(rerun_df):
    for well_fov, group in rerun_df.sort_values(
        ["well_fov", "compartment", "file_name"]
    ).groupby("well_fov"):
        rerun_section_lines.append(f"### {well_fov}")
        rerun_section_lines.append("")
        rerun_section_lines.append(
            group[
                [
                    "compartment",
                    "file_name",
                    "group_n_objects",
                    "group_last_modified",
                    "reference_n_objects",
                    "reference_n_files",
                    "reference_last_modified",
                ]
            ].to_markdown(index=False)
        )
        rerun_section_lines.append("")
else:
    rerun_section_lines.append("_None._")

whole_compartment_section_lines = []
misaligned_well_fovs = summary_df.loc[
    summary_df["object_ids_aligned_across_compartments"] == False, "well_fov"
].tolist()
if misaligned_well_fovs:
    whole_compartment_section_lines.append(
        "These well-FOVs have at least one single-cell compartment (`Nuclei`, `Cell`, "
        "`Cytoplasm`, `Nucleocentric`) whose object-ID set as a whole does not match the "
        "others. Unlike the file-level splits above, this isn't pinned to one feature "
        "file — it points to re-running that compartment's segmentation/featurization "
        "step entirely for that well-FOV. See `well_fov_object_id_mismatches.csv` for "
        "which compartment pair(s) disagree and the object-ID scheme involved."
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
        f"- Well-FOVs with specific files flagged to rerun: **{n_well_fovs_with_files_to_rerun}**",
        f"- Total individual files flagged to rerun: **{len(rerun_df)}**",
        f"- Of those, files whose object_id minimum is > {OBJECT_ID_SCHEME_THRESHOLD} "
        f"(the `z_slice_global` scheme): **{len(z_slice_global_rerun_df)}**",
        f"- Well-FOVs where a naive file-count majority vote would have picked the "
        f"*stale* group instead of the most-recently-regenerated one: "
        f"**{n_well_fovs_with_recency_vote_conflicts}**",
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
        "## Files to rerun, by well-FOV",
        "",
        "Each file listed here is stale relative to the most-recently-regenerated ",
        "`object_id` set in its own compartment for that well-FOV — rerun the ",
        "feature-extraction step that produced it (see the ",
        "`{Channel}_{FeatureType}_{Processor}` in the file name).",
        "",
    ]
    + rerun_section_lines
    + [
        "## Rerun-flagged files using the `z_slice_global` ID scheme (object_id > "
        f"{OBJECT_ID_SCHEME_THRESHOLD})",
        "",
        "These are the subset of the files-to-rerun list above whose object_id minimum is ",
        f"greater than {OBJECT_ID_SCHEME_THRESHOLD}, i.e. they were extracted with the ",
        "`z_slice_global` ID scheme while the rest of their compartment's files use the ",
        "`sequential` scheme (or vice versa) — the two schemes must not be mixed.",
        "",
        z_slice_global_section_lines,
        "",
        "## Files where a file-count majority vote would have picked the stale group",
        "",
        "For these files, the group with the most files is *not* the most recently",
        'regenerated group. This is the clearest signature of "masks were re-segmented, ',
        'only some feature types were rerun": the stale (not-yet-rerun) feature type is',
        "still the majority by file count, so a plain majority-vote check would have kept",
        'it as "correct" and missed the unfeaturized objects entirely.',
        "",
        conflict_section_lines,
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
        "- A file-level group split means specific files within one compartment disagree on",
        "  the object-ID set; the most-recently-modified group is treated as correct and",
        "  every file in another group is listed in `well_fov_files_to_rerun.csv` and the",
        "  section above. File count is not used to pick the reference group — a file-count",
        "  majority vote would often pick the stale group, since only some feature types get",
        "  rerun after a mask re-segmentation (see the section on recency-vs-vote conflicts).",
        "- Object-ID misalignment across whole compartments is a coarser rollup of the same",
        "  condition tracked row-by-row in `well_fov_object_id_mismatches.csv`; see that file",
        "  for the exact compartment pairs and object-ID scheme involved.",
        "- Full detail for every individual issue is in `well_fov_feature_merge_issues.csv`.",
        f"- `well_fov_files_to_rerun.csv` includes a `min_object_id` and `id_scheme_suspected`",
        f"  column for every flagged file, so the `z_slice_global` (object_id > "
        f"{OBJECT_ID_SCHEME_THRESHOLD}) vs `sequential` split can be filtered directly.",
    ]
)

report_text = "\n".join(report_lines)
report_out_path = logs_dir / "well_fov_feature_merge_report.md"
report_out_path.write_text(report_text)
print(f"Wrote {report_out_path}")


# In[15]:


from IPython.display import Markdown, display

display(Markdown(report_text))


# In[26]:


# investigate the F6-2 well fov
files = list(
    pathlib.Path(
        "/home/lippincm/Documents/NF1_3D_organoid_profiling_pipeline/data/NF0014_T1/extracted_features/F6-2"
    ).glob("*.parquet")
)
files = [f for f in files if "Organoid" in f.name]
new_df = pd.DataFrame()
for f in files:
    df = pd.read_parquet(f)
    new_df = pd.merge(new_df, df, on=MERGE_KEYS, how="left") if not new_df.empty else df
new_df
