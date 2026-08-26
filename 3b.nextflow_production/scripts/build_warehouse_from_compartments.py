#!/usr/bin/env python3
"""Validate and summarize one run's already-landed profile/image-asset
datasets.

Each image-set task (scripts/run_zedprofiler_image_set.py) writes its
finished profile parquet *and* its images.image_assets row directly into
this run's shared namespaced directories (warehouse/profiles/<table>/,
warehouse/images/image_assets/) -- there is no separate collection or
registration step, and no file is ever written twice, and nothing needs to
move to end up with a complete warehouse/ directory once every task lands.
That directory holds data files only; each task's validation.json and
run_record.json sidecars land under a parallel metadata/ tree instead,
mirroring the same relative path. This script's only job is to confirm
every expected file landed, cheaply cross-check schemas (metadata only, no
data materialized), aggregate validation status, write one run-level
summary (also under metadata/), and refresh warehouse/warehouse.duckdb, a
small view catalog over warehouse/ that stays portable with it (see
build_duckdb_views.py).

Takes either --manifest (repeatable, one YAML path per image set) or
--image-sets-index/--source-root (a CSV of patient,well_fov rows, each
manifest derived on the fly the same way FEATURIZE_IMAGE_SET does) -- see
build_manifest.py's resolve_manifest()/read_image_sets_index().
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from build_duckdb_views import build_views
from build_manifest import read_image_sets_index, resolve_manifest
from manifest_io import require_manifest_paths
from run_zedprofiler_image_set import (
    COMPARTMENTS,
    compartment_slug,
    git_commit,
    profile_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        dest="manifests",
        help="Repeatable. One manifest per image set landed in this run. "
        "Mutually exclusive with --image-sets-index.",
    )
    parser.add_argument(
        "--image-sets-index",
        type=Path,
        help="CSV of patient,well_fov rows -- one per image set landed in "
        "this run. Each manifest is derived on the fly via build_manifest(), "
        "same as FEATURIZE_IMAGE_SET does. Requires --source-root.",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    if bool(args.manifests) == bool(args.image_sets_index):
        raise SystemExit("Provide exactly one of --manifest (repeatable) or --image-sets-index")

    started = time.perf_counter()
    git_revision = git_commit(args.repo_root)

    manifests: list[tuple[str, dict[str, Any]]] = []
    if args.image_sets_index:
        if not args.source_root:
            raise SystemExit("--source-root is required with --image-sets-index")
        for patient, well_fov in read_image_sets_index(args.image_sets_index):
            label = f"{patient}/{well_fov}"
            manifest = resolve_manifest(None, patient, well_fov, args.source_root)
            path_errors = require_manifest_paths(manifest)
            if path_errors:
                raise SystemExit(f"{label}: " + "\n".join(path_errors))
            manifests.append((label, manifest))
    else:
        for manifest_path in args.manifests:
            manifest = resolve_manifest(manifest_path, None, None, None)
            path_errors = require_manifest_paths(manifest)
            if path_errors:
                raise SystemExit(f"{manifest_path}: " + "\n".join(path_errors))
            manifests.append((str(manifest_path), manifest))

    all_image_ids = [str(manifest["Metadata_Imaging_ImageID"]) for _, manifest in manifests]
    duplicate_ids = sorted({i for i in all_image_ids if all_image_ids.count(i) > 1})
    if duplicate_ids:
        raise SystemExit(
            f"Duplicate Metadata_Imaging_ImageID across image sets: {duplicate_ids}"
        )

    compartments = [
        str(c)
        for c in (manifests[0][1].get("compartments") or [manifests[0][1]["compartment"]])
    ]
    unknown = sorted(set(compartments) - set(COMPARTMENTS))
    if unknown:
        raise SystemExit(f"Unsupported compartments: {', '.join(unknown)}")
    for label, manifest in manifests[1:]:
        entry_compartments = [
            str(c) for c in (manifest.get("compartments") or [manifest["compartment"]])
        ]
        if entry_compartments != compartments:
            raise SystemExit(
                "All image sets in one run must share the same compartments list: "
                f"{label} has {entry_compartments}, expected {compartments}"
            )

    # Not every image set passed in necessarily landed output: FEATURIZE_
    # IMAGE_SET's errorStrategy 'ignore's a task that exhausts its retries
    # (e.g. a genuine source-data problem like a channel/mask z-slice
    # mismatch -- see conf/base.config), so it writes nothing for that image
    # set at all. Previously this hard-failed the whole warehouse build the
    # moment BUILD_WAREHOUSE hit one, discarding every other image set's
    # already-landed, valid output in the process. Determine landed-ness
    # upfront (same completeness definition plan_image_sets.py already
    # uses: images.image_assets plus every compartment's parquet, all
    # present) and only build the warehouse from what's actually there --
    # this stays a hard failure if *nothing* landed, since an empty
    # warehouse from an otherwise-populated index is a real problem, not a
    # partial-success case.
    assets_dir = args.outdir / "warehouse" / "images" / "image_assets"
    assets_metadata_dir = args.outdir / "metadata" / "images" / "image_assets"

    def is_landed(image_id: str) -> bool:
        if not (assets_dir / f"{image_id}.parquet").exists():
            return False
        if not (assets_metadata_dir / f"{image_id}.validation.json").exists():
            return False
        for compartment in compartments:
            table_slug = f"{compartment_slug(compartment)}_profiles"
            if not (
                args.outdir / "warehouse" / "profiles" / table_slug / f"{image_id}.parquet"
            ).exists():
                return False
            if not (
                args.outdir
                / "metadata"
                / "profiles"
                / table_slug
                / f"{image_id}.validation.json"
            ).exists():
                return False
        return True

    image_ids = [i for i in all_image_ids if is_landed(i)]
    skipped_image_ids = [i for i in all_image_ids if i not in set(image_ids)]
    if not image_ids:
        raise SystemExit(
            f"No image set landed complete output out of {len(all_image_ids)} passed in -- "
            "nothing to build a warehouse from. Check FEATURIZE_IMAGE_SET's own logs, not "
            "just this task's."
        )
    if skipped_image_ids:
        print(
            f"NF1_BUILD_WAREHOUSE skipping {len(skipped_image_ids)} image set(s) with no "
            f"landed output: {skipped_image_ids}",
            file=sys.stderr,
        )

    all_valid = True
    total_quality_warnings = 0
    output_tables: list[dict[str, Any]] = []

    for compartment in compartments:
        table_name = profile_table(compartment)
        slug = compartment_slug(compartment)
        table_slug = f"{slug}_profiles"
        target_dir = args.outdir / "warehouse" / "profiles" / table_slug
        compartment_metadata_dir = args.outdir / "metadata" / "profiles" / table_slug
        table_valid = True
        row_count = 0
        column_count = 0
        reference_types: dict[str, Any] = {}
        for image_id in image_ids:
            parquet_path = target_dir / f"{image_id}.parquet"
            validation_path = compartment_metadata_dir / f"{image_id}.validation.json"
            # No existence check here: image_ids was already filtered to
            # is_landed() image sets above, which requires this exact file.
            validation = json.loads(validation_path.read_text())["compartments"][
                compartment
            ]
            schema = pq.read_schema(parquet_path)
            field_types = {field.name: field.type for field in schema}
            # A file having fewer/more columns than another is expected now
            # (a compartment with zero detected objects for this image set
            # drops feature families that can't compute on zero objects
            # rather than padding with nulls -- see merge_feature_frames()),
            # and build_duckdb_views.py's union_by_name handles that at
            # query time. Zero-row files are skipped for type-conflict
            # purposes entirely: pandas/pyarrow infer an empty/all-NaN
            # column as double regardless of what a populated column of the
            # same name would be (observed: int64 columns like
            # Metadata_Object_ObjectID coming back double on a 0-object
            # file) -- with no actual values in that file, the type is
            # meaningless, not a real conflict, and DuckDB's union_by_name
            # widens int64/double for the same column name automatically.
            # Only a *populated* file disagreeing with another populated
            # file's type for the same column name is a genuine problem.
            if int(validation["row_count"]) > 0:
                conflicts = sorted(
                    name
                    for name, dtype in field_types.items()
                    if name in reference_types and reference_types[name] != dtype
                )
                if conflicts:
                    raise SystemExit(
                        f"Type mismatch for {table_name}/{image_id}, column(s) {conflicts}: "
                        "doesn't match the type already seen for these columns elsewhere "
                        f"in this table. Compare {parquet_path} against other files in "
                        f"{target_dir}."
                    )
                reference_types.update(field_types)
            table_valid = table_valid and bool(validation["valid"])
            row_count += int(validation["row_count"])
            column_count = max(column_count, int(validation["column_count"]))
            total_quality_warnings += len(validation.get("quality_warnings", []))
        all_valid = all_valid and table_valid
        output_tables.append(
            {
                "name": table_name,
                "path": str(target_dir),
                "join_keys": [
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Imaging_ImageID",
                    "Metadata_Compartment",
                    "Metadata_Object_ObjectID",
                ],
                "row_count": row_count,
                "column_count": column_count,
                "validation_status": "pass" if table_valid else "fail",
            }
        )

    assets_valid = True
    assets_row_count = 0
    for image_id in image_ids:
        parquet_path = assets_dir / f"{image_id}.parquet"
        validation_path = assets_metadata_dir / f"{image_id}.validation.json"
        if not parquet_path.exists() or not validation_path.exists():
            raise SystemExit(
                f"Missing image assets for {image_id}: expected {parquet_path} "
                f"and {validation_path} (FEATURIZE_IMAGE_SET output)"
            )
        validation = json.loads(validation_path.read_text())
        assets_valid = assets_valid and bool(validation["valid"])
        assets_row_count += int(validation["row_count"])
    all_valid = all_valid and assets_valid

    # Upper bound, not an exact count: one FEATURIZE_IMAGE_SET task per image
    # set (all compartments and images.image_assets handled in that one task
    # now), plus PLAN_IMAGE_SETS + BUILD_WAREHOUSE + coordinator. PLAN_IMAGE_
    # SETS skips FEATURIZE_IMAGE_SET for any image set whose warehouse output
    # already exists and validated, so the real count for a rerun over a
    # partially- or fully-complete warehouse is lower than this -- see that
    # task's own stderr summary (NF1_PLAN_IMAGE_SETS done=.../pending=...)
    # for what actually ran in a given invocation.
    workflow_slurm_jobs_expected = len(manifests) + 3

    # Views only, no data copied -- cheap to refresh here since this task
    # already has every table's path in hand from the scan above, at no
    # extra Slurm-job cost. Lives inside warehouse/ itself, with relative
    # view paths, so the whole warehouse/ directory stays portable as one
    # self-contained unit.
    warehouse_dir = args.outdir / "warehouse"
    db_path = warehouse_dir / "warehouse.duckdb"
    build_views(warehouse_dir, db_path)

    run_record: dict[str, Any] = {
        "run_id": args.run_id,
        "command": " ".join(sys.argv),
        "mode": "granularity_channel_fanout",
        "workflow_slurm_jobs_expected": workflow_slurm_jobs_expected,
        "git_commit": git_revision,
        "python_version": platform.python_version(),
        "image_sets": image_ids,
        "skipped_image_sets": skipped_image_ids,
        "outdir": str(args.outdir),
        "warehouse_dir": str(warehouse_dir),
        "tables": output_tables,
        "images.image_assets": {
            "path": str(assets_dir),
            "row_count": assets_row_count,
            "validation_status": "pass" if assets_valid else "fail",
        },
        "duckdb": str(db_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "exit_status": 0 if all_valid else 1,
        "validation_status": "pass" if all_valid else "fail",
        "quality_warning_count": total_quality_warnings,
    }
    validation_report: dict[str, Any] = {
        "valid": all_valid,
        "tables": output_tables,
        "images.image_assets": {
            "valid": assets_valid,
            "row_count": assets_row_count,
        },
        "skipped_image_sets": skipped_image_ids,
    }

    metadata_dir = args.outdir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    (metadata_dir / "validation.json").write_text(json.dumps(validation_report, indent=2))

    if not all_valid:
        print(json.dumps(validation_report, indent=2), file=sys.stderr)
        return 1
    print(
        "NF1_ZEDPROFILER_WAREHOUSE_OK "
        f"tables={len(output_tables)} image_sets={len(image_ids)} "
        f"skipped={len(skipped_image_ids)} "
        f"elapsed={run_record['elapsed_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
