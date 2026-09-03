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
import subprocess
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

    image_ids = [str(manifest["Metadata_Imaging_ImageID"]) for _, manifest in manifests]
    duplicate_ids = sorted({i for i in image_ids if image_ids.count(i) > 1})
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
        reference_schema = None
        reference_image_id = None
        for image_id in image_ids:
            parquet_path = target_dir / f"{image_id}.parquet"
            validation_path = compartment_metadata_dir / f"{image_id}.validation.json"
            if not parquet_path.exists() or not validation_path.exists():
                raise SystemExit(
                    f"Missing merged artifact for {compartment}/{image_id}: expected "
                    f"{parquet_path} and {validation_path} (FEATURIZE_IMAGE_SET output)"
                )
            schema = pq.read_schema(parquet_path)
            if reference_schema is None:
                reference_schema, reference_image_id = schema, image_id
            elif schema != reference_schema:
                raise SystemExit(
                    f"Schema mismatch for {table_name}: {image_id} does not match "
                    f"{reference_image_id}. Compare {parquet_path} against "
                    f"{target_dir / f'{reference_image_id}.parquet'}"
                )
            validation = json.loads(validation_path.read_text())["compartments"][
                compartment
            ]
            table_valid = table_valid and bool(validation["valid"])
            row_count += int(validation["row_count"])
            column_count = int(validation["column_count"])
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

    assets_dir = args.outdir / "warehouse" / "images" / "image_assets"
    assets_metadata_dir = args.outdir / "metadata" / "images" / "image_assets"
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
    }

    metadata_dir = args.outdir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    (metadata_dir / "validation.json").write_text(json.dumps(validation_report, indent=2))

    # Last step of the pipeline, run regardless of all_valid: koala is a
    # shared group allocation, and every file/dir under outdir needs to
    # stay group-accessible for others on the project, not just readable by
    # whoever's Slurm job happened to create it. beforeScript = 'umask 007'
    # (conf/base.config) gets new files/dirs to 660/770 as they're created,
    # but umask can never grant a file's execute bit, so a run that started
    # before this fix -- or any file a umask gap slipped through -- would
    # still land short of the 770 this project actually wants everywhere.
    # This sweep is the single point that guarantees it, independent of
    # umask working correctly at every creation site. check=False because a
    # permissions failure shouldn't mask a real validation failure below --
    # but it must still fail the run, not be silently swallowed, so the
    # return code is inspected explicitly instead.
    chmod_result = subprocess.run(
        ["chmod", "-R", "770", str(args.outdir)], check=False
    )
    if chmod_result.returncode != 0:
        print(
            f"WARNING: chmod -R 770 {args.outdir} exited "
            f"{chmod_result.returncode} -- some files/dirs under this run "
            "may not be group-writable on koala",
            file=sys.stderr,
        )

    if not all_valid or chmod_result.returncode != 0:
        print(json.dumps(validation_report, indent=2), file=sys.stderr)
        return 1
    print(
        "NF1_ZEDPROFILER_WAREHOUSE_OK "
        f"tables={len(output_tables)} image_sets={len(image_ids)} "
        f"elapsed={run_record['elapsed_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
