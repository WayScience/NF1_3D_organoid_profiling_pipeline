#!/usr/bin/env python3
"""Validate and summarize one run's already-landed profile/image-asset
datasets.

Each (image set, compartment) task (scripts/run_zedprofiler_image_set.py)
and each image set's image-assets build (scripts/build_image_assets.py)
write their finished parquet directly into this run's shared namespaced
directories (warehouse/profiles/<table>/, warehouse/images/image_assets/)
-- there is no separate collection or registration step, and no file is
ever written twice, and nothing needs to move to end up with a complete
warehouse/ directory once every task lands. That directory holds data files
only; each task's validation.json and run_record.json sidecars land under a
parallel metadata/ tree instead, mirroring the same relative path. This
script's only job is to confirm every expected file landed, cheaply
cross-check schemas (metadata only, no data materialized), aggregate
validation status, write one run-level summary (also under metadata/), and
refresh warehouse/warehouse.duckdb, a small view catalog over warehouse/
that stays portable with it (see build_duckdb_views.py).
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
from manifest_io import load_manifest, require_manifest_paths
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
        required=True,
        dest="manifests",
        help="Repeatable. One manifest per image set landed in this run.",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    git_revision = git_commit(args.repo_root)

    manifests: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in args.manifests:
        manifest = load_manifest(manifest_path)
        path_errors = require_manifest_paths(manifest)
        if path_errors:
            raise SystemExit(f"{manifest_path}: " + "\n".join(path_errors))
        manifests.append((manifest_path, manifest))

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
    for manifest_path, manifest in manifests[1:]:
        entry_compartments = [
            str(c) for c in (manifest.get("compartments") or [manifest["compartment"]])
        ]
        if entry_compartments != compartments:
            raise SystemExit(
                "All image sets in one run must share the same compartments list: "
                f"{manifest_path} has {entry_compartments}, expected {compartments}"
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
                f"and {validation_path} (BUILD_IMAGE_ASSETS output)"
            )
        validation = json.loads(validation_path.read_text())
        assets_valid = assets_valid and bool(validation["valid"])
        assets_row_count += int(validation["row_count"])
    all_valid = all_valid and assets_valid

    # One FEATURIZE_IMAGE_SET task and one BUILD_IMAGE_ASSETS task per image
    # set (all compartments handled in one task now), plus BUILD_WAREHOUSE +
    # coordinator.
    workflow_slurm_jobs_expected = len(manifests) * 2 + 2

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

    if not all_valid:
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
