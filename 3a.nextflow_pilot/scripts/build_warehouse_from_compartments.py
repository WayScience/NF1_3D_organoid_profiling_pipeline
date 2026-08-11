#!/usr/bin/env python3
"""Build final pilot artifacts from per-compartment extraction outputs."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import tifffile
from iceberg_warehouse import publish_iceberg_warehouse
from manifest_io import load_manifest, require_manifest_paths
from run_zedprofiler_image_set import (
    CHANNELS,
    COMPARTMENTS,
    add_metadata,
    add_segmentation_metadata,
    compartment_slug,
    git_commit,
    merge_feature_frames,
    profile_table,
    validate_aligned_shapes,
    validate_profiles,
)


def tiff_proxy(path: str) -> SimpleNamespace:
    """Return the image metadata needed by the warehouse image-assets table."""
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = series.shape
        dtype = series.dtype
    return SimpleNamespace(shape=tuple(shape), dtype=dtype)


def load_compartment_artifacts(
    compartment_root: Path, compartments: list[str], manifest: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, Any]]:
    profile_frames: dict[str, pd.DataFrame] = {}
    validations: dict[str, dict[str, Any]] = {}
    run_records: dict[str, Any] = {}

    for compartment in compartments:
        slug = compartment_slug(compartment)
        compartment_dir = compartment_root / slug
        profile_path = compartment_dir / f"{slug}_profiles.parquet"
        validation_path = compartment_dir / "validation.json"
        run_record_path = compartment_dir / "run_record.json"

        if not profile_path.exists():
            nongranularity_dir = compartment_dir / "nongranularity"
            nongranularity_path = (
                nongranularity_dir / f"{slug}_nongranularity_profiles.parquet"
            )
            nongranularity_validation_path = nongranularity_dir / "validation.json"
            nongranularity_run_record_path = nongranularity_dir / "run_record.json"
            granularity_paths = [
                compartment_dir
                / "granularity"
                / channel.lower()
                / f"{slug}_{channel.lower()}_granularity.parquet"
                for channel in CHANNELS
            ]
            granularity_validation_paths = [
                path.parent / "validation.json" for path in granularity_paths
            ]
            granularity_run_record_paths = [
                path.parent / "run_record.json" for path in granularity_paths
            ]
            missing = [
                str(path)
                for path in (
                    [nongranularity_path, nongranularity_validation_path]
                    + granularity_paths
                    + granularity_validation_paths
                )
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing split artifacts for {compartment}: "
                    f"{', '.join(missing)}"
                )

            frames = [pd.read_parquet(nongranularity_path)]
            frames.extend(pd.read_parquet(path) for path in granularity_paths)
            profiles = add_metadata(merge_feature_frames(frames), manifest)
            profiles.insert(0, "Metadata_Compartment", compartment)
            profiles = add_segmentation_metadata(profiles, manifest, compartment)

            mask = tifffile.imread(str(manifest["mask_paths"][compartment]))
            validation = validate_profiles(profiles, mask, manifest)
            partial_validations = [
                json.loads(nongranularity_validation_path.read_text())[
                    "compartments"
                ][compartment]
            ]
            partial_validations.extend(
                json.loads(path.read_text())["compartments"][compartment]
                for path in granularity_validation_paths
            )
            validation["compartment"] = compartment
            validation["quality_warnings"] = [
                warning
                for partial in partial_validations
                for warning in partial.get("quality_warnings", [])
            ]
            partial_records = {
                "nongranularity": json.loads(
                    nongranularity_run_record_path.read_text()
                )
            }
            for path in granularity_run_record_paths:
                if path.exists():
                    record = json.loads(path.read_text())
                    partial_records[f"granularity_{record.get('channel', '')}"] = record

            profile_frames[profile_table(compartment)] = profiles
            validations[compartment] = validation
            run_records[compartment] = {
                "mode": "compartment_split",
                "zedprofiler_version": partial_records["nongranularity"][
                    "zedprofiler_version"
                ],
                "elapsed_seconds": sum(
                    float(record.get("elapsed_seconds") or 0)
                    for record in partial_records.values()
                ),
                "validation_status": "pass" if validation["valid"] else "fail",
                "quality_warning_count": len(validation["quality_warnings"]),
                "timings_seconds": {
                    key: value
                    for record in partial_records.values()
                    for key, value in record.get("timings_seconds", {}).items()
                },
                "outputs": {
                    key: value
                    for record in partial_records.values()
                    for key, value in record.get("outputs", {}).items()
                },
                "partial_records": partial_records,
            }
            continue

        missing = [
            str(path)
            for path in (profile_path, validation_path, run_record_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts for {compartment}: {', '.join(missing)}"
            )

        profile_frames[profile_table(compartment)] = pd.read_parquet(profile_path)
        validation_report = json.loads(validation_path.read_text())
        run_records[compartment] = json.loads(run_record_path.read_text())
        validations[compartment] = validation_report["compartments"][compartment]

    return profile_frames, validations, run_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--compartment-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = load_manifest(args.manifest)
    path_errors = require_manifest_paths(manifest)
    if path_errors:
        raise SystemExit("\n".join(path_errors))

    args.outdir.mkdir(parents=True, exist_ok=True)
    compartments = [
        str(compartment)
        for compartment in (manifest.get("compartments") or [manifest["compartment"]])
    ]
    unknown = sorted(set(compartments) - set(COMPARTMENTS))
    if unknown:
        raise SystemExit(f"Unsupported compartments: {', '.join(unknown)}")

    profile_frames, validations, run_records = load_compartment_artifacts(
        args.compartment_root, compartments, manifest
    )

    top_level_outputs: dict[str, str] = {}
    output_tables: list[dict[str, Any]] = []
    for compartment in compartments:
        table_name = profile_table(compartment)
        slug = compartment_slug(compartment)
        target = args.outdir / f"{slug}_profiles.parquet"
        source = args.compartment_root / slug / f"{slug}_profiles.parquet"
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        elif not source.exists():
            profile_frames[table_name].to_parquet(target, index=False)
        validation = validations[compartment]
        top_level_outputs[table_name] = str(target)
        output_tables.append(
            {
                "name": table_name,
                "path": str(target),
                "schema_version": "0.1.0-pilot",
                "join_keys": [
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Imaging_ImageID",
                    "Metadata_Compartment",
                    "Metadata_Object_ObjectID",
                ],
                "source_image_root": manifest.get("source_image_root", ""),
                "run_id": args.run_id,
                "git_commit": git_commit(args.repo_root),
                "row_count": validation["row_count"],
                "column_count": validation["column_count"],
                "validation_status": "pass" if validation["valid"] else "fail",
            }
        )

    images = {
        channel: tiff_proxy(str(manifest["channel_paths"][channel]))
        for channel in CHANNELS
    }
    masks = {
        compartment: tiff_proxy(str(manifest["mask_paths"][compartment]))
        for compartment in compartments
    }
    shape_check = validate_aligned_shapes(images, masks)
    if not shape_check["valid"]:
        (args.outdir / "alignment_validation.json").write_text(
            json.dumps(shape_check, indent=2)
        )
        raise SystemExit(
            "Channel and mask arrays are not aligned by shape; see "
            f"{args.outdir / 'alignment_validation.json'}"
        )
    alignment = shape_check

    git_revision = git_commit(args.repo_root)
    first_record = next(iter(run_records.values()))
    warehouse_manifest = publish_iceberg_warehouse(
        outdir=args.outdir,
        run_id=args.run_id,
        git_commit=git_revision,
        manifest=manifest,
        profile_frames=profile_frames,
        images=images,
        masks=masks,
        validations=validations,
        alignment=alignment,
        zedprofiler_version=str(first_record["zedprofiler_version"]),
    )

    all_valid = all(validation["valid"] for validation in validations.values())
    all_timings: dict[str, float] = {}
    total_quality_warnings = 0
    compartment_records: dict[str, dict[str, Any]] = {}
    split_mode = any(
        record.get("mode") == "compartment_split" for record in run_records.values()
    )
    for compartment, record in run_records.items():
        all_timings.update(record.get("timings_seconds", {}))
        total_quality_warnings += int(record.get("quality_warning_count", 0))
        compartment_records[compartment] = {
            "elapsed_seconds": record.get("elapsed_seconds"),
            "validation_status": record.get("validation_status"),
            "quality_warning_count": record.get("quality_warning_count", 0),
            "outputs": record.get("outputs", {}),
        }

    run_record = {
        "run_id": args.run_id,
        "command": " ".join(sys.argv),
        "mode": "granularity_channel_fanout"
        if split_mode
        else "per_compartment_fanout",
        "workflow_slurm_jobs_expected": (
            len(compartments) * len(CHANNELS) + len(compartments) + 2
            if split_mode
            else len(compartments) + 2
        ),
        "git_commit": git_revision,
        "zedprofiler_version": first_record["zedprofiler_version"],
        "python_version": platform.python_version(),
        "manifest": manifest,
        "alignment": alignment,
        "outputs": top_level_outputs,
        "tables": output_tables,
        "warehouse_root": warehouse_manifest["warehouse_root"],
        "warehouse_manifest": str(args.outdir / "warehouse" / "warehouse_manifest.json"),
        "timings_seconds": all_timings,
        "compartment_records": compartment_records,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "exit_status": 0 if all_valid else 1,
        "validation_status": "pass" if all_valid else "fail",
        "quality_warning_count": total_quality_warnings,
    }
    validation_report = {
        "valid": all_valid,
        "alignment": alignment,
        "compartments": validations,
        "warehouse": {
            "valid": warehouse_manifest["validation_status"] == "pass",
            "manifest": str(args.outdir / "warehouse" / "warehouse_manifest.json"),
            "namespaces": warehouse_manifest["namespaces"],
            "tables": [table["table_name"] for table in warehouse_manifest["tables"]],
        },
    }

    (args.outdir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    (args.outdir / "validation.json").write_text(json.dumps(validation_report, indent=2))
    (args.outdir / "alignment_validation.json").write_text(json.dumps(alignment, indent=2))

    if not all_valid:
        print(json.dumps(validation_report, indent=2), file=sys.stderr)
        return 1
    print(
        "NF1_ZEDPROFILER_FANOUT_WAREHOUSE_OK "
        f"tables={len(output_tables)} elapsed={run_record['elapsed_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
