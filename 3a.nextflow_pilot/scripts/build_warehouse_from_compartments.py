#!/usr/bin/env python3
"""Build final pilot artifacts from per-compartment extraction outputs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import tifffile
from iceberg_warehouse import build_image_assets, publish_iceberg_warehouse
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


def load_image_set(
    manifest_path: Path, compartment_root: Path, run_id: str, git_revision: str
) -> dict[str, Any]:
    """Load one image set's artifacts, validate alignment, and build its assets."""
    manifest = load_manifest(manifest_path)
    path_errors = require_manifest_paths(manifest)
    if path_errors:
        raise SystemExit(f"{manifest_path}: " + "\n".join(path_errors))
    if "Metadata_Imaging_FieldID" in manifest:
        # A label, not a number -- but an unquoted numeric value in one
        # manifest's YAML (vs. a quoted string in another's) makes pandas
        # infer int64 for one image set's columns and object/str for
        # another's. Normalize at the source so every downstream use
        # (profile frames, image assets) is consistently typed before any
        # cross-image-set concatenation.
        manifest["Metadata_Imaging_FieldID"] = str(manifest["Metadata_Imaging_FieldID"])

    compartments = [
        str(compartment)
        for compartment in (manifest.get("compartments") or [manifest["compartment"]])
    ]
    unknown = sorted(set(compartments) - set(COMPARTMENTS))
    if unknown:
        raise SystemExit(
            f"Unsupported compartments in {manifest_path}: {', '.join(unknown)}"
        )

    profile_frames, validations, run_records = load_compartment_artifacts(
        compartment_root, compartments, manifest
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
        raise SystemExit(
            f"Channel and mask arrays are not aligned by shape for {manifest_path}:\n"
            f"{json.dumps(shape_check, indent=2)}"
        )

    image_assets = build_image_assets(manifest, images, masks, run_id, git_revision)
    first_record = next(iter(run_records.values()))

    compartment_records: dict[str, dict[str, Any]] = {}
    timings_seconds: dict[str, float] = {}
    quality_warnings = 0
    split_mode = any(
        record.get("mode") == "compartment_split" for record in run_records.values()
    )
    for compartment, record in run_records.items():
        timings_seconds.update(record.get("timings_seconds", {}))
        quality_warnings += int(record.get("quality_warning_count", 0))
        compartment_records[compartment] = {
            "elapsed_seconds": record.get("elapsed_seconds"),
            "validation_status": record.get("validation_status"),
            "quality_warning_count": record.get("quality_warning_count", 0),
            "outputs": record.get("outputs", {}),
        }

    return {
        "manifest_path": manifest_path,
        "compartment_root": compartment_root,
        "manifest": manifest,
        "compartments": compartments,
        "image_id": str(manifest["Metadata_Imaging_ImageID"]),
        "source_image_root": str(manifest.get("source_image_root", "")),
        "profile_frames": profile_frames,
        "validations": validations,
        "image_assets": image_assets,
        "alignment": shape_check,
        "zedprofiler_version": str(first_record["zedprofiler_version"]),
        "compartment_records": compartment_records,
        "timings_seconds": timings_seconds,
        "quality_warnings": quality_warnings,
        "split_mode": split_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--compartment-root", type=Path)
    parser.add_argument(
        "--image-set",
        nargs=2,
        metavar=("MANIFEST", "COMPARTMENT_ROOT"),
        action="append",
        dest="image_set",
        help="Repeatable. One (manifest, compartment-root) pair per image set. "
        "Mutually exclusive with --manifest/--compartment-root.",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    legacy_given = args.manifest is not None or args.compartment_root is not None
    if legacy_given and args.image_set:
        raise SystemExit(
            "Use either --manifest/--compartment-root or --image-set, not both"
        )
    if legacy_given:
        if args.manifest is None or args.compartment_root is None:
            raise SystemExit("--manifest and --compartment-root must both be given")
        specs = [(args.manifest, args.compartment_root)]
    elif args.image_set:
        specs = [(Path(m), Path(c)) for m, c in args.image_set]
    else:
        raise SystemExit(
            "Provide --manifest/--compartment-root or at least one --image-set"
        )

    started = time.perf_counter()
    git_revision = git_commit(args.repo_root)
    args.outdir.mkdir(parents=True, exist_ok=True)

    image_sets = [
        load_image_set(manifest_path, compartment_root, args.run_id, git_revision)
        for manifest_path, compartment_root in specs
    ]

    image_ids = [entry["image_id"] for entry in image_sets]
    duplicate_ids = sorted({i for i in image_ids if image_ids.count(i) > 1})
    if duplicate_ids:
        raise SystemExit(
            f"Duplicate Metadata_Imaging_ImageID across image sets: {duplicate_ids}"
        )

    compartments = image_sets[0]["compartments"]
    for entry in image_sets[1:]:
        if entry["compartments"] != compartments:
            raise SystemExit(
                "All image sets in one run must share the same compartments list: "
                f"{entry['manifest_path']} has {entry['compartments']}, "
                f"expected {compartments}"
            )

    zedprofiler_versions = sorted({entry["zedprofiler_version"] for entry in image_sets})
    if len(zedprofiler_versions) > 1:
        raise SystemExit(
            "zedprofiler_version differs across image sets in one run: "
            f"{zedprofiler_versions}"
        )
    zedprofiler_version = zedprofiler_versions[0]

    single = len(image_sets) == 1

    # Frames are kept as one-per-image-set lists all the way through, never
    # concatenated in this process's memory: peak memory stays bounded by
    # one image set's rows at a time instead of scaling with the total
    # number of image sets in a run. Each image set's frame becomes its own
    # parquet file below (a multi-file dataset directory) and its own
    # Iceberg data file (via write_table's per-frame append loop).
    profile_frames: dict[str, list[pd.DataFrame]] = {}
    for compartment in compartments:
        table_name = profile_table(compartment)
        frames = [entry["profile_frames"][table_name] for entry in image_sets]
        column_sets = {tuple(frame.columns) for frame in frames}
        if len(column_sets) > 1:
            raise SystemExit(
                f"Column mismatch across image sets for {table_name}: {column_sets}"
            )
        # A dtype mismatch in the same column across image sets (e.g. one
        # manifest's Metadata_Imaging_FieldID parsed as int64, another's as
        # str) still breaks this table even without concatenating, since
        # every appended frame must match the table's first-frame schema.
        # Fail loudly here with the offending column and per-image-set
        # dtypes named, rather than a cryptic PyArrow error deep inside
        # write_table.
        dtype_mismatches = sorted(
            column
            for column in frames[0].columns
            if len({str(frame[column].dtype) for frame in frames}) > 1
        )
        if dtype_mismatches:
            raise SystemExit(
                f"Dtype mismatch across image sets for {table_name}, "
                f"columns {dtype_mismatches}: "
                + ", ".join(
                    f"{entry['image_id']}="
                    f"{ {c: str(frame[c].dtype) for c in dtype_mismatches} }"
                    for entry, frame in zip(image_sets, frames)
                )
            )
        profile_frames[table_name] = frames

    image_assets = [entry["image_assets"] for entry in image_sets]

    source_roots = sorted({entry["source_image_root"] for entry in image_sets})
    source_image_root = source_roots[0] if len(source_roots) == 1 else ",".join(source_roots)

    validations_for_tables = {
        compartment: {
            "valid": all(
                entry["validations"][compartment]["valid"] for entry in image_sets
            ),
            "row_count": sum(
                entry["validations"][compartment]["row_count"] for entry in image_sets
            ),
            "column_count": image_sets[0]["validations"][compartment]["column_count"],
        }
        for compartment in compartments
    }

    top_level_outputs: dict[str, str] = {}
    output_tables: list[dict[str, Any]] = []
    for compartment in compartments:
        table_name = profile_table(compartment)
        slug = compartment_slug(compartment)
        target = args.outdir / f"{slug}_profiles"
        target.mkdir(parents=True, exist_ok=True)
        for entry, frame in zip(image_sets, profile_frames[table_name]):
            frame.to_parquet(target / f"{entry['image_id']}.parquet", index=False)
        validation = validations_for_tables[compartment]
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
                "source_image_root": source_image_root,
                "run_id": args.run_id,
                "git_commit": git_revision,
                "row_count": validation["row_count"],
                "column_count": validation["column_count"],
                "validation_status": "pass" if validation["valid"] else "fail",
            }
        )

    alignment = (
        image_sets[0]["alignment"]
        if single
        else {entry["image_id"]: entry["alignment"] for entry in image_sets}
    )

    warehouse_manifest = publish_iceberg_warehouse(
        outdir=args.outdir,
        run_id=args.run_id,
        git_commit=git_revision,
        image_assets=image_assets,
        profile_frames=profile_frames,
        validations=validations_for_tables,
        alignment=alignment,
        zedprofiler_version=zedprofiler_version,
        source_image_root=source_image_root,
    )

    all_valid = all(validation["valid"] for validation in validations_for_tables.values())
    total_quality_warnings = sum(entry["quality_warnings"] for entry in image_sets)
    workflow_slurm_jobs_expected = (
        sum(
            (
                len(entry["compartments"]) * len(CHANNELS) + len(entry["compartments"])
                if entry["split_mode"]
                else len(entry["compartments"])
            )
            for entry in image_sets
        )
        + 2
    )
    split_mode = any(entry["split_mode"] for entry in image_sets)

    run_record: dict[str, Any] = {
        "run_id": args.run_id,
        "command": " ".join(sys.argv),
        "mode": "granularity_channel_fanout"
        if split_mode
        else "per_compartment_fanout",
        "workflow_slurm_jobs_expected": workflow_slurm_jobs_expected,
        "git_commit": git_revision,
        "zedprofiler_version": zedprofiler_version,
        "python_version": platform.python_version(),
        "alignment": alignment,
        "outputs": top_level_outputs,
        "tables": output_tables,
        "warehouse_root": warehouse_manifest["warehouse_root"],
        "warehouse_manifest": str(args.outdir / "warehouse" / "warehouse_manifest.json"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "exit_status": 0 if all_valid else 1,
        "validation_status": "pass" if all_valid else "fail",
        "quality_warning_count": total_quality_warnings,
    }
    if single:
        entry = image_sets[0]
        run_record["manifest"] = entry["manifest"]
        run_record["timings_seconds"] = entry["timings_seconds"]
        run_record["compartment_records"] = entry["compartment_records"]
        validation_report = {
            "valid": all_valid,
            "alignment": alignment,
            "compartments": entry["validations"],
            "warehouse": {
                "valid": warehouse_manifest["validation_status"] == "pass",
                "manifest": str(args.outdir / "warehouse" / "warehouse_manifest.json"),
                "namespaces": warehouse_manifest["namespaces"],
                "tables": [table["table_name"] for table in warehouse_manifest["tables"]],
            },
        }
    else:
        run_record["image_sets"] = {
            entry["image_id"]: entry["manifest"] for entry in image_sets
        }
        run_record["timings_seconds"] = {
            entry["image_id"]: entry["timings_seconds"] for entry in image_sets
        }
        run_record["compartment_records"] = {
            entry["image_id"]: entry["compartment_records"] for entry in image_sets
        }
        validation_report = {
            "valid": all_valid,
            "image_sets": {
                entry["image_id"]: {
                    "alignment": entry["alignment"],
                    "compartments": entry["validations"],
                }
                for entry in image_sets
            },
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
        f"tables={len(output_tables)} image_sets={len(image_sets)} "
        f"elapsed={run_record['elapsed_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
