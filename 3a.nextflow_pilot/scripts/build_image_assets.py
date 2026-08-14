#!/usr/bin/env python3
"""Build one image set's images.image_assets rows and check whole-image-set
channel/mask alignment, writing directly into this run's shared warehouse
directory.

Runs as its own parallel Nextflow task per image set, independent of the
compartment/channel feature fan-out -- it only needs the manifest, not any
feature-extraction output, so it can run immediately alongside everything
else. Reads TIFF headers only (no pixel data) for all 4 channels and all
compartment masks, since that's the same set of files a whole-image-set
alignment check needs; doing both here means no other task has to open
these files again for that purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import tifffile
from manifest_io import load_manifest, require_manifest_paths
from run_zedprofiler_image_set import (
    CHANNELS,
    COMPARTMENTS,
    git_commit,
    validate_aligned_shapes,
)


def tiff_proxy(path: str) -> SimpleNamespace:
    """Return the image metadata needed by the image-assets table."""
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = series.shape
        dtype = series.dtype
    return SimpleNamespace(shape=tuple(shape), dtype=dtype)


def build_image_assets(
    manifest: dict[str, Any],
    images: dict[str, Any],
    masks: dict[str, Any],
    run_id: str,
    git_commit: str,
) -> pd.DataFrame:
    """Build the pilot `images.image_assets` table."""
    metadata = {
        key: manifest[key]
        for key in (
            "Metadata_Biology_PatientTumor",
            "Metadata_Biology_PatientID",
            "Metadata_Experiment_PlateID",
            "Metadata_Experiment_WellID",
            "Metadata_Imaging_FieldID",
            "Metadata_Imaging_ImageID",
        )
    }
    channel_paths = manifest.get("channel_paths") or {}
    channel_codes = manifest.get("channel_codes") or {}
    mask_paths = manifest.get("mask_paths") or {}
    primary_channels = manifest.get("compartment_primary_channels") or {}
    primary_codes = manifest.get("compartment_primary_channel_codes") or {}
    methods = manifest.get("compartment_segmentation_methods") or {}

    rows: list[dict[str, object]] = []
    for channel, array in images.items():
        shape = list(array.shape)
        rows.append(
            {
                **metadata,
                "Metadata_ImageAsset_AssetID": (
                    f"{metadata['Metadata_Imaging_ImageID']}::{channel}"
                ),
                "Metadata_ImageAsset_AssetType": "raw_image",
                "Metadata_ImageAsset_Channel": channel,
                "Metadata_ImageAsset_ChannelCode": str(channel_codes.get(channel, "")),
                "Metadata_ImageAsset_Compartment": "",
                "Metadata_ImageAsset_SegmentationMethod": "",
                "Metadata_ImageAsset_SourceURI": str(channel_paths.get(channel, "")),
                "Metadata_ImageAsset_DType": str(array.dtype),
                "Metadata_ImageAsset_SizeZ": int(shape[0]) if len(shape) > 0 else None,
                "Metadata_ImageAsset_SizeY": int(shape[1]) if len(shape) > 1 else None,
                "Metadata_ImageAsset_SizeX": int(shape[2]) if len(shape) > 2 else None,
                "Metadata_Run_RunID": run_id,
                "Metadata_Run_GitCommit": git_commit,
            }
        )

    for compartment, array in masks.items():
        shape = list(array.shape)
        channel = str(primary_channels.get(compartment, ""))
        rows.append(
            {
                **metadata,
                "Metadata_ImageAsset_AssetID": (
                    f"{metadata['Metadata_Imaging_ImageID']}::{compartment}_mask"
                ),
                "Metadata_ImageAsset_AssetType": "segmentation_mask",
                "Metadata_ImageAsset_Channel": channel,
                "Metadata_ImageAsset_ChannelCode": str(
                    primary_codes.get(compartment, "")
                ),
                "Metadata_ImageAsset_Compartment": compartment,
                "Metadata_ImageAsset_SegmentationMethod": str(
                    methods.get(compartment, "")
                ),
                "Metadata_ImageAsset_SourceURI": str(mask_paths.get(compartment, "")),
                "Metadata_ImageAsset_DType": str(array.dtype),
                "Metadata_ImageAsset_SizeZ": int(shape[0]) if len(shape) > 0 else None,
                "Metadata_ImageAsset_SizeY": int(shape[1]) if len(shape) > 1 else None,
                "Metadata_ImageAsset_SizeX": int(shape[2]) if len(shape) > 2 else None,
                "Metadata_Run_RunID": run_id,
                "Metadata_Run_GitCommit": git_commit,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="This run's shared outdir. Assets are written under "
        "<outdir>/images/image_assets/.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    path_errors = require_manifest_paths(manifest)
    if path_errors:
        raise SystemExit(f"{args.manifest}: " + "\n".join(path_errors))

    compartments = [
        str(compartment)
        for compartment in (manifest.get("compartments") or [manifest["compartment"]])
    ]
    unknown = sorted(set(compartments) - set(COMPARTMENTS))
    if unknown:
        raise SystemExit(
            f"Unsupported compartments in {args.manifest}: {', '.join(unknown)}"
        )

    channel_paths = manifest["channel_paths"]
    mask_paths = manifest.get("mask_paths") or {manifest["compartment"]: manifest["mask_path"]}
    images = {channel: tiff_proxy(str(channel_paths[channel])) for channel in CHANNELS}
    masks = {
        compartment: tiff_proxy(str(mask_paths[compartment]))
        for compartment in compartments
    }
    alignment = validate_aligned_shapes(images, masks)

    git_revision = git_commit(args.repo_root)
    assets = build_image_assets(manifest, images, masks, args.run_id, git_revision)

    image_id = str(manifest["Metadata_Imaging_ImageID"])
    target_dir = args.outdir / "images" / "image_assets"
    target_dir.mkdir(parents=True, exist_ok=True)
    assets.to_parquet(target_dir / f"{image_id}.parquet", index=False)
    validation = {
        "valid": bool(alignment["valid"]),
        "row_count": int(assets.shape[0]),
        "column_count": int(assets.shape[1]),
        "alignment": alignment,
    }
    (target_dir / f"{image_id}.validation.json").write_text(
        json.dumps(validation, indent=2)
    )

    status = "OK" if alignment["valid"] else "FAILED"
    print(
        f"NF1_BUILD_IMAGE_ASSETS_{status} image_id={image_id} "
        f"rows={assets.shape[0]} alignment_valid={alignment['valid']}"
    )
    if not alignment["valid"]:
        print(json.dumps(alignment, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
