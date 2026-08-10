#!/usr/bin/env python3
"""Run the one-image-set NF1 ZEDProfiler pilot and write validation artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tifffile
from manifest_io import load_manifest, require_manifest_paths

CHANNELS = ("DNA", "ER", "AGP", "Mito")
FEATURE_FAMILIES = (
    "VolumeSizeShape",
    "Intensity",
    "Texture",
    "Colocalization",
    "Neighbors",
    "Granularity",
)
PROFILE_TABLE = "profiles.nuclei_profiles"


def import_zedprofiler() -> dict[str, object]:
    """Import ZEDProfiler APIs across the 0.1.x module layout."""
    try:
        import zedprofiler
        from zedprofiler.featurization.colocalization import compute_colocalization
        from zedprofiler.featurization.granularity import compute_granularity
        from zedprofiler.featurization.intensity import compute_intensity
        from zedprofiler.featurization.neighbors import compute_neighbors
        from zedprofiler.featurization.texture import compute_texture
        from zedprofiler.featurization.volumesizeshape import compute_volume_size_shape
        from zedprofiler.IO.loading_classes import (
            ImageSetConfig,
            ImageSetLoader,
            ObjectLoader,
            TwoObjectLoader,
        )

        version = getattr(zedprofiler, "__version__", "unknown")
    except ImportError:
        import zedprofiler
        from zedprofiler.IO.loading_classes import (
            ImageSetConfig,
            ImageSetLoader,
            ObjectLoader,
            TwoObjectLoader,
        )

        compute_colocalization = zedprofiler.colocalization.compute_colocalization
        compute_granularity = zedprofiler.granularity.compute_granularity
        compute_intensity = zedprofiler.intensity.compute_intensity
        compute_neighbors = zedprofiler.neighbors.compute_neighbors
        compute_texture = zedprofiler.texture.compute_texture
        compute_volume_size_shape = (
            zedprofiler.volumesizeshape.compute_volume_size_shape
        )
        version = getattr(zedprofiler, "__version__", "unknown")

    return {
        "compute_colocalization": compute_colocalization,
        "compute_granularity": compute_granularity,
        "compute_intensity": compute_intensity,
        "compute_neighbors": compute_neighbors,
        "compute_texture": compute_texture,
        "compute_volume_size_shape": compute_volume_size_shape,
        "ImageSetConfig": ImageSetConfig,
        "ImageSetLoader": ImageSetLoader,
        "ObjectLoader": ObjectLoader,
        "TwoObjectLoader": TwoObjectLoader,
        "version": version,
    }


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def dataframe_numeric_is_finite(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    return bool(np.isfinite(numeric.to_numpy()).all())


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace(" ", "_") for col in df.columns]
    return df


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No feature frames were produced")

    merged = clean_columns(frames[0])
    for frame in frames[1:]:
        frame = clean_columns(frame)
        join_keys = [
            key
            for key in ("Metadata_Object_ObjectID", "Metadata_Experiment_ImageSet")
            if key in merged.columns and key in frame.columns
        ]
        if not join_keys:
            join_keys = ["Metadata_Object_ObjectID"]
        merged = merged.merge(frame, how="outer", on=join_keys)
    return merged


def add_metadata(df: pd.DataFrame, manifest: dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    metadata_keys = [
        "Metadata_Biology_PatientTumor",
        "Metadata_Biology_PatientID",
        "Metadata_Experiment_PlateID",
        "Metadata_Experiment_WellID",
        "Metadata_Imaging_FieldID",
        "Metadata_Imaging_ImageID",
    ]
    for key in reversed(metadata_keys):
        if key in df.columns:
            df[key] = manifest[key]
        else:
            df.insert(0, key, manifest[key])
    if "Metadata_Object_ObjectID" not in df.columns and "object_id" in df.columns:
        df = df.rename(columns={"object_id": "Metadata_Object_ObjectID"})
    return df


def feature_family_presence(columns: list[str]) -> dict[str, bool]:
    aliases = {
        "VolumeSizeShape": (
            "VolumeSizeShape",
            "AreaSizeShape",
            "Area_Shape",
            "SizeShape",
        ),
        "Intensity": ("Intensity",),
        "Texture": ("Texture",),
        "Colocalization": ("Colocalization",),
        "Neighbors": ("Neighbors",),
        "Granularity": ("Granularity",),
    }
    return {
        family: any(alias in column for alias in aliases[family] for column in columns)
        for family in FEATURE_FAMILIES
    }


def validate_profiles(
    profiles: pd.DataFrame, mask: np.ndarray, manifest: dict[str, object]
) -> dict[str, object]:
    object_ids = sorted(int(value) for value in np.unique(mask) if value != 0)
    observed_ids = sorted(
        int(value)
        for value in profiles["Metadata_Object_ObjectID"].dropna().unique()
        if not pd.isna(value)
    )
    feature_columns = [
        column for column in profiles.columns if not column.startswith("Metadata_")
    ]
    all_null = [column for column in feature_columns if profiles[column].isna().all()]
    family_presence = feature_family_presence(feature_columns)
    metadata_match = {
        key: bool((profiles[key].astype(str) == str(manifest[key])).all())
        for key in (
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_WellID",
            "Metadata_Imaging_FieldID",
            "Metadata_Imaging_ImageID",
        )
    }
    valid = (
        profiles.shape[0] == len(object_ids)
        and observed_ids == object_ids
        and all(family_presence.values())
        and not all_null
        and all(metadata_match.values())
    )
    return {
        "valid": valid,
        "row_count": int(profiles.shape[0]),
        "column_count": int(profiles.shape[1]),
        "mask_object_count": len(object_ids),
        "missing_object_ids": sorted(set(object_ids) - set(observed_ids)),
        "unexpected_object_ids": sorted(set(observed_ids) - set(object_ids)),
        "feature_family_presence": family_presence,
        "all_null_feature_columns": all_null,
        "metadata_match": metadata_match,
    }


def run_timed(
    name: str, fn: Callable[[], pd.DataFrame], timings: dict[str, float]
) -> pd.DataFrame:
    started = time.perf_counter()
    frame = fn()
    elapsed = time.perf_counter() - started
    timings[name] = round(elapsed, 3)
    if not dataframe_numeric_is_finite(frame):
        raise ValueError(f"{name} produced NaN or infinite numeric values")
    return frame


def make_image_set_loader(
    zapi: dict[str, object],
    image_set_name: str,
    channel_name: str,
    compartment: str,
    image: np.ndarray,
    mask: np.ndarray,
    z_spacing: float,
    xy_spacing: float,
):
    ImageSetConfig = zapi["ImageSetConfig"]
    ImageSetLoader = zapi["ImageSetLoader"]
    config = ImageSetConfig(
        image_set_name=image_set_name,
        raw_image_key_name=list(CHANNELS) + ["NoChannel"],
        label_key_name=[compartment],
    )
    return ImageSetLoader(
        image_set_path=None,
        label_set_path=None,
        image_set_array=image,
        label_set_array=mask,
        anisotropy_spacing=(z_spacing, xy_spacing, xy_spacing),
        channel_mapping={channel_name: channel_name, compartment: compartment},
        config=config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = load_manifest(args.manifest)
    path_errors = require_manifest_paths(manifest)
    if path_errors:
        raise SystemExit("\n".join(path_errors))

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    zapi = import_zedprofiler()
    compartment = str(manifest.get("compartment") or "Nuclei")
    image_set_name = str(manifest["Metadata_Imaging_ImageID"])
    z_spacing = float(manifest.get("z_spacing") or 1.0)
    xy_spacing = float(manifest.get("xy_spacing") or 0.1)
    channel_paths = manifest["channel_paths"]

    mask = tifffile.imread(str(manifest["mask_path"]))
    images = {
        channel: tifffile.imread(str(channel_paths[channel])) for channel in CHANNELS
    }
    no_channel = np.zeros_like(mask)

    ObjectLoader = zapi["ObjectLoader"]
    TwoObjectLoader = zapi["TwoObjectLoader"]
    timings: dict[str, float] = {}
    frames: list[pd.DataFrame] = []

    for channel in CHANNELS:
        loader = make_image_set_loader(
            zapi,
            image_set_name,
            channel,
            compartment,
            images[channel],
            mask,
            z_spacing,
            xy_spacing,
        )
        object_loader = ObjectLoader(
            image_set_loader=loader,
            channel_name=channel,
            compartment_name=compartment,
        )
        frames.append(
            run_timed(
                f"{channel}_Granularity",
                lambda object_loader=object_loader: zapi["compute_granularity"](
                    object_loader=object_loader,
                    radius=10,
                    granular_spectrum_length=16,
                    subsample_size=0.25,
                    image_sample_size=0.25,
                    mask_threshold=0.9,
                    verbose=False,
                ),
                timings,
            )
        )
        frames.append(
            run_timed(
                f"{channel}_Intensity",
                lambda object_loader=object_loader: zapi["compute_intensity"](
                    object_loader=object_loader
                ),
                timings,
            )
        )
        frames.append(
            run_timed(
                f"{channel}_Texture",
                lambda object_loader=object_loader: zapi["compute_texture"](
                    object_loader=object_loader,
                    distance=3,
                    grayscale=256,
                ),
                timings,
            )
        )

    no_channel_loader = make_image_set_loader(
        zapi,
        image_set_name,
        "NoChannel",
        compartment,
        no_channel,
        mask,
        z_spacing,
        xy_spacing,
    )
    no_channel_object_loader = ObjectLoader(
        image_set_loader=no_channel_loader,
        channel_name="NoChannel",
        compartment_name=compartment,
    )
    frames.append(
        run_timed(
            "NoChannel_VolumeSizeShape",
            lambda: zapi["compute_volume_size_shape"](
                image_set_loader=no_channel_loader,
                object_loader=no_channel_object_loader,
            ),
            timings,
        )
    )
    frames.append(
        run_timed(
            "NoChannel_Neighbors",
            lambda: zapi["compute_neighbors"](
                object_loader=no_channel_object_loader,
                distance_threshold=10,
                anisotropy_factor=z_spacing / xy_spacing,
            ),
            timings,
        )
    )

    for channel1, channel2 in itertools.combinations(CHANNELS, 2):
        coloc_loader = make_image_set_loader(
            zapi,
            image_set_name,
            channel1,
            compartment,
            images[channel1],
            mask,
            z_spacing,
            xy_spacing,
        )
        coloc_loader.image_set_dict[channel2] = images[channel2]
        two_object_loader = TwoObjectLoader(
            image_set_loader=coloc_loader,
            compartment=compartment,
            channel1=channel1,
            channel2=channel2,
        )
        frames.append(
            run_timed(
                f"{channel1}_{channel2}_Colocalization",
                lambda two_object_loader=two_object_loader, channel1=channel1, channel2=channel2: (
                    zapi["compute_colocalization"](
                        two_object_loader=two_object_loader,
                        thr=15,
                        fast_costes="Faster",
                        channel1=channel1,
                        channel2=channel2,
                    )
                ),
                timings,
            )
        )

    profiles = add_metadata(merge_feature_frames(frames), manifest)
    validation = validate_profiles(profiles, mask, manifest)

    profile_path = outdir / "nuclei_profiles.parquet"
    profiles.to_parquet(profile_path, index=False)

    elapsed = time.perf_counter() - started
    run_record = {
        "run_id": args.run_id,
        "command": " ".join(sys.argv),
        "git_commit": git_commit(args.repo_root),
        "zedprofiler_version": zapi["version"],
        "python_version": platform.python_version(),
        "manifest": manifest,
        "output": str(profile_path),
        "row_count": validation["row_count"],
        "column_count": validation["column_count"],
        "timings_seconds": timings,
        "elapsed_seconds": round(elapsed, 3),
        "exit_status": 0 if validation["valid"] else 1,
        "validation_status": "pass" if validation["valid"] else "fail",
    }
    warehouse_manifest = {
        "warehouse_root": str(outdir),
        "tables": [
            {
                "name": PROFILE_TABLE,
                "path": str(profile_path),
                "schema_version": "0.1.0-pilot",
                "join_keys": [
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Imaging_ImageID",
                    "Metadata_Object_ObjectID",
                ],
                "source_image_root": manifest.get("source_image_root", ""),
                "run_id": args.run_id,
                "git_commit": run_record["git_commit"],
                "row_count": validation["row_count"],
                "validation_status": run_record["validation_status"],
            }
        ],
    }

    (outdir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    (outdir / "warehouse_manifest.json").write_text(
        json.dumps(warehouse_manifest, indent=2)
    )
    (outdir / "validation.json").write_text(json.dumps(validation, indent=2))

    if not validation["valid"]:
        print(json.dumps(validation, indent=2), file=sys.stderr)
        return 1
    print(
        f"NF1_ZEDPROFILER_PILOT_OK {profile_path} rows={validation['row_count']} cols={validation['column_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
