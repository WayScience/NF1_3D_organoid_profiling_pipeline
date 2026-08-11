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
from iceberg_warehouse import publish_iceberg_warehouse
from manifest_io import load_manifest, require_manifest_paths

CHANNELS = ("DNA", "ER", "AGP", "Mito")
COMPARTMENTS = ("Nuclei", "Cell", "Cytoplasm", "Organoid")
CHANNEL_CODES = {
    "DNA": "405",
    "ER": "488",
    "AGP": "555",
    "Mito": "640",
}
COMPARTMENT_PRIMARY_CHANNELS = {
    "Nuclei": "DNA",
    "Cell": "AGP",
    "Cytoplasm": "AGP",
    "Organoid": "AGP",
}
COMPARTMENT_SEED_CHANNELS = {
    "Nuclei": "",
    "Cell": "DNA",
    "Cytoplasm": "DNA",
    "Organoid": "",
}
COMPARTMENT_SEGMENTATION_METHODS = {
    "Nuclei": "segmented_from_dna",
    "Cell": "agp_watershed_seeded_by_nuclei",
    "Cytoplasm": "cell_mask_minus_nuclei_mask",
    "Organoid": "segmented_from_agp",
}
COMPARTMENT_PRIMARY_CHANNEL_CODES = {
    compartment: CHANNEL_CODES[channel]
    for compartment, channel in COMPARTMENT_PRIMARY_CHANNELS.items()
}
COMPARTMENT_SEED_CHANNEL_CODES = {
    compartment: CHANNEL_CODES[channel] if channel else ""
    for compartment, channel in COMPARTMENT_SEED_CHANNELS.items()
}
FEATURE_FAMILIES = (
    "VolumeSizeShape",
    "Intensity",
    "Texture",
    "Colocalization",
    "Neighbors",
    "Granularity",
)
NONGRANULARITY_FEATURE_FAMILIES = tuple(
    family for family in FEATURE_FAMILIES if family != "Granularity"
)


def compartment_slug(compartment: str) -> str:
    return compartment.lower()


def profile_table(compartment: str) -> str:
    return f"profiles.{compartment_slug(compartment)}_profiles"


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


def array_shape_summary(
    images: dict[str, object], masks: dict[str, object]
) -> dict[str, dict[str, list[int]]]:
    return {
        "channels": {channel: list(array.shape) for channel, array in images.items()},
        "masks": {
            compartment: list(array.shape) for compartment, array in masks.items()
        },
    }


def validate_aligned_shapes(
    images: dict[str, object], masks: dict[str, object]
) -> dict[str, object]:
    shapes = array_shape_summary(images, masks)
    unique_shapes = {
        tuple(shape) for group in shapes.values() for shape in group.values()
    }
    return {
        "valid": len(unique_shapes) == 1,
        "reference_shape": list(next(iter(unique_shapes)))
        if len(unique_shapes) == 1
        else None,
        "shapes": shapes,
    }


def tiff_shape(path: Path | str) -> list[int]:
    """Read TIFF shape metadata without materializing the full image."""
    with tifffile.TiffFile(str(path)) as tiff:
        return list(tiff.series[0].shape)


def numeric_nonfinite_counts(df: pd.DataFrame) -> dict[str, int]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    finite = np.isfinite(numeric.to_numpy())
    if finite.all():
        return {}
    counts = (~finite).sum(axis=0)
    return {
        str(column): int(count)
        for column, count in zip(numeric.columns, counts, strict=True)
        if count
    }


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace(" ", "_") for col in df.columns]
    return df


def normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    join_candidates = {"Metadata_Object_ObjectID", "Metadata_Experiment_ImageSet"}
    df = clean_columns(df)
    drop_columns = [
        column
        for column in df.columns
        if column.startswith("Metadata_") and column not in join_candidates
    ]
    return df.drop(columns=drop_columns)


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No feature frames were produced")

    merged = normalize_feature_frame(frames[0])
    for frame in frames[1:]:
        frame = normalize_feature_frame(frame)
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


def manifest_mapping(
    manifest: dict[str, object], key: str, default: dict[str, str]
) -> dict[str, str]:
    value = manifest.get(key)
    if not isinstance(value, dict):
        return default
    return {str(k): str(v) for k, v in value.items()}


def add_segmentation_metadata(
    profiles: pd.DataFrame, manifest: dict[str, object], compartment: str
) -> pd.DataFrame:
    profiles = profiles.copy()
    primary_channels = manifest_mapping(
        manifest, "compartment_primary_channels", COMPARTMENT_PRIMARY_CHANNELS
    )
    primary_codes = manifest_mapping(
        manifest,
        "compartment_primary_channel_codes",
        COMPARTMENT_PRIMARY_CHANNEL_CODES,
    )
    seed_channels = manifest_mapping(
        manifest, "compartment_seed_channels", COMPARTMENT_SEED_CHANNELS
    )
    seed_codes = manifest_mapping(
        manifest, "compartment_seed_channel_codes", COMPARTMENT_SEED_CHANNEL_CODES
    )
    methods = manifest_mapping(
        manifest, "compartment_segmentation_methods", COMPARTMENT_SEGMENTATION_METHODS
    )
    metadata = [
        (
            "Metadata_Segmentation_PrimaryChannel",
            primary_channels.get(compartment, ""),
        ),
        (
            "Metadata_Segmentation_PrimaryChannelCode",
            primary_codes.get(compartment, ""),
        ),
        ("Metadata_Segmentation_SeedChannel", seed_channels.get(compartment, "")),
        ("Metadata_Segmentation_SeedChannelCode", seed_codes.get(compartment, "")),
        ("Metadata_Segmentation_Method", methods.get(compartment, "")),
    ]
    insert_at = 1 if "Metadata_Compartment" in profiles.columns else 0
    for offset, (column, value) in enumerate(metadata):
        if column in profiles.columns:
            profiles[column] = value
        else:
            profiles.insert(insert_at + offset, column, value)
    return profiles


def feature_family_presence(
    columns: list[str], required_families: tuple[str, ...] = FEATURE_FAMILIES
) -> dict[str, bool]:
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
        for family in required_families
    }


def validate_profiles(
    profiles: pd.DataFrame,
    mask: np.ndarray,
    manifest: dict[str, object],
    required_feature_families: tuple[str, ...] = FEATURE_FAMILIES,
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
    nonfinite_counts = numeric_nonfinite_counts(profiles[feature_columns])
    family_presence = feature_family_presence(
        feature_columns, required_families=required_feature_families
    )
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
        "nonfinite_numeric_feature_columns": nonfinite_counts,
        "metadata_match": metadata_match,
    }


def run_timed(
    name: str,
    fn: Callable[[], pd.DataFrame],
    timings: dict[str, float],
    quality_warnings: list[dict[str, object]],
) -> pd.DataFrame:
    started = time.perf_counter()
    frame = fn()
    elapsed = time.perf_counter() - started
    timings[name] = round(elapsed, 3)
    nonfinite_counts = numeric_nonfinite_counts(frame)
    if nonfinite_counts:
        quality_warnings.append(
            {
                "stage": name,
                "issue": "nonfinite_numeric_values",
                "columns": nonfinite_counts,
            }
        )
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


def extract_compartment_profile(
    zapi: dict[str, object],
    compartment: str,
    mask: np.ndarray,
    images: dict[str, np.ndarray],
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    image_set_name = str(manifest["Metadata_Imaging_ImageID"])
    z_spacing = float(manifest.get("z_spacing") or 1.0)
    xy_spacing = float(manifest.get("xy_spacing") or 0.1)

    ObjectLoader = zapi["ObjectLoader"]
    TwoObjectLoader = zapi["TwoObjectLoader"]
    timings: dict[str, float] = {}
    quality_warnings: list[dict[str, object]] = []
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
                f"{compartment}_{channel}_Granularity",
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
                quality_warnings,
            )
        )
        frames.append(
            run_timed(
                f"{compartment}_{channel}_Intensity",
                lambda object_loader=object_loader: zapi["compute_intensity"](
                    object_loader=object_loader
                ),
                timings,
                quality_warnings,
            )
        )
        frames.append(
            run_timed(
                f"{compartment}_{channel}_Texture",
                lambda object_loader=object_loader: zapi["compute_texture"](
                    object_loader=object_loader,
                    distance=3,
                    grayscale=256,
                ),
                timings,
                quality_warnings,
            )
        )

    no_channel = np.zeros_like(mask)
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
            f"{compartment}_NoChannel_VolumeSizeShape",
            lambda: zapi["compute_volume_size_shape"](
                image_set_loader=no_channel_loader,
                object_loader=no_channel_object_loader,
            ),
            timings,
            quality_warnings,
        )
    )
    frames.append(
        run_timed(
            f"{compartment}_NoChannel_Neighbors",
            lambda: zapi["compute_neighbors"](
                object_loader=no_channel_object_loader,
                distance_threshold=10,
                anisotropy_factor=z_spacing / xy_spacing,
            ),
            timings,
            quality_warnings,
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
                f"{compartment}_{channel1}_{channel2}_Colocalization",
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
                quality_warnings,
            )
        )

    profiles = add_metadata(merge_feature_frames(frames), manifest)
    profiles.insert(0, "Metadata_Compartment", compartment)
    profiles = add_segmentation_metadata(profiles, manifest, compartment)
    validation = validate_profiles(profiles, mask, manifest)
    validation["compartment"] = compartment
    validation["quality_warnings"] = quality_warnings
    return profiles, validation, timings


def extract_granularity_profile(
    zapi: dict[str, object],
    compartment: str,
    channel: str,
    mask: np.ndarray,
    image: np.ndarray,
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    image_set_name = str(manifest["Metadata_Imaging_ImageID"])
    z_spacing = float(manifest.get("z_spacing") or 1.0)
    xy_spacing = float(manifest.get("xy_spacing") or 0.1)

    ObjectLoader = zapi["ObjectLoader"]
    timings: dict[str, float] = {}
    quality_warnings: list[dict[str, object]] = []
    loader = make_image_set_loader(
        zapi,
        image_set_name,
        channel,
        compartment,
        image,
        mask,
        z_spacing,
        xy_spacing,
    )
    object_loader = ObjectLoader(
        image_set_loader=loader,
        channel_name=channel,
        compartment_name=compartment,
    )
    frame = run_timed(
        f"{compartment}_{channel}_Granularity",
        lambda: zapi["compute_granularity"](
            object_loader=object_loader,
            radius=10,
            granular_spectrum_length=16,
            subsample_size=0.25,
            image_sample_size=0.25,
            mask_threshold=0.9,
            verbose=False,
        ),
        timings,
        quality_warnings,
    )
    frame = clean_columns(frame)
    object_ids = sorted(int(value) for value in np.unique(mask) if value != 0)
    observed_ids = sorted(
        int(value)
        for value in frame["Metadata_Object_ObjectID"].dropna().unique()
        if not pd.isna(value)
    )
    feature_columns = [
        column for column in frame.columns if not column.startswith("Metadata_")
    ]
    validation = {
        "valid": observed_ids == object_ids
        and feature_family_presence(
            feature_columns, required_families=("Granularity",)
        )["Granularity"],
        "compartment": compartment,
        "channel": channel,
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "mask_object_count": len(object_ids),
        "missing_object_ids": sorted(set(object_ids) - set(observed_ids)),
        "unexpected_object_ids": sorted(set(observed_ids) - set(object_ids)),
        "feature_family_presence": feature_family_presence(
            feature_columns, required_families=("Granularity",)
        ),
        "nonfinite_numeric_feature_columns": numeric_nonfinite_counts(
            frame[feature_columns]
        ),
        "quality_warnings": quality_warnings,
    }
    return frame, validation, timings


def extract_compartment_nongranularity_profile(
    zapi: dict[str, object],
    compartment: str,
    mask: np.ndarray,
    channel_paths: dict[str, object],
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    image_set_name = str(manifest["Metadata_Imaging_ImageID"])
    z_spacing = float(manifest.get("z_spacing") or 1.0)
    xy_spacing = float(manifest.get("xy_spacing") or 0.1)

    ObjectLoader = zapi["ObjectLoader"]
    TwoObjectLoader = zapi["TwoObjectLoader"]
    timings: dict[str, float] = {}
    quality_warnings: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for channel in CHANNELS:
        image = tifffile.imread(str(channel_paths[channel]))
        loader = make_image_set_loader(
            zapi,
            image_set_name,
            channel,
            compartment,
            image,
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
                f"{compartment}_{channel}_Intensity",
                lambda object_loader=object_loader: zapi["compute_intensity"](
                    object_loader=object_loader
                ),
                timings,
                quality_warnings,
            )
        )
        frames.append(
            run_timed(
                f"{compartment}_{channel}_Texture",
                lambda object_loader=object_loader: zapi["compute_texture"](
                    object_loader=object_loader,
                    distance=3,
                    grayscale=256,
                ),
                timings,
                quality_warnings,
            )
        )
        del image, loader, object_loader

    no_channel = np.zeros_like(mask)
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
            f"{compartment}_NoChannel_VolumeSizeShape",
            lambda: zapi["compute_volume_size_shape"](
                image_set_loader=no_channel_loader,
                object_loader=no_channel_object_loader,
            ),
            timings,
            quality_warnings,
        )
    )
    frames.append(
        run_timed(
            f"{compartment}_NoChannel_Neighbors",
            lambda: zapi["compute_neighbors"](
                object_loader=no_channel_object_loader,
                distance_threshold=10,
                anisotropy_factor=z_spacing / xy_spacing,
            ),
            timings,
            quality_warnings,
        )
    )
    del no_channel, no_channel_loader, no_channel_object_loader

    for channel1, channel2 in itertools.combinations(CHANNELS, 2):
        image1 = tifffile.imread(str(channel_paths[channel1]))
        image2 = tifffile.imread(str(channel_paths[channel2]))
        coloc_loader = make_image_set_loader(
            zapi,
            image_set_name,
            channel1,
            compartment,
            image1,
            mask,
            z_spacing,
            xy_spacing,
        )
        coloc_loader.image_set_dict[channel2] = image2
        two_object_loader = TwoObjectLoader(
            image_set_loader=coloc_loader,
            compartment=compartment,
            channel1=channel1,
            channel2=channel2,
        )
        frames.append(
            run_timed(
                f"{compartment}_{channel1}_{channel2}_Colocalization",
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
                quality_warnings,
            )
        )
        del image1, image2, coloc_loader, two_object_loader

    profiles = add_metadata(merge_feature_frames(frames), manifest)
    profiles.insert(0, "Metadata_Compartment", compartment)
    profiles = add_segmentation_metadata(profiles, manifest, compartment)
    validation = validate_profiles(
        profiles,
        mask,
        manifest,
        required_feature_families=NONGRANULARITY_FEATURE_FAMILIES,
    )
    validation["compartment"] = compartment
    validation["quality_warnings"] = quality_warnings
    return profiles, validation, timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--compartment",
        choices=COMPARTMENTS,
        help="Extract only one compartment. Used by the Nextflow fan-out workflow.",
    )
    parser.add_argument(
        "--skip-warehouse",
        action="store_true",
        help="Write profile and validation artifacts without building Iceberg tables.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("all", "granularity", "nongranularity"),
        default="all",
        help="Select full extraction or one fan-out partial feature mode.",
    )
    parser.add_argument(
        "--channel",
        choices=CHANNELS,
        help="Channel to process when --feature-mode=granularity.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = load_manifest(args.manifest)
    path_errors = require_manifest_paths(manifest)
    if path_errors:
        raise SystemExit("\n".join(path_errors))

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    zapi = import_zedprofiler()
    channel_paths = manifest["channel_paths"]
    compartments = [
        str(compartment)
        for compartment in (manifest.get("compartments") or [manifest["compartment"]])
    ]
    if args.compartment:
        if args.compartment not in compartments:
            raise SystemExit(
                f"Compartment {args.compartment!r} is not listed in the manifest"
            )
        compartments = [args.compartment]
    mask_paths = manifest.get("mask_paths") or {
        manifest["compartment"]: manifest["mask_path"]
    }

    if args.feature_mode != "all":
        if not args.compartment:
            raise SystemExit("--compartment is required for partial feature modes")
        if args.feature_mode == "granularity" and not args.channel:
            raise SystemExit("--channel is required for --feature-mode=granularity")

        compartment = args.compartment
        mask = tifffile.imread(str(mask_paths[compartment]))
        channel_shapes = (
            {str(args.channel): tiff_shape(channel_paths[str(args.channel)])}
            if args.feature_mode == "granularity"
            else {
                channel: tiff_shape(channel_paths[channel])
                for channel in CHANNELS
            }
        )
        alignment = {
            "valid": len(
                {tuple(shape) for shape in channel_shapes.values()}
                | {tuple(mask.shape)}
            )
            == 1,
            "reference_shape": list(mask.shape),
            "shapes": {
                "channels": channel_shapes,
                "masks": {compartment: list(mask.shape)},
            },
        }
        if not alignment["valid"]:
            (outdir / "alignment_validation.json").write_text(
                json.dumps(alignment, indent=2)
            )
            raise SystemExit(
                "Channel and mask arrays are not aligned by shape; see "
                f"{outdir / 'alignment_validation.json'}"
            )

        git_revision = git_commit(args.repo_root)
        if args.feature_mode == "granularity":
            channel = str(args.channel)
            image = tifffile.imread(str(channel_paths[channel]))
            profiles, validation, timings = extract_granularity_profile(
                zapi=zapi,
                compartment=compartment,
                channel=channel,
                mask=mask,
                image=image,
                manifest=manifest,
            )
            profile_path = (
                outdir
                / f"{compartment_slug(compartment)}_{channel.lower()}_granularity.parquet"
            )
            output_name = f"granularity.{compartment}.{channel}"
        else:
            profiles, validation, timings = (
                extract_compartment_nongranularity_profile(
                    zapi=zapi,
                    compartment=compartment,
                    mask=mask,
                    channel_paths=channel_paths,
                    manifest=manifest,
                )
            )
            profile_path = (
                outdir / f"{compartment_slug(compartment)}_nongranularity_profiles.parquet"
            )
            output_name = f"nongranularity.{compartment}"

        profiles.to_parquet(profile_path, index=False)
        all_valid = bool(validation["valid"])
        run_record = {
            "run_id": args.run_id,
            "command": " ".join(sys.argv),
            "mode": args.feature_mode,
            "compartments": [compartment],
            "channel": args.channel or "",
            "git_commit": git_revision,
            "zedprofiler_version": zapi["version"],
            "python_version": platform.python_version(),
            "manifest": manifest,
            "alignment": alignment,
            "outputs": {output_name: str(profile_path)},
            "tables": [
                {
                    "name": output_name,
                    "path": str(profile_path),
                    "schema_version": "0.1.0-pilot",
                    "source_image_root": manifest.get("source_image_root", ""),
                    "run_id": args.run_id,
                    "git_commit": git_revision,
                    "row_count": validation["row_count"],
                    "column_count": validation["column_count"],
                    "validation_status": "pass" if all_valid else "fail",
                }
            ],
            "timings_seconds": timings,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "exit_status": 0 if all_valid else 1,
            "validation_status": "pass" if all_valid else "fail",
            "quality_warning_count": len(validation.get("quality_warnings", [])),
        }
        validation_report = {
            "valid": all_valid,
            "alignment": alignment,
            "compartments": {compartment: validation},
        }
        (outdir / "run_record.json").write_text(json.dumps(run_record, indent=2))
        (outdir / "validation.json").write_text(
            json.dumps(validation_report, indent=2)
        )
        (outdir / "alignment_validation.json").write_text(
            json.dumps(alignment, indent=2)
        )
        if not all_valid:
            print(json.dumps(validation_report, indent=2), file=sys.stderr)
            return 1
        print(
            "NF1_ZEDPROFILER_PARTIAL_OK "
            f"mode={args.feature_mode} elapsed={run_record['elapsed_seconds']}"
        )
        return 0

    images = {
        channel: tifffile.imread(str(channel_paths[channel])) for channel in CHANNELS
    }
    masks = {
        compartment: tifffile.imread(str(mask_paths[compartment]))
        for compartment in compartments
    }
    alignment = validate_aligned_shapes(images, masks)
    if not alignment["valid"]:
        (outdir / "alignment_validation.json").write_text(
            json.dumps(alignment, indent=2)
        )
        raise SystemExit(
            "Channel and mask arrays are not aligned by shape; see "
            f"{outdir / 'alignment_validation.json'}"
        )

    validations: dict[str, dict[str, object]] = {}
    profile_frames: dict[str, pd.DataFrame] = {}
    output_tables = []
    all_timings: dict[str, float] = {}
    total_quality_warnings = 0
    git_revision = git_commit(args.repo_root)

    for compartment in compartments:
        profiles, validation, timings = extract_compartment_profile(
            zapi=zapi,
            compartment=compartment,
            mask=masks[compartment],
            images=images,
            manifest=manifest,
        )
        profile_path = outdir / f"{compartment_slug(compartment)}_profiles.parquet"
        profiles.to_parquet(profile_path, index=False)
        validations[compartment] = validation
        profile_frames[profile_table(compartment)] = profiles
        all_timings.update(timings)
        total_quality_warnings += len(validation["quality_warnings"])
        output_tables.append(
            {
                "name": profile_table(compartment),
                "path": str(profile_path),
                "schema_version": "0.1.0-pilot",
                "join_keys": [
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Imaging_ImageID",
                    "Metadata_Compartment",
                    "Metadata_Object_ObjectID",
                ],
                "source_image_root": manifest.get("source_image_root", ""),
                "run_id": args.run_id,
                "git_commit": git_revision,
                "row_count": validation["row_count"],
                "column_count": validation["column_count"],
                "validation_status": "pass" if validation["valid"] else "fail",
            }
        )

    elapsed = time.perf_counter() - started
    all_valid = all(validation["valid"] for validation in validations.values())
    warehouse_manifest = None
    if not args.skip_warehouse:
        warehouse_manifest = publish_iceberg_warehouse(
            outdir=outdir,
            run_id=args.run_id,
            git_commit=git_revision,
            manifest=manifest,
            profile_frames=profile_frames,
            images=images,
            masks=masks,
            validations=validations,
            alignment=alignment,
            zedprofiler_version=str(zapi["version"]),
        )
    run_record = {
        "run_id": args.run_id,
        "command": " ".join(sys.argv),
        "mode": "single_compartment" if args.compartment else "image_set",
        "compartments": compartments,
        "git_commit": git_revision,
        "zedprofiler_version": zapi["version"],
        "python_version": platform.python_version(),
        "manifest": manifest,
        "alignment": alignment,
        "outputs": {table["name"]: table["path"] for table in output_tables},
        "tables": output_tables,
        "timings_seconds": all_timings,
        "elapsed_seconds": round(elapsed, 3),
        "exit_status": 0 if all_valid else 1,
        "validation_status": "pass" if all_valid else "fail",
        "quality_warning_count": total_quality_warnings,
    }
    if warehouse_manifest:
        run_record.update(
            {
                "warehouse_root": warehouse_manifest["warehouse_root"],
                "warehouse_manifest": str(
                    outdir / "warehouse" / "warehouse_manifest.json"
                ),
            }
        )
    validation_report = {
        "valid": all_valid,
        "alignment": alignment,
        "compartments": validations,
    }
    if warehouse_manifest:
        validation_report["warehouse"] = {
            "valid": warehouse_manifest["validation_status"] == "pass",
            "manifest": str(outdir / "warehouse" / "warehouse_manifest.json"),
            "namespaces": warehouse_manifest["namespaces"],
            "tables": [table["table_name"] for table in warehouse_manifest["tables"]],
        }

    (outdir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    (outdir / "validation.json").write_text(json.dumps(validation_report, indent=2))
    (outdir / "alignment_validation.json").write_text(json.dumps(alignment, indent=2))

    if not all_valid:
        print(json.dumps(validation_report, indent=2), file=sys.stderr)
        return 1
    print(
        "NF1_ZEDPROFILER_PILOT_OK "
        f"tables={len(output_tables)} elapsed={run_record['elapsed_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
