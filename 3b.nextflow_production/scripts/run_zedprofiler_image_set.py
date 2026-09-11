#!/usr/bin/env python3
"""Extract every compartment's full ZEDProfiler profile (nongranularity and
granularity features together, each channel loaded once and reused across
all compartments) for one image set, writing each compartment's result
directly into the run's shared warehouse directory -- its one and only
write. Also builds and writes images.image_assets (one row per channel plus
one per compartment mask) and checks whole-image-set channel/mask alignment
-- reusing the same pixel arrays already loaded for feature extraction, no
separate task or extra file I/O needed for either. Runs as one Nextflow
task per image set in the fan-out workflow; --compartment restricts it to a
single compartment for manual testing, which also skips the image_assets
build (that mode doesn't load every mask)."""

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
from build_manifest import resolve_manifest
from manifest_io import require_manifest_paths

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
    df = df.drop(columns=drop_columns)
    # zedprofiler>=0.1.4 (ZedProfiler#53) started returning ID columns on
    # otherwise-empty feature frames (e.g. Colocalization on a constant/
    # degenerate channel pair, or any feature family for a compartment with
    # zero detected objects) instead of omitting them entirely. An empty
    # column's dtype infers as float64 regardless of the field's real dtype
    # (str for Metadata_Experiment_ImageSet), which crashed merge_feature_
    # frames()'s merge() with a dtype-mismatch ValueError once aligned
    # against any other frame that carries this key with its real dtype --
    # including another all-empty frame from earlier in the same zero-
    # object compartment's accumulation, whose dtype had already drifted.
    # Normalize to str up front so every frame agrees on dtype for this key
    # regardless of whether it holds real values, confirmed via astype("str")
    # preserving NA-ness rather than stringifying it to a literal "nan".
    if "Metadata_Experiment_ImageSet" in df.columns:
        df["Metadata_Experiment_ImageSet"] = df["Metadata_Experiment_ImageSet"].astype(
            "str"
        )
    return df


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
            # Observed in production: a single feature computation (e.g.
            # Colocalization on a channel pair with a constant/degenerate
            # input -- ConstantInputWarning from scipy.stats.pearsonr) can
            # return a frame with no object identifier at all, regardless of
            # how many objects the compartment actually has. Blindly falling
            # back to "Metadata_Object_ObjectID" crashed merge() with
            # KeyError whenever that column wasn't actually in this specific
            # frame. Drop just this one feature family's columns instead of
            # failing the whole compartment -- every other feature family's
            # columns for this image set are still valid and worth keeping.
            if "Metadata_Object_ObjectID" not in frame.columns:
                print(
                    "merge_feature_frames: dropping a feature frame with no "
                    f"Metadata_Object_ObjectID column (columns: {list(frame.columns)})",
                    file=sys.stderr,
                )
                continue
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
    if not object_ids:
        # A compartment mask can legitimately contain zero objects (observed
        # in production: an Organoid mask with no detected organoids).
        # feature_family_presence/all_null are meaningless with nothing to
        # compute -- every column is trivially "all null" on zero rows, and
        # merge_feature_frames() already drops feature families that
        # returned no object identifier at all for this exact reason. Valid
        # here just means "correctly produced zero rows," not garbage.
        valid = profiles.shape[0] == 0 and all(metadata_match.values())
    else:
        # Structural correctness (row count, object IDs, metadata) is still
        # checked strictly. Whether *every* feature family/column has a
        # non-null value is not: observed in production, at a ~22% rate
        # across 6 patients, that a whole feature family can legitimately
        # come back all-null for real, correctly-identified objects --
        # dominant case is GLCM Texture needing a minimum object pixel
        # extent for its distance parameter (all 676 Texture columns null
        # for objects as large as 12, i.e. by count alone, not corrupted
        # data), plus rarer degenerate-signal cases (Colocalization/
        # Intensity stats undefined for a constant channel within one
        # object). Previously this failed the whole compartment and burned
        # 3 wasted Slurm attempts per image set before giving up permanently
        # -- the same object/channel data deterministically produces the
        # same nulls every retry. `all_null_feature_columns` below still
        # records exactly what came back null, so this stays fully visible;
        # it just no longer blocks. feature_family_presence is still required
        # in full -- every case observed so far has every family's columns
        # structurally present (just null-valued), never a family's columns
        # dropped from the schema entirely; keeping this check catches that
        # different, less-understood failure mode if it ever shows up,
        # rather than silently accepting it too. Only a compartment with
        # *zero* real signal anywhere (every feature column null) is
        # rejected on nullness alone -- that remains a sign of genuine, not
        # merely partial, failure.
        valid = (
            profiles.shape[0] == len(object_ids)
            and observed_ids == object_ids
            and all(family_presence.values())
            and len(all_null) < len(feature_columns)
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


def build_image_assets(
    manifest: dict[str, object],
    images: dict[str, object],
    masks: dict[str, object],
    run_id: str,
    git_commit_hash: str,
) -> pd.DataFrame:
    """Build the pilot `images.image_assets` table: one row per raw channel
    plus one row per compartment mask. ``images``/``masks`` values only need
    ``.shape``/``.dtype`` -- the full pixel arrays already loaded for
    feature extraction work directly, no separate header-only read needed.
    """
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
                "Metadata_Run_GitCommit": git_commit_hash,
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
                "Metadata_Run_GitCommit": git_commit_hash,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to a YAML manifest for this image set. Mutually "
        "exclusive with --patient/--well-fov/--source-root.",
    )
    parser.add_argument(
        "--patient",
        help="Patient/tumor ID, e.g. NF0055_T1. Used with --well-fov and "
        "--source-root to derive a manifest on the fly instead of reading "
        "a YAML file -- the index-driven path for large batches.",
    )
    parser.add_argument("--well-fov", help="Well+field, e.g. B10-1.")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Run's shared outdir. Each compartment's profile is written "
        "directly under <outdir>/warehouse/profiles/<compartment>_profiles/ "
        "-- its one and only write.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--compartment",
        choices=COMPARTMENTS,
        help="Restrict extraction to one compartment, for manual testing. "
        "The Nextflow fan-out workflow omits this to process every "
        "compartment in the manifest in one task.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = resolve_manifest(args.manifest, args.patient, args.well_fov, args.source_root)
    path_errors = require_manifest_paths(manifest)
    if path_errors:
        raise SystemExit("\n".join(path_errors))
    image_id = str(manifest["Metadata_Imaging_ImageID"])

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

    images = {
        channel: tifffile.imread(str(channel_paths[channel])) for channel in CHANNELS
    }
    masks = {
        compartment: tifffile.imread(str(mask_paths[compartment]))
        for compartment in compartments
    }
    # Cheap here (arrays are already loaded for feature extraction below) --
    # a fail-fast guard against computing features on misaligned data. This
    # result also becomes the images.image_assets validation record below.
    alignment = validate_aligned_shapes(images, masks)
    if not alignment["valid"]:
        raise SystemExit(
            f"Channel and mask arrays are not aligned by shape for {image_id}:\n"
            f"{json.dumps(alignment, indent=2)}"
        )

    git_revision = git_commit(args.repo_root)
    all_valid = True

    if not args.compartment:
        # Whole image set (the Nextflow fan-out's only mode): also build and
        # write images.image_assets here, reusing the channel/mask arrays
        # already loaded above -- no extra I/O, no separate task. Skipped
        # under --compartment since that mode only loads one mask, not the
        # full 4 image_assets needs.
        assets = build_image_assets(manifest, images, masks, args.run_id, git_revision)
        assets_target_dir = args.outdir / "warehouse" / "images" / "image_assets"
        assets_target_dir.mkdir(parents=True, exist_ok=True)
        assets.to_parquet(assets_target_dir / f"{image_id}.parquet", index=False)
        assets_validation = {
            "valid": bool(alignment["valid"]),
            "row_count": int(assets.shape[0]),
            "column_count": int(assets.shape[1]),
            "alignment": alignment,
        }
        assets_metadata_dir = args.outdir / "metadata" / "images" / "image_assets"
        assets_metadata_dir.mkdir(parents=True, exist_ok=True)
        (assets_metadata_dir / f"{image_id}.validation.json").write_text(
            json.dumps(assets_validation, indent=2)
        )

    for compartment in compartments:
        profiles, validation, timings = extract_compartment_profile(
            zapi=zapi,
            compartment=compartment,
            mask=masks[compartment],
            images=images,
            manifest=manifest,
        )
        slug = compartment_slug(compartment)
        table_slug = f"{slug}_profiles"
        target_dir = args.outdir / "warehouse" / "profiles" / table_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        profiles.to_parquet(target_dir / f"{image_id}.parquet", index=False)
        run_record = {
            "run_id": args.run_id,
            "command": " ".join(sys.argv),
            "mode": "compartment",
            "compartment": compartment,
            "git_commit": git_revision,
            "zedprofiler_version": zapi["version"],
            "python_version": platform.python_version(),
            "timings_seconds": timings,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "validation_status": "pass" if validation["valid"] else "fail",
            "quality_warning_count": len(validation.get("quality_warnings", [])),
        }
        # Sidecars live under metadata/, mirroring the data path, not
        # alongside the parquet -- the warehouse directories (profiles/,
        # images/) hold data files only.
        metadata_dir = args.outdir / "metadata" / "profiles" / table_slug
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / f"{image_id}.validation.json").write_text(
            json.dumps({"compartments": {compartment: validation}}, indent=2)
        )
        (metadata_dir / f"{image_id}.run_record.json").write_text(
            json.dumps(run_record, indent=2)
        )
        all_valid = all_valid and bool(validation["valid"])
        if not validation["valid"]:
            print(json.dumps(validation, indent=2), file=sys.stderr)

    if not all_valid:
        return 1
    print(
        "NF1_FEATURIZE_COMPARTMENT_OK "
        f"compartments={','.join(compartments)} "
        f"elapsed={round(time.perf_counter() - started, 3)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
