#!/usr/bin/env python3
"""Build or validate the one-image-set pilot manifest."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

from manifest_io import dump_manifest, load_manifest, require_manifest_paths

CHANNELS = ("DNA", "ER", "AGP", "Mito")
FEATURE_FAMILIES = (
    "VolumeSizeShape",
    "Intensity",
    "Texture",
    "Colocalization",
    "Neighbors",
    "Granularity",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_well_fov(well_fov: str) -> tuple[str, str]:
    match = re.match(r"^([A-Ha-h][0-9]{1,2})[-_]?(.+)$", well_fov)
    if not match:
        return well_fov, "1"
    return match.group(1).upper(), str(match.group(2))


def image_id(patient: str, plate: str, well: str, field: str) -> str:
    return "__".join([patient, plate, well, f"F{field}"])


def find_one(patterns: list[str], directory: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    return sorted(set(candidates))[0] if candidates else None


def channel_patterns(well_fov: str, token: str) -> list[str]:
    return [
        f"{well_fov}_{token}.tif",
        f"{well_fov}_{token}.tiff",
        f"*{token}*.tif",
        f"*{token}*.tiff",
    ]


def mask_patterns(compartment: str) -> list[str]:
    lower = compartment.lower()
    title = compartment[0].upper() + compartment[1:]
    return [
        f"{lower}_mask.tif",
        f"{lower}_mask.tiff",
        f"{title}_mask.tif",
        f"{title}_mask.tiff",
        f"*{lower}*mask*.tif",
        f"*{lower}*mask*.tiff",
    ]


def load_channel_mapping(path: Path) -> dict[str, str]:
    data = tomllib.loads(path.read_text())
    return data["channel_mapping"]


def complete_well_fovs(
    source_root: Path, patient: str, mapping: dict[str, str], requested: str | None
) -> list[tuple[str, Path, Path, dict[str, Path], Path]]:
    image_root = source_root / "data" / patient / "zstack_images"
    mask_root = source_root / "data" / patient / "segmentation_masks"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    candidates = (
        [requested]
        if requested
        else sorted(path.name for path in image_root.iterdir() if path.is_dir())
    )
    complete = []
    for well_fov in candidates:
        if not well_fov:
            continue
        image_dir = image_root / well_fov
        mask_dir = mask_root / well_fov
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        channel_paths: dict[str, Path] = {}
        for channel in CHANNELS:
            found = find_one(channel_patterns(well_fov, mapping[channel]), image_dir)
            if found is not None:
                channel_paths[channel] = found.resolve()
        mask_path = find_one(mask_patterns("nuclei"), mask_dir)
        if len(channel_paths) == len(CHANNELS) and mask_path is not None:
            complete.append(
                (
                    well_fov,
                    image_dir.resolve(),
                    mask_dir.resolve(),
                    channel_paths,
                    mask_path.resolve(),
                )
            )
    return complete


def build_manifest(
    source_root: Path, patient: str, well_fov: str | None
) -> dict[str, object]:
    mapping = load_channel_mapping(repo_root() / "config" / "channel_mapping.toml")
    complete = complete_well_fovs(source_root, patient, mapping, well_fov)
    if not complete:
        requested = f" for {well_fov}" if well_fov else ""
        raise FileNotFoundError(
            f"No complete image set found{requested} under {source_root}"
        )

    selected_well_fov, _image_dir, _mask_dir, channel_paths, mask_path = complete[0]
    well, field = parse_well_fov(selected_well_fov)
    patient_id = patient.split("_", maxsplit=1)[0]

    return {
        "schema_version": "0.1.0-pilot",
        "source_image_root": str(source_root.resolve()),
        "Metadata_Biology_PatientTumor": patient,
        "Metadata_Biology_PatientID": patient_id,
        "Metadata_Experiment_PlateID": patient,
        "Metadata_Experiment_WellID": well,
        "Metadata_Imaging_FieldID": field,
        "Metadata_Imaging_ImageID": image_id(patient, patient, well, field),
        "compartment": "Nuclei",
        "z_spacing": 1.0,
        "xy_spacing": 0.1,
        "mask_path": str(mask_path),
        "channel_paths": {channel: str(channel_paths[channel]) for channel in CHANNELS},
        "feature_families": list(FEATURE_FAMILIES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--patient", default="NF0014_T1")
    parser.add_argument("--well-fov")
    parser.add_argument(
        "--output", type=Path, default=Path("manifest/smoke_test_image_set.yaml")
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    output = args.output
    if not output.is_absolute():
        output = Path.cwd() / output

    if args.validate_only:
        manifest = load_manifest(output)
        errors = require_manifest_paths(manifest)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"Manifest paths are valid: {output}")
        return 0

    if args.source_root is None:
        parser.error("--source-root is required unless --validate-only is used")

    manifest = build_manifest(
        args.source_root.expanduser(), args.patient, args.well_fov
    )
    dump_manifest(manifest, output)
    print(f"Wrote manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
