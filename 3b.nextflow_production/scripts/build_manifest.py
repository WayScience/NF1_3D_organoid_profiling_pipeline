#!/usr/bin/env python3
"""Build or validate the one-image-set pilot manifest."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

from manifest_io import dump_manifest, load_manifest, require_manifest_paths

CHANNELS = ("DNA", "ER", "AGP", "Mito")
COMPARTMENTS = ("Nuclei", "Cell", "Cytoplasm", "Organoid")
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
FEATURE_FAMILIES = (
    "VolumeSizeShape",
    "Intensity",
    "Texture",
    "Colocalization",
    "Neighbors",
    "Granularity",
)
# One explanatory comment per top-level manifest field, written directly
# above that field by dump_manifest() -- added after review questions kept
# coming up about what individual fields meant (e.g. compartment_primary_
# channels). Keep in sync with build_manifest()'s returned keys below.
FIELD_COMMENTS: dict[str, str] = {
    "schema_version": "Manifest schema version, for future migrations.",
    "source_image_root": "Root directory this image set's files were staged under.",
    "Metadata_Biology_PatientTumor": "Patient/tumor identifier, e.g. NF0055_T1.",
    "Metadata_Biology_PatientID": "Patient identifier without the tumor suffix, e.g. NF0055.",
    "Metadata_Experiment_PlateID": "Plate identifier (matches PatientTumor in this pilot).",
    "Metadata_Experiment_WellID": "Well identifier, e.g. B10.",
    "Metadata_Imaging_FieldID": "Field-of-view identifier within the well, e.g. 1.",
    "Metadata_Imaging_ImageID": "Unique ID for this image set: patient__plate__well__field.",
    "compartment": "Legacy single-compartment fallback for older tooling -- see compartments below for the real list this pilot uses.",
    "compartments": "Every compartment ZEDProfiler extracts features for in this image set.",
    "z_spacing": "Z-axis voxel spacing (microns), used for anisotropy-aware features.",
    "xy_spacing": "XY-axis voxel spacing (microns), used for anisotropy-aware features.",
    "mask_path": "Legacy single-compartment fallback for older tooling -- see mask_paths below.",
    "mask_paths": "Segmentation mask TIFF path, one per compartment.",
    "channel_paths": "Raw channel image TIFF path, one per channel.",
    "channel_codes": "Wavelength/filter code per channel (e.g. DNA=405nm).",
    "compartment_primary_channels": "The channel each compartment's mask was actually segmented from (e.g. Nuclei from DNA, Cell/Cytoplasm/Organoid from AGP). Provenance metadata only -- not used in any computation, just stamped onto every profile row and the image_assets table.",
    "compartment_primary_channel_codes": "Wavelength code matching compartment_primary_channels.",
    "compartment_seed_channels": "The channel used to seed this compartment's segmentation, if any (e.g. Cell/Cytoplasm are watershed-seeded from the Nuclei/DNA segmentation). Empty string if the compartment isn't seeded from another one.",
    "compartment_seed_channel_codes": "Wavelength code matching compartment_seed_channels.",
    "compartment_segmentation_methods": "Short description of how each compartment's mask was produced.",
    "feature_families": "ZEDProfiler feature families this pilot expects and validates for every compartment.",
}


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
    entries = [path for path in directory.iterdir() if path.is_file()]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(
            path for path in entries if fnmatch.fnmatchcase(path.name, pattern)
        )
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
) -> list[tuple[str, Path, Path, dict[str, Path], dict[str, Path]]]:
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
        mask_paths: dict[str, Path] = {}
        for compartment in COMPARTMENTS:
            found = find_one(mask_patterns(compartment), mask_dir)
            if found is not None:
                mask_paths[compartment] = found.resolve()
        if len(channel_paths) == len(CHANNELS) and len(mask_paths) == len(COMPARTMENTS):
            complete.append(
                (
                    well_fov,
                    image_dir.resolve(),
                    mask_dir.resolve(),
                    channel_paths,
                    mask_paths,
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

    selected_well_fov, _image_dir, _mask_dir, channel_paths, mask_paths = complete[0]
    well, field = parse_well_fov(selected_well_fov)
    patient_id = patient.split("_", maxsplit=1)[0]

    return {
        "schema_version": "0.1.0-production",
        "source_image_root": str(source_root.resolve()),
        "Metadata_Biology_PatientTumor": patient,
        "Metadata_Biology_PatientID": patient_id,
        "Metadata_Experiment_PlateID": patient,
        "Metadata_Experiment_WellID": well,
        "Metadata_Imaging_FieldID": field,
        "Metadata_Imaging_ImageID": image_id(patient, patient, well, field),
        "compartment": "Nuclei",
        "compartments": list(COMPARTMENTS),
        "z_spacing": 1.0,
        "xy_spacing": 0.1,
        "mask_path": str(mask_paths["Nuclei"]),
        "mask_paths": {
            compartment: str(mask_paths[compartment]) for compartment in COMPARTMENTS
        },
        "channel_paths": {channel: str(channel_paths[channel]) for channel in CHANNELS},
        "channel_codes": {channel: str(mapping[channel]) for channel in CHANNELS},
        "compartment_primary_channels": dict(COMPARTMENT_PRIMARY_CHANNELS),
        "compartment_primary_channel_codes": {
            compartment: str(mapping[channel])
            for compartment, channel in COMPARTMENT_PRIMARY_CHANNELS.items()
        },
        "compartment_seed_channels": dict(COMPARTMENT_SEED_CHANNELS),
        "compartment_seed_channel_codes": {
            compartment: str(mapping[channel]) if channel else ""
            for compartment, channel in COMPARTMENT_SEED_CHANNELS.items()
        },
        "compartment_segmentation_methods": dict(COMPARTMENT_SEGMENTATION_METHODS),
        "feature_families": list(FEATURE_FAMILIES),
    }


def resolve_manifest(
    manifest_path: Path | None,
    patient: str | None,
    well_fov: str | None,
    source_root: Path | None,
) -> dict[str, object]:
    """Load a manifest from an explicit YAML path, or derive one on the fly
    from (source_root, patient, well_fov) via build_manifest() -- the
    index-driven path that avoids needing one YAML file per image set,
    since everything except patient/well/field/paths is a fixed pilot-wide
    constant and the paths themselves are already deterministically
    derivable by glob."""
    if manifest_path:
        return load_manifest(manifest_path)
    if patient and well_fov and source_root:
        return build_manifest(source_root, patient, well_fov)
    raise SystemExit(
        "Provide --manifest, or all of --patient/--well-fov/--source-root"
    )


def read_image_sets_index(path: Path) -> list[tuple[str, str]]:
    """Read one (patient, well_fov) pair per row from a CSV index -- the
    lightweight replacement for a whole YAML manifest file per image set.
    Expects a header row with at least 'patient' and 'well_fov' columns."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"patient", "well_fov"} - fieldnames
        if missing:
            raise SystemExit(
                f"{path}: CSV header missing required column(s): {', '.join(sorted(missing))}"
            )
        return [
            (row["patient"].strip(), row["well_fov"].strip())
            for row in reader
            if row.get("patient", "").strip()
        ]


def slug_for_image_set(patient: str, well_fov: str) -> str:
    """Nextflow-safe slug for a (patient, well_fov) pair, matching the
    lowercase/alnum-underscore convention featurize_image_set.nf already
    uses for manifest-filename-derived slugs."""
    return re.sub(r"[^a-z0-9_]+", "_", f"{patient}_{well_fov}".lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--patient")
    parser.add_argument("--well-fov")
    parser.add_argument(
        "--output", type=Path, default=Path("manifest/generated_manifest.yaml")
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
    if not args.patient:
        parser.error("--patient is required unless --validate-only is used")

    manifest = build_manifest(
        args.source_root.expanduser(), args.patient, args.well_fov
    )
    dump_manifest(manifest, output, field_comments=FIELD_COMMENTS)
    print(f"Wrote manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
