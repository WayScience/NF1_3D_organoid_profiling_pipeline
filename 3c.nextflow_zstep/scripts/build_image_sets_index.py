#!/usr/bin/env python3
"""Scan source_root for every complete z-step image set and write a small
CSV index -- the 3c.nextflow_zstep adaptation of 3b.nextflow_production's
build_image_sets_index.py, with two extra dimensions:

1. mask_group: every non-0.1um z-step directory carries two segmentation
   mask sets (native + from_0_1um; see build_manifest.py), so most
   (patient, well_fov) pairs emit TWO index rows -- one per mask set --
   and each row is one image set / one FEATURIZE_IMAGE_SET task. The 0.1
   um groups have only the native mask set (their own z-stack IS the
   0.1 um z-stack), so they emit one row.

2. z_step is stamped as a column too (derived from the patient
   directory's -Z-{step} suffix) -- featurize_image_set.nf reads it to
   size each task's Slurm memory/time, since a 0.1 um image set is ~10x
   the voxel count of a 1 um one and needs its own allocation.

Everything a manifest needs beyond (patient, well_fov, mask_group) is
either a fixed workflow-wide constant or deterministically derivable by
build_manifest() at task time via the same glob logic this script reuses
to discover image sets in the first place -- see build_manifest.py's
resolve_manifest() and read_image_sets_index(), which is what actually
reads the CSV this script writes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_manifest import (
    MASK_GROUPS,
    ZSTEP_SUFFIX_RE,
    complete_image_sets,
    load_channel_mapping,
    repo_root,
    z_step_for_patient,
)


def discover_image_sets(
    source_root: Path, patients: list[str] | None
) -> list[tuple[str, str, str, float]]:
    """Return (patient, well_fov, mask_group, z_step) for every complete
    image set found under source_root/data/<patient>/, reusing
    build_manifest.py's own completeness check (every channel and every
    compartment mask present) per mask group so this index only ever
    lists image sets that are actually ready to run."""
    data_root = source_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    mapping = load_channel_mapping(repo_root() / "config" / "channel_mapping.toml")

    patient_names = patients or sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and ZSTEP_SUFFIX_RE.search(path.name)
    )

    found: list[tuple[str, str, str, float]] = []
    for patient in patient_names:
        z_step = z_step_for_patient(patient)
        for mask_group in MASK_GROUPS:
            try:
                complete = complete_image_sets(
                    source_root, patient, mapping, None, mask_group
                )
            except FileNotFoundError:
                # This mask group's root directory does not exist for this
                # patient -- expected for from_0_1um on the 0.1 um groups.
                continue
            for well_fov, _channels, _masks in complete:
                found.append((patient, well_fov, mask_group, z_step))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Workflow db root containing data/<patient>/... "
        "(e.g. /pl/active/koala/nf1-3d-zstep-workflow-db).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="CSV index to write: one patient,well_fov,mask_group,z_step row "
        "per image set.",
    )
    parser.add_argument(
        "--patient",
        action="append",
        dest="patients",
        default=[],
        help="Restrict the scan to one patient directory (repeatable). "
        "Default: every *_T1-Z-* directory under data/.",
    )
    args = parser.parse_args()

    image_sets = discover_image_sets(args.source_root.expanduser(), args.patients)
    if not image_sets:
        raise SystemExit(f"No complete image sets found under {args.source_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient", "well_fov", "mask_group", "z_step"])
        writer.writerows(image_sets)

    print(f"Wrote {len(image_sets)} image sets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
