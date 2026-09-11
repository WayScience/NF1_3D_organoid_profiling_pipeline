#!/usr/bin/env python3
"""Scan source_root for every complete (patient, well_fov) image set and
write a small CSV index -- the lightweight replacement for one YAML
manifest file per image set.

Everything a manifest needs beyond patient/well_fov is either a fixed
pilot-wide constant (compartments, channel codes, segmentation methods,
feature families) or deterministically derivable by build_manifest() at
task time via the same glob logic this script reuses to discover image
sets in the first place -- see build_manifest.py's resolve_manifest() and
read_image_sets_index(), which is what actually reads the CSV this script
writes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_manifest import complete_well_fovs, load_channel_mapping, repo_root


def discover_image_sets(source_root: Path) -> list[tuple[str, str]]:
    """Return (patient, well_fov) for every complete image set found under
    source_root/data/<patient>/{zstack_images,segmentation_masks}/<well_fov>,
    reusing build_manifest.py's own completeness check (every channel and
    every compartment mask present) so this index only ever lists image
    sets that are actually ready to run."""
    data_root = source_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    mapping = load_channel_mapping(repo_root() / "config" / "channel_mapping.toml")

    found: list[tuple[str, str]] = []
    for patient_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        patient = patient_dir.name
        try:
            complete = complete_well_fovs(source_root, patient, mapping, None)
        except FileNotFoundError:
            continue
        for well_fov, *_rest in complete:
            found.append((patient, well_fov))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    image_sets = discover_image_sets(args.source_root.expanduser())
    if not image_sets:
        raise SystemExit(f"No complete image sets found under {args.source_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient", "well_fov"])
        writer.writerows(image_sets)

    print(f"Wrote {len(image_sets)} image sets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
