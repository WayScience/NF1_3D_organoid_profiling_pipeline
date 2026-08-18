#!/usr/bin/env python3
"""Check which image sets already have a complete, valid warehouse output
and which still need FEATURIZE_IMAGE_SET to run.

Read-only and cheap regardless of scale: stats parquet files and reads
small JSON validation sidecars only, never a parquet file's row data.
Nextflow's own `-resume` cache can't be relied on for this -- this pilot
never configures a persistent workDir across separate `make submit`
invocations, so there is nothing durable for `-resume` to resume from, and
even with one, its cache key is based on task inputs, not on whether a
finished, valid warehouse output already exists. This script checks the
warehouse directly instead: it is the source of truth for "already done".

Emits one "<slug>\\tdone" or "<slug>\\tpending" line per image set to
stdout, in the same order the image sets were given, so the caller (the
`PLAN_IMAGE_SETS` process in featurize_image_set.nf) can filter its
`FEATURIZE_IMAGE_SET` fan-out channel before ever submitting a Slurm task
for an image set that is already finished.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from build_manifest import read_image_sets_index, resolve_manifest, slug_for_image_set

COMPARTMENTS = ("Nuclei", "Cell", "Cytoplasm", "Organoid")


def slug_for_manifest_path(path: Path) -> str:
    """Mirrors featurize_image_set.nf's slugFor(): lowercased manifest
    filename stem, non-alnum/underscore runs collapsed to one underscore."""
    return re.sub(r"[^a-z0-9_]+", "_", path.stem.lower())


def is_image_set_complete(outdir: Path, manifest: dict[str, object]) -> bool:
    image_id = str(manifest["Metadata_Imaging_ImageID"])
    compartments = [
        str(c) for c in (manifest.get("compartments") or [manifest["compartment"]])
    ]

    assets_parquet = (
        outdir / "warehouse" / "images" / "image_assets" / f"{image_id}.parquet"
    )
    assets_validation = (
        outdir
        / "metadata"
        / "images"
        / "image_assets"
        / f"{image_id}.validation.json"
    )
    if not assets_parquet.is_file() or not assets_validation.is_file():
        return False
    try:
        if not json.loads(assets_validation.read_text()).get("valid"):
            return False
    except (json.JSONDecodeError, OSError):
        return False

    for compartment in compartments:
        table_slug = f"{compartment.lower()}_profiles"
        profile_parquet = (
            outdir / "warehouse" / "profiles" / table_slug / f"{image_id}.parquet"
        )
        validation_json = (
            outdir
            / "metadata"
            / "profiles"
            / table_slug
            / f"{image_id}.validation.json"
        )
        if not profile_parquet.is_file() or not validation_json.is_file():
            return False
        try:
            record = json.loads(validation_json.read_text())
            if not record.get("compartments", {}).get(compartment, {}).get("valid"):
                return False
        except (json.JSONDecodeError, OSError):
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        dest="manifests",
        default=[],
        help="Repeatable. Mutually exclusive with --image-sets-index.",
    )
    parser.add_argument("--image-sets-index", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    if bool(args.manifests) == bool(args.image_sets_index):
        raise SystemExit(
            "Provide either --manifest (repeatable) or --image-sets-index/--source-root"
        )

    entries: list[tuple[str, dict[str, object]]] = []
    if args.image_sets_index:
        if not args.source_root:
            raise SystemExit("--source-root is required with --image-sets-index")
        for patient, well_fov in read_image_sets_index(args.image_sets_index):
            slug = slug_for_image_set(patient, well_fov)
            manifest = resolve_manifest(None, patient, well_fov, args.source_root)
            entries.append((slug, manifest))
    else:
        for manifest_path in args.manifests:
            slug = slug_for_manifest_path(manifest_path)
            manifest = resolve_manifest(manifest_path, None, None, None)
            entries.append((slug, manifest))

    done_count = 0
    for slug, manifest in entries:
        complete = is_image_set_complete(args.outdir, manifest)
        done_count += int(complete)
        print(f"{slug}\t{'done' if complete else 'pending'}")

    print(
        f"NF1_PLAN_IMAGE_SETS done={done_count} pending={len(entries) - done_count} "
        f"total={len(entries)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
