#!/usr/bin/env python3
"""Check which image sets already have a complete, valid warehouse output
and which still need FEATURIZE_IMAGE_SET to run -- the
3c.nextflow_zstep adaptation of 3b.nextflow_production's
plan_image_sets.py, extended with the mask_group dimension (one image
set = one (patient, well_fov, mask_group) triple) and per-z-step task
resource sizing.

Read-only and cheap regardless of scale: stats parquet files and reads
small JSON validation sidecars only, never a parquet file's row data.
Nextflow's own `-resume` cache can't be relied on for this -- this
workflow never configures a persistent workDir across separate `make
submit` invocations, so there is nothing durable for `-resume` to resume
from, and even with one, its cache key is based on task inputs, not on
whether a finished, valid warehouse output already exists. This script
checks the warehouse directly instead: it is the source of truth for
"already done".

Emits one "<slug>\\t<pending|done>\\t<resources>" line per image set to
stdout, in the same order the image sets were given, so the caller (the
PLAN_IMAGE_SETS process in featurize_image_set.nf) can filter its
FEATURIZE_IMAGE_SET fan-out channel before ever submitting a Slurm task
for an image set that is already finished -- and can size each pending
task's Slurm request from the same line, without reimplementing the
sizing table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from build_manifest import read_image_sets_index, resolve_manifest, slug_for_image_set

COMPARTMENTS = ("Nuclei", "Cell", "Cytoplasm", "Organoid")

# Per-z-step Slurm sizing for one FEATURIZE_IMAGE_SET task. Anchored on
# the production run's observed numbers (3b full-run: a typical 63-slice
# image set peaked at ~14.5 GB RSS and ~10 min - 5 h wall with 2 cpus +
# 32 GB requested; zedprofiler holds 4 uint16 channels + 4 masks plus
# per-object float copies, so RSS scales ~linearly with voxel count):
#   z=1.0   ~53 slices  -> production-like:          48 GB,  2 cpus, 12h
#   z=0.5  ~106 slices  -> 2x voxels:                64 GB,  4 cpus, 18h
#   z=0.2  ~264 slices  -> 5x voxels:                128 GB,  8 cpus, 24h
#   z=0.1  ~527 slices  -> 10x voxels:              224 GB,  8 cpus, 24h
# Walltime is capped at acpu's 24h MaxTime. Memory ceilings are
# deliberately generous against node sizes (245 GB on the 64-cpu nodes):
# a single 0.1 um task at 224 GB still fits, and smaller tasks flow past
# it through the queue while a big task waits for a node. All of this is
# a starting point to be validated against the first real tasks' sacct
# numbers -- see README.md "Resource sizing".
Z_STEP_RESOURCES = {
    0.1: {"memory": "224 GB", "cpus": 8, "time": "24h"},
    0.2: {"memory": "128 GB", "cpus": 8, "time": "24h"},
    0.5: {"memory": "64 GB", "cpus": 4, "time": "18h"},
    1.0: {"memory": "48 GB", "cpus": 2, "time": "12h"},
}


def resources_for_z_step(z_step: float) -> dict[str, object]:
    """Slurm resource request for one image set at this z-step size.
    Unknown z-steps fall back to the nearest known step's sizing rather
    than crashing the planner -- the runner validates the actual data at
    task time anyway."""
    key = min(Z_STEP_RESOURCES, key=lambda candidate: abs(candidate - z_step))
    return dict(Z_STEP_RESOURCES[key])


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
        outdir / "metadata" / "images" / "image_assets" / f"{image_id}.validation.json"
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
        for patient, well_fov, mask_group in read_image_sets_index(
            args.image_sets_index
        ):
            slug = slug_for_image_set(patient, well_fov, mask_group)
            manifest = resolve_manifest(
                None, patient, well_fov, args.source_root, mask_group
            )
            entries.append((slug, manifest))
    else:
        for manifest_path in args.manifests:
            slug = slug_for_manifest_path(manifest_path)
            manifest = resolve_manifest(manifest_path, None, None, None, None)
            entries.append((slug, manifest))

    done_count = 0
    for slug, manifest in entries:
        complete = is_image_set_complete(args.outdir, manifest)
        done_count += int(complete)
        resources = resources_for_z_step(
            float(str(manifest["Metadata_Experiment_ZStep"]))
        )
        print(
            f"{slug}\t{'done' if complete else 'pending'}\t"
            f"mem={resources['memory']}\tcpus={resources['cpus']}\t"
            f"time={resources['time']}"
        )

    print(
        f"NF1_PLAN_IMAGE_SETS done={done_count} pending={len(entries) - done_count} "
        f"total={len(entries)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
