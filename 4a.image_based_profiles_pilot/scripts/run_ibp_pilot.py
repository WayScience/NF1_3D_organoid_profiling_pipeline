#!/usr/bin/env python3
"""Pilot driver: run IBP stage 4 step 3 against a ZedProfiler warehouse.

For each reference image set (manifest/reference_image_sets.yaml):

1. Build step 3's expected input parquet files directly from the warehouse
   (build_ibp_inputs_from_warehouse.py) -- this replaces IBP steps 00/0a/1/2,
   which exist only to produce the same shape of data from the older
   CellProfiler-style output.
2. Run `4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`
   completely unmodified, as a subprocess, with `utils/src` (the
   image_analysis_3D package step 3 imports from) added to PYTHONPATH.
   Only the lightweight submodules step 3 actually touches are exercised
   (pandas/numpy/matplotlib/scikit-image/tqdm) -- none of the heavier
   torch/napari/cellpose dependencies declared in utils/pyproject.toml are
   needed for this.
3. Copy step 3's *_related.parquet outputs into the source warehouse's own
   directory, under a new `ibp/` folder alongside `profiles/`/`images/` --
   same one-file-per-image-set convention as `profiles/<compartment>_profiles/`,
   additive only.

No formal validation.json/run_record.json for this pilot -- just a visible
pass/fail summary per image set.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PILOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PILOT_ROOT.parent
IBP_STEP3_SCRIPT = (
    REPO_ROOT
    / "4.processing_image_based_profiles"
    / "scripts"
    / "3.organoid_cell_relationship.py"
)
IBP_SUBPARENT_NAME = "image_based_profiles_pilot_zedprofiler"


def run_one_image_set(
    warehouse_dir: Path, patient: str, well_fov: str
) -> dict[str, object]:
    build_inputs = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_ibp_inputs_from_warehouse.py"),
            "--warehouse-dir",
            str(warehouse_dir),
            "--repo-root",
            str(REPO_ROOT),
            "--patient",
            patient,
            "--well-fov",
            well_fov,
            "--image-based-profiles-subparent-name",
            IBP_SUBPARENT_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(build_inputs.stdout.strip())

    env = dict(os.environ)
    utils_src = str(REPO_ROOT / "utils" / "src")
    env["PYTHONPATH"] = (
        f"{utils_src}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else utils_src
    )
    step3 = subprocess.run(
        [
            sys.executable,
            str(IBP_STEP3_SCRIPT),
            "--patient",
            patient,
            "--well_fov",
            well_fov,
            "--image_based_profiles_subparent_name",
            IBP_SUBPARENT_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    print(step3.stdout[-2000:])
    if step3.stderr.strip():
        print(step3.stderr[-2000:], file=sys.stderr)

    related_dir = (
        REPO_ROOT
        / "data"
        / patient
        / IBP_SUBPARENT_NAME
        / "1.related_profiles"
        / well_fov
    )
    sc_related = pd.read_parquet(related_dir / f"sc_profiles_{well_fov}_related.parquet")
    organoid_related = pd.read_parquet(
        related_dir / f"organoid_profiles_{well_fov}_related.parquet"
    )
    nucleocentric_related = pd.read_parquet(
        related_dir / f"nucleocentric_profiles_{well_fov}_related.parquet"
    )

    image_id = str(sc_related["Metadata_Imaging_ImageID"].iloc[0])

    ibp_dir = warehouse_dir / "ibp"
    for name, df in (
        ("sc_profiles_related", sc_related),
        ("organoid_profiles_related", organoid_related),
        ("nucleocentric_profiles_related", nucleocentric_related),
    ):
        table_dir = ibp_dir / name
        table_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(table_dir / f"{image_id}.parquet", index=False)

    assigned = int((sc_related["ParentOrganoid"] != -1).sum())
    unassigned = int((sc_related["ParentOrganoid"] == -1).sum())

    return {
        "patient": patient,
        "well_fov": well_fov,
        "image_id": image_id,
        "sc_rows": sc_related.shape[0],
        "organoid_rows": organoid_related.shape[0],
        "cells_assigned_to_organoid": assigned,
        "cells_unassigned": unassigned,
        "organoid_sc_count_max": (
            int(organoid_related["OrganoidSingleCellCount"].max())
            if not organoid_related.empty
            else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-dir",
        required=True,
        type=Path,
        help="Path to a ZedProfiler warehouse dir (contains warehouse.duckdb, profiles/, images/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PILOT_ROOT / "manifest" / "reference_image_sets.yaml",
    )
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    warehouse_dir = args.warehouse_dir.resolve(strict=True)

    results = []
    all_ok = True
    for entry in manifest["image_sets"]:
        patient, well_fov = entry["patient"], entry["well_fov"]
        try:
            result = run_one_image_set(warehouse_dir, patient, well_fov)
            result["status"] = "ok"
        except subprocess.CalledProcessError as error:
            all_ok = False
            print(
                f"FAILED {patient}/{well_fov}: {error.cmd} exited {error.returncode}\n"
                f"stdout: {error.stdout}\nstderr: {error.stderr}",
                file=sys.stderr,
            )
            result = {"patient": patient, "well_fov": well_fov, "status": "failed"}
        except Exception as error:  # noqa: BLE001 -- one bad image set shouldn't
            # abort the whole pilot run or drop the rest of the summary; this
            # is a driver over a small manifest of image sets, not a single
            # task, so failures here (a missing output file, a malformed
            # parquet, etc.) should be reported per-image-set like the
            # subprocess-failure case above rather than crashing the script.
            all_ok = False
            print(f"FAILED {patient}/{well_fov}: {error!r}", file=sys.stderr)
            result = {"patient": patient, "well_fov": well_fov, "status": "failed"}
        results.append(result)

    print("\n=== NF1_IBP_PILOT_SUMMARY ===")
    for result in results:
        print(result)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
