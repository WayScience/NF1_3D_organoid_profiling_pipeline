#!/usr/bin/env python3
"""Production driver: run IBP stage 4 step 3 against a ZEDProfiler warehouse
for the full staged dataset (4,134+ image sets), not just the pilot's 2
reference image sets.

For each image set in `--image-sets-index` (a CSV of `patient,well_fov`
rows -- same format `3b.nextflow_production`'s own index uses):

1. Skip if `warehouse/ibp/sc_profiles_related/<image_id>.parquet` already
   exists -- resumability, mirroring 3b's own `PLAN_IMAGE_SETS`
   skip-already-landed behavior. Matters here because a 4,000+-item run is
   long enough that interruption/resume is a real scenario, not a
   hypothetical.
2. Build step 3's expected input parquet files directly from the warehouse
   (build_ibp_inputs_from_warehouse.py) -- production-scale version, reads
   each image set's own parquet files directly by path rather than through
   a glob-based DuckDB view (see that script's own docstring for why: the
   view approach was measured at 35GB RAM / 9+ minutes per image-set query
   at this scale, vs. 0.8s reading files directly).
3. Run `4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py`
   as a subprocess, exactly as the pilot does -- unmodified, with
   `utils/src` on `PYTHONPATH`.
4. Copy step 3's sc_profiles/organoid_profiles `*_related.parquet` outputs
   into the source warehouse's own directory, under `ibp/`, same
   one-file-per-image-set convention as `profiles/<compartment>_profiles/`.
   Nucleocentric output is not copied -- see the pilot's own README for why
   (ZEDProfiler produces no real Nucleocentric data, so that output is
   always empty).

Unlike the pilot (2 image sets, no formal run record needed), this script:

- Runs image sets concurrently via a `ThreadPoolExecutor` (`--workers`,
  default 8 -- measured as a good throughput/predictability tradeoff on a
  16-core machine) rather than sequentially. Threads, not `multiprocessing`,
  because the actual work happens inside two `subprocess.run()` calls per
  image set (adapter + step 3), which release the GIL while blocked -- no
  process-spawn/pickling overhead needed.
- Prints one line per failure immediately, plus a periodic progress line
  (every `--progress-every`, default 100) rather than one line per
  success -- thousands of pilot-style per-image lines would flood the
  terminal.
- Writes `run_record.json` next to (not inside) `warehouse/ibp/`, so it
  isn't picked up by the `ibp.*` DuckDB views' own glob -- summarizes
  attempted/succeeded/failed/skipped counts and the full list of failures,
  matching `3b.nextflow_production`'s own `run_record.json` convention for
  a real production artifact.
- Accepts `--limit N` to process only the first N pending image sets --
  for a small dry run before committing to the full index.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_ibp_inputs_from_warehouse import image_id, parse_well_fov  # noqa: E402

PROD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROD_ROOT.parent
IBP_STEP3_SCRIPT = (
    REPO_ROOT
    / "4.processing_image_based_profiles"
    / "scripts"
    / "3.organoid_cell_relationship.py"
)
IBP_SUBPARENT_NAME = "image_based_profiles_production_zedprofiler"


def read_image_sets_index(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [(row["patient"], row["well_fov"]) for row in reader]


def _complete_marker_path(warehouse_dir: Path, iid: str) -> Path:
    # A dedicated marker directory, not either output table itself: judging
    # completion by whether sc_profiles_related/<iid>.parquet merely exists
    # doesn't guarantee organoid_profiles_related/<iid>.parquet was ever
    # written too (a kill between the two), nor that either file finished
    # writing cleanly rather than being left truncated by an interrupted
    # process. The marker is only ever created after both real outputs are
    # atomically in place -- see run_one_image_set().
    return warehouse_dir / "ibp" / ".complete" / iid


def is_complete(warehouse_dir: Path, iid: str) -> bool:
    return _complete_marker_path(warehouse_dir, iid).exists()


def run_one_image_set(
    warehouse_dir: Path, patient: str, well_fov: str
) -> dict[str, object]:
    subprocess.run(
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

    env = dict(os.environ)
    utils_src = str(REPO_ROOT / "utils" / "src")
    env["PYTHONPATH"] = (
        f"{utils_src}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else utils_src
    )
    subprocess.run(
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

    related_dir = (
        REPO_ROOT
        / "data"
        / patient
        / IBP_SUBPARENT_NAME
        / "1.related_profiles"
        / well_fov
    )
    sc_related = pd.read_parquet(
        related_dir / f"sc_profiles_{well_fov}_related.parquet"
    )
    organoid_related = pd.read_parquet(
        related_dir / f"organoid_profiles_{well_fov}_related.parquet"
    )

    # Computed directly from patient/well_fov, not read back from
    # sc_related's own Metadata_Imaging_ImageID column: when no real single
    # cells exist for this image set (Nuclei detected but Cell/Cytoplasm
    # empty -- a real, valid condition, see build_ibp_inputs_from_warehouse.py),
    # step 3's own "add an NA placeholder row" fallback sets every column,
    # Metadata_Imaging_ImageID included, to None -- reading it back would
    # silently name the output file "None.parquet" instead of failing loudly.
    well, field = parse_well_fov(well_fov)
    iid = image_id(patient, well, field)

    # Write both outputs to temp paths first, and only rename either one
    # into place once both writes have succeeded -- a process killed
    # mid-write (this loop has been interrupted for real during this
    # project's own runs) must never leave a half-written or single-table
    # parquet file sitting at a final path where a later run's resumability
    # check could mistake it for a complete, valid result. tmp_paths is
    # cleared once os.replace() has consumed each one, so the finally block
    # only ever cleans up files that didn't make it to a successful rename.
    ibp_dir = warehouse_dir / "ibp"
    tmp_paths: list[Path] = []
    final_paths: list[Path] = []
    try:
        for name, df in (
            ("sc_profiles_related", sc_related),
            ("organoid_profiles_related", organoid_related),
        ):
            table_dir = ibp_dir / name
            table_dir.mkdir(parents=True, exist_ok=True)
            final_path = table_dir / f"{iid}.parquet"
            tmp_path = table_dir / f".{iid}.{os.getpid()}.tmp.parquet"
            df.to_parquet(tmp_path, index=False)
            tmp_paths.append(tmp_path)
            final_paths.append(final_path)

        for tmp_path, final_path in zip(tmp_paths, final_paths):
            os.replace(tmp_path, final_path)  # atomic on the same filesystem
        tmp_paths = []

        marker_path = _complete_marker_path(warehouse_dir, iid)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
    finally:
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)

    # Same placeholder row also has object_id/ParentOrganoid set to None,
    # which `!= -1`/`== -1` would otherwise miscount as "assigned" (None/NaN
    # compares not-equal to -1) -- excluded from both counts by requiring a
    # real (non-null) object_id, so a placeholder-only image set correctly
    # reports 0 assigned and 0 unassigned rather than 1 spurious assignment.
    has_real_cell = sc_related["object_id"].notna()
    assigned = int(((sc_related["ParentOrganoid"] != -1) & has_real_cell).sum())
    unassigned = int(((sc_related["ParentOrganoid"] == -1) & has_real_cell).sum())

    return {
        "patient": patient,
        "well_fov": well_fov,
        "image_id": iid,
        # Real cell count, not the raw parquet row count -- a
        # placeholder-only image set has 1 raw row (all-None) but 0 real
        # cells, per the same has_real_cell distinction as assigned/unassigned above.
        "sc_rows": int(has_real_cell.sum()),
        "organoid_rows": organoid_related.shape[0],
        "cells_assigned_to_organoid": assigned,
        "cells_unassigned": unassigned,
        "organoid_sc_count_max": (
            int(organoid_related["OrganoidSingleCellCount"].max())
            if not organoid_related.empty
            else 0
        ),
        "status": "ok",
    }


_IBP_TABLES = (
    "sc_profiles_related",
    "organoid_profiles_related",
)


def create_ibp_views(warehouse_dir: Path) -> None:
    """(Re)create convenience views over ibp/<table>/*.parquet in the
    warehouse's existing warehouse.duckdb, same as the pilot's own
    create_ibp_views() -- one-time metadata operation per run, so its cost
    is unrelated to image-set count."""
    duckdb_path = warehouse_dir / "warehouse.duckdb"
    if not duckdb_path.exists():
        print(
            f"NOTE: {duckdb_path} does not exist -- skipping ibp.* view creation.",
            file=sys.stderr,
        )
        return

    previous_cwd = Path.cwd()
    os.chdir(warehouse_dir)
    try:
        with duckdb.connect("warehouse.duckdb") as con:
            con.execute('CREATE SCHEMA IF NOT EXISTS "ibp"')
            for table in _IBP_TABLES:
                if not any((warehouse_dir / "ibp" / table).glob("*.parquet")):
                    continue
                con.execute(
                    f'CREATE OR REPLACE VIEW "ibp"."{table}" AS '
                    f"SELECT * FROM read_parquet('ibp/{table}/*.parquet')"
                )
        print(f"ibp.* views (re)created in {duckdb_path}")
    finally:
        os.chdir(previous_cwd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-dir",
        required=True,
        type=Path,
        help="Path to a ZEDProfiler warehouse dir (contains warehouse.duckdb, profiles/, images/)",
    )
    parser.add_argument(
        "--image-sets-index",
        type=Path,
        default=PROD_ROOT / "manifest" / "image_sets_index.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent image sets in flight (ThreadPoolExecutor workers). "
        "8 was measured as a good throughput/predictability tradeoff on a "
        "16-core local machine -- 16 workers only cut per-image-set time "
        "another ~7% (subprocess-startup overhead dominates, not CPU).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a progress line every N completed image sets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pending (not-already-done) image sets -- for a dry run.",
    )
    args = parser.parse_args()

    warehouse_dir = args.warehouse_dir.resolve(strict=True)
    all_entries = read_image_sets_index(args.image_sets_index)

    pending: list[tuple[str, str, str]] = []
    skipped = 0
    for patient, well_fov in all_entries:
        well, field = parse_well_fov(well_fov)
        iid = image_id(patient, well, field)
        if is_complete(warehouse_dir, iid):
            skipped += 1
            continue
        pending.append((patient, well_fov, iid))

    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"NF1_IBP_PRODUCTION_PLAN total={len(all_entries)} "
        f"already_done={skipped} pending_this_run={len(pending)} "
        f"workers={args.workers}"
    )

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    all_ok = True
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_image_set, warehouse_dir, patient, well_fov): (
                patient,
                well_fov,
            )
            for patient, well_fov, _ in pending
        }
        for future in as_completed(futures):
            patient, well_fov = futures[future]
            try:
                result = future.result()
            except subprocess.CalledProcessError as error:
                all_ok = False
                print(
                    f"FAILED {patient}/{well_fov}: {error.cmd} exited {error.returncode}\n"
                    f"stdout: {error.stdout}\nstderr: {error.stderr}",
                    file=sys.stderr,
                )
                result = {
                    "patient": patient,
                    "well_fov": well_fov,
                    "status": "failed",
                    "error": f"{error.cmd} exited {error.returncode}: {error.stderr[-500:]}",
                }
            except Exception as error:  # noqa: BLE001 -- one bad image set
                # shouldn't abort the whole run or drop the rest of the
                # summary; this driver processes thousands of image sets,
                # not one task, so failures here (a missing warehouse file,
                # a malformed parquet, etc.) should be reported per-image-set
                # rather than crashing the script.
                all_ok = False
                print(f"FAILED {patient}/{well_fov}: {error!r}", file=sys.stderr)
                result = {
                    "patient": patient,
                    "well_fov": well_fov,
                    "status": "failed",
                    "error": repr(error),
                }
            results.append(result)
            completed += 1
            if completed % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"NF1_IBP_PRODUCTION_PROGRESS completed={completed}/{len(pending)} "
                    f"elapsed={elapsed:.1f}s"
                )

    create_ibp_views(warehouse_dir)

    # koala is a shared group allocation. Same reasoning as the pilot's own
    # final chmod sweep: running from a local machine (not Alpine, where
    # the coordinator scripts set umask 007) means new/rewritten files pick
    # up whatever the local process's default umask is. check=False so one
    # chmod failing doesn't crash the whole run, but the failure still
    # needs to surface and fail the script -- not be silently discarded.
    permissions_ok = True
    ibp_chmod = subprocess.run(
        ["chmod", "-R", "770", str(warehouse_dir / "ibp")], check=False
    )
    if ibp_chmod.returncode != 0:
        permissions_ok = False
        print(
            f"WARNING: chmod -R 770 {warehouse_dir / 'ibp'} exited "
            f"{ibp_chmod.returncode} -- some output may not be group-writable "
            "on koala",
            file=sys.stderr,
        )
    duckdb_path = warehouse_dir / "warehouse.duckdb"
    if duckdb_path.exists():
        duckdb_chmod = subprocess.run(["chmod", "770", str(duckdb_path)], check=False)
        if duckdb_chmod.returncode != 0:
            permissions_ok = False
            print(
                f"WARNING: chmod 770 {duckdb_path} exited "
                f"{duckdb_chmod.returncode} -- warehouse.duckdb may not be "
                "group-writable on koala",
                file=sys.stderr,
            )

    finished_at = datetime.now(timezone.utc)
    failed_results = [r for r in results if r.get("status") == "failed"]
    succeeded_results = [r for r in results if r.get("status") == "ok"]
    run_record = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "warehouse_dir": str(warehouse_dir),
        "image_sets_index": str(args.image_sets_index),
        "workers": args.workers,
        "total_in_index": len(all_entries),
        "already_done_before_this_run": skipped,
        "attempted_this_run": len(pending),
        "succeeded_this_run": len(succeeded_results),
        "failed_this_run": len(failed_results),
        "failures": failed_results,
        "permissions_ok": permissions_ok,
    }
    run_record_path = warehouse_dir.parent / "ibp_run_record.json"
    run_record_path.write_text(json.dumps(run_record, indent=2))
    # Can't be reflected in run_record's own permissions_ok field above --
    # the file has to exist before it can be chmod'd -- but a failure here
    # must still fail the run rather than being silently discarded, so it's
    # folded into the function's own return code below, same as the other
    # two chmod calls above.
    run_record_chmod = subprocess.run(
        ["chmod", "770", str(run_record_path)], check=False
    )
    if run_record_chmod.returncode != 0:
        permissions_ok = False
        print(
            f"WARNING: chmod 770 {run_record_path} exited "
            f"{run_record_chmod.returncode} -- ibp_run_record.json may not "
            "be group-writable on koala",
            file=sys.stderr,
        )

    print("\n=== NF1_IBP_PRODUCTION_SUMMARY ===")
    print(
        f"total_in_index={len(all_entries)} already_done={skipped} "
        f"attempted={len(pending)} succeeded={len(succeeded_results)} "
        f"failed={len(failed_results)} permissions_ok={permissions_ok}"
    )
    print(f"run record: {run_record_path}")
    if failed_results:
        print("Failed image sets:")
        for result in failed_results:
            print(f"  {result['patient']}/{result['well_fov']}: {result.get('error')}")

    return 0 if (all_ok and permissions_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
