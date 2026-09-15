# IBP production: stage 4 (organoid-cell relationships) on the full ZEDProfiler warehouse

## What this is

The production-scale sibling of `4a.image_based_profiles_pilot` -- same approach (a lightweight adapter + driver that runs stage 4 step 3, `3.organoid_cell_relationship.py`, directly against a ZEDProfiler warehouse instead of reimplementing IBP steps 00/0a/1/2), pointed at the **full production dataset** instead of 2 reference image sets.

Dataset: `nf1-production-full-run-2` (zedprofiler 0.1.4), staged as 4,136 patient/well-FOV entries (`manifest/image_sets_index.csv`, copied from `3b.nextflow_production/manifest/image_sets_index.csv`) -- see Findings below for exactly how many of those this pilot's own run actually produced output for, and why.
Warehouse: `/pl/active/koala/nf1-3d-production-workflow-db/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse`.

Same folder shape as the pilot -- `environments/`, `manifest/`, `scripts/`, this README -- copied and adapted, not imported/shared, matching this repo's own established convention for pilot/production sibling folders (`3a.nextflow_pilot` -> `3b.nextflow_production` is "the same code, copied and renamed", per that folder's own README).

## The one real change from the pilot: reading files directly, not through a glob view

The pilot's `build_ibp_inputs_from_warehouse.py` queried `warehouse.duckdb`'s `profiles.*` views, each one `read_parquet(glob)` over _every_ file in that table, filtered down to one image set per call.
At pilot scale (2 files/table) that's free.
At production scale (4,134 files/table) it is not: a single such query, measured directly against the real production warehouse before writing any of this folder's code, spiked to **35GB RAM and still hadn't finished after 9+ minutes** -- both locally over sshfs and natively on Alpine's login node (OOM-killed there at ~20s).
This is a query-planning cost problem, not a location problem or a data-volume problem.

The fix, verified against the same real warehouse: read each image set's own parquet files **directly by their known path** (`profiles/nuclei_profiles/{image_id}.parquet`, etc.) instead of through the glob view -- **0.8 seconds** for all 4 compartment files combined.
This also removes the per-image-set `WHERE Metadata_Biology_PatientTumor = ? AND ...` SQL filtering entirely: one file already _is_ one image set, so there's nothing to filter.
`build_ibp_inputs_from_warehouse.py` no longer opens `warehouse.duckdb` at all; the metadata dedup logic (which of Nuclei/Cell/Cytoplasm's copy of a shared/per-compartment metadata column to keep when merging them) is unchanged from the pilot, translated from SQL `EXCLUDE` clauses to pandas `.drop()`.

`image_id()` reproduces the exact convention `3b.nextflow_production/scripts/build_manifest.py`'s own `image_id()` already uses for this dataset: `"__".join([patient, patient, well, f"F{field}"])` (plate == patient here) -- confirmed against real filenames, e.g. `NF0014_T1__NF0014_T1__C10__F1.parquet`.
Reimplemented locally rather than imported, matching this repo's copy-not-share convention between sibling folders.

Everything else about the pilot's design carries over unchanged: step 3 (`3.organoid_cell_relationship.py`) is invoked completely unmodified; no column renaming happens anywhere (ZEDProfiler's `VolumeSizeShape` naming and `Metadata_Object_ObjectID` flow straight through, since step 3 itself was already widened in the pilot's own PR to accept them); no nucleocentric file is read or written (step 3 treats that input as optional).
See `4a.image_based_profiles_pilot/README.md` for the full history behind those decisions.

## What's different for production scale

- **Input**: `--image-sets-index manifest/image_sets_index.csv` (a CSV of `patient,well_fov` rows, same format 3b's own index uses) instead of a 2-entry YAML manifest.
- **Resumability**: before processing an image set, `run_ibp_production.py` skips it if a dedicated completion marker (`ibp/.complete/<image_id>`) already exists -- mirrors `3b.nextflow_production`'s own `PLAN_IMAGE_SETS` skip-already-landed behavior.
  This matters here because a 4,000+-item run is long enough that interruption/resume is a real scenario, not a hypothetical -- and in practice, every verification run below built on the previous one's output rather than redoing it.
  Both output parquet files are written to temp paths and atomically renamed into place before the marker is created, so a run interrupted mid-write can never be mistaken for complete (see the review-feedback update below for the full story -- this was a real gap, not a hypothetical one, since this script has genuinely been interrupted mid-run during this project's own work).
- **Light parallelism**: a `ThreadPoolExecutor` (`--workers`, default 8) runs image sets concurrently.
  Threads, not `multiprocessing`, because the actual work happens inside two `subprocess.run()` calls per image set (adapter + step 3), which release the GIL while blocked -- no process-spawn/pickling overhead needed.
  Measured directly against the real warehouse: 4 workers ~1.9s/image-set, 8 workers ~1.15s/image-set, 16 workers ~1.07s/image-set -- 8 is the practical default (subprocess-startup overhead dominates well before 16 cores saturate, so going higher buys little).
- **Progress + run record**: prints one line per failure immediately, plus a periodic progress line (`--progress-every`, default 100) instead of one line per success.
  Writes `ibp_run_record.json` **next to** (not inside) `warehouse/ibp/`, so it isn't picked up by the `ibp.*` DuckDB views' own glob -- summarizes attempted/succeeded/failed/already-skipped counts and the full list of failures, matching `3b.nextflow_production`'s own `run_record.json` convention for a real production artifact (the pilot explicitly skipped this, since 2 image sets don't need an audit trail).
- **`--limit N`**: process only the first N pending image sets -- used for the dry runs documented below before committing to the full index.
- Execution model: this machine, not Alpine/Slurm -- confirmed appropriate given the direct-file-read fix above makes per-image-set cost ~1s, dominated by Python/subprocess startup rather than I/O.
  A full Nextflow/Slurm DAG (`PLAN_IMAGE_SETS`/`FEATURIZE_IMAGE_SET`/`BUILD_WAREHOUSE`-style, as 3b uses) would be pure overhead here: step 3 has no heavy image I/O to amortize against, unlike ZEDProfiler's own ~19 min/image-set extraction, which is what justified that architecture for 3b.

## Environment

Same as the pilot's (`pandas`, `pyarrow`, `duckdb`, `scikit-image`, `matplotlib`, `tqdm`; `pyyaml` dropped here since the CSV index doesn't need it), plus `PYTHONPATH` pointed at `utils/src` so step 3's `import image_analysis_3D...` resolves without installing the heavier root/`utils` environment (torch/napari/cellpose not needed -- see the pilot's own README for the import trace behind that).

```bash
cd 4b.image_based_profiles_production
uv sync --project environments --locked
uv run --project environments python scripts/run_ibp_production.py \
  --warehouse-dir /pl/active/koala/nf1-3d-production-workflow-db/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse \
  --workers 8
```

## Warehouse layout addition

Same convention as the pilot: outputs land back in the **source warehouse's own directory**, under `ibp/` alongside `profiles/`/`images/`.

```text
warehouse/
  profiles/...                                  <- unchanged, ZEDProfiler's own output
  images/...                                     <- unchanged
  warehouse.duckdb                               <- unchanged base views; gains 2 new ibp.* views
  ibp/                                            <- this run's output
    sc_profiles_related/<image_id>.parquet        <- Nuclei+Cell+Cytoplasm + ParentOrganoid + shell/distance features
    organoid_profiles_related/<image_id>.parquet   <- Organoid + OrganoidSingleCellCount
```

`ibp_run_record.json` lands one level up, next to `warehouse/`, not inside `ibp/` -- `results-zedprofiler-0.1.4/nf1-production-full-run-2/ibp_run_record.json`.

## Findings

**Full run completed against all 4,136 staged image sets.**
Final result: **4,133 succeeded, 3 failed**, all `FileNotFoundError`s -- but checked individually rather than assumed identical, since 4,133 didn't quite match the source ZEDProfiler warehouse's own reported 4,134-image-set landed count:

- `NF0014_T1/F11-3` and `SARCO361_T1/D2-3` are genuine gaps -- well F11 only has fields 1-2 on disk, well D2 only has fields 1/2/4/5/6/7 (no field 3 either), confirmed by listing each well's actual files.
- `NF0018_T6/E-3` is **not** a missing-data gap: the real data exists as `NF0018_T6__NF0018_T6__E-3__F1.parquet` -- this plate has a well literally named `E-3` (containing a hyphen as part of the well name itself, field 1), which `well_fov.rsplit("-", 1)` (splitting "well-field" on the last hyphen) instead parses as well `E`, field `3`, a combination that doesn't exist.
  The ambiguity is in the source index's own `well-field` string convention for this one irregular well name, not something resolvable from the CSV row alone; left as a known, documented limitation rather than guessed at, since a wrong guess would be worse than a clear failure.
  4,133 + 3 = 4,136 -- every entry in the index is accounted for, nothing silently dropped, and the one entry that isn't a true data gap is called out precisely rather than lumped in with the real ones.
  `sc_profiles_related/` and `organoid_profiles_related/` each hold exactly 4,133 parquet files.
  Cross-checked against the pilot's own known-good reference image set, `NF0014_T1/C4-2`, which is part of this dataset too: 42 rows, 42 assigned, exact match.
  Spot-checked 25 random image sets for duplicate columns -- none found.
  The final invocation (after the bug fixes below) processed its 1,065 still-pending image sets in 1,143.8 seconds (~19 minutes) at 8 workers, per `ibp_run_record.json`'s own `elapsed_seconds` -- consistent with the ~1.15s/image-set throughput measured earlier.
  An earlier invocation processed most of the remaining ~3,071 image sets before being stopped partway through to fix the bugs below (no `ibp_run_record.json` was written for that interrupted run, so its own elapsed time isn't recorded, but its real, correct output was preserved and simply skipped by the resumability check on the next run rather than redone).

**One real bug found and fixed, discovered only at full-dataset scale**: the pilot's `if sc_df.empty: raise SystemExit(...)` (inherited unchanged from `4a.image_based_profiles_pilot`, since its 2 reference image sets never hit this condition) treated an empty single-cell merge as a hard failure.
At production scale, ~1.4% of image sets (about 42 of the first ~3,000 processed) have real, detected Nuclei but zero Cell/Cytoplasm objects -- a genuine, valid segmentation outcome, not an error.
Failing these discarded their organoid data too, even when organoid segmentation succeeded (e.g. one such image set had 7 real organoids that would have simply vanished from this pass).
Step 3 (`3.organoid_cell_relationship.py`) already has explicit, tested fallback handling for exactly this case -- its own "Empty dataframe fallbacks" section adds a single all-`None` placeholder row to an empty `sc_profile_df` and passes an empty `organoid_profile_df` through unchanged, specifically so `5.combining_profiles`'s `union_by_name` downstream can handle it.
Fixed by removing the hard-stop in `build_ibp_inputs_from_warehouse.py` and letting step 3 handle it as designed -- the same "let step 3 handle it" precedent already used for the optional nucleocentric input.
A genuinely missing image set (no files at all) still fails loudly via `FileNotFoundError` before this point is ever reached, so real gaps are unaffected.

**A second bug, exposed by the first**: with the placeholder row's every column (including `Metadata_Imaging_ImageID`) set to `None`, `run_ibp_production.py`'s original `iid = str(sc_related["Metadata_Imaging_ImageID"].iloc[0])` -- read back from the _result_ data rather than computed from the known patient/well/field -- silently named the output file `None.parquet`.
Caught by inspecting the very first post-fix test case rather than assumed to be correct; two stray `None.parquet` files (one in each `ibp/` table) had already landed in the real production warehouse from that one test and were deleted immediately.
Fixed by computing `iid` directly via `image_id(patient, well, field)` instead, which is correct regardless of whether step 3 emitted real data or a placeholder.
A related counting bug in the same code path -- `ParentOrganoid != -1`/`== -1` treats a placeholder row's `None` as "not equal to -1" (confirmed directly: `pd.Series([None]) != -1` is `True`), which would have miscounted a placeholder-only image set as "1 cell assigned" -- was fixed alongside it by excluding rows with a null `object_id` from both the assigned/unassigned counts and the reported `sc_rows`.
Verified against the real triggering image set (`NF0037_T1_CQ1/B2-17`) both before and after each fix.

**Convenience `ibp.*` DuckDB views work for exploration, not full-table aggregates, at this scale.** `SELECT ... LIMIT 5` against `ibp.sc_profiles_related` returns instantly.
`SELECT COUNT(*)` against the same view did not finish within a reasonable wait -- the identical glob-view query-planning cost documented above for `profiles.*`, since `ibp.*` uses the same `read_parquet(glob)` pattern over the same file count.
Row totals for these findings were confirmed by counting files on disk instead (`find ... | wc -l`), not by querying the view.

Not yet checked: whether `NF0014_T1/F11-3` or `SARCO361_T1/D2-3` (the 2 genuine gaps) are needed for a specific downstream analysis -- they'd need to be re-staged and re-extracted through ZEDProfiler first, since this folder can only process what already exists in the warehouse.

**Update (review feedback), `NF0018_T6/E-3` recovered:** per review, `well_fov.rsplit("-", 1)` was replaced with a `parse_well_fov()` regex-based split (same convention `3b.nextflow_production/scripts/build_manifest.py` already uses, reimplemented locally in `build_ibp_inputs_from_warehouse.py`), which correctly falls back to treating the whole string as the well name when it doesn't start with a `letter+digit(s)` pattern.
Verified against every row of `manifest/image_sets_index.csv` before adopting it: identical to the old parsing for 4,135 of 4,136 rows, and only differs -- correctly -- for `NF0018_T6/E-3`.
Re-running with the fix in place recovered this image set for real: **4,134 succeed now**, only the 2 genuine gaps remain.

**Update (review feedback), atomic writes + a real completion marker:** `run_one_image_set()` used to write both output parquet files directly to their final paths, and resumability checked only whether `sc_profiles_related/<id>.parquet` existed.
A process killed mid-write (this script _has_ been interrupted for real during this work) could leave a truncated file at a final path, or leave `sc_profiles_related` written but `organoid_profiles_related` missing for that image set -- either way, indistinguishable from a genuinely complete result to the old resumability check, silently skipped forever on every future run.
Fixed by writing both outputs to temp paths first, atomically `os.replace()`-ing both into place only once both writes succeed, then creating a dedicated marker file (`ibp/.complete/<image_id>`) that resumability now checks instead of either parquet file directly.
Verified directly: a simulated failure between the two writes leaves no trace at either final path, no marker, and no leftover temp file.
The existing 4,133 (now 4,134) already-correct image sets from before this fix don't have markers yet -- backfilled them in one pass (checked both output files exist per image set first; found zero inconsistent pairs) rather than needlessly reprocessing an already-correct hour of output.

**Update (review feedback), `chmod` on the run record itself now affects the exit code:** the final `chmod 770` on `ibp_run_record.json` discarded its return code entirely, unlike the two `chmod` calls just above it for `ibp/` and `warehouse.duckdb`.
Can't be reflected in the run record's own `permissions_ok` field -- the file has to exist before it can be `chmod`'d -- but a failure here now still flips `permissions_ok` to `False` and fails the script's own exit code, instead of being silently discarded.
