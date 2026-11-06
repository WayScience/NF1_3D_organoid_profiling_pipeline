# 3c.nextflow_zstep -- z-step experiment rollout plan

## Scope

Adapt `3b.nextflow_production`'s validated Nextflow/Slurm fan-out engine
(proven on the full 13-patient, ~4,136-image-set production batch) to the
z-step technical-analysis experiment: **8 patient groups x ~2 mask groups
x 3-6 FOVs = 63 image sets**, staged from bandicoot to a new, isolated
PetaLibrary root. This answers two experimental questions the
production run doesn't: how do extracted features change as the z-step
grows (0.1 -> 1 um), and does the mask source (native vs 0.1 um-derived)
change them comparably.

Explicitly **not** in this plan:

- Any change to ZEDProfiler feature logic, compartment/channel roles, or
  the warehouse table shape. Those are unchanged from production
  (including the zedprofiler==0.1.4 pin and both of 0.1.4's
  correctness fixes -- physical-space distances, empty-frame dtype
  handling -- verified during `nf1-production-full-run-2`).
- Any production-data rerun. The pilot/production warehouses are
  untouched; this run writes only under `nf1-3d-zstep-workflow-db`.
- GPU / deep-learning features. Same reserved, unimplemented seam the
  pilot and production left in `conf/curc_alpine.config`.
- Downstream z-step comparisons themselves (IOU/mAP-style analyses of the
  resulting profiles) -- this plan ends at a validated warehouse with the
  z-step/mask-group metadata stamped on every row.

## Data source and staging

Source of truth is bandicoot, z-step layout:

```text
~/mnt/bandicoot/NF1_organoid_data/data/{patient_group}/zstack_images/{well_fov}/
~/mnt/bandicoot/NF1_organoid_data/data/{patient_group}/segmentation_masks/{well_fov}/                      (native, most patients)
~/mnt/bandicoot/NF1_organoid_data/data/{patient_group}/segmentation_masks_from_0_1um/{well_fov}/           (native, NF0055_T1-Z-0.1 ONLY -- see staging/README.md)
~/mnt/bandicoot/NF1_organoid_data/data/{patient_group}/segmentation_masks_from_0_1um_adjusted/{well_fov}/  (segmentation 2, 0.2/0.5/1 um only)
```

Bandicoot's naming is messier than the technical-analysis README table
implies: a stale `segmentation_masks_from_0_1um/` directory (byte-
identical to native, confirmed via md5) exists alongside the real
`segmentation_masks_from_0_1um_adjusted/`, and `NF0055_T1-Z-0.1` has no
`segmentation_masks/` at all -- its native mask is the (confusingly
named) `segmentation_masks_from_0_1um/` directory instead. The staging
script handles all of this; see its header comment and
`staging/README.md` for the full evidence trail.

**Patient groups for this batch** (`staging/patients.txt`):
NF0037_T1-Z-{0.1,0.2,0.5,1} (FOVs F4-1, F4-2, F4-3) and
NF0055_T1-Z-{0.1,0.2,0.5,1} (FOVs E4-1-0, E4-3-0, E4-5-0, G5-1-0,
G5-2-0, G5-5-0) -- the groups from the technical-analysis README's
z-step table, with F4-1 included after confirming all complete FOVs
should run (~63 image sets vs the ~61 estimate).

**bandicoot is not reachable from Alpine** (verified Nov 2026: no
`~/mnt` on login nodes, compute nodes, or persistence1). Staging runs on
the lab Mac with bandicoot mounted (Isilon via `mount_isilon`) and
Alpine reached directly over `rsync -e ssh` -- NOT through an
sshfs/macfuse mount of koala, which proved unreliable (see
`staging/README.md` "Why not sshfs" for the two concrete failures this
caused on 2026-11-05). See `staging/README.md` for the exact hops and
the copy-back of results afterwards.

**Working copy: a new PetaLibrary root.** `nf1-3d-zstep-workflow-db` was
created alongside `nf1-3d-pilot-workflow-db` and
`nf1-3d-production-workflow-db`, matching their `data/` + `results/` +
`tools/` layout, so all three share the same operational pattern on
Alpine without any one touching another's files.

## Reused, unchanged from production

- `scripts/manifest_io.py`, the manifest shape beyond the new fields,
  `Metadata_*` identifier rules, warehouse directory shape
  (`warehouse/images/image_assets/`,
  `warehouse/profiles/<compartment>_profiles/`, `metadata/` sidecars),
  and compartment segmentation relationships.
- `run_zedprofiler_image_set.py`'s featurization call sequence (the
  exact per-family compute invocations, granularity parameters, and
  validation rules), `build_warehouse_from_compartments.py`'s
  landed-ness handling and run record, `build_duckdb_views.py`'s view
  construction, `smoke_synthetic.py`'s wiring check.
- The skip-already-completed-image-sets logic and the shared-`uv`-
  environment worker launch pattern in `workflows/featurize_image_set.nf`.
- The zedprofiler pin (`0.1.4`) and the `uv`-based isolated environment.

## Changed for the z-step experiment

- **Mask-group dimension**: one image set = one (patient, `well_fov`,
  `mask_group`) triple; the index CSV gains `mask_group` and `z_step`
  columns; manifests point at `segmentation_masks/` (native) or
  `segmentation_masks_from_0_1um_adjusted/` (`from_0_1um`);
  `Metadata_Imaging_ImageID` gains a `maskgroup_{group}` suffix.
- **Derived z_spacing**: parsed from the patient directory's
  `_T1-Z-{step}` suffix instead of hardcoded 1.0. Also stamped as
  `Metadata_Experiment_ZStep` on every row, with
  `Metadata_Segmentation_MaskGroup`/`..._MaskGroupLabel` beside it.
- **Per-z-step Slurm sizing** (`Z_STEP_RESOURCES` in
  `plan_image_sets.py`): 48 GB/2 cpu/12h at 1 um up to 224 GB/8 cpu/24h
  at 0.1 um, flowing through the plan channel into FEATURIZE's dynamic
  directives. This replaces production's flat 32 GB request, which was
  sized for ~63-slice stacks; the 0.1 um stacks carry ~8.4x those
  voxels. The table is an anchored estimate, deliberately validated
  (step 4 below) before the full batch.
- **Renamed namespace**: `nf1-3d-zstep-workflow-db` root, `NF1_ZSTEP_*`
  env vars, `bin/nf1-nextflow-zstep`, `schema_version: 0.1.0-zstep`,
  `queueSize = 24` (63 tasks, unequal memory, backfill-friendly).
- **Staging on the lab Mac** (see Data source) -- production's
  Alpine-side staging assumption no longer holds.

## Execution sequence

1. **Stage data** (Mac: `staging/stage_from_bandicoot.sh`) for all 8
   patient groups. Verify on Alpine with `make build-image-sets-index`
   -- expect **63 rows**.
2. **One-time Alpine setup**: repo checkout under
   `/pl/active/koala/nf1-3d-zstep-workflow-db/`, `uv sync --project
environments`, JDK 21 + Nextflow launcher under the new `tools/`
   (copy from the production root's `tools/`, same versions, so this
   folder stays isolated).
3. **Build the image-sets index** once staging is verified complete.
4. **Single-set wiring test**: `make run` on NF0037_T1-Z-1/F4-2 native
   (the smallest, ~53-slice set) end-to-end against real staged data;
   then one 0.1 um set to get a real MaxRSS/Elapsed point for the
   biggest sizing class. Adjust `Z_STEP_RESOURCES` from the measured
   numbers before the full batch -- the table's whole purpose is to be
   corrected by exactly this step.
5. **Full batch run** (`make submit`), all 63 image sets.
6. **Validate**: warehouse `validation_status: pass`; spot-check row
   counts per (patient, z-step, mask_group); confirm both mask groups of
   each non-0.1 um FOV are present and distinguishable via
   `Metadata_Segmentation_MaskGroup`; run a couple of DuckDB joined-view
   queries filtered by z-step/mask-group to confirm they're the
   apples-to-apples comparison keys downstream analysis expects.
7. **Copy results back to bandicoot** (Mac: see `staging/README.md`) so
   the z-step profiles live with the source data.

## Open risks

- The 0.1 um sizing (224 GB / 24h) has never been exercised; step 4
  closes this gap for memory, but a 527-slice Granularity pass could
  still push walltime past 24h on slow nodes. If it does, split
  granularity off per-channel (the pilot once benchmarked a granularity
  fan-out) rather than raising acpu's MaxTime, which cannot exceed 24h.
- `perJobMemLimit = true` (Slurm's `MaxMemPerCPU=3840` accounting) means
  the 224 GB request requires its 8 cpus to legally cover it
  (8 x 3840 MB = 30.7 GB per-cpu ceiling is NOT how `--mem` works, but
  Slurm validates `--mem` against the _node's_ real memory under
  perJobMemLimit, so 224 GB on a 245 GB node is legal; still, re-check
  `scontrol show partition acpu` before submit in case the limits
  changed since Nov 2026).
- Bandicoot's directory naming doesn't exactly match the technical-
  analysis README table (a stale `segmentation_masks_from_0_1um/` dir,
  and `NF0055_T1-Z-0.1` lacking `segmentation_masks/` entirely) --
  resolved via byte-level verification (see "Data source and staging"
  above), not just trusting directory names. If a future patient group
  is added, re-verify with the same md5/shape checks before assuming
  the pattern holds.
- Bandicoot's SMB mount on the Mac requires VPN + working credentials
  (a stale keychain entry had to be fixed during prep) -- staging can
  fail on auth, not just on network. Retry with `mount_isilon` if the
  mount is empty.
- An earlier sshfs-mediated staging path failed twice for real on
  2026-11-05 (silent mount drop writing to local disk instead of koala;
  then a macFUSE boundary bug killing transfers at ~3.68 GB per
  session) -- fixed by switching to direct `rsync -e ssh`, which has no
  local mount to drop and no FUSE layer to hit the boundary bug in. See
  `staging/README.md` "Why not sshfs". If staging ever needs to write
  through a local mount again for some reason, re-add the device-ID
  guard that used to live in `stage_from_bandicoot.sh`.
- Cluster contention: 63 tasks at mixed sizes is small, but the two
  0.1 um FOVs' worth of 224 GB tasks (12 total) can each wait a long
  time for a node with that much free memory. Not a correctness risk;
  just don't judge the run by its stragglers.

## Success criteria

- Staging verified: index builds with exactly 63 rows.
- Single-set test passes for both a 1 um and a 0.1 um image set, with
  measured MaxRSS/Elapsed recorded in this PLAN.md before the full run.
- Full batch completes, exit `0`, `validation_status: pass` on every
  table; each non-0.1 um FOV present under BOTH mask groups; 0.1 um
  FOVs under native only.
- DuckDB views queryable with z-step/mask-group filters returning the
  expected FOV counts (3 FOVs x 2 mask groups per non-0.1 um NF0037
  z-step, etc.).
- Results copied back to bandicoot.

## Explicitly deferred

- Any downstream comparison analysis of the profiles (that's the
  technical-analysis notebooks' job, reading this warehouse).
- Job arrays, GPU/deep-learning execution, Isilon publishing as the
  durable warehouse location -- same deferrals as the pilot and
  production plans.
