# Staging: bandicoot -> Mac -> PetaLibrary (koala)

This step copies the z-step experiment's raw data from the bandicoot
Isilon share into the z-step workflow's PetaLibrary (koala) data root,
before any Nextflow run can see it. `../workflows/featurize_image_set.nf`
and `../scripts/*.py` all read from the PetaLibrary copy, never from
bandicoot directly -- bandicoot is not reachable from Alpine at all
(checked Nov 2026: no `~/mnt` on login nodes, compute nodes, or
persistence1). The lab Mac can reach both, so staging runs there.

## The two hops

```text
1. lab Mac:   mount_isilon                      -> ~/mnt/bandicoot   (Isilon SMB share)
2. lab Mac:   staging/stage_from_bandicoot.sh    (rsync -e ssh: bandicoot mount -> alpine:$DEST_ROOT directly)
```

Writes go straight over `rsync -e ssh` to Alpine -- there is no local
mount of koala involved. An earlier version of this script wrote through
an sshfs/macfuse mount of `/pl/active` and hit two real failures on
2026-11-05 (see "Why not sshfs" below); going straight over SSH avoids
both, and was also meaningfully faster in practice (direct rsync protocol
vs. FUSE-mediated local writes).

After the run completes, results are copied back the other way (koala ->
bandicoot) so everything lives with the rest of the project's data; see
"Copying results back to bandicoot" below.

## Why not sshfs

The script used to write through `~/mnt/alpine/active` (`sshfs
alpine:/pl/active ...`). Two real failures on 2026-11-05 killed that
approach:

1. The sshfs mount silently dropped mid-session, leaving an ordinary
   local directory at the same path. rsync then "succeeded" writing
   ~75 GB to the Mac's own disk instead of koala -- nothing landed
   remotely, and local disk nearly filled (`df -h /` hit 138 MiB free).
2. Even with the mount freshly re-established, transfers reproducibly
   died with `Socket is not connected` / `Device not configured` at
   almost exactly 3.68 GB into a session (8 x 494 MB files) -- a
   macFUSE/sshfs boundary bug, not a data problem: the exact same file
   transferred fine via direct `rsync -e ssh`, bypassing the FUSE layer
   entirely, immediately afterward.

Going straight over SSH sidesteps both: there is no local mount to
silently drop, and no FUSE layer to hit a byte-count boundary bug in. If
you ever see either symptom again, it will be from something re-adding
an sshfs mount into the path -- don't.

## Source and destination layout

Bandicoot's actual directory names are **not** a clean 1:1 with the two
mask groups the experiment wants. Verified by hash-comparing real files
across every z-step/patient combination (2026-11-05):

```text
segmentation_masks/                       segmentation 1 (native).
  Present for every group EXCEPT NF0055_T1-Z-0.1.
segmentation_masks_from_0_1um/             STALE -- byte-identical to
  segmentation_masks/ everywhere both exist (confirmed on
  NF0037_T1-Z-1 and NF0037_T1-Z-0.1). An earlier run that apparently
  just reproduced the native mask. Never staged under its own name.
segmentation_masks_from_0_1um_adjusted/    segmentation 2, the real
  one -- present only for 0.2/0.5/1 um groups, dated 2026-10-27 (a
  later correction pass), genuinely different pixel content from
  native (confirmed via md5).
```

`NF0055_T1-Z-0.1` has no `segmentation_masks/` at all -- its only mask
dir is (confusingly) named `segmentation_masks_from_0_1um/`, but shape-
checked against its own 0.1um zstack (299 slices, matches) it IS the
native mask for this one patient. `stage_from_bandicoot.sh` renames it
on copy so every patient's native mask lands uniformly at
`segmentation_masks/` on the destination side, regardless of what it was
called on bandicoot. See the script's header comment for the full
evidence trail.

```text
bandicoot: ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/zstack_images/{well_fov}/
           ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/segmentation_masks/{well_fov}/                      (native, most patients)
           ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/segmentation_masks_from_0_1um/{well_fov}/           (native, NF0055_T1-Z-0.1 ONLY)
           ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/segmentation_masks_from_0_1um_adjusted/{well_fov}/  (segmentation 2, 0.2/0.5/1 um groups only)
alpine:    /pl/active/koala/nf1-3d-zstep-workflow-db/data/{patient}/zstack_images/{well_fov}/
           /pl/active/koala/nf1-3d-zstep-workflow-db/data/{patient}/segmentation_masks/{well_fov}/                      (native, always this name)
           /pl/active/koala/nf1-3d-zstep-workflow-db/data/{patient}/segmentation_masks_from_0_1um_adjusted/{well_fov}/   (segmentation 2, when it exists)
```

`nf1-3d-zstep-workflow-db` is a new, isolated PetaLibrary folder, parallel
to the pilot's `nf1-3d-pilot-workflow-db` and production's
`nf1-3d-production-workflow-db` but distinct from both -- nothing under
the pilot or production folders is touched by this workflow.

## Patient groups and expected counts

`patients.txt` lists the eight z-step groups (NF0037 and NF0055 at
0.1/0.2/0.5/1 um each). Expected image sets once staged:

```text
NF0037: 3 FOVs (F4-1, F4-2, F4-3)
  x (0.1um: 1 mask group; 0.2/0.5/1um: 2 mask groups each)
  = 3 x (1 + 2 + 2 + 2) = 21
NF0055: 6 FOVs (E4-1-0, E4-3-0, E4-5-0, G5-1-0, G5-2-0, G5-5-0)
  = 6 x 7 = 42
total: 63 image sets
```

The 0.1 um groups have only one real mask set (native -- "segmented from
the 0.1 um stack, subsampled to itself" is just identity) -- see "Source
and destination layout" above for exactly which directory supplies it
per patient. The staging script treats a missing
`segmentation_masks_from_0_1um_adjusted/` as an optional skip and a
missing native mask (both candidate directories absent) as a required-
directory failure.

Total transfer size is tens of GB (a 0.1 um channel TIFF alone is
~2.5 GB), so run the stage under `tmux`/`nohup`, not a short interactive
shell. rsync-based: safe to re-run, safe to interrupt and resume.

## Running the stage

```bash
cd 3c.nextflow_zstep

# bandicoot mounted, and alpine reachable over SSH?
ls ~/mnt/bandicoot/NF1_organoid_data/data >/dev/null && echo bandicoot OK
ssh -o ConnectTimeout=15 alpine true && echo alpine OK

# preview, then stage
staging/stage_from_bandicoot.sh --dry-run
staging/stage_from_bandicoot.sh

# one group only (e.g. to test the path end-to-end first)
staging/stage_from_bandicoot.sh --patient NF0037_T1-Z-1
```

Per-patient, per-subdirectory logs land in `staging/logs/` (gitignored).
Override `--bandicoot-root`/`--alpine-host`/`--dest-root` (or the
`BANDICOOT_ROOT`/`ALPINE_HOST`/`DEST_ROOT` environment variables) if
bandicoot mounts somewhere other than `~/mnt/bandicoot` or your SSH alias
for Alpine isn't `alpine`.

## Verifying completeness

Don't trust the script's exit code alone -- verify with the workflow's own
discovery tooling, ON ALPINE (where the workflow runs):

```bash
cd 3c.nextflow_zstep
python3 scripts/build_image_sets_index.py \
  --source-root /pl/active/koala/nf1-3d-zstep-workflow-db \
  --output manifest/image_sets_index.csv
```

That script only lists a `(patient, well_fov, mask_group)` triple once
every channel TIFF and every compartment mask is present, so **63 rows =
fully staged**. A shortfall usually means a well/FOV is missing one
channel or mask on bandicoot, not a transfer bug -- check the specific
well/FOV under `data/{patient}/` on both sides before re-running the
transfer for it.

## Copying results back to bandicoot

After a completed run (see `../README.md`), copy the run folder back so
the z-step results live alongside the rest of the project's data:

```bash
# from the lab Mac
rsync -a --partial -e ssh \
  alpine:/pl/active/koala/nf1-3d-zstep-workflow-db/results/<run_id>/ \
  ~/mnt/bandicoot/NF1_organoid_data/data/zstep_zedprofiler_results/<run_id>/
```

The exact destination under bandicoot is a lab convention choice; keep it
outside `data/{patient}/` so featurization outputs never get confused
with the input layout the staging script copies.
