# Staging: bandicoot -> PetaLibrary (koala)

This step copies raw data for the production patient set from the bandicoot
Isilon share into this workflow's PetaLibrary (koala) data root, before any
Nextflow run can see it. `../workflows/featurize_image_set.nf` and
`../scripts/*.py` all read from the PetaLibrary copy, never from bandicoot
directly (bandicoot is not reliably reachable from Alpine compute nodes; see
`../../3a.nextflow_pilot/PLAN.md`'s "Open decisions" section).

## Source and destination layout

```text
bandicoot:   ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/zstack_images/{well_fov}/
             ~/mnt/bandicoot/NF1_organoid_data/data/{patient}/segmentation_masks/{well_fov}/
petalibrary: ~/mnt/alpine/active/koala/nf1-3d-production-workflow-db/data/{patient}/zstack_images/{well_fov}/
             ~/mnt/alpine/active/koala/nf1-3d-production-workflow-db/data/{patient}/segmentation_masks/{well_fov}/
```

On Alpine itself, the PetaLibrary side of that same path is
`/pl/active/koala/nf1-3d-production-workflow-db/data/...` (same share,
different mount point -- see `../README.md`).

`nf1-3d-production-workflow-db` is a new, isolated PetaLibrary folder,
parallel to the pilot's `nf1-3d-pilot-workflow-db` but distinct from it --
nothing under the pilot folder is touched by this workflow.

## Patient list

`patients.txt` lists the initial production batch, one patient per line:

```text
NF0014_T1
NF0014_T2
NF0016_T1
NF0018_T6
NF0021_T1
NF0030_T1
NF0035_T1
NF0037_T1
NF0040_T1
```

Confirmed present on bandicoot at prep time, with these well/FOV counts
(`ls zstack_images/ | wc -l`, each patient's `segmentation_masks/` count is
one higher -- an extra non-well_fov entry in that listing, not investigated
further here):

| Patient | zstack well/FOV dirs |
| --- | --- |
| NF0014_T1 | 104 |
| NF0014_T2 | 350 |
| NF0016_T1 | 122 |
| NF0018_T6 | 160 |
| NF0021_T1 | 348 |
| NF0030_T1 | 207 |
| NF0035_T1 | 349 |
| NF0037_T1 | 420 |
| NF0040_T1 | 420 |

That's ~2,480 well/FOV directories across 9 patients -- large enough that
this transfer should run under `nohup`/`tmux`/`screen` or as its own job,
not a short interactive shell, and large enough to be worth the same
production-scale caution `3a.nextflow_pilot/PLAN.md`'s "Production-scale
(4200 image-set) time estimate" section already worked through for the
Nextflow run itself.

## Running the transfer

```bash
cd 3b.nextflow_production/staging

# see what would move without copying anything
./stage_from_bandicoot.sh --dry-run

# stage every patient in patients.txt
./stage_from_bandicoot.sh

# stage one patient only (e.g. to test the path end-to-end first)
./stage_from_bandicoot.sh --patient NF0014_T1
```

Uses `rsync -a --partial`, so it is safe to re-run: unchanged files are
skipped, and an interrupted run resumes rather than restarting. Per-patient,
per-subdirectory logs land in `staging/logs/` (gitignored).

Override source/destination roots with `--bandicoot-root`/`--dest-root` or
the `BANDICOOT_ROOT`/`DEST_ROOT` environment variables if running from a host
where the mounts land somewhere other than `~/mnt/...` (e.g. directly on
Alpine, where the PetaLibrary side is `/pl/active/koala/...`).

## Verifying completeness

Don't trust this script's exit code alone -- verify with the workflow's own
discovery tooling, which only counts a `(patient, well_fov)` pair once every
channel TIFF and every compartment mask is actually present:

```bash
cd ..
python3 scripts/build_image_sets_index.py \
  --source-root ~/mnt/alpine/active/koala/nf1-3d-production-workflow-db \
  --output manifest/image_sets_index.csv
```

Compare the per-patient row counts in the resulting CSV against the table
above. A shortfall usually means a well/FOV is missing one channel or mask
on bandicoot, not a transfer bug -- check the specific well/FOV under
`data/{patient}/{zstack_images,segmentation_masks}/` on both sides before
re-running the transfer for it.
