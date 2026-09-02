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

`patients.txt` lists the production batch, one patient per line. The first
nine were staged and verified first; `NF0055_T1`, `SARCO219_T2`, and
`SARCO361_T1` were added in a follow-up pass; `NF0037_T1_CQ1` was added
after that, per review request (WayScience/NF1_3D_organoid_profiling_pipeline#161):

```text
NF0014_T1
NF0014_T2
NF0016_T1
NF0018_T6
NF0021_T1
NF0030_T1
NF0035_T1
NF0037_T1
NF0037_T1_CQ1
NF0040_T1
NF0055_T1
SARCO219_T2
SARCO361_T1
```

Confirmed present on bandicoot at prep time, with these well/FOV counts
(`ls zstack_images/ | wc -l`, each patient's `segmentation_masks/` count is
one higher -- an extra non-well_fov entry in that listing, not investigated
further here):

| Patient | zstack well/FOV dirs | Status |
| --- | --- | --- |
| NF0014_T1 | 104 | staged, verified complete 104/104 |
| NF0014_T2 | 350 | staged, verified complete 350/350 |
| NF0016_T1 | 122 | staged, verified complete 122/122 |
| NF0018_T6 | 160 | staged, verified complete 160/160 |
| NF0021_T1 | 348 | staged, verified complete 348/348 |
| NF0030_T1 | 207 | staged, verified complete 203/207 (4 wells missing masks on bandicoot itself: F9-3, G2-1, G2-2, G2-3 -- see below) |
| NF0035_T1 | 349 | staged, verified complete 348/349 (1 well missing masks on bandicoot itself: C9-7 -- see below) |
| NF0037_T1 | 420 | staged, verified complete 420/420 |
| NF0037_T1_CQ1 | 693 | staged, verified complete 693/693 |
| NF0040_T1 | 420 | staged, verified complete 420/420 |
| NF0055_T1 | 420 | staged, verified complete 419/420 (1 well missing masks on bandicoot itself: D8-2 -- see below) |
| SARCO219_T2 | 199 | staged, verified complete 199/199 |
| SARCO361_T1 | 350 | staged, verified complete 350/350 |

All 13 patients staged and verified: **4,136 of 4,142 well/FOVs complete**.
That's ~4,142 well/FOV directories across 13 patients -- large enough that
this transfer should run under `nohup`/`tmux`/`screen` or as its own job,
not a short interactive shell, and large enough to be worth the same
production-scale caution `3a.nextflow_pilot/PLAN.md`'s "Production-scale
(4200 image-set) time estimate" section already worked through for the
Nextflow run itself.

The six-well shortfall across `NF0030_T1`/`NF0035_T1`/`NF0055_T1` is a
**source** data gap, not a transfer bug -- confirmed by checking bandicoot
directly: every one of these wells has only `nuclei_mask.tiff` under
`segmentation_masks/`, missing `cell_mask`, `cytoplasm_mask`, and
`organoid_mask`:

```text
~/mnt/bandicoot/NF1_organoid_data/data/NF0030_T1/segmentation_masks/F9-3/
~/mnt/bandicoot/NF1_organoid_data/data/NF0030_T1/segmentation_masks/G2-1/
~/mnt/bandicoot/NF1_organoid_data/data/NF0030_T1/segmentation_masks/G2-2/
~/mnt/bandicoot/NF1_organoid_data/data/NF0030_T1/segmentation_masks/G2-3/
~/mnt/bandicoot/NF1_organoid_data/data/NF0035_T1/segmentation_masks/C9-7/
~/mnt/bandicoot/NF1_organoid_data/data/NF0055_T1/segmentation_masks/D8-2/
```

Compare e.g. `.../NF0030_T1/segmentation_masks/F9-3/` (incomplete) against
`.../NF0030_T1/segmentation_masks/F9-1/` (complete, all four masks present).

## Running the transfer

```bash
cd 3b.nextflow_production/staging

# see what would move without copying anything
./stage_from_bandicoot.sh --dry-run

# stage every patient in patients.txt
./stage_from_bandicoot.sh

# stage one patient only (e.g. to test the path end-to-end first)
./stage_from_bandicoot.sh --patient NF0014_T1

# stage a specific subset (e.g. patients added in a follow-up pass) without
# re-running the full patients.txt list
printf 'NF0055_T1\nSARCO219_T2\nSARCO361_T1\n' > /tmp/new_patients.txt
./stage_from_bandicoot.sh --patients-file /tmp/new_patients.txt
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
