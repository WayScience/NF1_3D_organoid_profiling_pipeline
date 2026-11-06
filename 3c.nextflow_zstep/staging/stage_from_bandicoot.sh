#!/usr/bin/env bash
#
# stage_from_bandicoot.sh -- copy each listed z-step patient group's
# zstack_images/ and segmentation mask sets from the bandicoot Isilon share
# into the z-step workflow's PetaLibrary (koala) data root, preserving the
# well/FOV layout the manifest tooling expects.
#
# Source layout on bandicoot is NOT a clean 1:1 with the two mask groups
# the experiment wants (native / from_0_1um) -- verified by hash-comparing
# real files on 2026-11-05, across every z-step/patient combination:
#
#   segmentation_masks/                       segmentation 1 (native).
#     Present for every group EXCEPT NF0055_T1-Z-0.1.
#   segmentation_masks_from_0_1um/             STALE. Byte-identical to
#     segmentation_masks/ everywhere both exist (confirmed via md5 on
#     NF0037_T1-Z-1 and NF0037_T1-Z-0.1, nuclei_mask.tiff) -- an earlier,
#     apparently-buggy run that just reproduced the native mask instead of
#     a real 0.1um-derived one. NEVER staged as a distinct mask group.
#   segmentation_masks_from_0_1um_adjusted/    segmentation 2, the real
#     one: present only for 0.2/0.5/1 um groups (dated 2026-10-27, a
#     later correction pass over the above), genuinely different pixel
#     content from native (confirmed via md5 on NF0037_T1-Z-1).
#
# NF0055_T1-Z-0.1 is missing segmentation_masks/ entirely -- its only mask
# dir is (confusingly) named segmentation_masks_from_0_1um/, but shape-
# checked against its own 0.1um zstack (299 slices, matches) it IS the
# native mask for this one patient, just inconsistently named at the
# source. This script renames it on copy so the destination layout stays
# uniform: every patient's native mask lands at segmentation_masks/,
# regardless of what it was called on bandicoot.
#
# Destination layout (uniform across every patient, regardless of
# source-side naming quirks):
#
#   bandicoot:   {bandicoot_root}/{patient}/zstack_images/{well_fov}/
#                {bandicoot_root}/{patient}/segmentation_masks/{well_fov}/                      (native, most patients)
#                {bandicoot_root}/{patient}/segmentation_masks_from_0_1um/{well_fov}/            (native, NF0055_T1-Z-0.1 ONLY -- see above)
#                {bandicoot_root}/{patient}/segmentation_masks_from_0_1um_adjusted/{well_fov}/   (segmentation 2, 0.2/0.5/1 um groups only)
#   petalibrary: {dest_root}/data/{patient}/zstack_images/{well_fov}/
#                {dest_root}/data/{patient}/segmentation_masks/{well_fov}/                      (native, always this name)
#                {dest_root}/data/{patient}/segmentation_masks_from_0_1um_adjusted/{well_fov}/   (segmentation 2, when it exists)
#
# bandicoot is NOT reachable from Alpine (confirmed Nov 2026: no ~/mnt on
# login/compute/persistence1 nodes). It is reachable from a lab Mac, so
# this script runs ON THE MAC with both shares mounted:
#   ~/mnt/bandicoot   (Isilon, via mount_isilon)
#   ~/mnt/alpine      (sshfs: sshfs alpine:/pl/active ~/mnt/alpine/active)
# See staging/README.md for the full sequence and the post-run copy of
# results back to bandicoot.
#
# rsync-based: safe to re-run (skips unchanged files) and safe to
# interrupt and resume (--partial). The 0.1 um groups are ~2.5 GB per
# channel per FOV, so a full stage is tens of GB -- run it under
# nohup/tmux, not in a short interactive shell.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BANDICOOT_ROOT="${BANDICOOT_ROOT:-$HOME/mnt/bandicoot/NF1_organoid_data/data}"
DEST_ROOT="${DEST_ROOT:-$HOME/mnt/alpine/active/koala/nf1-3d-zstep-workflow-db/data}"
PATIENTS_FILE="${PATIENTS_FILE:-$SCRIPT_DIR/patients.txt}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"

DRY_RUN=0
ONE_PATIENT=""

usage() {
  cat <<'USAGE'
stage_from_bandicoot.sh

Usage:
  staging/stage_from_bandicoot.sh [--dry-run] [--patient PATIENT] \
    [--patients-file PATH] [--bandicoot-root PATH] [--dest-root PATH]

Options:
  --dry-run           Pass -n to rsync; report what would transfer without copying.
  --patient PATIENT   Stage exactly one patient instead of every line in --patients-file.
  --patients-file PATH  Default: staging/patients.txt (one patient per line, blank lines and lines starting with # ignored).
  --bandicoot-root PATH Default: $BANDICOOT_ROOT env var, or ~/mnt/bandicoot/NF1_organoid_data/data.
  --dest-root PATH    Default: $DEST_ROOT env var, or ~/mnt/alpine/active/koala/nf1-3d-zstep-workflow-db/data.
  -h, --help          Show this help.

zstack_images/ is always required (a missing one is an error). The native
mask is required too, but can come from either segmentation_masks/ or,
for NF0055_T1-Z-0.1 only, segmentation_masks_from_0_1um/ (see this
script's header comment for why) -- either source lands at the
destination's segmentation_masks/. segmentation_masks_from_0_1um_adjusted/
is optional: only the 0.2/0.5/1 um groups carry it, and a missing one is
reported as "skip (optional)", not an error.

Per-patient, per-subdirectory logs are written under staging/logs/. After
staging, verify completeness with the workflow's own discovery tooling
rather than trusting this script's exit code alone -- ON ALPINE, where
the workflow runs:

  cd 3c.nextflow_zstep
  python3 scripts/build_image_sets_index.py \
    --source-root /pl/active/koala/nf1-3d-zstep-workflow-db \
    --output manifest/image_sets_index.csv

That script only lists a (patient, well_fov, mask_group) triple once every
channel TIFF and every compartment mask is present, so its row count
against the expected 63 image sets (see README.md "The experiment, in
one table") is the real completeness check.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --patient) ONE_PATIENT="$2"; shift 2 ;;
    --patients-file) PATIENTS_FILE="$2"; shift 2 ;;
    --bandicoot-root) BANDICOOT_ROOT="$2"; shift 2 ;;
    --dest-root) DEST_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -d "$BANDICOOT_ROOT" ]]; then
  echo "Bandicoot source root not found or not mounted: $BANDICOOT_ROOT" >&2
  echo "On the lab Mac: run 'mount_isilon' and mount the 'bandicoot' share," >&2
  echo "then retry. See staging/README.md." >&2
  exit 1
fi

mkdir -p "$DEST_ROOT" "$LOG_DIR"

patients=()
if [[ -n "$ONE_PATIENT" ]]; then
  patients=("$ONE_PATIENT")
else
  if [[ ! -f "$PATIENTS_FILE" ]]; then
    echo "Patients file not found: $PATIENTS_FILE" >&2
    exit 1
  fi
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo -n "$line" | tr -d '[:space:]')"
    [[ -n "$line" ]] && patients+=("$line")
  done < "$PATIENTS_FILE"
fi

if [[ "${#patients[@]}" -eq 0 ]]; then
  echo "No patients to stage" >&2
  exit 1
fi

rsync_args=(-a --partial --exclude=".DS_Store" --exclude="._*")
# --info=progress2 is a GNU-rsync (3.x) feature; macOS's stock
# openrsync/rsync-2.6.9-compatible binary rejects it outright. Only add
# it when the installed rsync actually understands it.
if rsync --info=progress2 --version >/dev/null 2>&1; then
  rsync_args+=(--info=progress2)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_args+=(-n)
fi

echo "Staging ${#patients[@]} patient(s) from $BANDICOOT_ROOT to $DEST_ROOT"
echo "Patients: ${patients[*]}"
[[ "$DRY_RUN" -eq 1 ]] && echo "Dry run: no files will be copied"

# stage_one PATIENT SRC_SUBDIR DEST_SUBDIR REQUIRED
# Copies {bandicoot_root}/{patient}/{src_subdir}/ to
# {dest_root}/{patient}/{dest_subdir}/ (src and dest subdir names may
# differ -- see the NF0055_T1-Z-0.1 native-mask rename above). Sets
# overall_status=1 on a missing required source or an rsync failure.
# A missing optional source just logs "skip (optional)".
stage_one() {
  local patient="$1" src_subdir="$2" dest_subdir="$3" required="$4"
  local src="$BANDICOOT_ROOT/$patient/$src_subdir/"
  local dst="$DEST_ROOT/$patient/$dest_subdir/"
  local log="$LOG_DIR/${patient}.${dest_subdir}.log"

  if [[ ! -d "$src" ]]; then
    if [[ "$required" == "1" ]]; then
      echo "FAILED: required source missing for $patient/$dest_subdir (looked for $src_subdir/): $src" | tee -a "$log" >&2
      overall_status=1
    else
      echo "skip (optional): source missing for $patient/$dest_subdir (looked for $src_subdir/)" | tee -a "$log"
    fi
    return
  fi

  [[ "$DRY_RUN" -eq 0 ]] && mkdir -p "$dst"
  echo "=== $patient/$dest_subdir: $src -> $dst ===" | tee "$log"
  if rsync "${rsync_args[@]}" "$src" "$dst" 2>&1 | tee -a "$log"; then
    echo "done: $patient/$dest_subdir" | tee -a "$log"
  else
    echo "FAILED: $patient/$dest_subdir (see $log)" | tee -a "$log" >&2
    overall_status=1
  fi
}

overall_status=0
for patient in "${patients[@]}"; do
  stage_one "$patient" "zstack_images" "zstack_images" 1

  # Native mask: prefer segmentation_masks/; for the one patient missing
  # it (NF0055_T1-Z-0.1), fall back to segmentation_masks_from_0_1um/,
  # which IS the native mask there (verified by shape match against its
  # own 0.1um zstack) -- see this script's header comment. Never treat
  # segmentation_masks_from_0_1um/ as native when segmentation_masks/
  # already exists: on every patient checked, the two are byte-identical
  # stale duplicates, so the fallback only fires when there is nothing
  # else.
  if [[ -d "$BANDICOOT_ROOT/$patient/segmentation_masks" ]]; then
    stage_one "$patient" "segmentation_masks" "segmentation_masks" 1
  else
    stage_one "$patient" "segmentation_masks_from_0_1um" "segmentation_masks" 1
  fi

  # Segmentation 2 (the real one): only segmentation_masks_from_0_1um_adjusted/,
  # optional. The plain, unadjusted segmentation_masks_from_0_1um/ is
  # deliberately never staged under its own name -- it is a stale
  # duplicate of native everywhere, not real second-segmentation data.
  stage_one "$patient" "segmentation_masks_from_0_1um_adjusted" "segmentation_masks_from_0_1um_adjusted" 0
done

if [[ "$overall_status" -ne 0 ]]; then
  echo "Staging finished with errors; check staging/logs/" >&2
else
  echo "Staging finished for ${#patients[@]} patient(s)."
fi
exit "$overall_status"
