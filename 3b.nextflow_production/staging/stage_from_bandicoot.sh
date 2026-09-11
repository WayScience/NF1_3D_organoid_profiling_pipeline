#!/usr/bin/env bash
#
# stage_from_bandicoot.sh -- copy each listed patient's zstack_images/ and
# segmentation_masks/ from the bandicoot Isilon share into this production
# workflow's PetaLibrary (koala) data root, preserving the well/FOV layout
# both the pilot and this workflow's manifest tooling expect:
#
#   bandicoot:   {bandicoot_root}/{patient}/zstack_images/{well_fov}/
#                {bandicoot_root}/{patient}/segmentation_masks/{well_fov}/
#   petalibrary: {dest_root}/data/{patient}/zstack_images/{well_fov}/
#                {dest_root}/data/{patient}/segmentation_masks/{well_fov}/
#
# rsync-based: safe to re-run (skips unchanged files) and safe to interrupt
# and resume (--partial). This transfer is large -- thousands of well/FOV
# directories across the default patient list -- so run it under
# nohup/tmux/screen or as its own job, not in a short interactive shell.
# Requires both the bandicoot and PetaLibrary (koala) mounts to be reachable
# from wherever this runs.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BANDICOOT_ROOT="${BANDICOOT_ROOT:-$HOME/mnt/bandicoot/NF1_organoid_data/data}"
DEST_ROOT="${DEST_ROOT:-$HOME/mnt/alpine/active/koala/nf1-3d-production-workflow-db/data}"
PATIENTS_FILE="${PATIENTS_FILE:-$SCRIPT_DIR/patients.txt}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
SUBDIRS=(zstack_images segmentation_masks)

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
  --dest-root PATH    Default: $DEST_ROOT env var, or ~/mnt/alpine/active/koala/nf1-3d-production-workflow-db/data.
  -h, --help          Show this help.

Per-patient, per-subdirectory logs are written under staging/logs/. After
staging, verify completeness with the workflow's own discovery tooling
rather than trusting this script's exit code alone:

  cd ..
  python3 scripts/build_image_sets_index.py \
    --source-root ~/mnt/alpine/active/koala/nf1-3d-production-workflow-db \
    --output manifest/image_sets_index.csv

That script only lists a (patient, well_fov) pair once every channel TIFF
and every compartment mask is present, so its row count against each
patient's known well/FOV count is the real completeness check.
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

rsync_args=(-a --partial --info=progress2 --exclude=".DS_Store" --exclude="._*")
if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_args+=(-n)
fi

echo "Staging ${#patients[@]} patient(s) from $BANDICOOT_ROOT to $DEST_ROOT"
echo "Patients: ${patients[*]}"
[[ "$DRY_RUN" -eq 1 ]] && echo "Dry run: no files will be copied"

overall_status=0
for patient in "${patients[@]}"; do
  for subdir in "${SUBDIRS[@]}"; do
    src="$BANDICOOT_ROOT/$patient/$subdir/"
    dst="$DEST_ROOT/$patient/$subdir/"
    log="$LOG_DIR/${patient}.${subdir}.log"

    if [[ ! -d "$src" ]]; then
      echo "skip: source missing for $patient/$subdir: $src" | tee -a "$log" >&2
      overall_status=1
      continue
    fi

    [[ "$DRY_RUN" -eq 0 ]] && mkdir -p "$dst"
    echo "=== $patient/$subdir: $src -> $dst ===" | tee "$log"
    if rsync "${rsync_args[@]}" "$src" "$dst" 2>&1 | tee -a "$log"; then
      echo "done: $patient/$subdir" | tee -a "$log"
    else
      echo "FAILED: $patient/$subdir (see $log)" | tee -a "$log" >&2
      overall_status=1
    fi
  done
done

if [[ "$overall_status" -ne 0 ]]; then
  echo "Staging finished with errors; check staging/logs/" >&2
else
  echo "Staging finished for ${#patients[@]} patient(s)."
fi
exit "$overall_status"
