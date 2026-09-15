#!/usr/bin/env bash
#
# sync_pipeline_output_to_bandicoot.sh -- copy this workflow's full stage-4
# IBP pipeline output (steps 3-10 per-patient scratch/results under
# data/{patient}/image_based_profiles_production_zedprofiler/, plus step
# 11's cross-patient data/all_patient_profiles/) from this local checkout to
# a NEW, version-stamped folder on bandicoot -- never an existing/shared
# folder name.
#
# This replaces an earlier version of this script that wrote directly into
# bandicoot's existing NF1_organoid_data/data/{patient}/ and
# NF1_organoid_data/data/all_patient_profiles/ paths. That silently
# overwrote 8 files (sc_norm_*/organoid_norm_* across all 4 pycytominer
# stages) belonging to an unrelated, older CellProfiler-based production
# run that happened to use the exact same output filenames -- confirmed
# directly (new file sizes/timestamps matched this workflow's own output
# exactly). This script now writes to its own version-stamped root instead,
# and refuses to run at all if that destination already has content, so the
# same mistake can't repeat itself even with a version string collision.
#
#   local:      data/{patient}/image_based_profiles_production_zedprofiler/...
#               data/all_patient_profiles/...
#   bandicoot:  NF1_organoid_data/data/image_based_profiles_production_zedprofiler_pipeline_<VERSION>/{patient}/...
#               NF1_organoid_data/data/image_based_profiles_production_zedprofiler_pipeline_<VERSION>/all_patient_profiles/...
#
# VERSION defaults to <date>_<short git commit hash> (e.g. 20260915_f5a6f16)
# so the destination folder name alone identifies exactly which commit's
# code produced the data, not just when it was copied.
#
# Same rsync-based approach and same flag fix as sync_to_bandicoot.sh /
# sync_warehouse_to_bandicoot.sh: this bandicoot SMB/CIFS share rejects
# mkstemp() on the hidden dot-prefixed temp filenames rsync's default
# (atomic-rename) transfer mode creates. --inplace avoids that; --no-times/
# --no-perms/--no-owner/--no-group avoid a similar "failed to set times"
# error from utime() calls this share doesn't support.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SOURCE_DATA="${SOURCE_DATA:-$REPO_ROOT/data}"
DEST_DATA_ROOT="${DEST_DATA_ROOT:-$HOME/mnt/bandicoot/NF1_organoid_data/data}"
IBP_NAME="${IBP_NAME:-image_based_profiles_production_zedprofiler}"
VERSION="${VERSION:-$(date +%Y%m%d)_$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

DRY_RUN=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --source-data|--dest-data-root|--version)
      [[ $# -ge 2 ]] || { echo "Option $1 requires a value" >&2; exit 2; }
      case "$1" in
        --source-data) SOURCE_DATA="$2" ;;
        --dest-data-root) DEST_DATA_ROOT="$2" ;;
        --version) VERSION="$2" ;;
      esac
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
sync_pipeline_output_to_bandicoot.sh [--dry-run] [--force] [--source-data PATH]
    [--dest-data-root PATH] [--version STRING]

Writes to <dest-data-root>/image_based_profiles_production_zedprofiler_pipeline_<version>/,
a brand-new folder -- refuses to run if that destination already exists and
has content, unless --force is given.

Options:
  --dry-run             Pass -n to rsync; report what would transfer without copying.
  --force               Proceed even if the version destination already has content.
  --source-data PATH    Default: $SOURCE_DATA env var, or this repo's own data/ folder.
  --dest-data-root PATH Default: $DEST_DATA_ROOT env var, or
                        ~/mnt/bandicoot/NF1_organoid_data/data.
  --version STRING      Default: $VERSION env var, or <date>_<short git commit hash>.
USAGE
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$SOURCE_DATA" ]]; then
  echo "Source data/ folder not found: $SOURCE_DATA" >&2
  exit 1
fi

DEST_ROOT="$DEST_DATA_ROOT/${IBP_NAME}_pipeline_${VERSION}"

if [[ "$FORCE" -ne 1 ]] && [[ -d "$DEST_ROOT" ]] && [[ -n "$(ls -A "$DEST_ROOT" 2>/dev/null)" ]]; then
  echo "Destination already exists and is non-empty: $DEST_ROOT" >&2
  echo "Refusing to sync into it -- pass --force if this is deliberate (e.g. resuming" >&2
  echo "an interrupted transfer for the same version), or use a different --version." >&2
  exit 1
fi

echo "Version: $VERSION"
echo "Destination root: $DEST_ROOT"

rsync_args=(-rlD --inplace --no-times --no-perms --no-owner --no-group --partial --info=progress2 --exclude=".DS_Store" --exclude="._*")
[[ "$DRY_RUN" -eq 1 ]] && rsync_args+=(-n)

overall_status=0

for patient_dir in "$SOURCE_DATA"/*/; do
  patient="$(basename "$patient_dir")"
  src="$patient_dir$IBP_NAME/"
  [[ -d "$src" ]] || continue
  dst="$DEST_ROOT/$patient/"
  echo "Syncing $src -> $dst"
  [[ "$DRY_RUN" -eq 1 ]] || mkdir -p "$dst"
  if ! rsync "${rsync_args[@]}" "$src" "$dst"; then
    echo "FAILED: $patient" >&2
    overall_status=1
  fi
done

if [[ -d "$SOURCE_DATA/all_patient_profiles" ]]; then
  src="$SOURCE_DATA/all_patient_profiles/"
  dst="$DEST_ROOT/all_patient_profiles/"
  echo "Syncing $src -> $dst"
  [[ "$DRY_RUN" -eq 1 ]] || mkdir -p "$dst"
  if ! rsync "${rsync_args[@]}" "$src" "$dst"; then
    echo "FAILED: all_patient_profiles" >&2
    overall_status=1
  fi
fi

if [[ "$overall_status" -ne 0 ]]; then
  echo "Sync finished with errors." >&2
else
  echo "Sync finished. Version: $VERSION at $DEST_ROOT"
fi
exit "$overall_status"
