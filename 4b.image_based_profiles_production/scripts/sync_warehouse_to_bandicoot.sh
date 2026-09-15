#!/usr/bin/env bash
#
# sync_warehouse_to_bandicoot.sh -- copy the entire production ZEDProfiler
# warehouse (profiles/, images/, warehouse.duckdb, ibp/) from PetaLibrary
# (koala) to bandicoot, mirroring the same relative path koala itself uses:
#
#   koala:      .../nf1-3d-production-workflow-db/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse/
#   bandicoot:  NF1_organoid_data/data/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse/
#
# This duplicates ibp/ (already separately synced to
# image_based_profiles_production_zedprofiler/ibp/ by sync_to_bandicoot.sh)
# but keeps the rest of the warehouse (profiles/, images/, the duckdb file)
# in one natural, koala-path-shaped place rather than inventing a new
# structure for it.
#
# Same rsync-based approach and same flag fix as sync_to_bandicoot.sh: this
# bandicoot SMB/CIFS share rejects mkstemp() on the hidden dot-prefixed temp
# filenames rsync's default (safe, atomic-rename) transfer mode creates --
# confirmed directly (rsync reports 100% complete while writing zero actual
# bytes under -a). --inplace avoids that; --no-times/--no-perms/--no-owner/
# --no-group avoid a similar "failed to set times" error from utime() calls
# this share doesn't support (data content is unaffected either way).
set -euo pipefail

SOURCE_WAREHOUSE="${SOURCE_WAREHOUSE:-$HOME/mnt/alpine/active/koala/nf1-3d-production-workflow-db/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse}"
DEST_WAREHOUSE="${DEST_WAREHOUSE:-$HOME/mnt/bandicoot/NF1_organoid_data/data/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse}"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --source-warehouse) SOURCE_WAREHOUSE="$2"; shift 2 ;;
    --dest-warehouse) DEST_WAREHOUSE="$2"; shift 2 ;;
    -h|--help)
      cat <<'USAGE'
sync_warehouse_to_bandicoot.sh [--dry-run] [--source-warehouse PATH] [--dest-warehouse PATH]

Options:
  --dry-run              Pass -n to rsync; report what would transfer without copying.
  --source-warehouse PATH  Default: $SOURCE_WAREHOUSE env var, or the
                           production ZEDProfiler warehouse's default mount path.
  --dest-warehouse PATH  Default: $DEST_WAREHOUSE env var, or the same
                         relative path mirrored under bandicoot's
                         NF1_organoid_data/data/.
USAGE
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$SOURCE_WAREHOUSE" ]]; then
  echo "Source warehouse not found: $SOURCE_WAREHOUSE" >&2
  exit 1
fi

mkdir -p "$DEST_WAREHOUSE"

rsync_args=(-rlD --inplace --no-times --no-perms --no-owner --no-group --partial --info=progress2 --exclude=".DS_Store" --exclude="._*")
[[ "$DRY_RUN" -eq 1 ]] && rsync_args+=(-n)

echo "Syncing $SOURCE_WAREHOUSE/ -> $DEST_WAREHOUSE/"
[[ "$DRY_RUN" -eq 1 ]] && echo "Dry run: no files will be copied"

rsync "${rsync_args[@]}" \
  --exclude="ibp/.complete/" \
  "$SOURCE_WAREHOUSE/" "$DEST_WAREHOUSE/"

echo "Sync finished."
