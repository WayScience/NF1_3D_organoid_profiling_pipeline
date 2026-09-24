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
    --source-warehouse|--dest-warehouse)
      [[ $# -ge 2 ]] || { echo "Option $1 requires a path" >&2; exit 2; }
      if [[ "$1" == "--source-warehouse" ]]; then
        SOURCE_WAREHOUSE="$2"
      else
        DEST_WAREHOUSE="$2"
      fi
      shift 2
      ;;
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
if [[ ! -d "$SOURCE_WAREHOUSE/ibp/.complete" ]]; then
  echo "Source ibp/.complete/ marker folder not found: $SOURCE_WAREHOUSE/ibp/.complete" >&2
  exit 1
fi

mkdir -p "$DEST_WAREHOUSE"

rsync_args=(-rlD --inplace --checksum --no-times --no-perms --no-owner --no-group --partial --info=progress2 --exclude=".DS_Store" --exclude="._*")
[[ "$DRY_RUN" -eq 1 ]] && rsync_args+=(-n)

echo "Syncing $SOURCE_WAREHOUSE/ -> $DEST_WAREHOUSE/ (excluding ibp/, synced separately below)"
[[ "$DRY_RUN" -eq 1 ]] && echo "Dry run: no files will be copied"

rsync "${rsync_args[@]}" \
  --exclude="/ibp/" \
  "$SOURCE_WAREHOUSE/" "$DEST_WAREHOUSE/"

# ibp/ gets its own pass, filtered by completion marker: run_one_image_set()
# (run_ibp_production.py) renames sc_profiles_related and
# organoid_profiles_related sequentially via two separate os.replace()
# calls, creating the ibp/.complete/<image_id> marker only after both
# succeed. An interruption between the two renames can leave one final
# parquet on disk without its pair -- a plain recursive rsync of ibp/ would
# publish that incomplete, half-written image set to bandicoot. Building
# the transfer list from the completion markers instead means only image
# sets confirmed to have both files land on bandicoot.
file_list="$(mktemp)"
trap 'rm -f "$file_list"' EXIT
while IFS= read -r -d '' marker; do
  image_id="$(basename "$marker")"
  printf 'sc_profiles_related/%s.parquet\n' "$image_id"
  printf 'organoid_profiles_related/%s.parquet\n' "$image_id"
done < <(find "$SOURCE_WAREHOUSE/ibp/.complete" -mindepth 1 -maxdepth 1 -type f -print0) \
  > "$file_list"

marker_count=$(($(wc -l < "$file_list") / 2))
echo "Found $marker_count completed image set(s) via ibp/.complete/ markers"

ibp_rsync_args=(-rlD --inplace --checksum --no-times --no-perms --no-owner --no-group --partial --info=progress2 --files-from="$file_list")
[[ "$DRY_RUN" -eq 1 ]] && ibp_rsync_args+=(-n)

echo "Syncing $SOURCE_WAREHOUSE/ibp/ -> $DEST_WAREHOUSE/ibp/ (completion-marker-backed image sets only)"
mkdir -p "$DEST_WAREHOUSE/ibp"
rsync "${ibp_rsync_args[@]}" "$SOURCE_WAREHOUSE/ibp/" "$DEST_WAREHOUSE/ibp/"

echo "Sync finished."
