#!/usr/bin/env bash
#
# sync_to_bandicoot.sh -- copy this workflow's IBP step-3 output (the
# sc_profiles_related/organoid_profiles_related tables under the production
# ZEDProfiler warehouse's own ibp/ folder, plus the run record) from
# PetaLibrary (koala) to a new top-level folder on the bandicoot Isilon
# share, preserving the same internal layout:
#
#   koala:      {warehouse}/ibp/{sc_profiles_related,organoid_profiles_related}/*.parquet
#               {warehouse}/../ibp_run_record.json
#   bandicoot:  {dest_root}/ibp/{sc_profiles_related,organoid_profiles_related}/*.parquet
#               {dest_root}/ibp_run_record.json
#
# This is the reverse direction of ../../3b.nextflow_production/staging/
# stage_from_bandicoot.sh (which stages raw images bandicoot -> koala); here
# we're moving processed profiles koala -> bandicoot. Same rsync-based
# approach: safe to re-run (skips unchanged files) and safe to interrupt and
# resume (--partial).
#
# The warehouse's ibp/.complete/ marker directory (empty per-image-set
# files used only for this workflow's own local resumability bookkeeping)
# is intentionally excluded -- it carries no profile data and has no
# meaning on bandicoot.
set -euo pipefail

SOURCE_WAREHOUSE="${SOURCE_WAREHOUSE:-$HOME/mnt/alpine/active/koala/nf1-3d-production-workflow-db/results-zedprofiler-0.1.4/nf1-production-full-run-2/warehouse}"
DEST_ROOT="${DEST_ROOT:-$HOME/mnt/bandicoot/NF1_organoid_data/data/image_based_profiles_production_zedprofiler}"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --source-warehouse|--dest-root)
      [[ $# -ge 2 ]] || { echo "Option $1 requires a path" >&2; exit 2; }
      if [[ "$1" == "--source-warehouse" ]]; then
        SOURCE_WAREHOUSE="$2"
      else
        DEST_ROOT="$2"
      fi
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
sync_to_bandicoot.sh [--dry-run] [--source-warehouse PATH] [--dest-root PATH]

Options:
  --dry-run              Pass -n to rsync; report what would transfer without copying.
  --source-warehouse PATH  Default: $SOURCE_WAREHOUSE env var, or the
                           production ZEDProfiler warehouse's default mount path.
  --dest-root PATH       Default: $DEST_ROOT env var, or
                         ~/mnt/bandicoot/NF1_organoid_data/data/image_based_profiles_production_zedprofiler.
USAGE
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$SOURCE_WAREHOUSE/ibp" ]]; then
  echo "Source ibp/ folder not found: $SOURCE_WAREHOUSE/ibp" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_WAREHOUSE/ibp/.complete" ]]; then
  echo "Source ibp/.complete/ marker folder not found: $SOURCE_WAREHOUSE/ibp/.complete" >&2
  exit 1
fi

mkdir -p "$DEST_ROOT"

# run_one_image_set() (run_ibp_production.py) renames sc_profiles_related and
# organoid_profiles_related sequentially via two separate os.replace() calls,
# creating the ibp/.complete/<image_id> marker only after both succeed. An
# interruption between the two renames can leave one final parquet on disk
# without its pair -- a plain `rsync ibp/` would publish that incomplete,
# half-written image set to bandicoot. Build the transfer list from the
# completion markers instead, so only image sets confirmed to have both
# files land on bandicoot.
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

# Not -a: this bandicoot share is an SMB2/CIFS mount that rejects mkstemp()
# on the hidden dot-prefixed temp filenames rsync's default (safe,
# atomic-rename) transfer mode creates -- every file silently failed to
# write under -a (confirmed directly: rsync reported 100%/xfr#N complete
# with only a handful of "mkstemp ... Operation not permitted" lines
# visible, but the destination had zero actual files). --inplace writes
# directly to the target filename instead, which works; --no-times
# --no-perms --no-owner --no-group avoid a similar "failed to set
# times"/Operation not permitted error from this share rejecting utime()
# metadata calls it doesn't support over SMB (data content is unaffected
# either way -- confirmed by comparing transferred file sizes to source).
rsync_args=(-rlD --inplace --checksum --no-times --no-perms --no-owner --no-group --partial --info=progress2 --files-from="$file_list")
[[ "$DRY_RUN" -eq 1 ]] && rsync_args+=(-n)

echo "Syncing $SOURCE_WAREHOUSE/ibp/ -> $DEST_ROOT/ibp/ (completion-marker-backed image sets only)"
[[ "$DRY_RUN" -eq 1 ]] && echo "Dry run: no files will be copied"

mkdir -p "$DEST_ROOT/ibp"
rsync "${rsync_args[@]}" "$SOURCE_WAREHOUSE/ibp/" "$DEST_ROOT/ibp/"

run_record="$SOURCE_WAREHOUSE/../ibp_run_record.json"
if [[ -f "$run_record" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Would copy run record: $run_record -> $DEST_ROOT/ibp_run_record.json"
  else
    cp "$run_record" "$DEST_ROOT/ibp_run_record.json"
    echo "Copied run record: $run_record -> $DEST_ROOT/ibp_run_record.json"
  fi
else
  echo "No run record found at $run_record -- skipping" >&2
fi

echo "Sync finished."
