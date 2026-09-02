#!/usr/bin/env python3
"""Build (or refresh) a small DuckDB file with lightweight VIEWs over one
run's parquet warehouse.

Each of profiles/<table>/*.parquet and images/<table>/*.parquet becomes one
VIEW, mapped onto a DuckDB schema.table pair matching this pilot's existing
profiles.<compartment>_profiles / images.image_assets naming. A view is just
a stored query over read_parquet(glob) -- no data is copied, so the .duckdb
file stays tiny (a few KB) no matter how much parquet it points at, and
rerunning this picks up newly landed image sets for free without touching
anything already there.

Two further composite views live under a `joined` schema, each joining a
profile table (or tables) against images.image_assets with the image-asset
columns first (both the name and the column order put images first):
`joined.images_nuclei_cell_cytoplasm` joins the three single-cell-level
compartments together first (on Metadata_Imaging_ImageID +
Metadata_Object_ObjectID -- Cell and Cytoplasm are seeded from Nuclei's
segmentation, so their object IDs coincide) and then to image_assets (on
Metadata_Imaging_ImageID alone, so each object row is repeated once per
image asset); `joined.images_organoid` joins Organoid -- segmented
independently, its own object ID space -- directly to image_assets the same
way. Columns that carry identical values across every table (patient/plate/
well/field/image IDs) are kept exactly once, from image_assets; columns
that differ per compartment (Metadata_Compartment, the Segmentation_* ones)
are renamed with a per-compartment prefix so no two columns in the result
share a name.

The database file lives inside the warehouse directory itself
(<warehouse_dir>/warehouse.duckdb by default), one level above profiles/ and
images/, and every base view is defined with a path *relative* to that
directory (e.g. 'profiles/nuclei_profiles/*.parquet', never an absolute
PetaLibrary path) -- copy or rsync the whole warehouse directory anywhere
and the views still resolve, no environment-specific paths baked in.

DuckDB resolves relative paths against the *process's* current working
directory at query time, not the .duckdb file's own location, so querying
this later means running from inside the warehouse directory:

    cd <warehouse_dir> && duckdb warehouse.duckdb
    D SELECT * FROM profiles.nuclei_profiles LIMIT 5;
    D SELECT * FROM joined.images_nuclei_cell_cytoplasm LIMIT 5;

or in Python: chdir into <warehouse_dir> before calling duckdb.connect().

Can run standalone against any warehouse directory (finished or still
landing), or be called as a library function from
build_warehouse_from_compartments.py right after validation -- at no extra
Slurm-job cost, since that task already has everything it needs in memory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

# Candidates only -- when present, identical across every compartment table
# for the same image set, so kept exactly once (from images.image_assets)
# and excluded everywhere else. Not every table is guaranteed to carry all
# of these (e.g. synthetic test data doesn't), so _compartment_projection
# intersects this against each table's actual live columns rather than
# assuming.
DEDUPE_COLUMNS = (
    "Metadata_Biology_PatientTumor",
    "Metadata_Biology_PatientID",
    "Metadata_Experiment_PlateID",
    "Metadata_Experiment_WellID",
    "Metadata_Imaging_FieldID",
    "Metadata_Imaging_ImageID",
    "Metadata_Experiment_ImageSet",
)
# Differ per compartment (e.g. Metadata_Compartment='Nuclei' vs 'Cell') --
# renamed with a per-compartment prefix so the joined view never has two
# columns sharing one name.
PER_COMPARTMENT_COLUMNS = (
    "Metadata_Compartment",
    "Metadata_Segmentation_Method",
    "Metadata_Segmentation_PrimaryChannel",
    "Metadata_Segmentation_PrimaryChannelCode",
    "Metadata_Segmentation_SeedChannel",
    "Metadata_Segmentation_SeedChannelCode",
)


def discover_tables(warehouse_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (namespace, table, directory-relative-to-warehouse_dir) for
    each non-empty parquet dataset directory found directly under
    warehouse_dir/profiles and warehouse_dir/images."""
    tables: list[tuple[str, str, Path]] = []
    for namespace in ("profiles", "images"):
        namespace_dir = warehouse_dir / namespace
        if not namespace_dir.is_dir():
            continue
        for table_dir in sorted(namespace_dir.iterdir()):
            if table_dir.is_dir() and any(table_dir.glob("*.parquet")):
                tables.append((namespace, table_dir.name, table_dir.relative_to(warehouse_dir)))
    return tables


def _table_columns(conn: duckdb.DuckDBPyConnection, namespace: str, table: str) -> set[str]:
    """Actual column names of a view/table, queried live rather than
    assumed -- schemas can vary (e.g. synthetic smoke-test data doesn't
    carry every column a real run's does)."""
    return set(conn.execute(f'SELECT * FROM "{namespace}"."{table}" LIMIT 0').fetchdf().columns)


def _compartment_projection(
    conn: duckdb.DuckDBPyConnection,
    namespace: str,
    table: str,
    alias: str,
    compartment: str,
    drop_object_id: bool,
) -> str:
    """SQL fragment projecting one compartment's columns: dedupe columns
    dropped, per-compartment columns renamed with this compartment's
    prefix, feature columns (already compartment-prefixed by name)
    untouched. Only references columns confirmed present in this table --
    EXCLUDE/RENAME on a column that doesn't exist is a hard DuckDB error,
    not a no-op, so this can't just assume DEDUPE_COLUMNS/
    PER_COMPARTMENT_COLUMNS are always all there."""
    columns = _table_columns(conn, namespace, table)
    exclude = [column for column in DEDUPE_COLUMNS if column in columns]
    if drop_object_id:
        exclude.append("Metadata_Object_ObjectID")
    rename_pairs = [
        (column, f"Metadata_{compartment}_{column.removeprefix('Metadata_')}")
        for column in PER_COMPARTMENT_COLUMNS
        if column in columns
    ]

    fragment = f"{alias}.*"
    if exclude:
        fragment += f' EXCLUDE ({", ".join(exclude)})'
    if rename_pairs:
        rename_sql = ", ".join(f"{old} AS {new}" for old, new in rename_pairs)
        fragment += f" RENAME ({rename_sql})"
    return fragment


def build_joined_views(conn: duckdb.DuckDBPyConnection, tables: set[tuple[str, str]]) -> list[str]:
    """Create/refresh the composite joined.* views, skipping one if any of
    its required base tables aren't present in this warehouse (e.g. a run
    with fewer compartments)."""
    conn.execute('CREATE SCHEMA IF NOT EXISTS "joined"')
    created: list[str] = []

    single_cell_required = {
        ("profiles", "nuclei_profiles"),
        ("profiles", "cell_profiles"),
        ("profiles", "cytoplasm_profiles"),
        ("images", "image_assets"),
    }
    if single_cell_required <= tables:
        conn.execute(
            f"""
            CREATE OR REPLACE VIEW "joined"."images_nuclei_cell_cytoplasm" AS
            SELECT
                a.*,
                {_compartment_projection(conn, "profiles", "nuclei_profiles", "n", "Nuclei", drop_object_id=False)},
                {_compartment_projection(conn, "profiles", "cell_profiles", "c", "Cell", drop_object_id=True)},
                {_compartment_projection(conn, "profiles", "cytoplasm_profiles", "cy", "Cytoplasm", drop_object_id=True)}
            FROM images.image_assets a
            JOIN profiles.nuclei_profiles n
                ON n.Metadata_Imaging_ImageID = a.Metadata_Imaging_ImageID
            JOIN profiles.cell_profiles c
                ON c.Metadata_Imaging_ImageID = n.Metadata_Imaging_ImageID
                AND c.Metadata_Object_ObjectID = n.Metadata_Object_ObjectID
            JOIN profiles.cytoplasm_profiles cy
                ON cy.Metadata_Imaging_ImageID = n.Metadata_Imaging_ImageID
                AND cy.Metadata_Object_ObjectID = n.Metadata_Object_ObjectID
            """
        )
        created.append("joined.images_nuclei_cell_cytoplasm")

    organoid_required = {("profiles", "organoid_profiles"), ("images", "image_assets")}
    if organoid_required <= tables:
        conn.execute(
            f"""
            CREATE OR REPLACE VIEW "joined"."images_organoid" AS
            SELECT
                a.*,
                {_compartment_projection(conn, "profiles", "organoid_profiles", "o", "Organoid", drop_object_id=False)}
            FROM images.image_assets a
            JOIN profiles.organoid_profiles o
                ON o.Metadata_Imaging_ImageID = a.Metadata_Imaging_ImageID
            """
        )
        created.append("joined.images_organoid")

    return created


def build_views(
    warehouse_dir: Path, db_path: Path
) -> tuple[list[tuple[str, str, int]], list[str]]:
    """Create/refresh one VIEW per discovered table plus the joined.* views,
    all defined with paths relative to warehouse_dir so the resulting
    .duckdb file stays portable with the directory it points at. Returns
    ((namespace, table, file_count) per base view, joined view names)."""
    warehouse_dir = warehouse_dir.resolve()
    db_path = db_path.resolve()
    tables = discover_tables(warehouse_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    previous_cwd = Path.cwd()
    try:
        # CREATE VIEW resolves read_parquet's relative glob against the
        # process cwd immediately (to infer the view's schema), so it has
        # to run from inside warehouse_dir -- restored below regardless of
        # outcome.
        os.chdir(warehouse_dir)
        for namespace in {namespace for namespace, _table, _dir in tables}:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{namespace}"')
        created: list[tuple[str, str, int]] = []
        for namespace, table, rel_dir in tables:
            glob = (rel_dir / "*.parquet").as_posix().replace("'", "''")
            # union_by_name: some image sets (a compartment mask with zero
            # detected objects -- observed in production) legitimately land
            # a reduced-column file, since merge_feature_frames() drops
            # feature families that can't compute on zero objects rather
            # than padding with nulls. Reading the dataset by column name
            # instead of by position means those files just contribute NULL
            # for the columns they don't have, instead of every other file
            # in the same table being misaligned against them.
            conn.execute(
                f'CREATE OR REPLACE VIEW "{namespace}"."{table}" AS '
                f"SELECT * FROM read_parquet('{glob}', union_by_name = true)"
            )
            created.append((namespace, table, len(list((warehouse_dir / rel_dir).glob("*.parquet")))))

        joined = build_joined_views(conn, {(namespace, table) for namespace, table, _dir in tables})
    finally:
        os.chdir(previous_cwd)
        conn.close()
    return created, joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-dir",
        required=True,
        type=Path,
        help="Directory directly containing profiles/ and images/.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="DuckDB file to create/refresh. Default: <warehouse-dir>/warehouse.duckdb",
    )
    args = parser.parse_args()

    warehouse_dir = args.warehouse_dir.resolve()
    db_path = (args.db or (warehouse_dir / "warehouse.duckdb")).resolve()
    created, joined = build_views(warehouse_dir, db_path)
    if not created:
        raise SystemExit(f"No parquet-dataset directories found under {warehouse_dir}")

    for namespace, table, file_count in created:
        print(f"{namespace}.{table} -> {file_count} file(s)")
    for view_name in joined:
        print(f"{view_name} -> joined view")
    print(f"DuckDB views written to {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
