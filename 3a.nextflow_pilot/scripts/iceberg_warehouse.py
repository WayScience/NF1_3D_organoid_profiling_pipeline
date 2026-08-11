"""Write pilot outputs as a small namespaced Apache Iceberg warehouse."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import load_catalog

WAREHOUSE_SPEC_VERSION = "0.1.0"
CATALOG_NAME = "nf1_pilot"


def file_uri(path: Path) -> str:
    """Return an absolute file URI for PyIceberg catalog configuration."""
    return path.resolve().as_uri()


def split_table_name(table_name: str) -> tuple[str, str]:
    """Split a namespace-qualified table identifier."""
    namespace, _, table = table_name.partition(".")
    if not namespace or not table or "." in table:
        raise ValueError(f"Expected <namespace>.<table>, got {table_name!r}")
    return namespace, table


def table_location(warehouse_root: Path, table_name: str) -> Path:
    """Map a dotted table name to the local warehouse namespace path."""
    namespace, table = split_table_name(table_name)
    return warehouse_root / namespace / table


def column_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    """Return manifest-safe column metadata from an Arrow schema."""
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def build_image_assets(
    manifest: dict[str, Any],
    images: dict[str, Any],
    masks: dict[str, Any],
    run_id: str,
    git_commit: str,
) -> pd.DataFrame:
    """Build the pilot `images.image_assets` table."""
    metadata = {
        key: manifest[key]
        for key in (
            "Metadata_Biology_PatientTumor",
            "Metadata_Biology_PatientID",
            "Metadata_Experiment_PlateID",
            "Metadata_Experiment_WellID",
            "Metadata_Imaging_FieldID",
            "Metadata_Imaging_ImageID",
        )
    }
    channel_paths = manifest.get("channel_paths") or {}
    channel_codes = manifest.get("channel_codes") or {}
    mask_paths = manifest.get("mask_paths") or {}
    primary_channels = manifest.get("compartment_primary_channels") or {}
    primary_codes = manifest.get("compartment_primary_channel_codes") or {}
    methods = manifest.get("compartment_segmentation_methods") or {}

    rows: list[dict[str, object]] = []
    for channel, array in images.items():
        shape = list(array.shape)
        rows.append(
            {
                **metadata,
                "Metadata_ImageAsset_AssetID": (
                    f"{metadata['Metadata_Imaging_ImageID']}::{channel}"
                ),
                "Metadata_ImageAsset_AssetType": "raw_image",
                "Metadata_ImageAsset_Channel": channel,
                "Metadata_ImageAsset_ChannelCode": str(channel_codes.get(channel, "")),
                "Metadata_ImageAsset_Compartment": "",
                "Metadata_ImageAsset_SegmentationMethod": "",
                "Metadata_ImageAsset_SourceURI": str(channel_paths.get(channel, "")),
                "Metadata_ImageAsset_DType": str(array.dtype),
                "Metadata_ImageAsset_SizeZ": int(shape[0]) if len(shape) > 0 else None,
                "Metadata_ImageAsset_SizeY": int(shape[1]) if len(shape) > 1 else None,
                "Metadata_ImageAsset_SizeX": int(shape[2]) if len(shape) > 2 else None,
                "Metadata_Run_RunID": run_id,
                "Metadata_Run_GitCommit": git_commit,
            }
        )

    for compartment, array in masks.items():
        shape = list(array.shape)
        channel = str(primary_channels.get(compartment, ""))
        rows.append(
            {
                **metadata,
                "Metadata_ImageAsset_AssetID": (
                    f"{metadata['Metadata_Imaging_ImageID']}::{compartment}_mask"
                ),
                "Metadata_ImageAsset_AssetType": "segmentation_mask",
                "Metadata_ImageAsset_Channel": channel,
                "Metadata_ImageAsset_ChannelCode": str(
                    primary_codes.get(compartment, "")
                ),
                "Metadata_ImageAsset_Compartment": compartment,
                "Metadata_ImageAsset_SegmentationMethod": str(
                    methods.get(compartment, "")
                ),
                "Metadata_ImageAsset_SourceURI": str(mask_paths.get(compartment, "")),
                "Metadata_ImageAsset_DType": str(array.dtype),
                "Metadata_ImageAsset_SizeZ": int(shape[0]) if len(shape) > 0 else None,
                "Metadata_ImageAsset_SizeY": int(shape[1]) if len(shape) > 1 else None,
                "Metadata_ImageAsset_SizeX": int(shape[2]) if len(shape) > 2 else None,
                "Metadata_Run_RunID": run_id,
                "Metadata_Run_GitCommit": git_commit,
            }
        )

    return pd.DataFrame(rows)


def write_table(
    catalog: Any,
    warehouse_root: Path,
    table_name: str,
    frame: pd.DataFrame,
    properties: dict[str, str] | None = None,
) -> tuple[Any, pa.Schema]:
    """Create or replace a single local Iceberg table from a DataFrame."""
    arrow_table = pa.Table.from_pandas(frame, preserve_index=False)
    location = table_location(warehouse_root, table_name)
    namespace, _table = split_table_name(table_name)
    catalog.create_namespace_if_not_exists(namespace)

    if catalog.table_exists(table_name):
        catalog.purge_table(table_name)
    if location.exists():
        shutil.rmtree(location)

    table = catalog.create_table(
        identifier=table_name,
        schema=arrow_table.schema,
        location=file_uri(location),
        properties=properties or {},
    )
    if arrow_table.num_rows:
        table.append(arrow_table)
        table = catalog.load_table(table_name)
    return table, arrow_table.schema


def publish_iceberg_warehouse(
    outdir: Path,
    run_id: str,
    git_commit: str,
    manifest: dict[str, Any],
    profile_frames: dict[str, pd.DataFrame],
    images: dict[str, Any],
    masks: dict[str, Any],
    validations: dict[str, dict[str, Any]],
    alignment: dict[str, Any],
    zedprofiler_version: str,
) -> dict[str, Any]:
    """Publish profile/image tables and return the warehouse manifest."""
    warehouse_root = outdir / "warehouse"
    warehouse_root.mkdir(parents=True, exist_ok=True)
    catalog_db = warehouse_root / "catalog.db"
    catalog = load_catalog(
        CATALOG_NAME,
        type="sql",
        uri=f"sqlite:///{catalog_db}",
        warehouse=file_uri(warehouse_root),
    )

    tables: list[dict[str, Any]] = []
    table_properties = {
        "write.parquet.compression-codec": "zstd",
        "producer": "nf1-nextflow-pilot",
        "run_id": run_id,
    }

    image_assets = build_image_assets(manifest, images, masks, run_id, git_commit)
    table, schema = write_table(
        catalog,
        warehouse_root,
        "images.image_assets",
        image_assets,
        properties={**table_properties, "role": "image_assets"},
    )
    tables.append(
        {
            "table_name": "images.image_assets",
            "name": "images.image_assets",
            "namespace": "images",
            "table": "image_assets",
            "role": "image_assets",
            "format": "iceberg",
            "location": table.location(),
            "metadata_location": table.metadata_location,
            "schema_version": "0.1.0-pilot",
            "join_keys": [
                "Metadata_Biology_PatientTumor",
                "Metadata_Imaging_ImageID",
                "Metadata_ImageAsset_AssetID",
            ],
            "columns": column_manifest(schema),
            "row_count": int(image_assets.shape[0]),
            "validation_status": "pass",
            "source_image_root": manifest.get("source_image_root", ""),
            "run_id": run_id,
            "git_commit": git_commit,
        }
    )

    for table_name, frame in profile_frames.items():
        _namespace, table_slug = split_table_name(table_name)
        compartment = table_slug.removesuffix("_profiles").capitalize()
        validation = validations.get(compartment, {})
        table, schema = write_table(
            catalog,
            warehouse_root,
            table_name,
            frame,
            properties={**table_properties, "role": "profiles"},
        )
        tables.append(
            {
                "table_name": table_name,
                "name": table_name,
                "namespace": "profiles",
                "table": table_slug,
                "role": "profiles",
                "format": "iceberg",
                "location": table.location(),
                "metadata_location": table.metadata_location,
                "schema_version": "0.1.0-pilot",
                "profile_level": "object",
                "compartment": compartment,
                "join_keys": [
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Imaging_ImageID",
                    "Metadata_Compartment",
                    "Metadata_Object_ObjectID",
                ],
                "columns": column_manifest(schema),
                "row_count": int(frame.shape[0]),
                "column_count": int(frame.shape[1]),
                "validation_status": "pass" if validation.get("valid") else "fail",
                "source_image_root": manifest.get("source_image_root", ""),
                "run_id": run_id,
                "git_commit": git_commit,
            }
        )

    warehouse_manifest = {
        "warehouse_spec_version": WAREHOUSE_SPEC_VERSION,
        "warehouse_root": str(warehouse_root),
        "catalog": {
            "name": CATALOG_NAME,
            "type": "sql",
            "uri": f"sqlite:///{catalog_db}",
            "warehouse": file_uri(warehouse_root),
        },
        "namespaces": sorted({table["namespace"] for table in tables}),
        "tables": tables,
        "run_id": run_id,
        "git_commit": git_commit,
        "zedprofiler_version": zedprofiler_version,
        "source_image_root": manifest.get("source_image_root", ""),
        "alignment": alignment,
        "validation_status": "pass"
        if all(table["validation_status"] == "pass" for table in tables)
        else "fail",
    }

    manifest_text = json.dumps(warehouse_manifest, indent=2)
    (warehouse_root / "warehouse_manifest.json").write_text(manifest_text)
    (outdir / "warehouse_manifest.json").write_text(manifest_text)
    return warehouse_manifest
