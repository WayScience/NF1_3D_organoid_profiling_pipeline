#!/usr/bin/env python3
"""Create a tiny synthetic artifact set to check pilot wiring without data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iceberg_warehouse import publish_iceberg_warehouse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    image_id = "SMOKE__SMOKE__A01__F1"
    manifest = {
        "schema_version": "0.1.0-pilot",
        "source_image_root": str(args.outdir / "source"),
        "Metadata_Biology_PatientTumor": "SMOKE",
        "Metadata_Biology_PatientID": "SMOKE",
        "Metadata_Experiment_PlateID": "SMOKE",
        "Metadata_Experiment_WellID": "A01",
        "Metadata_Imaging_FieldID": "1",
        "Metadata_Imaging_ImageID": image_id,
        "channel_paths": {
            "DNA": str(args.outdir / "source" / "A01-1_405.tif"),
            "ER": str(args.outdir / "source" / "A01-1_488.tif"),
            "AGP": str(args.outdir / "source" / "A01-1_555.tif"),
            "Mito": str(args.outdir / "source" / "A01-1_640.tif"),
        },
        "mask_paths": {
            "Nuclei": str(args.outdir / "source" / "nuclei_mask.tiff"),
            "Cell": str(args.outdir / "source" / "cell_mask.tiff"),
            "Cytoplasm": str(args.outdir / "source" / "cytoplasm_mask.tiff"),
            "Organoid": str(args.outdir / "source" / "organoid_mask.tiff"),
        },
        "channel_codes": {
            "DNA": "405",
            "ER": "488",
            "AGP": "555",
            "Mito": "640",
        },
        "compartment_primary_channels": {
            "Nuclei": "DNA",
            "Cell": "AGP",
            "Cytoplasm": "AGP",
            "Organoid": "AGP",
        },
        "compartment_primary_channel_codes": {
            "Nuclei": "405",
            "Cell": "555",
            "Cytoplasm": "555",
            "Organoid": "555",
        },
        "compartment_seed_channels": {
            "Nuclei": "",
            "Cell": "DNA",
            "Cytoplasm": "DNA",
            "Organoid": "",
        },
        "compartment_seed_channel_codes": {
            "Nuclei": "",
            "Cell": "405",
            "Cytoplasm": "405",
            "Organoid": "",
        },
        "compartment_segmentation_methods": {
            "Nuclei": "segmented_from_dna",
            "Cell": "agp_watershed_seeded_by_nuclei",
            "Cytoplasm": "cell_mask_minus_nuclei_mask",
            "Organoid": "segmented_from_agp",
        },
    }
    images = {
        "DNA": np.ones((2, 4, 4), dtype=np.uint16),
        "ER": np.ones((2, 4, 4), dtype=np.uint16) * 2,
        "AGP": np.ones((2, 4, 4), dtype=np.uint16) * 3,
        "Mito": np.ones((2, 4, 4), dtype=np.uint16) * 4,
    }
    masks = {
        "Nuclei": np.ones((2, 4, 4), dtype=np.uint16),
        "Cell": np.ones((2, 4, 4), dtype=np.uint16),
        "Cytoplasm": np.ones((2, 4, 4), dtype=np.uint16),
        "Organoid": np.ones((2, 4, 4), dtype=np.uint16),
    }
    profile_frames = {}
    validations = {}
    for compartment in masks:
        frame = pd.DataFrame(
            {
                "Metadata_Compartment": [compartment],
                "Metadata_Biology_PatientTumor": ["SMOKE"],
                "Metadata_Biology_PatientID": ["SMOKE"],
                "Metadata_Experiment_PlateID": ["SMOKE"],
                "Metadata_Experiment_WellID": ["A01"],
                "Metadata_Imaging_FieldID": ["1"],
                "Metadata_Imaging_ImageID": [image_id],
                "Metadata_Object_ObjectID": [1],
                f"{compartment}_DNA_Intensity_Mean": [1.0],
            }
        )
        table_name = f"profiles.{compartment.lower()}_profiles"
        profile_frames[table_name] = frame
        frame.to_parquet(args.outdir / f"{compartment.lower()}_profiles.parquet")
        validations[compartment] = {
            "valid": True,
            "row_count": 1,
            "column_count": int(frame.shape[1]),
        }
    alignment = {
        "valid": True,
        "reference_shape": [2, 4, 4],
        "shapes": {
            "channels": {channel: [2, 4, 4] for channel in images},
            "masks": {compartment: [2, 4, 4] for compartment in masks},
        },
    }
    warehouse_manifest = publish_iceberg_warehouse(
        outdir=args.outdir,
        run_id=args.run_id,
        git_commit="smoke",
        manifest=manifest,
        profile_frames=profile_frames,
        images=images,
        masks=masks,
        validations=validations,
        alignment=alignment,
        zedprofiler_version="synthetic",
    )
    validation = {
        "valid": True,
        "mode": "synthetic-wiring-only",
        "note": "This does not execute ZEDProfiler; it verifies the isolated pilot artifact layout.",
        "warehouse": {
            "valid": warehouse_manifest["validation_status"] == "pass",
            "manifest": str(args.outdir / "warehouse" / "warehouse_manifest.json"),
            "namespaces": warehouse_manifest["namespaces"],
            "tables": [table["table_name"] for table in warehouse_manifest["tables"]],
        },
    }
    run_record = {
        "run_id": args.run_id,
        "validation_status": "pass",
        "mode": "synthetic-wiring-only",
        "warehouse_root": warehouse_manifest["warehouse_root"],
        "warehouse_manifest": str(
            args.outdir / "warehouse" / "warehouse_manifest.json"
        ),
    }
    (args.outdir / "validation.json").write_text(json.dumps(validation, indent=2))
    (args.outdir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    print(f"Smoke artifacts written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
