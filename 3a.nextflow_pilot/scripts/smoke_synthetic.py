#!/usr/bin/env python3
"""Create a tiny synthetic artifact set to check pilot wiring without data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    image_id = "SMOKE__SMOKE__A01__F1"
    compartments = ("Nuclei", "Cell", "Cytoplasm", "Organoid")

    tables = []
    for compartment in compartments:
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
        target_dir = args.outdir / "profiles" / f"{compartment.lower()}_profiles"
        target_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target_dir / f"{image_id}.parquet", index=False)
        validation = {
            "valid": True,
            "row_count": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
            "quality_warnings": [],
        }
        (target_dir / f"{image_id}.validation.json").write_text(
            json.dumps({"compartments": {compartment: validation}}, indent=2)
        )
        tables.append(
            {
                "name": table_name,
                "path": str(target_dir),
                "row_count": validation["row_count"],
                "column_count": validation["column_count"],
                "validation_status": "pass",
            }
        )

    assets_dir = args.outdir / "images" / "image_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    assets = pd.DataFrame(
        {
            "Metadata_Biology_PatientTumor": ["SMOKE"] * 8,
            "Metadata_Imaging_ImageID": [image_id] * 8,
            "Metadata_ImageAsset_AssetID": [
                f"{image_id}::{name}"
                for name in ("DNA", "ER", "AGP", "Mito", "Nuclei_mask", "Cell_mask", "Cytoplasm_mask", "Organoid_mask")
            ],
        }
    )
    assets.to_parquet(assets_dir / f"{image_id}.parquet", index=False)
    assets_validation = {
        "valid": True,
        "row_count": int(assets.shape[0]),
        "column_count": int(assets.shape[1]),
        "alignment": {"valid": True, "reference_shape": [2, 4, 4]},
    }
    (assets_dir / f"{image_id}.validation.json").write_text(
        json.dumps(assets_validation, indent=2)
    )

    validation = {
        "valid": True,
        "mode": "synthetic-wiring-only",
        "note": "This does not execute ZEDProfiler; it verifies the isolated pilot artifact layout.",
        "tables": tables,
        "images.image_assets": {
            "valid": True,
            "row_count": assets_validation["row_count"],
        },
    }
    run_record = {
        "run_id": args.run_id,
        "validation_status": "pass",
        "mode": "synthetic-wiring-only",
        "outdir": str(args.outdir),
        "tables": tables,
    }
    (args.outdir / "validation.json").write_text(json.dumps(validation, indent=2))
    (args.outdir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    print(f"Smoke artifacts written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
