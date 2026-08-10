#!/usr/bin/env python3
"""Create a tiny synthetic artifact set to check pilot wiring without data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    validation = {
        "valid": True,
        "mode": "synthetic-wiring-only",
        "note": "This does not execute ZEDProfiler; it verifies the isolated pilot artifact layout.",
    }
    run_record = {
        "run_id": args.run_id,
        "validation_status": "pass",
        "mode": "synthetic-wiring-only",
    }
    warehouse_manifest = {
        "warehouse_root": str(args.outdir),
        "tables": [],
        "validation_status": "pass",
    }
    (args.outdir / "validation.json").write_text(json.dumps(validation, indent=2))
    (args.outdir / "run_record.json").write_text(json.dumps(run_record, indent=2))
    (args.outdir / "warehouse_manifest.json").write_text(
        json.dumps(warehouse_manifest, indent=2)
    )
    print(f"Smoke artifacts written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
