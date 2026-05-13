#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.marginally_exact_ou_bridge_sweep.common import (  # noqa: E402
    collect_job_results,
    load_jobs_from_manifest,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect completed sweep result JSON files into CSV/JSON summaries."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = load_jobs_from_manifest(args.manifest)
    output_dir = args.manifest.parent
    output_csv = args.output_csv or output_dir / "summary.csv"
    output_json = args.output_json or output_dir / "summary.json"
    rows = collect_job_results(jobs)
    write_summary(rows, output_csv=output_csv, output_json=output_json)
    print(f"Wrote summary CSV: {output_csv}")
    print(f"Wrote summary JSON: {output_json}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
