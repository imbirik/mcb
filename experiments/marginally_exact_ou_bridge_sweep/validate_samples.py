#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.marginally_exact_ou_bridge_sweep.common import (  # noqa: E402
    job_is_complete,
    load_jobs_from_manifest,
    validate_samples_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate saved sweep samples.jsonl files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--samples-jsonl", type=Path)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Required when validating one samples.jsonl file directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    if args.manifest is not None:
        jobs = load_jobs_from_manifest(args.manifest)
        for job in jobs:
            complete, job_errors = job_is_complete(job)
            if not complete:
                errors.extend(job_errors)
    else:
        if args.expected_count is None:
            raise SystemExit("--expected-count is required with --samples-jsonl")
        errors.extend(
            validate_samples_jsonl(
                args.samples_jsonl,
                expected_count=args.expected_count,
            )
        )

    if errors:
        for error in errors[:100]:
            print(error, file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors", file=sys.stderr)
        raise SystemExit(1)
    print("samples JSONL validation: OK")


if __name__ == "__main__":
    main()
