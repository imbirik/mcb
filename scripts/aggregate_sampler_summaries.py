#!/usr/bin/env python3

import argparse
import glob
import json
import math
from pathlib import Path


METRICS = [
    "generation_seconds",
    "perplexity",
    "generative_ppl",
    "diversity",
    "distinct_2",
    "distinct_3",
    "distinct_4",
    "mauve",
    "entropy",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed sampler summaries into mean/std metrics."
    )
    parser.add_argument("--summary-glob", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def _extract_seed(summary_path: Path):
    for part in summary_path.parts:
        if part.startswith("seed-"):
            try:
                return int(part.split("-", 1)[1])
            except ValueError:
                return part
    return summary_path.parent.name


def main():
    args = parse_args()
    summary_paths = sorted(Path(path) for path in glob.glob(args.summary_glob))
    if not summary_paths:
        raise FileNotFoundError(
            f"No summary files matched pattern: {args.summary_glob}"
        )

    summaries = [json.loads(path.read_text()) for path in summary_paths]
    first = summaries[0]

    sampler_names = [result["sampler"] for result in first["results"]]
    aggregated_results = []
    for sampler_name in sampler_names:
        sampler_runs = []
        for path, summary in zip(summary_paths, summaries):
            result = next(
                result for result in summary["results"]
                if result["sampler"] == sampler_name
            )
            sampler_runs.append({
                "seed": _extract_seed(path),
                "summary_path": str(path.resolve()),
                **result,
            })

        metric_values = {}
        for metric in METRICS:
            values = [
                run[metric] for run in sampler_runs
                if run.get(metric) is not None
            ]
            if values:
                metric_values[f"{metric}_mean"] = _mean(values)
                metric_values[f"{metric}_std"] = _std(values)

        aggregate = {
            "sampler": sampler_name,
            "num_seeds": len(sampler_runs),
            "per_seed": sampler_runs,
            **metric_values,
        }
        if any("q_t_act" in run for run in sampler_runs):
            aggregate["q_t_act"] = next(
                run["q_t_act"] for run in sampler_runs if "q_t_act" in run
            )
        if any("q_start_t" in run for run in sampler_runs):
            aggregate["q_start_t"] = next(
                run["q_start_t"] for run in sampler_runs if "q_start_t" in run
            )
        aggregated_results.append(aggregate)

    output = {
        "algo": first["algo"],
        "dataset": first["dataset"],
        "model": first["model"],
        "length": first["length"],
        "compare_samplers": first["compare_samplers"],
        "num_seeds": len(summary_paths),
        "summary_paths": [str(path.resolve()) for path in summary_paths],
        "results": aggregated_results,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=4) + "\n")
    print(f"Wrote aggregate summary: {args.output_path}")


if __name__ == "__main__":
    main()
