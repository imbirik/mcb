#!/usr/bin/env python3

import argparse
import glob
import json
import re
from pathlib import Path


METRIC_CONFIG = {
    "perplexity": {
        "label": "Perplexity",
        "maximize": False,
    },
    "diversity": {
        "label": "Diversity",
        "maximize": True,
    },
    "mauve": {
        "label": "MAUVE",
        "maximize": True,
    },
}

SAMPLER_ORDER = [
    "ode",
    "sde",
    "marginally_exact",
    "marginally_exact_deterministic",
    "q",
]

SAMPLER_LABELS = {
    "ode": "ODE",
    "sde": "SDE",
    "marginally_exact": "Marginally exact",
    "marginally_exact_deterministic": "Marginally exact, no bridge noise",
    "q": "Q-sampler",
}

SAMPLER_COLORS = {
    "ode": "#1f77b4",
    "sde": "#9467bd",
    "marginally_exact": "#d62728",
    "marginally_exact_deterministic": "#ff7f0e",
    "q": "#2ca02c",
}

DISABLED_BY_DEFAULT = {
    "marginally_exact_deterministic",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create Pareto-style speed/quality plots from sampler summary JSON "
            "files produced at different step counts."
        )
    )
    parser.add_argument(
        "--summary-glob",
        action="append",
        required=True,
        help=(
            "Glob pattern matching summary.json or aggregate summary JSON files. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["perplexity", "diversity", "mauve"],
        choices=sorted(METRIC_CONFIG.keys()),
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=None,
        help="Optional sampler-name filter.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Defaults to a dataset-derived title.",
    )
    parser.add_argument(
        "--annotate-steps",
        dest="annotate_steps",
        action="store_true",
        help="Annotate each point with its step count.",
    )
    parser.add_argument(
        "--no-annotate-steps",
        dest="annotate_steps",
        action="store_false",
        help="Disable step-count annotations.",
    )
    parser.set_defaults(annotate_steps=True)
    return parser.parse_args()


def _expand_summary_paths(glob_patterns):
    paths = set()
    for pattern in glob_patterns:
        for match in glob.glob(pattern):
            paths.add(Path(match))
    return sorted(paths)


def _extract_metric(result, metric_name):
    if metric_name in result and result[metric_name] is not None:
        return result[metric_name]
    mean_key = f"{metric_name}_mean"
    if mean_key in result and result[mean_key] is not None:
        return result[mean_key]
    return None


def _steps_from_path(path):
    match = re.search(r"steps-(\d+)", str(path))
    if match is None:
        return None
    return int(match.group(1))


def _extract_steps(result, summary_path):
    if result.get("steps") is not None:
        return int(result["steps"])
    per_seed = result.get("per_seed", [])
    per_seed_steps = {
        int(run["steps"])
        for run in per_seed
        if run.get("steps") is not None
    }
    if len(per_seed_steps) == 1:
        return per_seed_steps.pop()
    return _steps_from_path(summary_path)


def _load_records(summary_paths, samplers=None):
    records = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text())
        dataset = summary.get("dataset")
        for result in summary.get("results", []):
            sampler_name = result["sampler"]
            if samplers is not None and sampler_name not in samplers:
                continue
            if samplers is None and sampler_name in DISABLED_BY_DEFAULT:
                continue

            record = {
                "dataset": dataset,
                "summary_path": str(summary_path.resolve()),
                "sampler": sampler_name,
                "steps": _extract_steps(result, summary_path),
                "generation_seconds": _extract_metric(result, "generation_seconds"),
                "perplexity": _extract_metric(result, "perplexity"),
                "diversity": _extract_metric(result, "diversity"),
                "mauve": _extract_metric(result, "mauve"),
            }
            if record["generation_seconds"] is None or record["steps"] is None:
                continue
            records.append(record)
    return records


def _sampler_sort_key(name):
    if name in SAMPLER_ORDER:
        return (0, SAMPLER_ORDER.index(name))
    return (1, name)


def _dominates(left, right, metric_name, maximize):
    left_time = left["generation_seconds"]
    right_time = right["generation_seconds"]
    left_metric = left[metric_name]
    right_metric = right[metric_name]

    time_ok = left_time <= right_time
    metric_ok = left_metric >= right_metric if maximize else left_metric <= right_metric
    strictly_better = (
        left_time < right_time
        or (left_metric > right_metric if maximize else left_metric < right_metric)
    )
    return time_ok and metric_ok and strictly_better


def _pareto_frontier(records, metric_name, maximize):
    valid = [record for record in records if record.get(metric_name) is not None]
    frontier = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            if _dominates(other, candidate, metric_name=metric_name, maximize=maximize):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda record: (record["generation_seconds"], record[metric_name]),
    )


def _default_title(records):
    datasets = sorted({record["dataset"] for record in records if record.get("dataset")})
    if len(datasets) == 1:
        return f"Sampler Pareto sweep ({datasets[0]})"
    return "Sampler Pareto sweep"


def main():
    args = parse_args()
    summary_paths = _expand_summary_paths(args.summary_glob)
    if not summary_paths:
        raise FileNotFoundError(
            f"No summary files matched patterns: {args.summary_glob}"
        )

    sampler_filter = None if args.samplers is None else set(args.samplers)
    records = _load_records(summary_paths, samplers=sampler_filter)
    if not records:
        raise ValueError("No plot records were extracted from the provided summaries.")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for Pareto plotting. "
            "Install it with `pip install matplotlib` or `pip install -r requirements.txt`."
        ) from exc

    sampler_names = sorted(
        {record["sampler"] for record in records},
        key=_sampler_sort_key,
    )

    figure, axes = plt.subplots(
        1,
        len(args.metrics),
        figsize=(5.5 * len(args.metrics), 4.5),
        squeeze=False,
    )
    axes = axes[0]

    for axis, metric_name in zip(axes, args.metrics):
        metric_info = METRIC_CONFIG[metric_name]
        valid_records = [
            record for record in records if record.get(metric_name) is not None
        ]
        for sampler_name in sampler_names:
            sampler_records = sorted(
                [
                    record
                    for record in valid_records
                    if record["sampler"] == sampler_name
                ],
                key=lambda record: record["steps"],
            )
            if not sampler_records:
                continue

            x_values = [record["generation_seconds"] for record in sampler_records]
            y_values = [record[metric_name] for record in sampler_records]
            color = SAMPLER_COLORS.get(sampler_name)
            label = SAMPLER_LABELS.get(sampler_name, sampler_name)
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.8,
                markersize=6,
                color=color,
                label=label,
                alpha=0.9,
            )
            if args.annotate_steps:
                for record in sampler_records:
                    axis.annotate(
                        str(record["steps"]),
                        (record["generation_seconds"], record[metric_name]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                    )

        frontier = _pareto_frontier(
            valid_records,
            metric_name=metric_name,
            maximize=metric_info["maximize"],
        )
        if frontier:
            axis.plot(
                [record["generation_seconds"] for record in frontier],
                [record[metric_name] for record in frontier],
                linestyle="--",
                linewidth=1.5,
                color="black",
                alpha=0.8,
                label="Pareto frontier",
            )
            axis.scatter(
                [record["generation_seconds"] for record in frontier],
                [record[metric_name] for record in frontier],
                color="black",
                s=30,
                zorder=4,
            )

        axis.set_xlabel("Generation time (s)")
        direction = "↑" if metric_info["maximize"] else "↓"
        axis.set_ylabel(f"{metric_info['label']} {direction}")
        axis.set_title(metric_info["label"])
        axis.grid(True, alpha=0.25)

    handles = []
    labels = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    seen_labels = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        figure.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(unique_labels), 4),
            frameon=False,
        )

    figure.suptitle(args.title or _default_title(records), y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=200, bbox_inches="tight")
    print(f"Wrote Pareto plot: {args.output_path}")


if __name__ == "__main__":
    main()
