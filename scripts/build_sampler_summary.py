#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble per-sampler JSON files into one comparison summary."
    )
    parser.add_argument("--compare-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument(
        "--compare-samplers",
        nargs="+",
        default=["ode", "sde", "marginally_exact", "q"],
    )
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    for sampler_name in args.compare_samplers:
        sampler_path = args.compare_dir / f"{sampler_name}.json"
        if not sampler_path.exists():
            raise FileNotFoundError(f"Missing sampler result: {sampler_path}")
        result = json.loads(sampler_path.read_text())
        results.append({
            "sampler": sampler_name,
            "samples_path": str(sampler_path.resolve()),
            "steps": result["steps"],
            "num_sample_batches": result["num_sample_batches"],
            "generation_seconds": result["generation_seconds"],
            "perplexity": result.get("perplexity", result.get("generative_ppl")),
            "generative_ppl": result.get("generative_ppl"),
            "diversity": result.get("diversity"),
            "distinct_2": result.get("distinct_2"),
            "distinct_3": result.get("distinct_3"),
            "distinct_4": result.get("distinct_4"),
            "mauve": result.get("mauve"),
            "entropy": result.get("entropy"),
            **({"q_t_act": result["q_t_act"]} if "q_t_act" in result else {}),
            **({"q_start_t": result["q_start_t"]} if "q_start_t" in result else {}),
        })

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "algo": "flm",
        "dataset": args.dataset,
        "model": args.model,
        "length": args.length,
        "compare_samplers": args.compare_samplers,
        "results": results,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=4) + "\n")
    print(f"Wrote summary: {args.output_path}")


if __name__ == "__main__":
    main()
