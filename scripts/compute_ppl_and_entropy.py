#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import dataloader
import metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute generative perplexity and token entropy for a saved set "
            "of generations."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help=(
            "Path to a sampler JSON file containing generated_seqs, a JSON list "
            "of strings, or a plain text file with one sample per line."
        ),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/openwebtext-split.yaml"),
        help="Hydra data config used to construct the tokenizer.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/small.yaml"),
        help="Hydra model config used for max sequence length.",
    )
    parser.add_argument(
        "--algo-config",
        type=Path,
        default=Path("configs/algo/flm.yaml"),
        help="Hydra algo config.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Base Hydra config.",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help=(
            "Override the autoregressive evaluator used for perplexity. "
            "Defaults to eval.gen_ppl_eval_model_name_or_path from the base config."
        ),
    )
    parser.add_argument(
        "--perplexity-batch-size",
        type=int,
        default=None,
        help="Override eval.perplexity_batch_size from the config.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override model.length from the model config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for perplexity evaluation, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON file to write the computed metrics to.",
    )
    return parser.parse_args()


def load_text_samples(input_path: Path):
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(input_path.read_text())
        if isinstance(payload, dict):
            if "generated_seqs" in payload:
                return list(payload["generated_seqs"])
            raise ValueError(
                f"JSON object at {input_path} does not contain generated_seqs"
            )
        if isinstance(payload, list):
            if all(isinstance(item, str) for item in payload):
                return payload
            raise ValueError(
                f"JSON list at {input_path} must contain only strings"
            )
        raise ValueError(f"Unsupported JSON structure in {input_path}")

    if suffix == ".jsonl":
        texts = []
        for line in input_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, str):
                texts.append(record)
            elif isinstance(record, dict) and "text" in record:
                texts.append(record["text"])
            else:
                raise ValueError(
                    f"Unsupported JSONL record in {input_path}: {record}"
                )
        return texts

    return [line.strip() for line in input_path.read_text().splitlines() if line.strip()]


def build_runtime_config(args):
    config = OmegaConf.load(args.base_config)
    config.data = OmegaConf.load(args.data_config)
    config.model = OmegaConf.load(args.model_config)
    config.algo = OmegaConf.load(args.algo_config)

    if args.eval_model is not None:
        config.eval.gen_ppl_eval_model_name_or_path = args.eval_model
    if args.perplexity_batch_size is not None:
        config.eval.perplexity_batch_size = int(args.perplexity_batch_size)
    if args.max_length is not None:
        config.model.length = int(args.max_length)
    return config


def compute_entropy_from_texts(tokenizer, text_samples, max_length):
    encoded = tokenizer(
        text_samples,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    entropies = []
    for input_ids, attention_mask in zip(
        encoded["input_ids"],
        encoded["attention_mask"],
    ):
        valid_tokens = input_ids[attention_mask.bool()]
        if valid_tokens.numel() == 0:
            continue
        _, counts = torch.unique(valid_tokens, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
        entropies.append(entropy)
    if not entropies:
        return 0.0
    return float(sum(entropies) / len(entropies))


def main():
    args = parse_args()
    text_samples = load_text_samples(args.input_path)
    if not text_samples:
        raise ValueError(f"No text samples found in {args.input_path}")

    config = build_runtime_config(args)
    tokenizer = dataloader.get_tokenizer(config)

    eval_metrics = metrics.Metrics(
        gen_ppl_eval_model_name_or_path=config.eval.gen_ppl_eval_model_name_or_path,
        eval_ppl_batch_size=config.eval.perplexity_batch_size,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_metrics.to(device)
    eval_metrics.gen_ppl.reset()
    eval_metrics.record_generative_perplexity(
        text_samples=text_samples,
        max_length=int(config.model.length),
        device=device,
    )

    result = {
        "input_path": str(args.input_path.resolve()),
        "num_samples": len(text_samples),
        "eval_model": config.eval.gen_ppl_eval_model_name_or_path,
        "max_length": int(config.model.length),
        "perplexity": float(eval_metrics.gen_ppl.compute().item()),
        "entropy": compute_entropy_from_texts(
            tokenizer=tokenizer,
            text_samples=text_samples,
            max_length=int(config.model.length),
        ),
    }

    print(json.dumps(result, indent=2))
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote metrics: {args.output_path}")


if __name__ == "__main__":
    main()
