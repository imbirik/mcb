#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.marginally_exact_ou_bridge_sweep.common import (  # noqa: E402
    DATASET_PRESETS,
    JobSpec,
    collect_job_results,
    endpoint_pairs,
    find_repo_root,
    job_is_complete,
    make_job,
    parse_float_list,
    parse_gpu_list,
    parse_int_list,
    powers_of_two_upto,
    write_manifest,
    write_summary,
)
from scripts.download_flm_checkpoints import (  # noqa: E402
    DEFAULT_FOLDER_URL,
    download_selected_checkpoints,
)


class NullProgress:
    def __init__(self, *args, **kwargs):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def write(self, message: str) -> None:
        print(message)

    def refresh(self) -> None:
        pass


def make_progress_bar(total: int):
    if tqdm is None:
        return NullProgress()
    return tqdm(total=total, desc="Sweep jobs", unit="job", dynamic_ncols=True)


def make_generation_bar(total: int, desc: str):
    if tqdm is None:
        return NullProgress()
    return tqdm(total=total, desc=desc, unit="sample", dynamic_ncols=True, leave=False)


def compute_text_diversity(text_samples: list[str], n_values: tuple[int, ...] = (2, 3, 4)) -> dict:
    distinct_scores: dict[str, float] = {}
    product = 1.0
    for n in n_values:
        unique_ngrams = set()
        total_ngrams = 0
        for text in text_samples:
            tokens = text.split()
            if len(tokens) < n:
                continue
            for idx in range(len(tokens) - n + 1):
                unique_ngrams.add(tuple(tokens[idx : idx + n]))
            total_ngrams += len(tokens) - n + 1
        score = len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0
        distinct_scores[f"distinct_{n}"] = score
        product *= score
    distinct_scores["diversity"] = product
    return distinct_scores


def split_batches(total_batches: int, num_shards: int) -> list[int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    base = total_batches // num_shards
    remainder = total_batches % num_shards
    return [base + (1 if idx < remainder else 0) for idx in range(num_shards)]


def gpu_label(gpu: str | None) -> str:
    if gpu is None:
        return "cpu"
    return "gpu" + "".join(ch if ch.isalnum() else "_" for ch in str(gpu))


def result_sample_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    samples = payload.get("generated_seqs")
    if not isinstance(samples, list) or not all(isinstance(text, str) for text in samples):
        return None
    return len(samples)


def shard_paths(job: JobSpec, shard_index: int, gpu: str | None) -> tuple[Path, Path, Path, Path]:
    shard_dir = Path(job.job_dir) / "shards" / f"shard{shard_index:02d}_{gpu_label(gpu)}"
    return (
        shard_dir,
        shard_dir / "result.json",
        shard_dir / "run.log",
        shard_dir / "progress.json",
    )


def read_progress_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    return max(0, int(payload.get("generated_samples", 0)))


def shard_overrides(
    job: JobSpec,
    *,
    shard_dir: Path,
    shard_result_path: Path,
    num_sample_batches: int,
) -> list[str]:
    replacements = {
        "eval.generated_samples_path": str(shard_result_path),
        "hydra.run.dir": str(shard_dir / "hydra"),
        "sampling.num_sample_batches": str(num_sample_batches),
    }
    overrides = []
    seen = set()
    for item in job.overrides:
        key = item.split("=", 1)[0]
        if key in replacements:
            overrides.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            overrides.append(item)
    for key, value in replacements.items():
        if key not in seen:
            overrides.append(f"{key}={value}")
    return overrides


def weighted_mean(payloads: list[dict], counts: list[int], key: str, default: float = 0.0) -> float:
    pairs = [
        (count, float(payload[key]))
        for payload, count in zip(payloads, counts)
        if count > 0 and payload.get(key) is not None
    ]
    if not pairs:
        return default
    total = sum(count for count, _ in pairs)
    return sum(count * value for count, value in pairs) / total


def weighted_geometric_mean(payloads: list[dict], counts: list[int], key: str, default: float = 0.0) -> float:
    pairs = [
        (count, float(payload[key]))
        for payload, count in zip(payloads, counts)
        if count > 0 and payload.get(key) is not None and float(payload[key]) > 0.0
    ]
    if not pairs:
        return default
    total = sum(count for count, _ in pairs)
    return math.exp(sum(count * math.log(value) for count, value in pairs) / total)


def format_config_value(value) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def option_was_provided(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def set_from_config(args: argparse.Namespace, argv: list[str], option: str, attr: str, value) -> None:
    if value is None or option_was_provided(argv, option):
        return
    setattr(args, attr, value)


def load_experiment_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Experiment config does not exist: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return parse_simple_yaml(path.read_text())


def strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def parse_yaml_scalar(value: str):
    value = value.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML at line {line_number}: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_yaml_scalar(value)
    return root


def apply_experiment_config(args: argparse.Namespace, argv: list[str]) -> None:
    cfg = load_experiment_config(args.experiment_config)
    if not cfg:
        return

    checkpoint = cfg.get("checkpoint", {})
    generation = cfg.get("generation", {})
    metrics_cfg = cfg.get("metrics", {})
    sweep = cfg.get("sweep", {})
    runtime = cfg.get("runtime", {})

    set_from_config(args, argv, "--dataset", "dataset", cfg.get("dataset"))
    set_from_config(args, argv, "--seed", "seed", cfg.get("seed"))
    set_from_config(args, argv, "--checkpoint-path", "checkpoint_path", checkpoint.get("path"))
    set_from_config(args, argv, "--checkpoint-dir", "checkpoint_dir", checkpoint.get("dir"))
    set_from_config(args, argv, "--download-checkpoints", "download_checkpoints", checkpoint.get("download"))
    set_from_config(args, argv, "--checkpoint-folder-url", "checkpoint_folder_url", checkpoint.get("folder_url"))
    set_from_config(args, argv, "--output-root", "output_root", cfg.get("output_root"))
    set_from_config(args, argv, "--text-length", "text_length", generation.get("text_length"))
    set_from_config(args, argv, "--target-samples", "target_samples", generation.get("target_samples"))
    set_from_config(args, argv, "--eval-batch-size", "eval_batch_size", generation.get("eval_batch_size"))
    set_from_config(args, argv, "--ppl-batch-size", "ppl_batch_size", metrics_cfg.get("ppl_batch_size"))
    set_from_config(args, argv, "--mauve-batch-size", "mauve_batch_size", metrics_cfg.get("mauve_batch_size"))
    set_from_config(args, argv, "--gpus", "gpus", runtime.get("gpus"))
    set_from_config(args, argv, "--gpu-scheduling", "gpu_scheduling", runtime.get("gpu_scheduling"))
    set_from_config(args, argv, "--jobs-per-gpu", "jobs_per_gpu", runtime.get("jobs_per_gpu"))
    set_from_config(args, argv, "--python", "python", runtime.get("python"))

    if not option_was_provided(argv, "--compute-mauve"):
        args.compute_mauve = bool(metrics_cfg.get("compute_mauve", args.compute_mauve))
    if not option_was_provided(argv, "--compute-generative-perplexity"):
        args.compute_generative_perplexity = bool(
            metrics_cfg.get(
                "compute_generative_perplexity",
                args.compute_generative_perplexity,
            )
        )
    if not option_was_provided(argv, "--no-ode"):
        args.no_ode = not bool(sweep.get("include_ode", not args.no_ode))
    if not option_was_provided(argv, "--steps"):
        steps = sweep.get("steps")
        args.steps = None if steps is None or steps == "powers_of_two" else format_config_value(steps)
    if not option_was_provided(argv, "--temperatures"):
        args.temperatures = format_config_value(sweep.get("temperatures", args.temperatures))
    if not option_was_provided(argv, "--nucleus-ps"):
        args.nucleus_ps = format_config_value(sweep.get("nucleus_ps", args.nucleus_ps))

    if args.checkpoint_path is not None:
        args.checkpoint_path = Path(args.checkpoint_path)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = Path(args.checkpoint_dir)
    if args.output_root is not None:
        args.output_root = Path(args.output_root)


def group_key(job: JobSpec) -> tuple:
    return (
        job.dataset,
        job.sampler,
        job.temperature,
        job.nucleus_p,
        job.seed,
        job.text_length,
        job.target_samples,
        job.eval_batch_size,
    )


def group_name(jobs: list[JobSpec]) -> str:
    first = jobs[0]
    parts = [first.dataset, first.sampler]
    if first.sampler == "marginally_exact":
        parts.extend([f"temp{first.temperature:g}", f"p{first.nucleus_p:g}"])
    return "__".join(parts)


def grouped_jobs(jobs: list[JobSpec]) -> list[list[JobSpec]]:
    groups: dict[tuple, list[JobSpec]] = {}
    order: list[tuple] = []
    for job in jobs:
        key = group_key(job)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(job)
    return [sorted(groups[key], key=lambda item: item.steps) for key in order]


def select_group_shard(
    groups: list[list[JobSpec]],
    *,
    shard_index: int,
    num_shards: int,
) -> list[list[JobSpec]]:
    if num_shards < 1:
        raise ValueError(f"--num-group-shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--group-shard-index must be in [0, {num_shards}), got {shard_index}"
        )
    return [
        group
        for group_index, group in enumerate(groups)
        if group_index % num_shards == shard_index
    ]


def flatten_groups(groups: list[list[JobSpec]]) -> list[JobSpec]:
    return [job for group in groups for job in group]


def run_config_summary(group: list[JobSpec], args: argparse.Namespace) -> dict:
    first = group[0]
    gpus = parse_gpu_list(args.gpus)
    shard_batches = split_batches(first.num_sample_batches, len(gpus))
    return {
        "run": group_name(group),
        "dataset": first.dataset,
        "sampler": first.sampler,
        "temperature": first.temperature,
        "nucleus_p": first.nucleus_p,
        "steps": [job.steps for job in group],
        "seed": first.seed,
        "text_length": first.text_length,
        "target_samples_per_step": first.target_samples,
        "eval_batch_size": first.eval_batch_size,
        "num_sample_batches_per_step": first.num_sample_batches,
        "checkpoint_path": first.checkpoint_path,
        "compute_generative_perplexity": args.compute_generative_perplexity,
        "compute_mauve": args.compute_mauve,
        "gpu_scheduling": args.gpu_scheduling,
        "gpus": args.gpus,
        "sample_shards_per_step": [
            {
                "gpu": gpu,
                "num_sample_batches": num_batches,
                "samples": num_batches * first.eval_batch_size,
            }
            for gpu, num_batches in zip(gpus, shard_batches)
            if num_batches > 0
        ],
    }


def step_config_summary(job: JobSpec, args: argparse.Namespace) -> dict:
    return {
        "job": job.name,
        "sampler": job.sampler,
        "steps": job.steps,
        "temperature": job.temperature,
        "nucleus_p": job.nucleus_p,
        "target_samples": job.target_samples,
        "eval_batch_size": job.eval_batch_size,
        "num_sample_batches": job.num_sample_batches,
        "compute_generative_perplexity": args.compute_generative_perplexity,
        "compute_mauve": args.compute_mauve,
        "gpus": args.gpus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable temperature/nucleus-p sweep for the marginally "
            "exact OU-bridge sampler and the ODE baseline."
        )
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        help="YAML experiment config with sweep axes and runtime defaults.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="lm1b")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/flm"),
        help="Directory for released FLM checkpoints when --checkpoint-path is not set.",
    )
    parser.add_argument(
        "--download-checkpoints",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Checkpoint download policy for the default checkpoint directory. "
            "'auto' downloads only when the required checkpoint is missing."
        ),
    )
    parser.add_argument(
        "--checkpoint-folder-url",
        default=None,
        help="Optional Google Drive folder URL used for automatic checkpoint download.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments/marginally_exact_ou_bridge_sweep"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--text-length", type=int, default=None)
    parser.add_argument("--target-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--ppl-batch-size", type=int, default=None)
    parser.add_argument("--mauve-batch-size", type=int, default=2)
    parser.add_argument(
        "--compute-generative-perplexity",
        action="store_true",
        help="Compute generative PPL during sampling. The default sweep is text-only.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step counts. Defaults to powers of two up to text length.",
    )
    parser.add_argument(
        "--temperatures",
        default="0.1,0.3,0.5,0.7,0.9,1.0",
        help="Comma-separated marginal endpoint temperatures.",
    )
    parser.add_argument(
        "--nucleus-ps",
        default="0.9,0.95,1.0",
        help="Comma-separated marginal endpoint nucleus-p values.",
    )
    parser.add_argument("--no-ode", action="store_true", help="Skip ODE baseline jobs.")
    parser.add_argument("--compute-mauve", action="store_true")
    parser.add_argument(
        "--gpus",
        default=os.environ.get("GPU_LIST", "0"),
        help="Comma-separated GPU ids, or 'cpu'. Example: 0,1,2,3",
    )
    parser.add_argument(
        "--gpu-scheduling",
        choices=["shards", "jobs"],
        default="shards",
        help=(
            "'shards' uses all listed GPUs for one logical job and merges the "
            "generated samples. 'jobs' runs different logical jobs on different GPUs."
        ),
    )
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Rerun even valid completed jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest and print jobs only.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N jobs.")
    parser.add_argument(
        "--group-shard-index",
        type=int,
        default=0,
        help=(
            "Run only this modulo shard of grouped runs. Use with --num-group-shards "
            "to split the sweep across multiple launcher processes."
        ),
    )
    parser.add_argument(
        "--num-group-shards",
        type=int,
        default=1,
        help="Total number of grouped-run shards used with --group-shard-index.",
    )
    parser.add_argument(
        "--skip-manifest-write",
        action="store_true",
        help="Do not write manifest.json. Useful for parallel Slurm workers after rank 0 prepared it.",
    )
    parser.add_argument(
        "--skip-final-summary",
        action="store_true",
        help="Do not rebuild summary.csv/summary.json at the end of this process.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override passed to every job. May be repeated.",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def ensure_checkpoint(args: argparse.Namespace, repo_root: Path, preset: dict) -> Path:
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Explicit --checkpoint-path does not exist: {checkpoint_path}")
        return checkpoint_path

    checkpoint_dir = (repo_root / args.checkpoint_dir).resolve()
    checkpoint_path = checkpoint_dir / preset["checkpoint_name"]
    needs_download = args.download_checkpoints == "always" or (
        args.download_checkpoints == "auto" and not checkpoint_path.exists()
    )

    if needs_download:
        download_selected_checkpoints(
            output_dir=checkpoint_dir,
            folder_url=args.checkpoint_folder_url or DEFAULT_FOLDER_URL,
            expected=[preset["checkpoint_name"]],
            force=args.download_checkpoints == "always",
        )

    if not checkpoint_path.exists():
        if args.download_checkpoints == "never":
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}. "
                "Rerun without --download-checkpoints never to download it automatically."
            )
        raise FileNotFoundError(f"Missing checkpoint after download attempt: {checkpoint_path}")
    return checkpoint_path


def build_jobs(args: argparse.Namespace, repo_root: Path) -> tuple[list[JobSpec], Path, dict]:
    preset = DATASET_PRESETS[args.dataset]
    text_length = int(args.text_length or preset["text_length"])
    eval_batch_size = int(args.eval_batch_size or preset["eval_batch_size"])
    if args.target_samples % eval_batch_size != 0:
        raise ValueError(
            f"--target-samples={args.target_samples} must be divisible by "
            f"--eval-batch-size={eval_batch_size}"
        )
    num_sample_batches = args.target_samples // eval_batch_size
    ppl_batch_size = int(args.ppl_batch_size or eval_batch_size)

    checkpoint_path = ensure_checkpoint(args, repo_root, preset)

    steps = parse_int_list(args.steps) if args.steps else powers_of_two_upto(text_length)
    temperatures = parse_float_list(args.temperatures)
    nucleus_ps = parse_float_list(args.nucleus_ps)
    output_dir = (repo_root / args.output_root / args.dataset).resolve()

    jobs: list[JobSpec] = []
    if not args.no_ode:
        for step_count in steps:
            jobs.append(
                make_job(
                    output_dir=output_dir,
                    dataset=args.dataset,
                    preset=preset,
                    sampler="ode",
                    steps=step_count,
                    temperature=None,
                    nucleus_p=None,
                    seed=args.seed,
                    text_length=text_length,
                    target_samples=args.target_samples,
                    eval_batch_size=eval_batch_size,
                    num_sample_batches=num_sample_batches,
                    ppl_batch_size=ppl_batch_size,
                    mauve_batch_size=args.mauve_batch_size,
                    checkpoint_path=checkpoint_path,
                    compute_mauve=args.compute_mauve,
                    compute_generative_perplexity=args.compute_generative_perplexity,
                    extra_overrides=args.extra_override,
                )
            )

    for step_count in steps:
        for temperature, nucleus_p in endpoint_pairs(temperatures, nucleus_ps):
            jobs.append(
                make_job(
                    output_dir=output_dir,
                    dataset=args.dataset,
                    preset=preset,
                    sampler="marginally_exact",
                    steps=step_count,
                    temperature=temperature,
                    nucleus_p=nucleus_p,
                    seed=args.seed,
                    text_length=text_length,
                    target_samples=args.target_samples,
                    eval_batch_size=eval_batch_size,
                    num_sample_batches=num_sample_batches,
                    ppl_batch_size=ppl_batch_size,
                    mauve_batch_size=args.mauve_batch_size,
                    checkpoint_path=checkpoint_path,
                    compute_mauve=args.compute_mauve,
                    compute_generative_perplexity=args.compute_generative_perplexity,
                    extra_overrides=args.extra_override,
                )
            )

    if args.limit is not None:
        jobs = jobs[: args.limit]

    metadata = {
        "dataset": args.dataset,
        "checkpoint_path": str(checkpoint_path),
        "text_length": text_length,
        "steps": steps,
        "temperatures": temperatures,
        "nucleus_ps": nucleus_ps,
        "target_samples": args.target_samples,
        "eval_batch_size": eval_batch_size,
        "num_sample_batches": num_sample_batches,
        "ppl_batch_size": ppl_batch_size,
        "seed": args.seed,
        "compute_mauve": bool(args.compute_mauve),
        "compute_generative_perplexity": bool(args.compute_generative_perplexity),
        "gpu_scheduling": args.gpu_scheduling,
        "gpus": args.gpus,
        "experiment_config": str(args.experiment_config) if args.experiment_config else None,
        "output_dir": str(output_dir),
    }
    return jobs, output_dir, metadata


def run_job(job: JobSpec, *, gpu: str | None, args: argparse.Namespace, repo_root: Path) -> dict:
    complete, validation_errors = job_is_complete(job) if not args.force else (False, [])
    if complete:
        return {
            "name": job.name,
            "status": "cached",
            "gpu": gpu,
            "returncode": 0,
            "validation_errors": [],
        }
    if validation_errors and Path(job.result_path).exists():
        print(f"RERUN {job.name}: {'; '.join(validation_errors[:3])}")

    job_dir = Path(job.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [args.python, "-u", "-m", "main", *job.overrides]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    Path(job.log_path).write_text(proc.stdout)

    complete, validation_errors = job_is_complete(job) if proc.returncode == 0 else (False, [])
    status = "ok" if proc.returncode == 0 and complete else "failed"
    if status == "failed":
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        print(f"FAILED {job.name} on gpu={gpu}\n{tail}")
        if validation_errors:
            print("Validation errors:", "; ".join(validation_errors[:5]))
    return {
        "name": job.name,
        "status": status,
        "gpu": gpu,
        "returncode": proc.returncode,
        "seconds": elapsed,
        "validation_errors": validation_errors,
    }


def run_job_shard(
    job: JobSpec,
    *,
    shard_index: int,
    gpu: str | None,
    num_sample_batches: int,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict:
    shard_dir, shard_result_path, shard_log_path, shard_progress_path = shard_paths(
        job,
        shard_index,
        gpu,
    )
    expected_samples = num_sample_batches * job.eval_batch_size

    if not args.force and result_sample_count(shard_result_path) == expected_samples:
        return {
            "shard_index": shard_index,
            "status": "cached",
            "gpu": gpu,
            "returncode": 0,
            "num_sample_batches": num_sample_batches,
            "expected_samples": expected_samples,
            "result_path": str(shard_result_path),
            "log_path": str(shard_log_path),
            "progress_path": str(shard_progress_path),
            "seconds": 0.0,
        }

    shard_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["FLM_PROGRESS_PATH"] = str(shard_progress_path)
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    overrides = shard_overrides(
        job,
        shard_dir=shard_dir,
        shard_result_path=shard_result_path,
        num_sample_batches=num_sample_batches,
    )
    cmd = [args.python, "-u", "-m", "main", *overrides]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    shard_log_path.write_text(proc.stdout)

    sample_count = result_sample_count(shard_result_path)
    status = "ok" if proc.returncode == 0 and sample_count == expected_samples else "failed"
    return {
        "shard_index": shard_index,
        "status": status,
        "gpu": gpu,
        "returncode": proc.returncode,
        "num_sample_batches": num_sample_batches,
        "expected_samples": expected_samples,
        "num_generated_samples": sample_count,
        "result_path": str(shard_result_path),
        "log_path": str(shard_log_path),
        "progress_path": str(shard_progress_path),
        "seconds": elapsed,
    }


def merge_shard_results(job: JobSpec, shard_results: list[dict], *, wall_seconds: float) -> None:
    ordered = sorted(shard_results, key=lambda item: int(item["shard_index"]))
    payloads = [json.loads(Path(item["result_path"]).read_text()) for item in ordered]
    samples_by_shard = [payload.get("generated_seqs", []) for payload in payloads]
    counts = [len(samples) for samples in samples_by_shard]
    samples = [sample for shard_samples in samples_by_shard for sample in shard_samples]

    diversity_stats = compute_text_diversity(samples)
    first = payloads[0] if payloads else {}
    result = {
        "sampler": job.sampler,
        "steps": job.steps,
        "num_sample_batches": job.num_sample_batches,
        "generation_seconds": wall_seconds,
        "perplexity": weighted_geometric_mean(payloads, counts, "perplexity"),
        "generative_ppl": weighted_geometric_mean(payloads, counts, "generative_ppl"),
        "diversity": diversity_stats.get("diversity", 0.0),
        "distinct_2": diversity_stats.get("distinct_2", 0.0),
        "distinct_3": diversity_stats.get("distinct_3", 0.0),
        "distinct_4": diversity_stats.get("distinct_4", 0.0),
        "mauve": None,
        "entropy": weighted_mean(payloads, counts, "entropy"),
        "generated_seqs": samples,
        "sharded": True,
        "shards": [
            {
                "shard_index": int(item["shard_index"]),
                "gpu": item["gpu"],
                "num_sample_batches": int(item["num_sample_batches"]),
                "num_generated_samples": int(item.get("num_generated_samples") or item["expected_samples"]),
                "result_path": item["result_path"],
                "log_path": item["log_path"],
                "seconds": item["seconds"],
            }
            for item in ordered
        ],
    }
    if first.get("q_t_act") is not None:
        result["q_t_act"] = float(first["q_t_act"])
    if first.get("q_start_t") is not None:
        result["q_start_t"] = float(first["q_start_t"])

    Path(job.result_path).write_text(json.dumps(result, indent=4) + "\n")


def run_job_sharded(job: JobSpec, *, args: argparse.Namespace, repo_root: Path) -> dict:
    complete, validation_errors = job_is_complete(job) if not args.force else (False, [])
    if complete:
        return {
            "name": job.name,
            "status": "cached",
            "gpu": args.gpus,
            "returncode": 0,
            "validation_errors": [],
        }
    if validation_errors and Path(job.result_path).exists():
        print(f"RERUN {job.name}: {'; '.join(validation_errors[:3])}")

    gpus = parse_gpu_list(args.gpus)
    batch_splits = split_batches(job.num_sample_batches, len(gpus))
    shards = [
        (shard_index, gpu, num_batches)
        for shard_index, (gpu, num_batches) in enumerate(zip(gpus, batch_splits))
        if num_batches > 0
    ]
    if not shards:
        raise ValueError(f"{job.name}: no shard has any sample batches")

    Path(job.job_dir).mkdir(parents=True, exist_ok=True)
    progress_paths: dict[int, Path] = {}
    expected_by_shard: dict[int, int] = {}
    for shard_index, gpu, num_batches in shards:
        shard_dir, _, _, shard_progress_path = shard_paths(job, shard_index, gpu)
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_progress_path.write_text(
            json.dumps(
                {
                    "generated_samples": 0,
                    "total_samples": num_batches * job.eval_batch_size,
                    "sampler": job.sampler,
                    "steps": job.steps,
                }
            )
        )
        progress_paths[shard_index] = shard_progress_path
        expected_by_shard[shard_index] = num_batches * job.eval_batch_size

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = {
            executor.submit(
                run_job_shard,
                job,
                shard_index=shard_index,
                gpu=gpu,
                num_sample_batches=num_batches,
                args=args,
                repo_root=repo_root,
            ): shard_index
            for shard_index, gpu, num_batches in shards
        }
        shard_results = []
        generated_by_shard = {shard_index: 0 for shard_index, _, _ in shards}
        completed_shards: set[int] = set()
        with make_generation_bar(job.target_samples, desc=f"Generating {job.name}") as progress:
            while futures:
                done, _ = wait(futures.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
                for shard_index, progress_path in progress_paths.items():
                    if shard_index in completed_shards:
                        continue
                    generated_by_shard[shard_index] = min(
                        read_progress_count(progress_path),
                        expected_by_shard[shard_index],
                    )
                current_generated = sum(generated_by_shard.values())
                if current_generated > progress.n:
                    progress.update(current_generated - progress.n)

                for future in done:
                    shard_index = futures.pop(future)
                    result = future.result()
                    shard_results.append(result)
                    if result["status"] in {"ok", "cached"}:
                        generated_by_shard[shard_index] = expected_by_shard[shard_index]
                    completed_shards.add(shard_index)

                current_generated = sum(generated_by_shard.values())
                if current_generated > progress.n:
                    progress.update(current_generated - progress.n)
    elapsed = time.perf_counter() - started

    failed = [item for item in shard_results if item["status"] == "failed"]
    if failed:
        lines = [f"{job.name} failed in {len(failed)} shard(s)."]
        for item in failed:
            log_path = Path(item["log_path"])
            tail = "\n".join(log_path.read_text().splitlines()[-30:]) if log_path.exists() else ""
            lines.append(
                f"\n--- shard={item['shard_index']} gpu={item['gpu']} "
                f"returncode={item['returncode']} samples={item.get('num_generated_samples')} "
                f"expected={item['expected_samples']} ---\n{tail}"
            )
        message = "\n".join(lines)
        Path(job.log_path).write_text(message)
        print(f"FAILED {job.name} sharded gpus={args.gpus}\n{message}")
        return {
            "name": job.name,
            "status": "failed",
            "gpu": args.gpus,
            "returncode": 1,
            "seconds": elapsed,
            "validation_errors": [f"{len(failed)} shard(s) failed"],
            "shards": shard_results,
        }

    merge_shard_results(job, shard_results, wall_seconds=elapsed)
    complete, validation_errors = job_is_complete(job)
    status = "ok" if complete else "failed"
    log_lines = [
        f"{job.name} sharded run status={status} seconds={elapsed:.2f}",
        f"target_samples={job.target_samples}",
        f"eval_batch_size={job.eval_batch_size}",
        f"num_sample_batches={job.num_sample_batches}",
    ]
    for item in sorted(shard_results, key=lambda result: int(result["shard_index"])):
        log_lines.append(
            f"shard={item['shard_index']} gpu={item['gpu']} "
            f"status={item['status']} batches={item['num_sample_batches']} "
            f"samples={item.get('num_generated_samples') or item['expected_samples']} "
            f"seconds={item['seconds']:.2f} result={item['result_path']}"
        )
    if validation_errors:
        log_lines.append("validation_errors=" + "; ".join(validation_errors[:5]))
    Path(job.log_path).write_text("\n".join(log_lines) + "\n")
    if validation_errors:
        print("Validation errors:", "; ".join(validation_errors[:5]))
    return {
        "name": job.name,
        "status": status,
        "gpu": args.gpus,
        "returncode": 0 if status == "ok" else 1,
        "seconds": elapsed,
        "validation_errors": validation_errors,
        "shards": shard_results,
    }


def run_jobs(jobs: list[JobSpec], *, args: argparse.Namespace, repo_root: Path) -> list[dict]:
    if args.gpu_scheduling == "shards":
        groups = grouped_jobs(jobs)
        results: list[dict] = []
        stats = {"ok": 0, "cached": 0, "failed": 0}
        with make_progress_bar(len(groups)) as progress:
            progress.set_postfix(
                queued=len(groups),
                running=0,
                ok=0,
                cached=0,
                failed=0,
            )
            for index, group in enumerate(groups, start=1):
                name = group_name(group)
                steps = ",".join(str(job.steps) for job in group)
                progress.write("Running this configuration...")
                progress.write(json.dumps(run_config_summary(group, args), indent=2))
                progress.write(
                    f"START {name} steps={steps} gpus={args.gpus} scheduling=shards"
                )
                progress.set_postfix(
                    queued=len(groups) - index,
                    running=1,
                    ok=stats["ok"],
                    cached=stats["cached"],
                    failed=stats["failed"],
                )
                group_results = []
                for job in group:
                    progress.write("Running this step job...")
                    progress.write(json.dumps(step_config_summary(job, args), indent=2))
                    progress.write(f"  STEP {job.name} gpus={args.gpus}")
                    group_results.append(run_job_sharded(job, args=args, repo_root=repo_root))
                results.extend(group_results)
                group_status = (
                    "failed"
                    if any(result["status"] == "failed" for result in group_results)
                    else "cached"
                    if all(result["status"] == "cached" for result in group_results)
                    else "ok"
                )
                stats[group_status] = stats.get(group_status, 0) + 1
                progress.update(1)
                progress.write(
                    f"DONE {index}/{len(groups)} {name} "
                    f"status={group_status} gpus={args.gpus}"
                )
                progress.set_postfix(
                    queued=len(groups) - index,
                    running=0,
                    ok=stats["ok"],
                    cached=stats["cached"],
                    failed=stats["failed"],
                )
        return results

    pending = list(jobs)
    slots: list[str | None] = []
    for gpu in parse_gpu_list(args.gpus):
        slots.extend([gpu] * max(1, int(args.jobs_per_gpu)))
    if not slots:
        slots = [None]

    results: list[dict] = []
    running = {}
    completed = 0
    total = len(pending)
    stats = {"ok": 0, "cached": 0, "failed": 0}

    def update_postfix(progress) -> None:
        progress.set_postfix(
            queued=len(pending),
            running=len(running),
            ok=stats["ok"],
            cached=stats["cached"],
            failed=stats["failed"],
        )

    with (
        ThreadPoolExecutor(max_workers=len(slots)) as executor,
        make_progress_bar(total) as progress,
    ):
        update_postfix(progress)
        for slot_index, gpu in enumerate(slots):
            if not pending:
                break
            job = pending.pop(0)
            progress.write(f"START {job.name} gpu={gpu}")
            future = executor.submit(run_job, job, gpu=gpu, args=args, repo_root=repo_root)
            running[future] = (slot_index, gpu, job)
            update_postfix(progress)

        while running:
            done, _ = wait(running.keys(), timeout=5.0, return_when=FIRST_COMPLETED)
            if not done:
                update_postfix(progress)
                progress.refresh()
                continue
            for future in done:
                slot_index, gpu, job = running.pop(future)
                result = future.result()
                results.append(result)
                completed += 1
                stats[result["status"]] = stats.get(result["status"], 0) + 1
                progress.update(1)
                progress.write(
                    f"DONE {completed}/{total} {job.name} "
                    f"status={result['status']} gpu={gpu}"
                )
                if pending:
                    next_job = pending.pop(0)
                    progress.write(f"START {next_job.name} gpu={gpu}")
                    next_future = executor.submit(
                        run_job,
                        next_job,
                        gpu=gpu,
                        args=args,
                        repo_root=repo_root,
                    )
                    running[next_future] = (slot_index, gpu, next_job)
                update_postfix(progress)
    return results


def main() -> None:
    args = parse_args()
    if args.experiment_config is None:
        args.experiment_config = (
            Path("experiments/marginally_exact_ou_bridge_sweep/configs")
            / f"{args.dataset}.yaml"
        )
    apply_experiment_config(args, sys.argv[1:])
    if args.compute_mauve and args.gpu_scheduling == "shards":
        raise ValueError(
            "--compute-mauve is not supported with --gpu-scheduling shards because "
            "MAUVE must be computed on the merged sample set. Use --gpu-scheduling jobs "
            "or compute MAUVE later from the saved samples."
        )
    repo_root = find_repo_root()
    jobs, output_dir, metadata = build_jobs(args, repo_root)
    all_jobs = jobs
    all_groups = grouped_jobs(all_jobs)
    groups = select_group_shard(
        all_groups,
        shard_index=args.group_shard_index,
        num_shards=args.num_group_shards,
    )
    jobs = flatten_groups(groups)
    metadata.update(
        {
            "num_total_groups": len(all_groups),
            "group_shard_index": args.group_shard_index,
            "num_group_shards": args.num_group_shards,
            "selected_groups": [group_name(group) for group in groups],
        }
    )
    manifest_path = output_dir / "manifest.json"
    status_path = (
        output_dir
        / (
            f"run_status.shard{args.group_shard_index:02d}-of-{args.num_group_shards:02d}.json"
            if args.num_group_shards > 1
            else "run_status.json"
        )
    )
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    if not args.skip_manifest_write:
        write_manifest(output_path=manifest_path, jobs=all_jobs, metadata=metadata)
    print(f"Experiment config: {args.experiment_config}")
    print(f"Manifest: {manifest_path}")
    if args.gpu_scheduling == "shards":
        print(f"Total runs: {len(all_groups)}")
        print(f"Selected runs: {len(groups)}")
        print(f"Selected step jobs: {len(jobs)}")
        print(f"Samples per step job: {metadata['target_samples']}")
    else:
        print(f"Total jobs: {len(all_jobs)}")
        print(f"Selected jobs: {len(jobs)}")
        print(f"Samples per job: {metadata['target_samples']}")
    if args.num_group_shards > 1:
        print(
            f"Group shard: {args.group_shard_index}/{args.num_group_shards} "
            f"selected={[group_name(group) for group in groups]}"
        )
    print(f"GPU scheduling: {args.gpu_scheduling} ({args.gpus})")
    if args.gpu_scheduling == "shards":
        gpus = parse_gpu_list(args.gpus)
        shard_batches = split_batches(metadata["num_sample_batches"], len(gpus))
        shard_samples = [num_batches * metadata["eval_batch_size"] for num_batches in shard_batches]
        print(f"Samples per shard per step job: {shard_samples}")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        if args.gpu_scheduling == "shards":
            for group in groups[:20]:
                steps = ",".join(str(job.steps) for job in group)
                print(f"{group_name(group)} steps={steps}")
                print(json.dumps(run_config_summary(group, args), indent=2))
            if len(groups) > 20:
                print(f"... {len(groups) - 20} more runs")
        else:
            for job in jobs[:20]:
                print(job.name)
            if len(jobs) > 20:
                print(f"... {len(jobs) - 20} more")
        return

    if not jobs:
        print("No jobs selected for this group shard.")
        status_path.write_text(json.dumps({"jobs": []}, indent=2) + "\n")
        return

    status = run_jobs(jobs, args=args, repo_root=repo_root)
    status_path.write_text(json.dumps({"jobs": status}, indent=2) + "\n")

    if not args.skip_final_summary:
        rows = collect_job_results(all_jobs)
        write_summary(rows, output_csv=summary_csv, output_json=summary_json)
    failed = [item for item in status if item["status"] == "failed"]
    print(f"Wrote status: {status_path}")
    if not args.skip_final_summary:
        print(f"Wrote summary CSV: {summary_csv}")
        print(f"Wrote summary JSON: {summary_json}")
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed")


if __name__ == "__main__":
    main()
