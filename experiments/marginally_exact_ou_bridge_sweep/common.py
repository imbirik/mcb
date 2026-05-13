#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "lm1b": {
        "data_config": "lm1b-wrap",
        "checkpoint_name": "lm1b_flm.ckpt",
        "text_length": 128,
        "eval_batch_size": 4,
        "summary_dataset": "lm1b",
    },
    "owt": {
        "data_config": "openwebtext-split",
        "checkpoint_name": "owt_flm.ckpt",
        "text_length": 1024,
        "eval_batch_size": 2,
        "summary_dataset": "openwebtext-split",
    },
}

SUMMARY_FIELDS = [
    "name",
    "dataset",
    "sampler",
    "steps",
    "temperature",
    "nucleus_p",
    "seed",
    "text_length",
    "target_samples",
    "num_generated_samples",
    "num_sample_batches",
    "eval_batch_size",
    "generation_seconds",
    "perplexity",
    "generative_ppl",
    "entropy",
    "diversity",
    "distinct_2",
    "distinct_3",
    "distinct_4",
    "mauve",
    "result_path",
    "samples_json_path",
    "samples_jsonl_path",
    "samples_txt_path",
    "log_path",
]


@dataclass(frozen=True)
class JobSpec:
    name: str
    dataset: str
    sampler: str
    steps: int
    temperature: float | None
    nucleus_p: float | None
    seed: int
    text_length: int
    target_samples: int
    eval_batch_size: int
    num_sample_batches: int
    checkpoint_path: str
    job_dir: str
    result_path: str
    samples_json_path: str
    samples_jsonl_path: str
    samples_txt_path: str
    log_path: str
    overrides: list[str]


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "main.py").exists() and (path / "configs").exists():
            return path
    raise FileNotFoundError(f"Could not find repo root from {current}")


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_gpu_list(value: str) -> list[str | None]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts or parts == ["cpu"]:
        return [None]
    return parts


def powers_of_two_upto(limit: int) -> list[int]:
    return [2**idx for idx in range(int(math.floor(math.log2(limit))) + 1)]


def fmt_float(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def endpoint_pairs(temperatures: list[float], nucleus_ps: list[float]) -> list[tuple[float, float]]:
    return [(float(temp), float(top_p)) for temp in temperatures for top_p in nucleus_ps]


def make_job(
    *,
    output_dir: Path,
    dataset: str,
    preset: dict[str, Any],
    sampler: str,
    steps: int,
    temperature: float | None,
    nucleus_p: float | None,
    seed: int,
    text_length: int,
    target_samples: int,
    eval_batch_size: int,
    num_sample_batches: int,
    ppl_batch_size: int,
    mauve_batch_size: int,
    checkpoint_path: Path,
    compute_mauve: bool,
    compute_generative_perplexity: bool,
    extra_overrides: list[str],
) -> JobSpec:
    parts = [dataset, sampler, f"steps{steps}"]
    if sampler == "marginally_exact":
        if temperature is None or nucleus_p is None:
            raise ValueError("marginally_exact jobs require temperature and nucleus_p")
        parts.extend([f"temp{fmt_float(temperature)}", f"p{fmt_float(nucleus_p)}"])
    name = "__".join(parts)
    job_dir = output_dir / "jobs" / name
    result_path = job_dir / "result.json"
    samples_json_path = job_dir / "samples.json"
    samples_jsonl_path = job_dir / "samples.jsonl"
    samples_txt_path = job_dir / "samples.txt"
    log_path = job_dir / "run.log"

    overrides = [
        "mode=sample_eval",
        f"seed={seed}",
        f"data={preset['data_config']}",
        "model=small",
        "algo=flm",
        f"model.length={text_length}",
        f"eval.checkpoint_path={checkpoint_path}",
        f"eval.generated_samples_path={result_path}",
        f"hydra.run.dir={job_dir / 'hydra'}",
        "hydra.job.chdir=false",
        f"sampling.flm_sampler={sampler}",
        f"sampling.steps={steps}",
        f"sampling.num_sample_batches={num_sample_batches}",
        f"loader.eval_global_batch_size={eval_batch_size}",
        f"loader.eval_batch_size={eval_batch_size}",
        f"eval.perplexity_batch_size={ppl_batch_size}",
        f"eval.mauve_batch_size={mauve_batch_size}",
        f"eval.compute_mauve={str(bool(compute_mauve)).lower()}",
        f"eval.compute_generative_perplexity={str(bool(compute_generative_perplexity)).lower()}",
        "sampling.use_float64=true",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        *extra_overrides,
    ]
    if sampler == "marginally_exact":
        overrides.extend(
            [
                f"sampling.temperature={temperature}",
                f"sampling.p_nucleus={nucleus_p}",
                f"sampling.marginally_exact_temperature={temperature}",
                f"sampling.marginally_exact_p_nucleus={nucleus_p}",
            ]
        )

    return JobSpec(
        name=name,
        dataset=dataset,
        sampler=sampler,
        steps=steps,
        temperature=temperature,
        nucleus_p=nucleus_p,
        seed=seed,
        text_length=text_length,
        target_samples=target_samples,
        eval_batch_size=eval_batch_size,
        num_sample_batches=num_sample_batches,
        checkpoint_path=str(checkpoint_path),
        job_dir=str(job_dir),
        result_path=str(result_path),
        samples_json_path=str(samples_json_path),
        samples_jsonl_path=str(samples_jsonl_path),
        samples_txt_path=str(samples_txt_path),
        log_path=str(log_path),
        overrides=overrides,
    )


def export_samples(job: JobSpec) -> dict[str, Any]:
    result_path = Path(job.result_path)
    if not result_path.exists():
        return {"num_generated_samples": 0}

    payload = json.loads(result_path.read_text())
    samples = payload.get("generated_seqs", [])
    if not isinstance(samples, list) or not all(isinstance(text, str) for text in samples):
        raise ValueError(f"{result_path} does not contain generated_seqs as a list of strings")

    samples_json_path = Path(job.samples_json_path)
    samples_jsonl_path = Path(job.samples_jsonl_path)
    samples_txt_path = Path(job.samples_txt_path)
    samples_json_path.parent.mkdir(parents=True, exist_ok=True)

    samples_json_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")
    with samples_jsonl_path.open("w", encoding="utf-8") as handle:
        for sample_index, text in enumerate(samples):
            handle.write(
                json.dumps(
                    {
                        "sample_index": sample_index,
                        "text": text,
                        "dataset": job.dataset,
                        "sampler": job.sampler,
                        "steps": job.steps,
                        "temperature": job.temperature,
                        "nucleus_p": job.nucleus_p,
                        "seed": job.seed,
                        "text_length": job.text_length,
                        "target_samples": job.target_samples,
                        "checkpoint_path": job.checkpoint_path,
                        "result_path": str(result_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    samples_txt_path.write_text("\n\n".join(samples) + "\n", encoding="utf-8")
    return {
        "num_generated_samples": len(samples),
        "samples_json_path": str(samples_json_path),
        "samples_jsonl_path": str(samples_jsonl_path),
        "samples_txt_path": str(samples_txt_path),
    }


def validate_samples_jsonl(
    path: Path,
    *,
    expected_count: int,
    expected_metadata: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    expected_metadata = expected_metadata or {}
    if not path.exists():
        return [f"missing samples JSONL: {path}"]

    count = 0
    required_keys = {
        "sample_index",
        "text",
        "dataset",
        "sampler",
        "steps",
        "temperature",
        "nucleus_p",
        "seed",
        "text_length",
        "target_samples",
        "checkpoint_path",
        "result_path",
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(f"{path}:{line_number}: empty line")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                continue

            missing = required_keys.difference(record)
            if missing:
                errors.append(f"{path}:{line_number}: missing keys {sorted(missing)}")
            if record.get("sample_index") != count:
                errors.append(
                    f"{path}:{line_number}: sample_index={record.get('sample_index')} expected {count}"
                )
            if not isinstance(record.get("text"), str):
                errors.append(f"{path}:{line_number}: text must be a string")
            if not isinstance(record.get("steps"), int):
                errors.append(f"{path}:{line_number}: steps must be an int")
            if not isinstance(record.get("text_length"), int):
                errors.append(f"{path}:{line_number}: text_length must be an int")
            if not isinstance(record.get("target_samples"), int):
                errors.append(f"{path}:{line_number}: target_samples must be an int")
            for key, expected_value in expected_metadata.items():
                if record.get(key) != expected_value:
                    errors.append(
                        f"{path}:{line_number}: {key}={record.get(key)!r} expected {expected_value!r}"
                    )
            count += 1

    if count != expected_count:
        errors.append(f"{path}: contains {count} records, expected {expected_count}")
    return errors


def job_is_complete(job: JobSpec) -> tuple[bool, list[str]]:
    result_path = Path(job.result_path)
    if not result_path.exists():
        return False, [f"missing result JSON: {result_path}"]

    try:
        sample_info = export_samples(job)
    except Exception as exc:  # noqa: BLE001 - surface as validation error
        return False, [f"failed to export samples for {job.name}: {exc}"]

    expected_count = job.num_sample_batches * job.eval_batch_size
    if sample_info.get("num_generated_samples") != expected_count:
        return False, [
            f"{job.name}: result has {sample_info.get('num_generated_samples')} samples, expected {expected_count}"
        ]

    errors = validate_samples_jsonl(
        Path(job.samples_jsonl_path),
        expected_count=expected_count,
        expected_metadata={
            "dataset": job.dataset,
            "sampler": job.sampler,
            "steps": job.steps,
            "temperature": job.temperature,
            "nucleus_p": job.nucleus_p,
            "seed": job.seed,
            "text_length": job.text_length,
            "target_samples": job.target_samples,
        },
    )
    return not errors, errors


def load_jobs_from_manifest(manifest_path: Path) -> list[JobSpec]:
    payload = json.loads(manifest_path.read_text())
    return [JobSpec(**job) for job in payload["jobs"]]


def write_manifest(
    *,
    output_path: Path,
    jobs: list[JobSpec],
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "jobs": [asdict(job) for job in jobs],
            },
            indent=2,
        )
        + "\n"
    )


def collect_job_results(jobs: list[JobSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        result_path = Path(job.result_path)
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text())
        generated = payload.pop("generated_seqs", [])
        sample_info = export_samples(job)
        rows.append(
            {
                "name": job.name,
                "dataset": job.dataset,
                "sampler": job.sampler,
                "steps": job.steps,
                "temperature": job.temperature,
                "nucleus_p": job.nucleus_p,
                "seed": job.seed,
                "text_length": job.text_length,
                "target_samples": job.target_samples,
                "num_generated_samples": len(generated),
                "num_sample_batches": payload.get("num_sample_batches", job.num_sample_batches),
                "eval_batch_size": job.eval_batch_size,
                "result_path": str(result_path),
                "log_path": job.log_path,
                **sample_info,
                **payload,
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], *, output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(SUMMARY_FIELDS)
    extra_fields = sorted({key for row in rows for key in row}.difference(all_fields))
    fields = all_fields + extra_fields
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    output_json.write_text(json.dumps({"results": rows}, indent=2) + "\n")
