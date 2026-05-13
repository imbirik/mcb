import functools
import json
import os
import time
from datetime import datetime

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import metrics
import utils
from samplers import build_sampler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

torch.load = functools.partial(torch.load, weights_only=False)

# Allow torch.load(weights_only=True) to safely unpickle Hydra configs stored in checkpoints
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata,
])

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    "device_count", lambda: max(torch.cuda.device_count(), 1)
)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver(
    "div_up", lambda x, y: (x + y - 1) // y
)


def _build_sampler(config, tokenizer):
    return build_sampler(
        config=config,
        tokenizer=tokenizer,
        checkpoint_path=config.eval.checkpoint_path,
    )


def _normalize_num_steps(config):
    steps = config.sampling.steps
    if isinstance(steps, (list, omegaconf.ListConfig)):
        if len(steps) != 1:
            raise ValueError(
                "sample_eval/sample_compare expects a single sampling step count. "
                f"Got: {list(steps)}"
            )
        return int(steps[0])
    return int(steps)


def _decode_reference_batch(batch, tokenizer):
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    texts = []
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    for ids, mask in zip(input_ids, attention_mask):
        valid_len = int(mask.sum().item())
        text = tokenizer.decode(
            ids[:valid_len],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        texts.append(text.strip())
    return texts


def _collect_reference_texts(config, tokenizer, num_texts):
    _, valid_loader = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )
    reference_texts = []
    for batch in valid_loader:
        reference_texts.extend(_decode_reference_batch(batch, tokenizer))
        if len(reference_texts) >= num_texts:
            break
    return reference_texts[:num_texts]


def _compute_optional_text_metrics(
    config,
    sampler,
    generated_texts,
    reference_texts=None,
):
    diversity_stats = metrics.compute_text_diversity(
        generated_texts,
        n_values=tuple(config.eval.diversity_ngrams),
    )
    mauve_score = None
    if config.eval.compute_mauve:
        if reference_texts is None:
            reference_texts = _collect_reference_texts(
                config, sampler.tokenizer, len(generated_texts)
            )
        device_id = 0 if sampler.device.type == "cuda" else -1
        mauve_score = metrics.compute_mauve_score(
            reference_texts=reference_texts,
            generated_texts=generated_texts,
            device_id=device_id,
            max_text_length=int(config.eval.mauve_max_text_length),
            featurize_model_name=config.eval.mauve_featurize_model_name,
            batch_size=int(config.eval.mauve_batch_size),
            verbose=False,
        )
    return diversity_stats, mauve_score


def _run_sample_eval(model, config, reference_texts=None):
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    all_samples = []
    num_steps = _normalize_num_steps(config)
    sampler_name = getattr(config.sampling, "flm_sampler", "ode")
    total_samples = int(config.sampling.num_sample_batches) * int(config.loader.eval_batch_size)
    progress_path = os.environ.get("FLM_PROGRESS_PATH")

    def write_progress(generated_samples: int) -> None:
        if not progress_path:
            return
        tmp_path = f"{progress_path}.tmp"
        payload = {
            "generated_samples": int(generated_samples),
            "total_samples": total_samples,
            "sampler": sampler_name,
            "steps": num_steps,
        }
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, progress_path)

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.perf_counter()
    write_progress(0)

    progress = None
    if tqdm is not None and not progress_path:
        progress = tqdm(total=total_samples, desc="Generating", unit="sample", dynamic_ncols=True)

    try:
        for _ in range(config.sampling.num_sample_batches):
            samples = model.sample(
                num_samples=config.loader.eval_batch_size,
                num_steps=num_steps,
            )
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            all_samples.extend(list(text_samples))
            write_progress(len(all_samples))
            if progress is not None:
                progress.update(len(text_samples))
    finally:
        if progress is not None:
            progress.close()

    elapsed = time.perf_counter() - start_time
    write_progress(len(all_samples))
    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = 0.0
    entropy = 0.0
    diversity_stats = {}
    mauve_score = None
    if config.eval.compute_generative_perplexity:
        print("generative perplexity start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        model.metrics.record_generative_perplexity(
            all_samples,
            config.model.length,
            device=model.device,
        )
        print("generative perplexity end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    diversity_stats, mauve_score = _compute_optional_text_metrics(
        config=config,
        sampler=model,
        generated_texts=all_samples,
        reference_texts=reference_texts,
    )
    print("Generative perplexity:", generative_ppl)
    print("Diversity:", diversity_stats.get("diversity", 0.0))
    print("MAUVE:", mauve_score)
    print("Sample entropy:", entropy)

    result = {
        "sampler": sampler_name,
        "steps": num_steps,
        "num_sample_batches": int(config.sampling.num_sample_batches),
        "generation_seconds": elapsed,
        "perplexity": generative_ppl,
        "generative_ppl": generative_ppl,
        "diversity": diversity_stats.get("diversity", 0.0),
        "distinct_2": diversity_stats.get("distinct_2", 0.0),
        "distinct_3": diversity_stats.get("distinct_3", 0.0),
        "distinct_4": diversity_stats.get("distinct_4", 0.0),
        "mauve": mauve_score,
        "entropy": entropy,
        "generated_seqs": all_samples,
    }
    if sampler_name == "q":
        q_t_act = getattr(config.sampling, "q_t_act", None)
        if q_t_act is not None:
            result["q_t_act"] = float(q_t_act)
            result["q_start_t"] = 1.0 - float(q_t_act)
        else:
            result["q_start_t"] = float(config.sampling.q_start_t)
    return result


def _print_config(config, resolve=True, save_cfg=True):
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section,
                resolve=resolve,
            )
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if save_cfg:
        with fsspec.open(os.path.join(os.getcwd(), "config_tree.txt"), "w") as fp:
            rich.print(tree, file=fp)


def _generate_samples(config, logger, tokenizer):
    logger.info("Starting sample_eval.")
    model = _build_sampler(config=config, tokenizer=tokenizer)
    result = _run_sample_eval(model, config)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, "w") as f:
        json.dump(result, f, indent=4)
    print("Samples saved at:", samples_path)


def _compare_flm_samplers(config, logger, tokenizer):
    logger.info("Starting sample_compare.")

    compare_dir = config.eval.sampler_compare_dir
    os.makedirs(compare_dir, exist_ok=True)

    summary = {
        "checkpoint_path": config.eval.checkpoint_path,
        "algo": config.algo.name,
        "dataset": config.data.train,
        "model": config.model.name,
        "length": int(config.model.length),
        "compare_samplers": list(config.sampling.compare_samplers),
        "results": [],
    }
    reference_texts = None
    if config.eval.compute_mauve:
        total_samples = int(config.loader.eval_batch_size) * int(
            config.sampling.num_sample_batches
        )
        reference_texts = _collect_reference_texts(
            config, tokenizer, total_samples
        )

    original_sampler = config.sampling.flm_sampler
    for sampler_name in config.sampling.compare_samplers:
        logger.info(f"Running sampler={sampler_name}")
        config.sampling.flm_sampler = sampler_name
        model = _build_sampler(config=config, tokenizer=tokenizer)
        result = _run_sample_eval(model, config, reference_texts=reference_texts)
        sampler_path = os.path.join(compare_dir, f"{sampler_name}.json")
        with fsspec.open(sampler_path, "w") as f:
            json.dump(result, f, indent=4)
        summary["results"].append({
            "sampler": sampler_name,
            "samples_path": sampler_path,
            "steps": result["steps"],
            "num_sample_batches": result["num_sample_batches"],
            "generation_seconds": result["generation_seconds"],
            "perplexity": result["perplexity"],
            "generative_ppl": result["generative_ppl"],
            "diversity": result["diversity"],
            "distinct_2": result["distinct_2"],
            "distinct_3": result["distinct_3"],
            "distinct_4": result["distinct_4"],
            "mauve": result["mauve"],
            "entropy": result["entropy"],
            **({"q_t_act": result["q_t_act"]} if "q_t_act" in result else {}),
            **({"q_start_t": result["q_start_t"]} if "q_start_t" in result else {}),
        })
        print("Samples saved at:", sampler_path)

    config.sampling.flm_sampler = original_sampler
    with fsspec.open(config.eval.sampler_comparison_path, "w") as f:
        json.dump(summary, f, indent=4)
    print("Sampler comparison saved at:", config.eval.sampler_comparison_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    if config.algo.name != "flm":
        raise ValueError(
            "This repository has been cleaned to FLM inference only. "
            f"Unsupported algo: {config.algo.name}"
        )
    if config.mode not in {"sample_eval", "sample_compare"}:
        raise ValueError(
            "This repository has been cleaned to inference only. "
            f"Unsupported mode: {config.mode}"
        )

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == "sample_eval":
        _generate_samples(config=config, tokenizer=tokenizer, logger=logger)
    else:
        _compare_flm_samplers(config=config, tokenizer=tokenizer, logger=logger)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
