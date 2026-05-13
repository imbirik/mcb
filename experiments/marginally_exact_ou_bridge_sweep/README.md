# Marginally Exact OU-Bridge Sweep

This experiment runs the paper-matched 1,024-sample evaluation sweep for the
stochastic `marginally_exact` OU-bridge sampler and the ODE baseline.

## Prepare

The runner checks for the released FLM checkpoint under `checkpoints/flm/` and
downloads only the checkpoint required by `--dataset` when it is missing.
Downloading is done in-process and requires `gdown`.

Expected default paths:

- `checkpoints/flm/lm1b_flm.ckpt`
- `checkpoints/flm/owt_flm.ckpt`

You can also pass `--checkpoint-path /path/to/checkpoint.ckpt` to use an
explicit checkpoint, or `--download-checkpoints never` to disable automatic
downloads.

## Run

LM1B:

```bash
python experiments/marginally_exact_ou_bridge_sweep/run_sweep.py \
  --dataset lm1b \
  --gpus 0,1,2,3 \
  --target-samples 1024
```

OpenWebText:

```bash
python experiments/marginally_exact_ou_bridge_sweep/run_sweep.py \
  --dataset owt \
  --gpus 0,1,2,3 \
  --target-samples 1024
```

The default run reads:

- `experiments/marginally_exact_ou_bridge_sweep/configs/lm1b.yaml`
- `experiments/marginally_exact_ou_bridge_sweep/configs/owt.yaml`

Use `--experiment-config path/to/config.yaml` to run a different sweep. The
config file contains the dataset, generation settings, metrics, runtime, and
the sweep axes (`steps`, `temperatures`, and `nucleus_ps`). Command-line
arguments override the config.

By default, `--gpu-scheduling shards` groups the sweep into 19 visible runs:
one ODE run and one run for each temperature/nucleus-p pair. Within each run,
the script evaluates all powers-of-two step counts. For each step count,
`--target-samples 1024 --gpus 0,1,2,3` runs four shards of 256 samples and
merges them into that step job's final `result.json` and sample files. To run
different step jobs on different GPUs instead, pass `--gpu-scheduling jobs`.

Before each run, the launcher prints the exact configuration it is about to
execute. During each step job, it shows an aggregate generation progress bar
over the full sample count, for example `Generating ... 256/1024`.

The sweep is text-generation only by default:

```yaml
metrics:
  compute_generative_perplexity: false
  compute_mauve: false
```

This keeps sampling decoupled from metric evaluation. The generated texts are
saved to `samples.json`, `samples.jsonl`, and `samples.txt`; run metric
evaluation later from those files. To opt back into PPL during sampling, pass
`--compute-generative-perplexity`.

The default sweep is:

- ODE baseline
- `marginally_exact` temperatures: `0.1,0.3,0.5,0.7,0.9,1.0`
- nucleus-p values: `0.9,0.95,1.0`
- powers-of-two step counts up to the generated text length

Each job writes:

- `result.json`: metrics and generated text
- `samples.json`: plain JSON list of generated texts
- `samples.jsonl`: one generated text per line with metadata
- `samples.txt`: readable text dump
- `run.log`: generation log

The runner supports resume. A job is skipped only when `result.json` exists and
`samples.jsonl` validates to the expected sample count and metadata.
During execution it shows a `tqdm` progress bar with queued, running, cached,
completed, and failed job counts.

## Validate

```bash
python experiments/marginally_exact_ou_bridge_sweep/validate_samples.py \
  --manifest outputs/experiments/marginally_exact_ou_bridge_sweep/lm1b/manifest.json
```

## Rebuild Summaries

```bash
python experiments/marginally_exact_ou_bridge_sweep/collect_results.py \
  --manifest outputs/experiments/marginally_exact_ou_bridge_sweep/lm1b/manifest.json
```
