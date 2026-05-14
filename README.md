# Sampling from Flow Language Models via Marginal-Conditioned Bridges

[![arXiv](https://img.shields.io/badge/arXiv-2605.13681-b31b1b.svg)](https://arxiv.org/abs/2605.13681)

Official implementation of MCB sampler for flow language models, based on [david3684/flm](https://github.com/david3684/flm).

We propose a new sampler for FLMs which exploits additional probabilistic  structure of the denoiser, and admits temprature scaled and nucleus (top-p) sampling.

## Setup

Install dependencies with `uv`:

```bash
uv sync
# install flash attention
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Running Experiments

To reproduce the experiments, run notebooks in `notebooks/` folder.

### Reference
```
@article{azangulov2026mcb,
  title={Sampling from Flow Language Models via Marginal-Conditioned Bridges},
  author={Iskander, Azangulov and Leo Zhang},
  journal={arXiv preprint arXiv:2605.13681},
  year={2026}
}
```
