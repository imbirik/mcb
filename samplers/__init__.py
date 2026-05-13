from samplers.base import BaseSampler
from samplers.marginally_exact import (
    MarginallyExactDeterministicSampler,
    MarginallyExactSampler,
)
from samplers.ode import ODESampler
from samplers.sde import SDESampler


SAMPLER_REGISTRY = {
    "ode": ODESampler,
    "naive_ode": ODESampler,
    "sde": SDESampler,
    "bridge_sde": SDESampler,
    "mean_bridge": SDESampler,
    "euler_maruyama": SDESampler,
    "marginally_exact": MarginallyExactSampler,
    "marginally_exact_sampler": MarginallyExactSampler,
    "marginally_exact_deterministic": MarginallyExactDeterministicSampler,
    "marginally_exact_no_noise": MarginallyExactDeterministicSampler,
    "corrected": MarginallyExactSampler,
    "corrected_sampler": MarginallyExactSampler,
}


def build_sampler(config, tokenizer, checkpoint_path=None, device=None):
    sampler_name = getattr(config.sampling, "flm_sampler", "ode")
    sampler_cls = SAMPLER_REGISTRY.get(sampler_name)
    if sampler_cls is None:
        raise ValueError(f"Unsupported sampler: {sampler_name}")
    return sampler_cls.from_checkpoint(
        config=config,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        device=device,
    )
