import torch

from samplers.base import BaseSampler


class ODESampler(BaseSampler):
    """Deterministic Euler / ODE FLM sampler."""

    def step(
        self,
        latent,
        clean_probs,
        tau_curr,
        tau_next,
        t_curr,
        t_next,
        dt,
        final_step,
    ):
        del tau_curr, tau_next, t_next
        clean_probs = clean_probs.to(latent.dtype)
        if final_step:
            return clean_probs
        denom = (1.0 - t_curr).clamp_min(1e-5).view(-1, 1, 1)
        velocity = (clean_probs - latent) / denom
        return latent + dt.view(-1, 1, 1) * velocity
