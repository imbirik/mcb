import torch

from samplers.base import BaseSampler


class SDESampler(BaseSampler):
    """Conditional-mean SI bridge sampler with Gaussian bridge noise."""

    def noise_scale(self):
        return float(getattr(self.config.sampling, "sde_noise_scale", 1.0))

    def bridge_coefficients(self, t_curr, t_next):
        t_curr = t_curr.clamp(0.0, 1.0)
        t_next = t_next.clamp(0.0, 1.0)
        if (t_next < t_curr).any():
            raise ValueError("SDE bridge sampler requires t_next >= t_curr")

        delta = (t_next - t_curr).clamp_min(0.0)
        remaining = (1.0 - t_curr).clamp_min(1e-8)
        t_next_safe = t_next.clamp_min(1e-8)
        mix_term = t_next * (1.0 - t_curr) + t_curr * (1.0 - t_next)

        coef_xt = (
            t_curr
            * (1.0 - t_next).square()
            / (t_next_safe * remaining.square())
        )
        coef_x1 = delta * mix_term / (t_next_safe * remaining.square())
        var = (
            (1.0 - t_next).square()
            * delta
            * mix_term
            / (t_next_safe.square() * remaining.square())
        )
        return coef_xt, coef_x1, var.clamp_min(0.0)

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
        del tau_curr, tau_next, dt
        clean_probs = clean_probs.to(latent.dtype)
        if final_step:
            return clean_probs

        coef_xt, coef_x1, var = self.bridge_coefficients(
            t_curr=t_curr,
            t_next=t_next,
        )
        updated = (
            coef_xt.view(-1, 1, 1) * latent
            + coef_x1.view(-1, 1, 1) * clean_probs
        )

        noise_std = (
            self.noise_scale()
            * torch.sqrt(var).to(dtype=latent.dtype).view(-1, 1, 1)
        )
        return updated + noise_std * torch.randn_like(latent)
