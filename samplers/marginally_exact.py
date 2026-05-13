import torch
import torch.nn.functional as F

import utils
from samplers.base import BaseSampler


class _MarginallyExactBaseSampler(BaseSampler):
    """Marginally exact FLM bridge sampler in SI coordinates.

    The FLM interpolation is
        x_t = t x_1 + (1 - t) eps.

    Conditioning on a sampled endpoint x_1 and simplifying the matched VP
    bridge directly in SI coordinates gives

        x_s | x_t, x_1 ~ N(a_{t,s} x_t + b_{t,s} x_1, v_{t,s} I),

    for 0 <= t < s <= 1, where

        a_{t,s} = t (1 - s)^2 / (s (1 - t)^2)
        b_{t,s} = (s - t) (s + t - 2 s t) / (s (1 - t)^2)
        v_{t,s} = (1 - s)^2 (s - t) (s + t - 2 s t) / (s^2 (1 - t)^2).

    This class implements the per-token marginal endpoint draw used in the
    paper and applies those coefficients directly, without routing through an
    explicit SI -> VP -> SI conversion at runtime.
    """

    def marginally_exact_temperature(self):
        value = getattr(self.config.sampling, "marginally_exact_temperature", None)
        if value is None:
            value = getattr(self.config.sampling, "corrected_temperature", None)
        if value is None:
            value = self.config.sampling.temperature
        return float(value)

    def marginally_exact_top_p(self):
        value = getattr(self.config.sampling, "marginally_exact_p_nucleus", None)
        if value is None:
            value = getattr(self.config.sampling, "corrected_p_nucleus", None)
        if value is None:
            value = self.config.sampling.p_nucleus
        return float(value)

    def sample_clean_one_hot(self, clean_probs):
        sample_dtype = (
            torch.float64 if bool(self.config.sampling.use_float64) else torch.float32
        )
        probs = clean_probs.to(sample_dtype)
        probs = probs.clamp_min(torch.finfo(sample_dtype).tiny)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(sample_dtype).tiny
        )
        logits = probs.log()

        temperature = self.marginally_exact_temperature()
        if temperature < 0.0:
            raise ValueError(
                "marginally_exact_temperature must be non-negative; "
                "use 0.0 for greedy endpoint selection."
            )
        if temperature != 1.0:
            if temperature == 0.0:
                top_p = self.marginally_exact_top_p()
                if top_p < 1.0:
                    logits = utils.top_k_top_p_filtering(logits, top_p=top_p)
                token_ids = logits.argmax(dim=-1)
                return F.one_hot(token_ids, self.vocab_size).to(
                    dtype=clean_probs.dtype
                )
            logits = logits / temperature

        top_p = self.marginally_exact_top_p()
        if top_p < 1.0:
            logits = utils.top_k_top_p_filtering(logits, top_p=top_p)

        uniform = torch.rand_like(logits, dtype=sample_dtype)
        uniform = uniform.clamp_(
            min=torch.finfo(sample_dtype).tiny,
            max=1.0 - torch.finfo(sample_dtype).eps,
        )
        gumbels = -torch.log(-torch.log(uniform))
        token_ids = (logits + gumbels).argmax(dim=-1)
        return F.one_hot(token_ids, self.vocab_size).to(dtype=clean_probs.dtype)

    def bridge_coefficients(self, t_curr, t_next):
        t_curr = t_curr.clamp(0.0, 1.0)
        t_next = t_next.clamp(0.0, 1.0)
        if (t_next < t_curr).any():
            raise ValueError("Marginally exact sampler requires t_next >= t_curr")

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

    def bridge_step(self, latent, clean_probs, t_curr, t_next, final_step, add_noise):
        clean_sample = self.sample_clean_one_hot(clean_probs).to(latent.dtype)
        if final_step:
            return clean_sample

        coef_xt, coef_x1, var = self.bridge_coefficients(
            t_curr=t_curr,
            t_next=t_next,
        )
        updated = (
            coef_xt.view(-1, 1, 1) * latent
            + coef_x1.view(-1, 1, 1) * clean_sample
        )
        if add_noise:
            updated = updated + torch.sqrt(var).view(-1, 1, 1) * torch.randn_like(
                latent
            )
        return updated


class MarginallyExactSampler(_MarginallyExactBaseSampler):
    """Stochastic marginally exact sampler with Gaussian bridge noise."""

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
        return self.bridge_step(
            latent=latent,
            clean_probs=clean_probs,
            t_curr=t_curr,
            t_next=t_next,
            final_step=final_step,
            add_noise=True,
        )


class MarginallyExactDeterministicSampler(_MarginallyExactBaseSampler):
    """Deterministic marginally exact sampler without Gaussian bridge noise."""

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
        return self.bridge_step(
            latent=latent,
            clean_probs=clean_probs,
            t_curr=t_curr,
            t_next=t_next,
            final_step=final_step,
            add_noise=False,
        )
