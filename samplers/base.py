from __future__ import annotations

from abc import ABC, abstractmethod

import omegaconf
import torch
from tqdm.auto import tqdm

import metrics
import models
import utils


class BaseSampler(ABC):
    """Shared FLM inference runtime.

    This class owns checkpoint loading, EMA restoration, tokenizer access,
    and the time discretization used by all concrete samplers.
    """

    def __init__(self, config, tokenizer, device=None):
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.num_tokens = int(config.model.length)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if config.algo.backbone != "dit":
            raise ValueError(
                "This inference workspace only supports the DIT FLM backbone. "
                f"Got: {config.algo.backbone}"
            )

        self.backbone = models.dit.DIT(config, vocab_size=self.vocab_size)
        self.backbone.to(self.device)
        self.backbone.eval()

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=(
                config.eval.gen_ppl_eval_model_name_or_path
            ),
            eval_ppl_batch_size=config.eval.perplexity_batch_size,
        )
        self.metrics.to(self.device)

        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)

    @property
    def dtype(self):
        return next(self.backbone.parameters()).dtype

    @classmethod
    def from_checkpoint(
        cls,
        config,
        tokenizer,
        checkpoint_path=None,
        device=None,
    ):
        sampler = cls(config=config, tokenizer=tokenizer, device=device)
        sampler.load_checkpoint(checkpoint_path or config.eval.checkpoint_path)
        return sampler

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("teacher."):
                continue
            cleaned_state_dict[key.replace("._orig_mod.", ".")] = value

        backbone_state = {
            key[len("backbone.") :]: value
            for key, value in cleaned_state_dict.items()
            if key.startswith("backbone.")
        }
        missing, unexpected = self.backbone.load_state_dict(
            backbone_state,
            strict=False,
        )
        if unexpected:
            print(f"[WARNING] Unexpected backbone keys: {unexpected}")
        if missing:
            print(f"[WARNING] Missing backbone keys: {missing}")

        use_ema = (
            not bool(getattr(self.config.eval, "disable_ema", False))
            and "ema" in checkpoint
        )
        if use_ema:
            self._apply_ema_state(checkpoint["ema"])

        self.backbone.to(self.device)
        self.backbone.eval()

    def _apply_ema_state(self, ema_state):
        shadow_params = list(ema_state.get("shadow_params", []))
        if not shadow_params:
            print("[WARNING] EMA state found but shadow_params is empty")
            return

        trainable_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if len(shadow_params) != len(trainable_params):
            print(
                "[WARNING] EMA parameter count mismatch. "
                f"Checkpoint has {len(shadow_params)}, model expects {len(trainable_params)}. "
                "Using checkpoint backbone weights without EMA override."
            )
            return

        with torch.no_grad():
            for param, shadow in zip(trainable_params, shadow_params):
                if param.shape != shadow.shape:
                    print(
                        "[WARNING] EMA parameter shape mismatch. "
                        "Using checkpoint backbone weights without EMA override."
                    )
                    return
            for param, shadow in zip(trainable_params, shadow_params):
                param.copy_(shadow.to(param.device, dtype=param.dtype))

    def tau_to_t(self, tau):
        return utils.alpha_to_gamma(tau, self.lut_a2g)

    def t_to_tau(self, t):
        return utils.gamma_to_alpha(t, self.lut_g2a)

    def discretization(self, num_steps=None):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if isinstance(num_steps, (list, tuple, omegaconf.ListConfig)):
            if len(num_steps) != 1:
                raise ValueError(f"Expected a single step count, got {num_steps}")
            num_steps = num_steps[0]
        num_steps = int(num_steps)
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        return num_steps, tau_vals

    def process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def process_model_output(self, model_output, cap_value=30.0):
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)

    def predict_clean_log_probs(self, latent, tau, tau_prime=None):
        sigma = self.process_sigma(tau)
        sigma_prime = None if tau_prime is None else self.process_sigma(tau_prime)
        autocast_enabled = self.device.type == "cuda"
        with torch.amp.autocast(
            device_type="cuda" if autocast_enabled else "cpu",
            dtype=torch.float32,
            enabled=autocast_enabled,
        ):
            if sigma_prime is None:
                model_output = self.backbone(latent, sigma)
            else:
                model_output = self.backbone(latent, sigma, sigma_prime)
        return self.process_model_output(model_output)

    def predict_clean_probs(self, latent, tau, tau_prime=None):
        return self.predict_clean_log_probs(latent, tau, tau_prime=tau_prime).exp()

    def initial_latent(self, num_samples):
        return torch.randn(
            (int(num_samples), self.num_tokens, self.vocab_size),
            device=self.device,
            dtype=self.dtype,
        )

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids)

    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_steps=None,
        show_progress=True,
        progress_desc=None,
    ):
        num_steps, tau_vals = self.discretization(num_steps=num_steps)
        latent = self.initial_latent(num_samples)
        batch_size = int(num_samples)

        step_iterator = range(num_steps)
        if show_progress:
            step_iterator = tqdm(
                step_iterator,
                total=num_steps,
                desc=progress_desc or self.__class__.__name__,
                leave=False,
            )

        for step_index in step_iterator:
            tau_curr = tau_vals[step_index].expand(batch_size)
            tau_next = tau_vals[step_index + 1].expand(batch_size)
            t_curr = self.tau_to_t(tau_curr)
            t_next = self.tau_to_t(tau_next)
            dt = t_next - t_curr
            clean_probs = self.predict_clean_probs(latent, tau_curr)
            latent = self.step(
                latent=latent,
                clean_probs=clean_probs,
                tau_curr=tau_curr,
                tau_next=tau_next,
                t_curr=t_curr,
                t_next=t_next,
                dt=dt,
                final_step=step_index == num_steps - 1,
            )

        return latent.argmax(dim=-1)

    @abstractmethod
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
        raise NotImplementedError
