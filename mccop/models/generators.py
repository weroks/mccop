import torch
from dima.diffusion.dima import DiMAModel


class DiMAWrapper(DiMAModel):
    """A wrapper around DiMAModel that adds encode and decode capabilities
    the counterfactual editors manifold projection.

    Assumes input tensors are always [Batch, SeqLen, Dim].
    """

    def encode(self, x: torch.Tensor, t_frac: float = 1.0, noise: torch.Tensor = None) -> torch.Tensor:
        """Encodes a sequence by adding noise, corresponding to q(x_t | x_0).

        Args:
            x: Input tensor [B, L, D]
            t_frac: Fraction of total timesteps to diffuse to (0.0 to 1.0)
            noise: Optional noise tensor [B, L, D]

        Returns:
            Noisy tensor x_t [B, L, D]
        """
        device = x.device
        batch_size = x.shape[0]

        t_val = t_frac * self.dynamic.T
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)

        params = self.dynamic.marginal_params(t)
        mu, std = params["mu"], params["std"]

        if noise is None:
            noise = torch.randn_like(x)

        x_t = x * mu + noise * std
        return x_t

    def decode(self, z: torch.Tensor, t_frac: float = 1.0, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Decode a latent tensor by progressively denoising it.

        Args:
            z: Latent tensor [B, L, D]
            t_frac: Fraction of total timesteps to denoise from
            mask: Attention mask [B, L] indicating valid sequence positions
            **kwargs: Additional arguments passed to solver

        Returns:
            Denoised tensor [B, L, D]
        """
        if mask is None:
            raise ValueError("Mask is required for DiMA decoding to handle variable lengths correctly.")

        total_steps = self.config.generation.N_steps
        n_steps_reverse = int(total_steps * t_frac)

        if n_steps_reverse == 0:
            return z

        t_start = t_frac * self.dynamic.T
        t_min = self.config.generation.t_min

        timesteps = torch.linspace(t_start, t_min, n_steps_reverse + 1, device=z.device)

        x = z
        x_0_self_cond = torch.zeros_like(x)

        for idx in range(n_steps_reverse):
            t_curr = timesteps[idx]
            t_next = timesteps[idx + 1]

            t_curr_batch = torch.full((x.shape[0],), t_curr, device=z.device)
            t_next_batch = torch.full((x.shape[0],), t_next, device=z.device)

            output = self.solver.step(
                x_t=x, t=t_curr_batch, next_t=t_next_batch, mask=mask, x_0_self_cond=x_0_self_cond, **kwargs
            )

            x = output["x"]
            x_0_self_cond = output.get("x_0", torch.zeros_like(x))

        return x
