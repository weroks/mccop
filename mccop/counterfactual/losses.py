import torch
from torch import nn


class MarginLoss(nn.Module):
    """Computes a margin-based loss for binary classification counterfactuals."""

    def __init__(self, margin: float = 2.2) -> None:
        """Initializes the MarginLoss.

        Args:
            margin: The desired margin between classes.
        """
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the margin loss.

        Args:
            logits: The model's output logits for the counterfactual samples.
            target: The desired target classes (0 or 1).
        """
        y_sign = 2 * target - 1
        return nn.functional.softplus(self.margin - y_sign * logits).mean()


class KLSparsityLoss(nn.Module):
    """Encapsulates KL-based sparsity logic as a stateful module."""

    def __init__(
        self,
        target_edits: int = 2,
        temperature: float = 5.0,
        penalty_too_many: float = 5.0,
        penalty_too_few: float = 10.0,
        sharpness: float = 2.0,
    ) -> None:
        super().__init__()
        self.target_edits = target_edits
        self.temperature = temperature
        self.penalty_too_many = penalty_too_many
        self.penalty_too_few = penalty_too_few
        self.sharpness = sharpness

    def forward(self, cf_logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes the sparsity loss based on KL divergence.

        Args:
            cf_logits: The counterfactual logits from the CHEAP model.
            ref_logits: The reference logits from the original input.
            mask: The attention mask for the sequence.

        Returns:
            The sparsity loss as a mean over the batch.
        """
        p_cf = torch.softmax(cf_logits / self.temperature, dim=-1)
        p_ref = torch.softmax(ref_logits / self.temperature, dim=-1)

        kl = torch.sum(p_ref * (torch.log(p_ref + 1e-6) - torch.log(p_cf + 1e-6)), dim=-1)
        if mask is not None:
            kl = kl * mask

        soft_changed = torch.sigmoid(self.sharpness * (kl - 0.5)) + (0.01 * kl)
        estimated_edits = soft_changed.sum(dim=-1)

        diff = estimated_edits - self.target_edits
        loss = torch.where(diff > 0, self.penalty_too_many * (diff**2), self.penalty_too_few * (diff**2))
        return loss.mean()
