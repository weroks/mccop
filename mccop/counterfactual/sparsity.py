from abc import ABC, abstractmethod
import torch
from torch import nn


class BaseSparsityMechanism(nn.Module, ABC):
    """Abstract base class for sparsity mechanisms."""

    def setup(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        predictor: nn.Module,
        loss_fn: nn.Module,
        target_classes: torch.Tensor,
        cheap_model: nn.Module | None = None,
    ) -> None:
        """Called once before the optimization loop starts to initialize state (e.g. masks, ref logits)."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        cf: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        cheap_model: nn.Module | None = None,
    ) -> torch.Tensor:
        """Called inside the loop. Returns a scalar loss tensor (default 0)."""
        return torch.tensor(0.0, device=cf.device)

    @abstractmethod
    def apply_gradients(self, cf: torch.Tensor) -> None:
        """Called after backward() but before optimizer.step(). Useful for masking gradients."""
        pass

    @abstractmethod
    def apply_constraints(self, cf: torch.Tensor, x: torch.Tensor) -> None:
        """Called after optimizer.step(). Useful for hard constraints (resetting values)."""
        pass


class GradientMasking(BaseSparsityMechanism):
    """Enforces sparsity by calculating sensitivity and masking gradients for non-top-k tokens."""

    def __init__(self, k: int = 2) -> None:
        super().__init__()
        self.k = k
        self.grad_mask: torch.Tensor | None = None

    def setup(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        predictor: nn.Module,
        loss_fn: nn.Module,
        target_classes: torch.Tensor,
        cheap_model: nn.Module | None = None,  # noqa: ARG002
    ) -> None:
        """Computes the gradient mask based on sensitivity of the predictor's loss to input tokens.

        Args:
            x: Original input tensor of shape (batch_size, seq_len, embed_dim).
            mask: Attention mask tensor of shape (batch_size, seq_len).
            predictor: The predictor model used to compute gradients.
            loss_fn: The loss function used to compute the loss.
            target_classes: The target classes for the counterfactuals.
            cheap_model: Optional CHEAP model (not used here).
        """
        x_check = x.clone().detach().requires_grad_(True)
        init_pred = predictor(x_check)
        init_loss = loss_fn(init_pred, target_classes)
        init_loss.backward()

        sensitivity = x_check.grad.norm(dim=-1)
        sensitivity = sensitivity * mask

        topk_values, topk_indices = torch.topk(sensitivity, k=self.k, dim=1)

        self.grad_mask = torch.zeros_like(sensitivity)
        self.grad_mask.scatter_(1, topk_indices, 1.0)
        self.grad_mask = self.grad_mask.unsqueeze(-1)

    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """Returns 0 loss as this mechanism only affects gradients and constraints."""
        cf = kwargs.get("cf")
        if cf is None:
            if len(args) > 0:
                cf = args[0]
            else:
                raise ValueError("Counterfactual tensor 'cf' not found in args or kwargs.")

        return torch.tensor(0.0, device=cf.device)

    def apply_gradients(self, cf: torch.Tensor) -> None:
        """Applies the gradient mask to the gradients of the counterfactual tensor."""
        if cf.grad is not None and self.grad_mask is not None:
            cf.grad.data.mul_(self.grad_mask)

    def apply_constraints(self, cf: torch.Tensor, x: torch.Tensor) -> None:
        """Resets non-top-k tokens in the counterfactual to their original values."""
        if self.grad_mask is not None:
            with torch.no_grad():
                cf.data = x * (1 - self.grad_mask) + cf.data * self.grad_mask


class KLSparsity(BaseSparsityMechanism):
    """Enforces sparsity via KL divergence on the decoded logits."""

    def __init__(self, loss_fn: nn.Module, weight: float = 1.0) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.weight = weight
        self.ref_logits: torch.Tensor | None = None

    def setup(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cheap_model: nn.Module | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Precomputes reference logits using the CHEAP model.

        Args:
            x: Original input tensor of shape (batch_size, seq_len, embed_dim).
            mask: Attention mask tensor of shape (batch_size, seq_len).
            cheap_model: Required CHEAP model to compute reference logits.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        if cheap_model is None:
            raise ValueError("KLSparsityMechanism requires a cheap_model to be passed in setup.")

        with torch.no_grad():
            self.ref_logits = cheap_model.get_logits(x, mask)

    def compute_loss(
        self,
        cf: torch.Tensor,
        x: torch.Tensor,  # noqa: ARG002
        mask: torch.Tensor,
        cheap_model: nn.Module | None = None,
    ) -> torch.Tensor:
        """Computes the KL sparsity loss comparing current logits to reference logits.

        Args:
            cf: Counterfactual tensor of shape (batch_size, seq_len, embed_dim).
            x: Original input tensor of shape (batch_size, seq_len, embed_dim), ignored.
            mask: Attention mask tensor of shape (batch_size, seq_len).
            cheap_model: Required CHEAP model to compute logits.

        Returns:
            The computed KL sparsity loss as a scalar tensor.
        """
        if cheap_model is None:
            raise ValueError("KLSparsityMechanism requires a cheap_model to be passed in compute_loss.")

        if self.ref_logits is None:
            raise ValueError("KLSparsityMechanism setup() must be called before compute_loss().")

        curr_logits = cheap_model.get_logits(cf, mask)
        return self.weight * self.loss_fn(curr_logits, self.ref_logits, mask)

    def apply_gradients(self, cf: torch.Tensor) -> None:
        """No-op for loss-based mechanism."""
        pass

    def apply_constraints(self, cf: torch.Tensor, x: torch.Tensor) -> None:
        """No-op for loss-based mechanism."""
        pass


class DistanceSparsity(BaseSparsityMechanism):
    """Enforces sparsity via direct distance metrics (e.g., L2) in embedding space."""

    def __init__(self, loss_fn: nn.Module, weight: float = 1.0) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.weight = weight

    def compute_loss(
        self,
        cf: torch.Tensor,
        x: torch.Tensor,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        """Computes the distance-based sparsity loss between counterfactual and original input.

        Args:
            cf: Counterfactual tensor of shape (batch_size, seq_len, embed_dim).
            x: Original input tensor of shape (batch_size, seq_len, embed_dim).
            **kwargs: Additional arguments (ignored, for compatibility).

        Returns:
            The computed distance loss as a scalar tensor.
        """
        return self.weight * self.loss_fn(cf, x)

    def apply_gradients(self, cf: torch.Tensor) -> None:
        """No-op for loss-based mechanism."""
        pass

    def apply_constraints(self, cf: torch.Tensor, x: torch.Tensor) -> None:
        """No-op for loss-based mechanism."""
        pass


class CompositeSparsity(BaseSparsityMechanism):
    """Combines multiple sparsity mechanisms."""

    def __init__(self, mechanisms: list[BaseSparsityMechanism]) -> None:
        super().__init__()
        self.mechanisms = nn.ModuleList(mechanisms)

    def setup(self, *args, **kwargs) -> None:
        """Sets up all constituent mechanisms."""
        for m in self.mechanisms:
            m.setup(*args, **kwargs)

    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """Computes the total sparsity loss from all constituent mechanisms."""
        cf = kwargs.get("cf")
        if cf is None:
            if len(args) > 0:
                cf = args[0]
            else:
                raise ValueError("Counterfactual tensor 'cf' not found in args or kwargs.")

        loss = torch.tensor(0.0, device=cf.device)
        for m in self.mechanisms:
            loss += m.compute_loss(*args, **kwargs)
        return loss

    def apply_gradients(self, *args, **kwargs) -> None:
        """Applies gradients for all constituent mechanisms."""
        for m in self.mechanisms:
            m.apply_gradients(*args, **kwargs)

    def apply_constraints(self, *args, **kwargs) -> None:
        """Applies constraints for all constituent mechanisms."""
        for m in self.mechanisms:
            m.apply_constraints(*args, **kwargs)
