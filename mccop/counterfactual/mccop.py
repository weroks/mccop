import time
from dataclasses import dataclass, field

import torch
from dima import get_config_path
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from mccop.counterfactual.base import BaseExplainer
from mccop.counterfactual.sparsity import BaseSparsityMechanism, GradientMasking
from mccop.models.generators import DiMAWrapper
from mccop.smoothing.smoother import Smoother
from mccop.utils.helpers import logger


@dataclass
class _OptimizationState:
    """Tracks mutable state across optimization steps."""

    cf: nn.Parameter
    optimizer: torch.optim.Adam
    best_cf: torch.Tensor
    best_confidences: torch.Tensor
    still_optimizing: torch.Tensor
    ref_tokens: torch.Tensor
    diffusion_time: float = 0.0
    last_step: int = field(default=0, init=False)


class MCCOP(BaseExplainer):
    """Generates counterfactual embeddings using a predictor-guided optimization approach.

    This editor refines an input sample to produce a counterfactual that changes the
    predictor's output to a target class. It supports projecting the counterfactual
    onto the data manifold using a pre-trained generative model.
    """

    def __init__(
        self,
        dataset: Dataset,
        loss_fn: nn.Module | None = None,
        sparsity_mechanism: BaseSparsityMechanism | None = None,
        smoother: Smoother | None = None,
        project_on_manifold: bool = True,
        sampling_time_fraction: float = 0.1,
        learning_rate: float = 1e-2,
        gradient_steps: int = 100,
        projection_alpha: float = 0.3,
        **kwargs,
    ) -> None:
        """Initializes the editor.

        Args:
            dataset: The dataset to use for counterfactual generation.
            loss_fn: The loss function to guide optimization. Defaults to BCEWithLogitsLoss.
            sparsity_mechanism: Mechanism to encourage sparse edits. Defaults to GradientMasking with k=2.
            smoother: Optional smoother to apply to the predictor.
            project_on_manifold: Whether to project counterfactuals onto the data manifold.
            sampling_time_fraction: Fraction of the diffusion process to use for projection.
            learning_rate: Learning rate for the optimizer.
            gradient_steps: Number of gradient steps for optimization.
            projection_alpha: Blending factor for manifold projection.
            **kwargs: Additional arguments for the base explainer.
        """
        super().__init__(dataset, **kwargs)
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.sparsity_mechanism = sparsity_mechanism or GradientMasking(k=2)
        self.project_on_manifold = project_on_manifold
        self.learning_rate = learning_rate
        self.gradient_steps = gradient_steps
        self.sampling_time_fraction = sampling_time_fraction
        self.projection_alpha = projection_alpha
        self.optimization_predictor = self.predictor

        if smoother:
            smoother.decoder = self.cheap_model
            self.optimization_predictor = (
                smoother.smooth(self.predictor, self.dataset, self.wandb_logger).to(self.device).eval()
            )

        self.generator = None
        if self.project_on_manifold:
            dima_config = get_config_path()
            self.generator = DiMAWrapper(config_path=dima_config, device=self.device)
            self.generator.load_pretrained()

    def generate_batch(self, batch: dict, batch_idx: int) -> dict:
        """Generates counterfactuals for a batch of input samples.

        Args:
            batch: The input batch containing embeddings, masks, and labels.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the following keys:
                - "original_input": The original input tensor.
                - "original_predicted_classes": The original predicted classes.
                - "target_classes": The target classes used for optimization.
                - "best_confidences": The best confidence scores achieved.
                - "counterfactuals": The generated counterfactual tensors.
        """
        x, mask, target_classes = self._unpack_batch(batch)
        original_classes = self._predict_classes(x)
        state = self._init_optimization_state(x, mask, target_classes)

        for i in tqdm(range(self.gradient_steps), desc="Optimizing Batch", leave=False):
            state.last_step = i
            self._optimization_step(state, x, mask, target_classes, batch_idx, i)
            if not state.still_optimizing.any():
                logger.info(f"All samples reached threshold at step {i + 1}")
                break

        return self._build_result(x, original_classes, target_classes, state)

    def _unpack_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts and moves batch tensors to the target device."""
        x = batch["embedding"].to(self.device)
        mask = batch["mask"].to(self.device)
        target_classes = (1 - batch["label"]).float().to(self.device)
        return x, mask, target_classes

    @torch.no_grad()
    def _predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Returns binary predicted classes for the original inputs."""
        return (self.predictor(x) > 0).int()

    def _init_optimization_state(
        self, x: torch.Tensor, mask: torch.Tensor, target_classes: torch.Tensor
    ) -> _OptimizationState:
        """Creates the initial optimization state and configures the sparsity mechanism."""
        with torch.no_grad():
            ref_tokens = self.cheap_model.get_logits(x, mask).argmax(dim=-1)

        self.sparsity_mechanism.setup(
            x=x,
            mask=mask,
            predictor=self.optimization_predictor,
            loss_fn=self.loss_fn,
            target_classes=target_classes,
            cheap_model=self.cheap_model,
        )

        cf = nn.Parameter(x.clone(), requires_grad=True)
        return _OptimizationState(
            cf=cf,
            optimizer=torch.optim.Adam([cf], lr=self.learning_rate),
            best_cf=x.clone(),
            best_confidences=torch.zeros(x.shape[0], device=self.device),
            still_optimizing=torch.ones(x.shape[0], dtype=torch.bool, device=self.device),
            ref_tokens=ref_tokens,
        )

    def _optimization_step(
        self,
        state: _OptimizationState,
        x: torch.Tensor,
        mask: torch.Tensor,
        target_classes: torch.Tensor,
        batch_idx: int,
        step: int,
    ) -> None:
        """Executes a single gradient-based optimization step."""
        state.optimizer.zero_grad()

        with torch.no_grad():
            state.cf.data.clamp_(-10.0, 10.0)

        active_loss, pred_loss, sparsity_loss = self._compute_active_loss(
            state.cf, x, mask, target_classes, state.still_optimizing
        )
        self._record_step(batch_idx, step, active_loss, pred_loss, sparsity_loss, state.still_optimizing)

        active_loss.backward()
        self.sparsity_mechanism.apply_gradients(state.cf)
        state.optimizer.step()
        self.sparsity_mechanism.apply_constraints(state.cf, x)

        if self.project_on_manifold:
            state.diffusion_time += self._project_on_manifold(state.cf, mask)

        self._update_best(state, mask, target_classes)

    def _compute_active_loss(
        self,
        cf: nn.Parameter,
        x: torch.Tensor,
        mask: torch.Tensor,
        target_classes: torch.Tensor,
        still_optimizing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the combined loss, weighted by which samples are still active.

        Returns:
            A tuple of (active_loss, prediction_loss, sparsity_loss).
        """
        prediction_loss = self.loss_fn(self.optimization_predictor(cf), target_classes)
        sparsity_loss = self.sparsity_mechanism.compute_loss(cf=cf, x=x, mask=mask, cheap_model=self.cheap_model)
        total_loss = prediction_loss + sparsity_loss
        active_loss = (total_loss * still_optimizing).sum() / (still_optimizing.sum() + 1e-8)
        return active_loss, prediction_loss, sparsity_loss

    def _record_step(
        self,
        batch_idx: int,
        step: int,
        active_loss: torch.Tensor,
        prediction_loss: torch.Tensor,
        sparsity_loss: torch.Tensor | float,
        still_optimizing: torch.Tensor,
    ) -> None:
        """Logs progress and appends metrics to the training history."""
        pred_val = prediction_loss.mean().item()
        sparsity_val = sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else 0.0

        if step % 10 == 0:
            logger.info(
                f"Batch {batch_idx} step {step}: loss={active_loss.item():.4f}, "
                f"pred={pred_val:.4f}, active={still_optimizing.sum().item()}"
            )

        self.history.append(
            {
                "batch_id": batch_idx,
                "step": step,
                "loss": active_loss.item(),
                "pred_loss": pred_val,
                "sparsity_loss": sparsity_val,
                "active_samples": still_optimizing.sum().item(),
            }
        )

    @torch.no_grad()
    def _update_best(self, state: _OptimizationState, mask: torch.Tensor, target_classes: torch.Tensor) -> None:
        """Updates best counterfactuals when confidence improves and tokens have changed."""
        confidences = self.target_confidence(self.predictor(state.cf), target_classes)
        current_tokens = self.cheap_model.get_logits(state.cf, mask).argmax(dim=-1)
        has_changed = (current_tokens != state.ref_tokens).any(dim=-1)
        improved = (confidences > state.best_confidences) & has_changed

        state.best_cf[improved] = state.cf[improved].detach()
        state.best_confidences[improved] = confidences[improved]
        state.still_optimizing = state.best_confidences < self.confidence_threshold

    @staticmethod
    def _build_result(
        x: torch.Tensor,
        original_classes: torch.Tensor,
        target_classes: torch.Tensor,
        state: _OptimizationState,
    ) -> dict:
        """Packages optimization outputs into the expected result dict."""
        return {
            "original_input": x.cpu(),
            "original_predicted_classes": original_classes.cpu().flatten().tolist(),
            "target_classes": target_classes.cpu().flatten().tolist(),
            "best_confidences": state.best_confidences.cpu().tolist(),
            "counterfactuals": state.best_cf.detach().cpu(),
            "n_optimization_steps": state.last_step + 1,
            "diffusion_time": state.diffusion_time,
            "encoding_time": 0,
        }

    def _project_on_manifold(self, cf: nn.Parameter, mask: torch.Tensor) -> float:
        """Projects the counterfactual onto the data manifold in-place and returns elapsed seconds.

        Args:
            cf: The counterfactual parameter to project.
            mask: The attention mask for the sequence.

        Returns:
            Wall-clock seconds spent on the projection.
        """
        if self.generator is None:
            return 0.0

        self._sync_cuda()
        t0 = time.perf_counter()

        t_frac = self.sampling_time_fraction
        noise = torch.randn_like(cf)
        with torch.no_grad():
            cf_encoded = self.generator.encode(cf, t_frac=t_frac, noise=noise)
            cf_projected = self.generator.decode(cf_encoded, t_frac=t_frac, mask=mask)

        cf.data = (1 - self.projection_alpha) * cf.data + self.projection_alpha * cf_projected

        self._sync_cuda()
        return time.perf_counter() - t0
