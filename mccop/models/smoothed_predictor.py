"""A wrapper class to apply smoothing techniques to a BasePredictor model."""

from copy import deepcopy

import torch
from torch import nn
from torchmetrics import MetricCollection

from mccop.models.predictors import BasePredictor
from mccop.smoothing.smoothing_steps import (
    Distillation,
    apply_spectral_norm,
    jacobian_regularization_loss,
    replace_relu_with_softplus,
)
from mccop.smoothing.smoothness_metrics import avg_input_grad_norm, local_lipschitz_adv


class SmoothedPredictor(BasePredictor):
    """A LightningModule wrapper for training a smoothed model."""

    def __init__(
        self,
        predictor: BasePredictor,
        metrics: MetricCollection | None = None,
        use_spectral_norm: bool = False,
        use_jacobian_reg: bool = False,
        use_distillation: bool = False,
        use_smooth_activations: bool = False,
        distillation_temp: float = 2.0,
        distillation_alpha: float = 0.5,
        learning_rate: float = 1e-5,
        reinitialize: bool = False,
        loading_from_checkpoint: bool = False,
        **kwargs,  # For dataset related components from the config, will be ignored
    ) -> None:
        """Initialize the SmoothedPredictor."""
        loss_clone = deepcopy(predictor.loss_fn) if isinstance(predictor.loss_fn, nn.Module) else predictor.loss_fn
        super().__init__(
            seq_len=predictor.hparams.seq_len,
            embed_dim=predictor.hparams.embed_dim,
            loss_fn=loss_clone,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.save_hyperparameters(ignore=["predictor", "metrics", "loss_fn"])

        self.train_metrics = metrics.clone() if metrics else None
        self.val_metrics = metrics.clone() if metrics else None
        self.test_metrics = metrics.clone() if metrics else None

        self.model = deepcopy(predictor.model)
        self.teacher_model = None

        if not loading_from_checkpoint:
            for p in self.model.parameters():
                p.requires_grad = True

            self.model.train()

            if reinitialize:
                _reinitialize_module_parameters(self.model)

            if self.hparams.use_distillation:
                self.teacher_model = deepcopy(predictor.model)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

                self.distillation = Distillation(
                    temperature=self.hparams.distillation_temp,
                    alpha=self.hparams.distillation_alpha,
                )

            if self.hparams.use_spectral_norm:
                apply_spectral_norm(self.model)

            if self.hparams.use_smooth_activations:
                replace_relu_with_softplus(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch_size,).
        """
        return self.model(x).squeeze(-1)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for the smoothed model.

        Args:
            batch: A batch of data containing 'embedding' and 'label'.
            batch_idx: Index of the batch.

        Returns:
            The computed loss for the batch.
        """
        x, y = batch["embedding"], batch["label"]

        if self.hparams.use_jacobian_reg:
            x.requires_grad_(True)

        y_hat = self(x)

        if self.hparams.use_distillation:
            loss = self.distillation(
                student_logits=y_hat,
                inputs=x,
                labels=y.float(),
                original_loss_fn=self.loss_fn,
                teacher_model=self.teacher_model,
            )
        else:
            loss = self.loss_fn(y_hat, y.float())

        if self.hparams.use_jacobian_reg:
            jac = jacobian_regularization_loss(x, y_hat)
            self.log("jac_loss", jac, on_step=True, on_epoch=True, prog_bar=False)
            self.log("bce_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
            loss += jac

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.train_metrics:
            self.train_metrics(torch.sigmoid(y_hat), y.int())

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
        """
        loss, y_hat, y = self._step(batch, batch_idx)
        x = batch["embedding"]

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        with torch.enable_grad():
            x_new = x.detach().requires_grad_(True)

            v_grad_norm = avg_input_grad_norm(self.model, x_new)
            try:
                v_local_lip = local_lipschitz_adv(self.model, x_new, eps=1e-3)
            except Exception:
                v_local_lip = torch.tensor(float('nan'))

        self.log("val_grad_norm", v_grad_norm, on_step=False, on_epoch=True)
        self.log("val_local_lip", v_local_lip, on_step=False, on_epoch=True)

        if self.val_metrics:
            self.val_metrics(torch.sigmoid(y_hat), y.int())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the model."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def _reinitialize_module_parameters(module: nn.Module) -> None:
    """Reinitialize a module by calling reset_parameters() where available."""
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
