"""A wrapper class to apply smoothing techniques to a BasePredictor model."""

import hashlib
import json
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MetricCollection

from mccop.models.predictors import BasePredictor
from mccop.models.smoothed_predictor import SmoothedPredictor
from mccop.smoothing.smoothing_steps import (
    AdversarialAugmentation,
)
from mccop.utils.helpers import logger, extract_metric_from_checkpoint


class Smoother:
    """This class orchestrates the process of applying various smoothing techniques
    to a base predictor model. It handles configuration management, caching of
    smoothed models to avoid re-training, and the training process itself.
    """

    def __init__(
        self,
        # Predictor options
        use_spectral_norm: bool = False,
        use_jacobian_reg: bool = False,
        use_distillation: bool = False,
        use_smooth_activations: bool = False,
        distillation_temp: float = 2.0,
        distillation_alpha: float = 0.5,
        learning_rate: float = 1e-5,
        reinitialize: bool = False,
        # Dataset augmentation options
        use_adversarial_aug: bool = False,
        adv_epsilon: float = 0.01,
        # Training options
        max_epochs: int = 20,
        metrics: MetricCollection | None = None,
        force_recompute: bool = False,
    ) -> None:
        """Initializes the Smoother with a specific smoothing configuration.

        Args:
            use_spectral_norm: If True, applies spectral normalization to the model.
            use_jacobian_reg: If True, adds Jacobian regularization to the loss.
            use_distillation: If True, uses knowledge distillation from the original model.
            use_smooth_activations: If True, replaces ReLU with Softplus activations.
            distillation_temp: Temperature for distillation loss.
            distillation_alpha: Weighting factor for the distillation loss component.
            learning_rate: Learning rate for the optimizer.
            reinitialize: If True, reinitializes model parameters before training.
            use_adversarial_aug: If True, applies adversarial augmentation to the dataset.
            adv_epsilon: Epsilon value for adversarial augmentation.
            max_epochs: Maximum number of epochs for training the smoothed model.
            metrics: A collection of metrics to monitor during training.
            force_recompute: If True, forces retraining even if a cached model exists.
        """
        self.config = {
            "use_spectral_norm": use_spectral_norm,
            "use_adversarial_aug": use_adversarial_aug,
            "use_jacobian_reg": use_jacobian_reg,
            "use_distillation": use_distillation,
            "use_smooth_activations": use_smooth_activations,
            "distillation_temp": distillation_temp,
            "distillation_alpha": distillation_alpha,
            "adv_epsilon": adv_epsilon,
            "learning_rate": learning_rate,
            "reinitialize": reinitialize,
        }
        self.decoder = None
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.force_recompute = force_recompute

    def smooth(
        self, predictor: BasePredictor, dataset: torch.utils.data.Dataset, wandb_logger: WandbLogger | None = None
    ) -> BasePredictor:
        """Orchestrates the smoothing process."""
        if not hasattr(predictor, "checkpoint_path"):
            raise ValueError("The provided predictor does not have a 'checkpoint_path' attribute.")

        config_hash = self._get_config_hash()
        smoothed_dir = predictor.checkpoint_path.parent.parent / "smoothed"
        checkpoint_name = f"{config_hash}"

        cached_model = self._handle_caching(smoothed_dir, checkpoint_name, predictor, wandb_logger)
        if cached_model:
            return cached_model

        if self.config["use_adversarial_aug"]:
            dataset = self._apply_adversarial_augmentation(predictor, dataset)

        best_model_path = self._run_training(predictor, dataset, smoothed_dir, checkpoint_name, wandb_logger)

        self._verify_performance(Path(best_model_path), wandb_logger)
        return SmoothedPredictor.load_from_checkpoint(
            best_model_path, predictor=predictor, loading_from_checkpoint=True, strict=False
        )

    def _handle_caching(
        self,
        smoothed_dir: Path,
        checkpoint_name: str,
        predictor: BasePredictor,
        wandb_logger: WandbLogger | None = None,
    ) -> SmoothedPredictor | None:
        """Checks for existing models and handles the force_recompute logic."""
        existing_checkpoints = list(smoothed_dir.glob(f"{checkpoint_name}-*.ckpt"))
        if not existing_checkpoints:
            logger.info(f"No cached model found for config {checkpoint_name}. Starting new smoothing process.")
            smoothed_dir.mkdir(exist_ok=True)
            return None

        if self.force_recompute:
            logger.info("Force recompute enabled. Deleting and retraining.")
            for checkpoint in existing_checkpoints:
                checkpoint.unlink()
            return None

        checkpoint_path = existing_checkpoints[0]
        logger.info(f"Loading existing smoothed model from {checkpoint_path}")
        self._verify_performance(checkpoint_path, wandb_logger)

        return SmoothedPredictor.load_from_checkpoint(
            checkpoint_path, predictor=predictor, loading_from_checkpoint=True, strict=False
        )

    def _get_config_hash(self) -> str:
        """Generates a unique hash for the smoothing configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:6]

    def _apply_adversarial_augmentation(
        self, predictor: BasePredictor, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.Dataset:
        """Applies adversarial augmentation to the provided dataset."""
        logger.info("Applying adversarial augmentation to the dataset.")
        adv_augmenter = AdversarialAugmentation(
            model=predictor,
            loss_fn=predictor.loss_fn,
            epsilon=self.config["adv_epsilon"],
            decoder=self.decoder,
        )
        return adv_augmenter.augment_dataset(dataset)

    def _run_training(
        self,
        predictor: BasePredictor,
        dataset: torch.utils.data.Dataset,
        smoothed_dir: Path,
        checkpoint_name: str,
        wandb_logger: WandbLogger | None = None,
    ) -> str:
        """Sets up the Lightning Trainer and runs the fit process."""
        smoothed_model = SmoothedPredictor(predictor=predictor, metrics=self.metrics, **self.config)

        checkpoint_callback = ModelCheckpoint(
            dirpath=smoothed_dir,
            filename=f"{checkpoint_name}-{{val_auroc:.4f}}",
            save_top_k=1,
            monitor="val_auroc",
            mode="max",
        )
        early_stopping_callback = EarlyStopping(monitor="val_auroc", mode="max", patience=4, min_delta=1e-5)

        trainer = Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator="auto",
            devices="auto",
            deterministic=True,
            logger=wandb_logger,
        )

        train_loader, val_loader, test_loader = dataset.get_loaders()
        trainer.fit(smoothed_model, train_loader, val_loader)
        trainer.test(smoothed_model, test_loader)

        logger.info(f"Smoothing complete. Model saved to {checkpoint_callback.best_model_path}")
        return checkpoint_callback.best_model_path

    def _verify_performance(self, checkpoint_path: Path, wandb_logger: WandbLogger | None = None) -> None:
        """Extracts metrics from checkpoint and validates threshold."""
        metric_value = extract_metric_from_checkpoint(checkpoint_path)
        if metric_value is not None:
            if wandb_logger:
                wandb_logger.log_metrics({"smoothed_val_auroc": metric_value})
                logger.info(f"Logged smoothed model val_auroc: {metric_value:.4f}")

            if metric_value < 0.9:
                logger.warning(
                    f"The smoothed model has a low val_auroc of {metric_value:.4f}. "
                    "Please check the model or consider retraining."
                )
