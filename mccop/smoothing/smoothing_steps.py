"""Implementations of various model smoothing techniques."""

import gc
from collections.abc import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import grad
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mccop.data.datasets import BaseDMSDataset
from mccop.utils.constants import ColumnNames, Splits
from mccop.utils.helpers import logger
from mccop.models.autoencoders import CHEAPModel


class Distillation:
    """Applies knowledge distillation to a model.

    This technique uses temperature-scaled soft targets from a teacher.
    For binary classification, we use a temperature-scaled BCE-with-logits setup:
      - scale both student and teacher logits by T
      - multiply the distillation loss by T^2 to preserve gradient magnitudes
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> None:
        """Args:
        temperature: The temperature for softening the logits (T > 1 recommended).
        alpha: Weight for distillation term; (1 - alpha) is weight for original supervised loss.
        """
        self.temperature = temperature
        self.alpha = alpha

    @torch.no_grad()
    def _teacher_logits(self, teacher_model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return teacher_model(inputs).squeeze(-1)

    def __call__(
        self,
        student_logits: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        original_loss_fn: Callable,
        teacher_model: nn.Module,
    ) -> torch.Tensor:
        """Calculates the combined distillation + original loss.

        Args:
            student_logits: Logits from the student model (shape: [N]).
            inputs: Inputs to feed the teacher (same batch).
            labels: Ground-truth labels for the original loss.
            original_loss_fn: The original supervised loss function (e.g., BCEWithLogitsLoss).
            teacher_model: The teacher model (eval, no grad, on right device).

        Returns:
            A scalar tensor: alpha * distill_loss + (1 - alpha) * original_loss.
        """
        T = self.temperature

        with torch.no_grad():
            teacher_logits = self._teacher_logits(teacher_model, inputs)

        student_logits_T = student_logits / T
        teacher_probs_T = torch.sigmoid(teacher_logits / T)

        distill_loss = F.binary_cross_entropy_with_logits(student_logits_T, teacher_probs_T) * (T * T)

        original_loss = original_loss_fn(student_logits, labels)

        return self.alpha * distill_loss + (1 - self.alpha) * original_loss


def apply_spectral_norm(model: nn.Module) -> nn.Module:
    """Recursively applies spectral normalization to all Conv and Linear layers.

    Args:
        model: The model to which spectral normalization will be applied.

    Returns:
        The model with spectral normalization applied.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d | nn.Linear):
            setattr(model, name, spectral_norm(module))
        else:
            apply_spectral_norm(module)
    return model


def replace_relu_with_softplus(model: nn.Module) -> nn.Module:
    """Recursively replaces all ReLU activation functions with Softplus.

    Args:
        model: The model in which ReLU activations will be replaced.

    Returns:
        The model with ReLU activations replaced by Softplus.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.Softplus())
        else:
            replace_relu_with_softplus(module)
    return model


def jacobian_regularization_loss(
    inputs: torch.Tensor, outputs: torch.Tensor, lambda_reg: float = 1000.0
) -> torch.Tensor:
    """Computes the Jacobian regularization loss.

    This loss penalizes the norm of the Jacobian of the output with respect
    to the input, encouraging the model to be less sensitive to input perturbations.

    Args:
        inputs: The model inputs.
        outputs: The model outputs.
        lambda_reg: The regularization strength.

    Returns:
        The Jacobian regularization loss.
    """
    gradients = grad(
        outputs=outputs.sum(),
        inputs=inputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    jacobian_norm_sq = gradients.pow(2).mean()

    return lambda_reg * jacobian_norm_sq


class AdversarialAugmentation:
    """Generates adversarial examples for data augmentation using FGSM."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        decoder: CHEAPModel | None = None,
        epsilon: float = 0.01,
    ) -> None:
        """Initializes the AdversarialAugmentation class.

        Args:
            model: The model to attack.
            loss_fn: The loss function to use for generating perturbations.
            decoder: Optional component to decode embeddings back to sequences.
            epsilon: The perturbation magnitude.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.decoder = decoder

    def _generate_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]] | None:
        """Generates adversarial examples for a single batch.

        Returns:
            A tuple containing perturbed inputs, labels, masks, and original sequences
            of valid augmentations, or None if no valid augmentations are generated.
        """
        inputs = batch["embedding"]
        labels = batch["label"]
        masks = batch["mask"]
        original_seqs = batch["seq"]

        device = next(self.model.parameters()).device
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels.float())
        loss.backward()

        # FGSM attack
        with torch.no_grad():
            perturbed_inputs = inputs + self.epsilon * inputs.grad.sign()
            perturbed_inputs = torch.clamp(perturbed_inputs, -1, 1)

        # Check if the augmented sample is still valid (sequence remains the same)
        if self.decoder is None:
            raise ValueError("Decoder not provided for adversarial augmentation.")
        else:
            augmented_seqs = self.decoder.embedding_to_seq(perturbed_inputs, masks)

        valid_indices = [
            i for i, (orig, aug) in enumerate(zip(original_seqs, augmented_seqs, strict=True)) if orig == aug
        ]

        if not valid_indices:
            return None

        valid_original_seqs = [original_seqs[i] for i in valid_indices]

        return (
            perturbed_inputs[valid_indices].cpu(),
            labels[valid_indices].cpu(),
            masks[valid_indices].cpu(),
            valid_original_seqs,
        )

    def augment_dataset(self, dataset: BaseDMSDataset) -> BaseDMSDataset:
        """Applies adversarial augmentation to the training and validation sets of a dataset."""
        augmented_dir = dataset.base_path / "augmented"
        augmented_dir.mkdir(exist_ok=True)
        augmented_parquet_path = augmented_dir / f"{dataset.name}.parquet"

        if augmented_parquet_path.exists():
            logger.info(f"Found cached augmented dataset at {augmented_dir}. Loading it.")
            return dataset.get_augmented_self(augmented_dir)

        original_df = dataset.data.copy()
        original_df = original_df[[ColumnNames.SEQ, dataset.label_col, ColumnNames.SPLIT]]
        original_df[ColumnNames.IS_AUGMENTED] = False

        logger.info("Generating augmented dataset...")
        augmented_embeddings = []
        augmented_masks = []
        augmented_rows = []

        for split in [Splits.TRAIN, Splits.VAL]:
            split_indices = dataset.data[dataset.data[ColumnNames.SPLIT] == split.value].index.tolist()
            split_subset = Subset(dataset, split_indices)
            loader = DataLoader(split_subset, batch_size=dataset.batch_size, shuffle=False)

            for batch in tqdm(loader, desc=f"Augmenting {split.label} set"):
                aug_result = self._generate_batch(batch)
                if aug_result:
                    aug_embeddings, aug_labels, aug_masks, aug_seqs = aug_result

                    augmented_rows.extend(
                        [
                            {
                                ColumnNames.SEQ: aug_seqs[i],
                                dataset.label_col: aug_labels[i].item(),
                                ColumnNames.IS_AUGMENTED: True,
                                ColumnNames.SPLIT: split.value,
                            }
                            for i in range(len(aug_seqs))
                        ]
                    )

                    augmented_embeddings.append(aug_embeddings)
                    augmented_masks.append(aug_masks)

        if not augmented_rows:
            raise RuntimeError("No valid adversarial augmentations were generated.")

        augmented_df = pd.DataFrame(augmented_rows)

        final_df = pd.concat([dataset.data, augmented_df], ignore_index=True)
        del augmented_rows, augmented_df

        final_embeddings = torch.cat([dataset.embeddings, *augmented_embeddings], dim=0)
        del augmented_embeddings

        final_masks = torch.cat([dataset.masks, *augmented_masks], dim=0)
        del augmented_masks

        gc.collect()

        final_df.to_parquet(augmented_parquet_path)
        torch.save(final_embeddings, augmented_dir / f"{dataset.name}_embeddings.pt")
        torch.save(final_masks, augmented_dir / f"{dataset.name}_masks.pt")

        return dataset.get_augmented_self(augmented_dir)
