import importlib
import re
from pathlib import Path

import torch
from lightning import LightningModule

from mccop.utils.constants import Paths
from mccop.utils.helpers import logger


def load_model_from_checkpoint(checkpoint_path: Path) -> LightningModule:
    """Loads a PyTorch Lightning model from a checkpoint without needing to know the
    specific model class beforehand.

    This function inspects the checkpoint to find the model's class path,
    dynamically imports the class, and then uses the class's `load_from_checkpoint`
    method to instantiate the model.

    Args:
        checkpoint_path: The path to the .ckpt file.

    Returns:
        The loaded PyTorch Lightning model.

    Raises:
        ValueError: If the model's class path cannot be found in the checkpoint.
        ImportError: If the model's class cannot be imported.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)

    hparams = checkpoint.get("hyper_parameters", {})
    class_path = None

    class_path_key = next((key for key in hparams if key.endswith("_target_")), None)
    if class_path_key:
        class_path = hparams[class_path_key]

    if not class_path:
        raise ValueError("Could not determine model class path from checkpoint.")

    module_path, class_name = class_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model class '{class_name}' from '{module_path}'.") from e

    logger.info(f"Found model class: {model_class.__name__}")

    model = model_class.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.checkpoint_path = checkpoint_path

    return model


def load_best_model(
    dataset_name: str,
    embed_dim: int,
    task_name: str,
    metric: str,
    mode: str,
    model_class: LightningModule | None = None,
) -> torch.nn.Module:
    """Loads the best model for a given dataset and task.

    It first looks for a run directory starting with 'best'. If found, it loads that model.
    Otherwise, it scans all run directories, finds the checkpoint with the best metric,
    renames the run directory to 'best_*', and then loads the model.

    Args:
        dataset_name (str): The name of the dataset.
        embed_dim (int): The embedding dimension used.
        task_name (str): The name of the task (e.g., 'classification', 'ddpm').
        model_class (Type[torch.nn.Module]): The model class to load (must have `load_from_checkpoint`).
        metric (str): The metric to evaluate for finding the best model (e.g., 'val_auroc').
        mode (str): 'max' to find the highest metric, 'min' for the lowest.

    Returns:
        torch.nn.Module: The loaded model with a `checkpoint_path` attribute.
    """
    checkpoints_path = Path(Paths.OUTPUTS_DIR) / f"{dataset_name}_{embed_dim}_{task_name}"

    if not checkpoints_path.exists():
        raise FileNotFoundError(f"No checkpoints found for {dataset_name}/{task_name} at {checkpoints_path}")

    try:
        best_run_dir = next(d for d in checkpoints_path.iterdir() if d.is_dir() and d.name.startswith("best"))
        logger.info(f"Found existing 'best' run: {best_run_dir.name}. Loading model from this directory.")
        checkpoint_files = list((best_run_dir / "checkpoints").glob("*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint file found in {best_run_dir}")
        best_model_path = checkpoint_files[0]

    except StopIteration:
        logger.info(f"No 'best' run found. Searching for the best run based on '{metric}'.")
        best_model_path = find_and_rename_best_run(checkpoints_path=checkpoints_path, metric=metric, mode=mode)

    logger.info(f"Loading model from {best_model_path}")
    if model_class is not None:
        model = model_class.load_from_checkpoint(best_model_path, weights_only=False)
        model.checkpoint_path = best_model_path
        return model
    return load_model_from_checkpoint(best_model_path)


def _find_best_checkpoint(
    checkpoints_path: Path, metric: str, mode: str
) -> tuple[Path | None, float | None, list[str]]:
    metric_pattern = re.compile(rf"{metric}=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    best_score = -float("inf") if mode == "max" else float("inf")
    best_model_path = None
    tied_runs = []

    for run_dir in checkpoints_path.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("best"):
            continue
        checkpoint_dir = run_dir / "checkpoints"
        for ckpt_file in checkpoint_dir.glob("*.ckpt"):
            match = metric_pattern.search(ckpt_file.name)
            if not match:
                continue
            try:
                score = float(match.group(1))
            except ValueError:
                logger.warning(f"Skipping checkpoint with unparsable metric in filename: {ckpt_file.name}")
                continue
            is_better = (mode == "max" and score > best_score) or (mode == "min" and score < best_score)
            if is_better:
                best_score = score
                best_model_path = ckpt_file
                tied_runs = [run_dir.name]
            elif score == best_score:
                tied_runs.append(run_dir.name)
    return best_model_path, best_score, tied_runs


def find_and_rename_best_run(checkpoints_path: Path, metric: str, mode: str) -> Path:
    """Finds the best run based on a metric, renames its directory, and returns the model path.

    Args:
        checkpoints_path (Path): Path to the directory containing run folders.
        metric (str): The metric to evaluate (e.g., 'val_auroc').
        mode (str): 'max' to find the highest metric, 'min' for the lowest.

    Returns:
        Path: The path to the best checkpoint file.

    Raises:
        ValueError: If mode is not 'max' or 'min'.
        FileNotFoundError: If no valid checkpoints are found.
    """
    if mode not in ["max", "min"]:
        raise ValueError("mode must be 'max' or 'min'")

    best_model_path, best_score, tied_runs = _find_best_checkpoint(checkpoints_path, metric, mode)

    if not best_model_path:
        raise FileNotFoundError(f"No valid checkpoint files found in {checkpoints_path} for metric '{metric}'")

    if len(tied_runs) > 1:
        logger.warning(
            f"Multiple runs found with the same best {metric} ({best_score}): {tied_runs}. "
            f"Selecting from run '{Path(best_model_path).parent.parent.name}'."
        )

    best_run_dir = best_model_path.parent.parent
    run_id = best_run_dir.name
    new_run_dir_name = f"best_{run_id}"
    new_run_dir_path = best_run_dir.parent / new_run_dir_name
    best_run_dir.rename(new_run_dir_path)

    updated_model_path = new_run_dir_path / "checkpoints" / best_model_path.name

    logger.info(f"Found best model with {metric}={best_score}. Renamed run dir to '{new_run_dir_name}'.")

    return updated_model_path
