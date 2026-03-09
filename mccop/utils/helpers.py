import logging
import os
import random
import sys
from pathlib import Path
from typing import Final
import re

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig


def set_logger() -> logging.Logger:
    """Set up the logger.

    Returns:
        Logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s\n%(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = set_logger()


class ConfigKeys:
    """Keys present in configs."""

    CONFIG: Final[str] = "cfg"
    SEED: Final[str] = "seed"
    WANDB: Final[str] = "wandb"
    JOB: Final[str] = "job"
    STORE: Final[str] = "store"


def seed_everything(seed: int) -> None:
    """Seeds all random number generators.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    """Returns the available device for torch.

    Returns:
        The GPU or the MPS device when available and the CPU device as a fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_output_dir() -> Path:
    """Return the hydra output directory.

    Returns:
        Path to the hydra output directory.
    """
    return Path(HydraConfig.get().runtime.output_dir)


def extract_metric_from_checkpoint(checkpoint_path: Path, metric_name: str = "val_auroc") -> float | None:
    """Extracts the metric value from a checkpoint filename.

    Args:
        checkpoint_path: Path to the checkpoint file.
        metric_name: Name of the metric to extract (default: "val_auroc").

    Returns:
        The extracted metric value, or None if not found.
    """
    pattern = rf"{metric_name}[=:]?([\d.]+)"
    match = re.search(pattern, checkpoint_path.stem)
    return float(match.group(1)) if match else None
