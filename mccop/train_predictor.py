import torch
from hydra_zen.typing import Partial
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from mccop.data.datasets import BaseDMSDataset
from mccop.models.predictors import BasePredictor
from mccop.utils.config import run
from mccop.utils.constants import Paths
from mccop.utils.runs import Run


def main(
    cfg: Run,
    model: Partial[BasePredictor],
    trainer: Partial[Trainer],
    dataset: BaseDMSDataset,
) -> None:
    """Main function to train a unimodal model.

    Args:
        cfg: Training config holding e.g. global parameters.
        model: Model class to be trained.
        trainer: Trainer class for training the model.
        dataset: Dataset object containing the data.
    """
    wandb_logger = WandbLogger(save_dir=Paths.OUTPUTS_DIR, log_model=True) if cfg.wandb is not None else False

    # Trade precision for performace when using tensor cores
    torch.set_float32_matmul_precision("medium")

    train_dl, val_dl, test_dl = dataset.get_loaders()

    model = model(seq_len=dataset.seq_len, embed_dim=dataset.embed_dim)

    trainer = trainer(logger=wandb_logger, default_root_dir=Paths.OUTPUTS_DIR)
    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    from mccop.configs.tasks import train_predictor  # noqa: F401

    run(main)
