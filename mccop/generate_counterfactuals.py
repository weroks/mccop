from hydra_zen.typing import Partial
from lightning.pytorch.loggers import WandbLogger

from mccop.counterfactual.base import BaseExplainer
from mccop.data.datasets import BaseDMSDataset
from mccop.utils.config import run
from mccop.utils.constants import Paths
from mccop.utils.runs import Run
from mccop.utils.constants import Splits

def main(
    cfg: Run,
    editor: Partial[BaseExplainer],
    dataset: BaseDMSDataset,
) -> None:
    """Main function to generate counterfactuals.

    Args:
        cfg: Run config holding e.g. global parameters.
        editor: BaseExplainer class with pre-filled parameters.
        dataset: Dataset object containing the data.
    """
    wandb_logger = (
        WandbLogger(save_dir=Paths.OUTPUTS_DIR, log_model=True)
        if cfg.wandb is not None
        else None
    )
    editor_instance = editor(dataset=dataset, wandb_logger=wandb_logger)
    editor_instance.run(split=Splits.TEST)


if __name__ == "__main__":
    from mccop.configs.tasks import generate_counterfactuals  # noqa: F401

    run(main)
