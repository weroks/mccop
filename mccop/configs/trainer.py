from hydra_zen import builds, store
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

DefaultTrainerConfig = builds(
    Trainer,
    max_epochs=50,
    accelerator="auto",
    devices="auto",
    deterministic=True,
    callbacks=[
        builds(EarlyStopping, monitor="val_loss", mode="min", min_delta=1e-5, patience=5),
        builds(
            ModelCheckpoint,
            monitor="val_auroc",
            mode="max",
            save_top_k=1,
            filename="best-model-{val_auroc:.2f}",
        ),
    ],
    zen_partial=True,
)

store(
    DefaultTrainerConfig,
    name="default",
    group="trainer",
)
