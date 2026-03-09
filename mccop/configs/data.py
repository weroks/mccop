from pathlib import Path

from hydra_zen import builds, store

from mccop.configs.preprocessing import (
    TapeFluorescencePipelineConfig,
    TapeStabilityPipelineConfig,
    Ube4bPipelineConfig,
)
from mccop.data.datasets import TapeFluorescenceDataset, TapeStabilityDataset, Ube4bDataset

BASE_PATH = Path("datasets")

TapeFluorescenceDatasetConfig = builds(
    TapeFluorescenceDataset,
    base_path=BASE_PATH / "tape_fluorescence",
    preprocessing_pipeline=TapeFluorescencePipelineConfig,
    embed_dim=1024,
    batch_size=32,
    num_workers=4,
    populate_full_signature=True,
)

TapeStabilityDatasetConfig = builds(
    TapeStabilityDataset,
    base_path=BASE_PATH / "tape_stability",
    preprocessing_pipeline=TapeStabilityPipelineConfig,
    embed_dim=1024,
    batch_size=32,
    num_workers=4,
    populate_full_signature=True,
)

Ube4bDatasetConfig = builds(
    Ube4bDataset,
    base_path=BASE_PATH / "ube4b",
    preprocessing_pipeline=Ube4bPipelineConfig,
    embed_dim=1024,
    batch_size=32,
    num_workers=4,
    populate_full_signature=True,
)

store(TapeFluorescenceDatasetConfig, name="tape_fluorescence", group="dataset")
store(TapeStabilityDatasetConfig, name="tape_stability", group="dataset")
store(Ube4bDatasetConfig, name="ube4b", group="dataset")
