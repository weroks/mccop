from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from mccop.preprocessing.pipeline import PreprocessingPipeline
from mccop.utils.constants import ColumnNames, Splits
from mccop.utils.helpers import logger


class BaseDMSDataset(Dataset):
    """Base class for deep mutational scanning datasets."""

    def __init__(
        self,
        base_path: str,
        name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        task: str = "classification",
        embed_dim: int = 1024,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.base_path = Path(base_path)
        self.name = name
        self.embed_dim = embed_dim
        self.processed_path = self.base_path / f"processed_{self.embed_dim}"
        self.task = task
        self.label_col = ColumnNames.BINARY_LABEL if task == "classification" else ColumnNames.LABEL
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_pipeline = preprocessing_pipeline

        if (
            not (self.processed_path / f"{self.name}.parquet").exists()
            or not (self.processed_path / f"{self.name}_embeddings.pt").exists()
            or not (self.processed_path / f"{self.name}_masks.pt").exists()
        ):
            logger.info(f"Processed file for '{self.name}' not found. Running preprocessing pipeline...")
            preprocessing_pipeline.run(self)
            logger.info("Preprocessing complete.")

        self._load_data(self.processed_path)
        if self.embeddings.shape[2] != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embed_dim}, "
                f"but got {self.embeddings.shape[2]}."
            )
        self.seq_len = self.embeddings.shape[1]
        logger.info(
            f"Loaded dataset '{self.name}' with {len(self)} samples, "
            f"max sequence length {self.seq_len}, embedding dimension {self.embed_dim}."
        )

    def _load_data(self, data_path: Path) -> None:
        """Load the dataset from the specified path and file prefix."""
        self.data = pd.read_parquet(data_path / f"{self.name}.parquet")
        self.embeddings = torch.load(data_path / f"{self.name}_embeddings.pt")
        self.masks = torch.load(data_path / f"{self.name}_masks.pt")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample from the dataset."""
        row = self.data.iloc[idx]
        embedding = self.embeddings[idx]
        mask = self.masks[idx]

        return {
            "seq": row[ColumnNames.SEQ],
            "embedding": embedding,
            "mask": mask,
            "label": row[self.label_col],
        }

    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return train, val, and test data loaders."""
        return self.get_loader(Splits.TRAIN), self.get_loader(Splits.VAL), self.get_loader(Splits.TEST)

    def get_loader(self, split: Splits, target_label: int | None = None) -> DataLoader:
        """Return a DataLoader for the specified split."""
        condition = self.data[ColumnNames.SPLIT] == split.value

        if target_label is not None:
            condition &= self.data[self.label_col] == target_label

        split_indices = self.data[condition].index

        split_subset = Subset(self, split_indices)

        return DataLoader(
            split_subset,
            batch_size=self.batch_size,
            shuffle=(split == Splits.TRAIN),
            num_workers=self.num_workers,
        )

    def get_augmented_self(self, augmented_dir: Path) -> "BaseDMSDataset":
        """Creates a new dataset instance with augmented data."""
        # Create a new instance of the same dataset class
        augmented_dataset = self.__class__(
            base_path=self.base_path.as_posix(),
            preprocessing_pipeline=self.preprocessing_pipeline,
            task=self.task,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            embed_dim=self.embed_dim,
        )
        # Override its data with the augmented data
        augmented_dataset._load_data(augmented_dir)
        return augmented_dataset


class TapeFluorescenceDataset(BaseDMSDataset):
    """Dataset class for tape fluorescence data. Embeddings have shape 256x128."""

    def __init__(self, base_path: str, preprocessing_pipeline: PreprocessingPipeline, **kwargs) -> None:
        super().__init__(base_path, name="tape_fluorescence", preprocessing_pipeline=preprocessing_pipeline, **kwargs)


class TapeStabilityDataset(BaseDMSDataset):
    """Dataset class for tape stability data. Embeddings have shape 64x128."""

    def __init__(self, base_path: str, preprocessing_pipeline: PreprocessingPipeline, **kwargs) -> None:
        super().__init__(base_path, name="tape_stability", preprocessing_pipeline=preprocessing_pipeline, **kwargs)


class Ube4bDataset(BaseDMSDataset):
    """Dataset class for UBE4B data. Embeddings have shape 128x128."""

    def __init__(self, base_path: str, preprocessing_pipeline: PreprocessingPipeline, **kwargs) -> None:
        super().__init__(base_path, name="ube4b", preprocessing_pipeline=preprocessing_pipeline, **kwargs)
