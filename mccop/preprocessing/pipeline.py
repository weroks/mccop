from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mccop.data.datasets import BaseDMSDataset


class PreprocessingStep(ABC):
    """Abstract base class for a step in the preprocessing pipeline.
    Each step transforms a DataFrame and returns the modified DataFrame.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame | None, dataset: "BaseDMSDataset") -> pd.DataFrame:
        """Apply the preprocessing step.

        Args:
            df (pd.DataFrame | None): The DataFrame from the previous step.
                                      `None` if this is the first step.
            dataset (BaseDMSDataset): The dataset object, providing context like paths.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class PreprocessingPipeline:
    """A class to manage and run a preprocessing pipeline in-memory."""

    def __init__(self, steps: list[PreprocessingStep]) -> None:
        self.steps = steps

    def run(self, dataset: "BaseDMSDataset") -> None:
        """Run the preprocessing pipeline.

        Data is loaded by the first step and then passed through subsequent
        steps in memory. The final DataFrame is saved once at the end.

        Args:
            dataset (BaseDMSDataset): The dataset object to be processed.
        """
        processed_df: pd.DataFrame | None = None

        for step in self.steps:
            processed_df = step.apply(processed_df, dataset)

        if processed_df is not None:
            dataset.processed_path.mkdir(parents=True, exist_ok=True)
            output_path = dataset.processed_path / f"{dataset.name}.parquet"
            processed_df.to_parquet(output_path)
