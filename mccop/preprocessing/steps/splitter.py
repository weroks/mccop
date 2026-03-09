import pandas as pd
from sklearn.model_selection import train_test_split

from mccop.data.datasets import BaseDMSDataset
from mccop.preprocessing.pipeline import PreprocessingStep
from mccop.utils.constants import ColumnNames, Splits


class Splitter(PreprocessingStep):
    """A preprocessing step that adds a split column to the DataFrame."""

    def apply(self, df: pd.DataFrame | None, dataset: BaseDMSDataset) -> pd.DataFrame:  # noqa: ARG002
        """Add a split column to the DataFrame.

        Args:
            df (pd.DataFrame | None): The input DataFrame, if any. Should not be None.
            dataset (BaseDMSDataset): The dataset object containing the base path.

        Returns:
            pd.DataFrame: The DataFrame with the split column added.
        """
        if df is None:
            raise ValueError("Splitter cannot be the first step. It requires an input DataFrame.")

        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df[ColumnNames.BINARY_LABEL], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df[ColumnNames.BINARY_LABEL], random_state=42
        )

        train_df[ColumnNames.SPLIT] = Splits.TRAIN.value
        val_df[ColumnNames.SPLIT] = Splits.VAL.value
        test_df[ColumnNames.SPLIT] = Splits.TEST.value

        return pd.concat([train_df, val_df, test_df]).sort_index()
