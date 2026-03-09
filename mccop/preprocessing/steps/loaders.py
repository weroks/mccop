from abc import ABC, abstractmethod

import pandas as pd
from skimage.filters import threshold_otsu

from mccop.data.datasets import BaseDMSDataset
from mccop.preprocessing.pipeline import PreprocessingStep
from mccop.utils.constants import ColumnNames


class BaseLoader(PreprocessingStep, ABC):
    """Base class for steps that load and perform initial processing of data."""

    def apply(self, df: pd.DataFrame | None, dataset: BaseDMSDataset) -> pd.DataFrame:
        """Apply the loading and processing step to the dataset.

        Args:
            df (pd.DataFrame | None): The input DataFrame, if any. Should be None for the first step.
            dataset (BaseDMSDataset): The dataset object containing the base path.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        if df is not None:
            raise ValueError("Loaders must be the first step in the pipeline.")
        raw_df = self._load_data(dataset)
        return self._process_data(raw_df)

    @abstractmethod
    def _load_data(self, dataset: BaseDMSDataset) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement data loading.")

    @abstractmethod
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement data processing.")


class TapeBaseLoader(BaseLoader, ABC):
    """Base loader for TAPE datasets (Fluorescence and Stability)."""

    SOURCE_COLUMN: str
    FILE_PREFIX: str

    def _load_data(self, dataset: BaseDMSDataset) -> pd.DataFrame:
        original_path = dataset.base_path / "original"
        split_dfs = []
        for split in ["train", "valid", "test"]:
            split_path = original_path / f"{self.FILE_PREFIX}_{split}.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Split path {split_path} does not exist.")
            split_dfs.append(pd.read_json(split_path))
        return pd.concat(split_dfs).reset_index(drop=True)

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["id"])
        df = df.rename(
            columns={
                "primary": ColumnNames.SEQ,
                "protein_length": ColumnNames.SEQ_LEN,
                self.SOURCE_COLUMN: ColumnNames.LABEL,
            }
        )
        df[ColumnNames.LABEL] = df[ColumnNames.LABEL].apply(lambda x: x[0] if isinstance(x, list) else x)
        return self._create_binary_label(df)

    @abstractmethod
    def _create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement binary label creation.")


class TapeFluorescenceLoader(TapeBaseLoader):
    """Loads and processes the TAPE Fluorescence dataset."""

    SOURCE_COLUMN = "log_fluorescence"
    FILE_PREFIX = "fluorescence"

    def _create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = threshold_otsu(df[ColumnNames.LABEL].values)
        df[ColumnNames.BINARY_LABEL] = (df[ColumnNames.LABEL] > threshold).astype(int)
        return df


class TapeStabilityLoader(TapeBaseLoader):
    """Loads and processes the TAPE Stability dataset."""

    SOURCE_COLUMN = "stability_score"
    FILE_PREFIX = "stability"

    def _create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df[ColumnNames.BINARY_LABEL] = pd.qcut(df[ColumnNames.LABEL], q=3, labels=[0, 0.5, 1]).astype(float)
        return df[df[ColumnNames.BINARY_LABEL] != 0.5].reset_index(drop=True)


class Ube4bLoader(BaseLoader):
    """Loads and processes the UBE4B dataset."""

    def _load_data(self, dataset: BaseDMSDataset) -> pd.DataFrame:
        original_path = dataset.base_path / "original"
        df = pd.read_excel(original_path / "sd01.xlsx")
        df = df[~df.seqID.str.contains(r"\*|NA", na=False)]

        seq_info = pd.read_csv(original_path / "original_seq_info.csv")
        original_seq = seq_info["extended_U_box_domain"].iloc[0]

        df[ColumnNames.SEQ] = df["seqID"].apply(lambda x: self._apply_mutations(original_seq, x))
        return df

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df[ColumnNames.SEQ_LEN] = df[ColumnNames.SEQ].str.len()
        df = df.rename(columns={"nscor_log2_ratio": ColumnNames.LABEL})
        df[ColumnNames.BINARY_LABEL] = pd.qcut(df[ColumnNames.LABEL], q=3, labels=[0, 0.5, 1]).astype(float)
        df = df.dropna(subset=[ColumnNames.SEQ, ColumnNames.LABEL, ColumnNames.BINARY_LABEL])
        return df[df[ColumnNames.BINARY_LABEL] != 0.5].reset_index(drop=True)

    def _apply_mutations(self, sequence: str, seqid: str) -> str:
        seq_list = list(sequence)
        if "-" in seqid:
            pos_str, mut_str = seqid.split("-")
            positions = [int(p) for p in pos_str.split(",")]
            mutations = mut_str.split(",")
            for pos, mut in zip(positions, mutations, strict=False):
                seq_list[pos - 1] = mut
        return "".join(seq_list)
