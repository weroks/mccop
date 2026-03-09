from abc import ABC
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from mccop.data.datasets import BaseDMSDataset
from mccop.preprocessing.pipeline import PreprocessingStep
from mccop.utils.constants import ColumnNames
from mccop.utils.helpers import logger, get_device
from mccop.utils.protein_utils import load_cheap_model

from dima.encoders.enc_normalizer import EncNormalizer


class Embedder(PreprocessingStep, ABC):
    """Creates CHEAP embeddings, normalizes them, and pads to a fixed length."""

    def __init__(self, stats_dir: Path, batch_size: int = 64, embed_dim: int = 1024, max_len: int = 254) -> None:
        """Initializes the Embedder.

        Args:
            stats_dir: Directory containing DIMA statistics files for normalization.
            batch_size: The batch size for processing sequences.
            embed_dim: The embedding dimension for the model.
            max_len: The fixed length to pad sequences to (must match DIMA config).
        """
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.stats_dir = stats_dir

        self.model_name = f"CHEAP_shorten_1_dim_{self.embed_dim}"

    def _get_normalizer(self) -> EncNormalizer:
        """Loads the normalizer if the stats file exists."""
        stats_path = self.stats_dir / f"encodings-{self.model_name}.pth"

        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics file not found at {stats_path}. Cannot normalize embeddings.")

        return EncNormalizer(str(stats_path))

    def apply(self, df: pd.DataFrame | None, dataset: BaseDMSDataset) -> pd.DataFrame:
        """Apply the embedding step and save raw embeddings and masks separately.

        Args:
            df: The input DataFrame from a previous pipeline step.
            dataset: The dataset object, used to determine save paths.

        Returns:
            The DataFrame without the large embedding/mask columns.

        Raises:
            ValueError: If the input DataFrame is None or is missing the sequence column.
        """
        if df is None:
            raise ValueError("Embedders must not be the first step in the pipeline.")
        if ColumnNames.SEQ not in df.columns:
            raise ValueError(f"Input DataFrame must contain a '{ColumnNames.SEQ}' column.")

        sequences = df[ColumnNames.SEQ].tolist()
        num_samples = len(sequences)

        embeddings_path = dataset.processed_path / f"{dataset.name}_embeddings.pt"
        masks_path = dataset.processed_path / f"{dataset.name}_masks.pt"

        if embeddings_path.exists() and masks_path.exists():
            logger.info(f"Embeddings already exist at {dataset.processed_path}. Skipping generation.")
            return df

        pipeline, _ = load_cheap_model(embed_dim=self.embed_dim, device=get_device())
        normalizer = self._get_normalizer()

        final_embeddings = torch.zeros((num_samples, self.max_len, self.embed_dim), dtype=torch.float32)
        final_masks = torch.zeros((num_samples, self.max_len), dtype=torch.int8)

        logger.info(f"Generating embeddings for {dataset.name} using {self.model_name}...")
        logger.info(f"Target format: [Batch, {self.max_len}, {self.embed_dim}]")

        for i in tqdm(range(0, num_samples, self.batch_size), desc=f"Embedding with {self.model_name}"):
            batch_sequences = sequences[i : i + self.batch_size]

            with torch.no_grad():
                output = pipeline(batch_sequences)
                emb = output[0]

                if normalizer:
                    emb = normalizer.normalize(emb)

                emb = emb.cpu()

                for j, seq in enumerate(batch_sequences):
                    original_len = len(seq)
                    if original_len > self.max_len:
                        logger.warning(
                            f"Sequence at index {i + j} exceeds max length of {self.max_len}. "
                            f"Truncating from {original_len} to {self.max_len}."
                        )

                    valid_embedding_part = emb[j, :original_len]

                    idx = i + j
                    final_embeddings[idx, :original_len, :] = valid_embedding_part
                    final_masks[idx, :original_len] = 1

        dataset.processed_path.mkdir(parents=True, exist_ok=True)
        torch.save(final_embeddings, embeddings_path)
        torch.save(final_masks, masks_path)

        logger.info(f"Processed (normalized & padded) embeddings saved to {dataset.processed_path}")

        return df
