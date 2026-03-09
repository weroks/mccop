import importlib
from typing import Any
from pathlib import Path
import torch
from skimage.filters import threshold_otsu

from mccop.utils.helpers import logger

from dima.encoders.enc_normalizer import EncNormalizer


class CHEAPModel:
    """A wrapper around the CHEAP model that handles (de-)normalization, encoding, decoding, and logits extraction.
    Ensures compatibility with DiMA in particular regarding the (technically unnecessary) normalization.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        device: str = "cpu",
        normalization_stats_path: Path | None = None,
    ) -> None:
        self.device = device
        self.embed_dim = embed_dim
        self.normalizer = None
        self.model_name = f"CHEAP_shorten_1_dim_{self.embed_dim}"

        if normalization_stats_path:
            normalization_stats_path = Path(normalization_stats_path) / f"encodings-{self.model_name}.pth"
            logger.info(f"Loading normalization statistics from {normalization_stats_path}")
            self.normalizer = EncNormalizer(Path(normalization_stats_path))

        self.pipeline, self.latent_to_seq = self._load_components(device)

    def _load_components(self, device: str) -> tuple[Any, Any]:
        pipeline = None
        latent_to_seq = None

        try:
            module = importlib.import_module("cheap.pretrained")
            model_func = getattr(module, self.model_name)
            pipeline = model_func(return_pipeline=True).to(device)
            logger.info(f"Pipeline loaded for model: {self.model_name}")

        except (ImportError, AttributeError) as e:
            raise ValueError(f"Model '{self.model_name}' not found in 'cheap.pretrained'") from e

        try:
            latent_to_seq_module = importlib.import_module("cheap.proteins")
            latent_to_seq = latent_to_seq_module.LatentToSequence().to(device)
        except (ImportError, AttributeError) as e:
            raise ValueError("LatentToSequence not found in 'cheap.proteins'") from e

        return pipeline, latent_to_seq

    def embedding_to_seq(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> list[str]:
        """Deconde embeddings back to protein sequences.

        Args:
            embeddings: The input embedding tensor (shape: [B, H, W]).
            attention_mask: Mask indicating valid positions in the sequence (shape: [B, H]).
        """
        embeddings = embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.normalizer is not None:
            embeddings = self.normalizer.denormalize(embeddings)

        esm_embeddings = self.pipeline.decode(embeddings, attention_mask)

        _, _, sequence = self.latent_to_seq.to_sequence(latent=esm_embeddings, mask=attention_mask, return_logits=True)

        final_sequences = []
        for s, m in zip(sequence, attention_mask, strict=True):
            length = int(m.sum().item())
            final_sequences.append(s[:length])

        return final_sequences

    def get_logits(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get decoder logits from latent encodings.

        Args:
            embeddings: The input embedding tensor (shape: [B, H, W]).
            attention_mask: Mask indicating valid positions in the sequence (shape: [B, H]
        """
        embeddings = embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.normalizer is not None:
            embeddings = self.normalizer.denormalize(embeddings)

        esm_embeddings = self.pipeline.decode(embeddings, attention_mask)

        logits, _, _ = self.latent_to_seq.to_sequence(latent=esm_embeddings, mask=attention_mask, return_logits=True)
        return logits


def load_cheap_model(embed_dim: int = 1024, device: str = "cpu") -> Any:
    """Loads the CHEAP model with specified embedding dimension.

    Args:
        embed_dim: Embedding dimension (512 or 1024).
        device: Device to load the model onto.

    Returns:
        The loaded CHEAP model.
    """
    pipeline = None
    latent_to_seq = None
    model_name = f"CHEAP_shorten_1_dim_{embed_dim}"

    try:
        module = importlib.import_module("cheap.pretrained")
        model_func = getattr(module, model_name)
        pipeline = model_func(return_pipeline=True).to(device)
        logger.info(f"Pipeline loaded for model: {model_name}")

    except (ImportError, AttributeError) as e:
        raise ValueError(f"Model '{model_name}' not found in 'cheap.pretrained'") from e

    try:
        latent_to_seq_module = importlib.import_module("cheap.proteins")
        latent_to_seq = latent_to_seq_module.LatentToSequence().to(device)
    except (ImportError, AttributeError) as e:
        raise ValueError("LatentToSequence not found in 'cheap.proteins'") from e

    return pipeline, latent_to_seq


def embedding_to_sequence(
    embedding: torch.Tensor, mask: torch.Tensor | None, decoder: Any, latent_to_seq: Any, device: str
) -> str:
    """Converts an embedding back to a protein sequence using the CHEAP model.

    Args:
        embedding: The input embedding tensor (shape: [B, C, H, W]).
        mask: Mask indicating valid positions in the sequence (shape: [B, H]).
        decoder: The CHEAP model decoder (returns uncompressed ESMfold embeddings).
        latent_to_seq: The latent to sequence converter from CHEAP.
        device: The device to run the computations on.

    Returns:
        The decoded protein sequence as a string.
    """
    mask = mask.to(device) if mask is not None else infer_mask_from_embedding(embedding)
    embedding = embedding.to(device).squeeze(1)  # Remove channel dimension (always 1)

    uncompressed = decoder.decode(embedding, mask)
    seqs = latent_to_seq.to_sequence(uncompressed)[-1]

    # Remove padding using the mask
    seq_lens = [m.sum().item() for m in mask]
    seqs = [seq[:seq_len] for seq, seq_len in zip(seqs, seq_lens, strict=True)]

    # Return list of sequences if batch size > 1, else return single sequence
    return seqs[0] if len(seqs) == 1 else seqs


def infer_mask_from_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Infers a mask from the embedding by identifying non-zero rows.

    Args:
        embedding: The input embedding tensor (shape: [B, C, H, W]).

    Returns:
        A binary mask tensor indicating valid positions (shape: [B, H]).
    """
    embedding = embedding.squeeze(1)  # Remove channel dimension (always 1)
    row_wise_std = torch.std(embedding, dim=-1).cpu().numpy()
    thresh = threshold_otsu(row_wise_std)
    mask = torch.tensor(row_wise_std > thresh, dtype=torch.int)

    # Check that each mask is contiguous
    if not torch.all(mask[:, :-1] >= mask[:, 1:]):
        logger.warning("Non-contiguous masks detected, adjusting to make them contiguous.")
        for i in range(mask.shape[0]):
            first_zero = torch.where(mask[i] == 0)[0]
            if len(first_zero) > 0:
                mask[i, first_zero[0] :] = 0

    return mask.bool().to(embedding.device)
