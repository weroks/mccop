import importlib
from typing import Any
from pathlib import Path
import torch

from mccop.utils.helpers import logger

from dima.encoders.enc_normalizer import EncNormalizer
from dima import get_stats_path


class CHEAPModel:
    """A wrapper around the CHEAP model that handles (de-)normalization, encoding, decoding, and logits extraction.
    Ensures compatibility with DiMA in particular regarding the (technically unnecessary) normalization.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        device: str = "cpu",
        max_len: int = 254,
        normalization_stats_path: str | Path | None = None,
    ) -> None:
        self.device = device
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.normalizer = None
        self.model_name = f"CHEAP_shorten_1_dim_{self.embed_dim}"

        if normalization_stats_path is None:
            normalization_stats_path = get_stats_path(f"CHEAP_shorten_1_dim_{self.embed_dim}")

        if normalization_stats_path and Path(normalization_stats_path).exists():
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

    def encode(self, sequences: list[str]) -> torch.Tensor:
        """Encodes a list of protein sequences into embeddings using CHEAP.

        Applies normalization if a normalizer is configured.

        Args:
            sequences: List of protein strings.

        Returns:
            Tensor of shape [Batch, Length, EmbedDim].
        """
        padded_embeddings = torch.zeros((len(sequences), self.max_len, self.embed_dim), dtype=torch.float32)
        seq_lens = [len(seq) for seq in sequences]

        with torch.no_grad():
            output = self.pipeline(sequences)
            embeddings = output[0]

            if self.normalizer is not None:
                embeddings = self.normalizer.normalize(embeddings)
            for i, length in enumerate(seq_lens):
                padded_embeddings[i, :length, :] = embeddings[i, :length, :]

        return padded_embeddings

    def embedding_to_seq(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> list[str]:
        """Decode embeddings back to protein sequences.

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
