from abc import ABC, abstractmethod
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset
from tqdm import tqdm

from mccop.counterfactual.utils import load_best_model
from mccop.data.datasets import BaseDMSDataset
from mccop.eval.metrics import StructurePredictor, sequence_plausibility, hamming_distance
from mccop.models.autoencoders import CHEAPModel
from mccop.models.predictors import BasePredictor
from mccop.utils.constants import Splits
from mccop.utils.helpers import get_device, logger, seed_everything


class BaseExplainer(ABC):
    """Abstract base class for counterfactual generation and baselines.

    Handles common setup, data iteration, and evaluation logic.
    """

    def __init__(
        self,
        dataset: Dataset,
        predictor: BasePredictor | None = None,
        confidence_threshold: float = 0.95,
        max_samples: int | None = None,
        original_class_filter: int | None = None,
        wandb_logger: WandbLogger | None = None,
        normalization_stats_path: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        torch.set_float32_matmul_precision("medium")
        seed_everything(seed)

        self.device = get_device()
        self.dataset: BaseDMSDataset = dataset
        self.wandb_logger = wandb_logger
        self.history = []
        self.timing_results = []
        self.seed = seed

        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        self.original_class_filter = original_class_filter

        self.predictor = (
            (
                predictor
                if predictor is not None
                else load_best_model(
                    dataset_name=dataset.name,
                    embed_dim=dataset.embed_dim,
                    task_name="classification",
                    metric="val_auroc",
                    mode="max",
                )
            )
            .to(self.device)
            .eval()
        )

        self.cheap_model = CHEAPModel(
            embed_dim=dataset.embed_dim, device=self.device, normalization_stats_path=normalization_stats_path
        )

    @abstractmethod
    def generate_batch(self, batch: dict, batch_idx: int) -> dict:
        """Abstract method to generate counterfactuals for a specific batch.

        Must return a dict with keys: 'original_input', 'original_predicted_classes',
        'target_classes', 'best_confidences', 'counterfactuals'.
        """
        pass

    def run(self, split: str = Splits.TEST) -> pd.DataFrame:
        """Run the counterfactual generation or baseline.

        Args:
            split (str): Dataset split to use ('train', 'val', 'test').

        Returns:
            pd.DataFrame: DataFrame containing counterfactual results and metrics.
        """
        data_loader = self.dataset.get_loader(split, target_label=self.original_class_filter)
        results = []

        pbar = tqdm(total=self.max_samples or len(data_loader.dataset), desc=f"Running {self.__class__.__name__}")
        samples_processed = 0

        for batch_idx, batch in enumerate(data_loader):
            self._sync_cuda()
            start_time = time.perf_counter()
            batch_results = self.generate_batch(batch, batch_idx)
            self._sync_cuda()
            batch_time = time.perf_counter() - start_time

            processed_results, decoding_time = self._process_batch_results(batch, batch_results)

            self.timing_results.append(
                {
                    "batch_idx": batch_idx,
                    "time_seconds": batch_time,
                    "batch_size": len(batch["seq"]),
                    "diffusion_time": batch_results.get("diffusion_time", 0),
                    "encoding_time": batch_results.get("encoding_time", 0),
                    "decoding_time": decoding_time,
                    "n_optimization_steps": batch_results.get("n_optimization_steps", 0),
                }
            )

            results.extend(processed_results)

            samples_processed += len(processed_results)
            pbar.update(len(processed_results))

            if self.max_samples and samples_processed >= self.max_samples:
                break

        pbar.close()
        return self._compute_and_save_metrics(results)

    def target_confidence(self, logits: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
        """Computes the confidence of the prediction towards the target class.

        Args:
            logits: Raw logits from the predictor, shape ``(batch_size, 1)`` or ``(batch_size,)``.
            target_classes: Binary target classes, shape ``(batch_size,)``.

        Returns:
            Confidence scores in ``[0, 1]``, shape ``(batch_size,)``.
        """
        probs = torch.sigmoid(logits)
        return torch.where(target_classes == 1, probs, 1.0 - probs).flatten()

    def _process_batch_results(self, batch: dict, batch_results: dict) -> tuple[list, float]:
        """Helper to transform batch tensors into result dictionaries.

        Args:
            batch: The original input batch.
            batch_results: The results from optimize_batch.

        Returns:
            A tuple of (result list, decoding_time) where decoding_time is the time
            spent decoding embeddings to sequences (0 if sequences were already provided).
        """
        batch_list = []
        decoding_time = 0.0
        for j in range(len(batch_results["target_classes"])):
            cf_single = batch_results["counterfactuals"][j : j + 1].to(self.device)
            m_single = batch["mask"][j : j + 1]

            if "counterfactual_sequences" in batch_results:
                cf_seq = batch_results["counterfactual_sequences"][j]
            else:
                self._sync_cuda()
                t0 = time.perf_counter()
                cf_seq = self.cheap_model.embedding_to_seq(cf_single, m_single)[0]
                self._sync_cuda()
                decoding_time += time.perf_counter() - t0

            res = {
                "original_sequence": batch["seq"][j],
                "counterfactual_sequence": cf_seq,
                "original_predicted_class": batch_results["original_predicted_classes"][j],
                "original_true_class": batch["label"][j].item(),
                "target_class": batch_results["target_classes"][j],
                "best_confidence": batch_results["best_confidences"][j],
                "original_input": batch_results["original_input"][j].tolist(),
                "counterfactual": batch_results["counterfactuals"][j].tolist(),
            }
            batch_list.append(res)
        return batch_list, decoding_time

    def _compute_and_save_metrics(self, results: list) -> pd.DataFrame:
        """Computes final metrics, logs to WandB, and saves result files."""
        if not results:
            logger.warning("No counterfactuals were generated.")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        output_dir = Path("results") / self.dataset.name / self.__class__.__name__ / f"seed_{self.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        structures_dir = output_dir / "structures"
        structures_dir.mkdir(exist_ok=True)

        df = self._filter_correct_predictions(df)
        df = self._compute_distance_metrics(df)
        df = self._compute_sequence_metrics(df)
        df = self._compute_structural_metrics(df, structures_dir)

        summary_stats = self._build_summary_stats(df)
        logger.info(f"Results for {self.__class__.__name__}: {summary_stats}")

        self._save_artifacts(df, summary_stats, output_dir)
        return df

    @staticmethod
    def _filter_correct_predictions(df: pd.DataFrame) -> pd.DataFrame:
        """Removes samples where the predictor's original prediction was wrong."""
        old_len = len(df)
        df = df[df["original_predicted_class"] == df["original_true_class"]]
        n_removed = old_len - len(df)
        if n_removed:
            logger.info(f"Removed {n_removed}/{old_len} samples due to incorrect original predictions.")
        return df

    @staticmethod
    def _compute_distance_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Computes edit distance and cosine similarity between originals and counterfactuals."""
        df["edit_distance"] = df.apply(
            lambda row: hamming_distance(row["original_sequence"], row["counterfactual_sequence"]), axis=1
        )

        orig_tensors = torch.tensor(df["original_input"].tolist())
        cf_tensors = torch.tensor(df["counterfactual"].tolist())
        df["cosine_similarity"] = F.cosine_similarity(orig_tensors.flatten(1), cf_tensors.flatten(1), dim=1).numpy()
        return df.drop(columns=["original_input", "counterfactual"])

    @staticmethod
    def _compute_sequence_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Computes sequence plausibility metrics for originals and counterfactuals."""
        logger.info("Computing sequence properties...")
        for col, prefix in [("counterfactual_sequence", "cf_"), ("original_sequence", "orig_")]:
            metrics = df[col].apply(sequence_plausibility).apply(pd.Series)
            df = pd.concat([df, metrics.add_prefix(prefix)], axis=1)
        return df

    def _compute_structural_metrics(self, df: pd.DataFrame, structures_dir: Path) -> pd.DataFrame:
        """Computes structural metrics (pLDDT, SASA, etc.) for successful counterfactuals."""
        logger.info("Computing structural metrics...")
        structure_predictor = StructurePredictor()
        tqdm.pandas(desc="Computing Structural Metrics")

        def compute_row_structure(row: pd.Series) -> pd.Series:
            if row.get("best_confidence", 1.0) < self.confidence_threshold:
                return pd.Series(dtype=float)

            orig_scores = structure_predictor.compute_scores(
                row["original_sequence"], structures_dir / f"orig_{row.name}.pdb"
            )
            cf_scores = structure_predictor.compute_scores(
                row["counterfactual_sequence"], structures_dir / f"cf_{row.name}.pdb"
            )

            combined = {f"orig_{k}": v for k, v in orig_scores.items()}
            combined.update({f"cf_{k}": v for k, v in cf_scores.items()})
            return pd.Series(combined)

        structure_metrics = df.progress_apply(compute_row_structure, axis=1)
        return pd.concat([df, structure_metrics], axis=1)

    def _build_summary_stats(self, df: pd.DataFrame) -> dict:
        """Builds summary statistics dict from the results dataframe."""
        successful = df[df["best_confidence"] >= self.confidence_threshold]
        n_successful = len(successful)
        n_adversarial = len(successful[successful["edit_distance"] == 0])
        total_time = sum(t["time_seconds"] for t in self.timing_results)
        total_diffusion_time = sum(t["diffusion_time"] for t in self.timing_results)
        total_encoding_time = sum(t["encoding_time"] for t in self.timing_results)
        total_decoding_time = sum(t["decoding_time"] for t in self.timing_results)
        total_samples_processed = sum(t["batch_size"] for t in self.timing_results)

        def _safe_mean(series_df: pd.DataFrame, col: str) -> float | None:
            return series_df[col].mean() if col in series_df.columns else None

        summary = {
            "method": self.__class__.__name__,
            "metrics/success_rate": n_successful / len(df) if len(df) > 0 else 0,
            "metrics/adversarial_rate": n_adversarial / len(df) if len(df) > 0 else 0,
            "metrics/mean_edit_distance": _safe_mean(successful, "edit_distance"),
            "metrics/mean_cf_plddt": _safe_mean(successful, "cf_plddt"),
            "metrics/mean_cf_sasa": _safe_mean(successful, "cf_sasa"),
            "metrics/mean_cf_rg": _safe_mean(successful, "cf_radius_of_gyration"),
            "metrics/total_time_seconds": total_time,
            "metrics/total_diffusion_time": total_diffusion_time,
            "metrics/total_encoding_time": total_encoding_time,
            "metrics/total_decoding_time": total_decoding_time,
            "metrics/time_per_successful_edit": total_time / n_successful if n_successful > 0 else None,
        }

        if self.timing_results:
            timing_df = pd.DataFrame(self.timing_results)
            n = total_samples_processed
            summary.update(
                {
                    "metrics/mean_optimization_steps": timing_df["n_optimization_steps"].mean(),
                    "metrics/median_optimization_steps": timing_df["n_optimization_steps"].median(),
                    "metrics/total_samples_processed": n,
                    "metrics/time_per_sample": total_time / n if n > 0 else None,
                    "metrics/diffusion_time_per_sample": total_diffusion_time / n if n > 0 else None,
                    "metrics/encoding_time_per_sample": total_encoding_time / n if n > 0 else None,
                    "metrics/decoding_time_per_sample": total_decoding_time / n if n > 0 else None,
                    "metrics/diffusion_time_fraction": total_diffusion_time / total_time if total_time > 0 else None,
                    "metrics/encoding_time_fraction": total_encoding_time / total_time if total_time > 0 else None,
                    "metrics/decoding_time_fraction": total_decoding_time / total_time if total_time > 0 else None,
                }
            )

        return summary

    def _save_artifacts(self, df: pd.DataFrame, summary_stats: dict, output_dir: Path) -> None:
        """Saves summary CSV, timing results, training log, and optionally logs to WandB."""
        df.to_parquet(output_dir / "counterfactuals.parquet")

        if self.wandb_logger:
            self.wandb_logger.log_metrics(summary_stats)

        pd.DataFrame([summary_stats]).to_csv(output_dir / "summary.csv")

        if self.timing_results:
            pd.DataFrame(self.timing_results).to_csv(output_dir / "timing_results.csv", index=False)

        if self.history:
            pd.DataFrame(self.history).to_parquet(output_dir / "training_log.parquet", index=False)

    @staticmethod
    def _sync_cuda() -> None:
        """Synchronizes CUDA if available, for accurate timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
