import random
from copy import deepcopy

import torch
from torch import nn
import time

from mccop.counterfactual.base import BaseExplainer
from mccop.data.datasets import BaseDMSDataset
from mccop.eval.metrics import hamming_distance
from mccop.utils.helpers import logger


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


class GradientAscentBaseline(BaseExplainer):
    """Baseline: Standard Gradient Ascent (PGD-like) in embedding space.

    No smoothing, no manifold projection, no complex sparsity mechanisms.
    Just optimizes cross-entropy loss directly on the embedding.
    """

    def __init__(
        self,
        dataset: BaseDMSDataset,
        learning_rate: float = 1e-2,
        gradient_steps: int = 50,
        confidence_threshold: float = 0.95,
        **kwargs,
    ) -> None:
        super().__init__(dataset, confidence_threshold=confidence_threshold, **kwargs)
        self.learning_rate = learning_rate
        self.gradient_steps = gradient_steps
        self.loss_fn = nn.BCEWithLogitsLoss()

    def generate_batch(
        self,
        batch: dict,
        batch_idx: int,
    ) -> dict:
        """Generates counterfactuals/adversarial examples for a batch of input samples."""
        x = batch["embedding"].to(self.device)
        batch_size = x.shape[0]

        target_classes = (1 - batch["label"]).float().to(self.device)

        cf = nn.Parameter(x.clone(), requires_grad=True)

        optimizer = torch.optim.Adam([cf], lr=self.learning_rate)

        best_cf = x.clone()
        best_confidences = torch.zeros(batch_size, device=self.device)
        solved_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            original_classes = (self.predictor(x) > 0).int()

        for i in range(self.gradient_steps):
            if solved_mask.all():
                logger.info(f"Batch {batch_idx}: All samples solved at step {i}")
                break

            optimizer.zero_grad()

            pred_logits = self.predictor(cf)
            loss = self.loss_fn(pred_logits, target_classes)

            if i % 10 == 0:
                logger.info(f"Batch {batch_idx}, Step {i + 1}/{self.gradient_steps}, Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                current_conf = self.target_confidence(self.predictor(cf), target_classes)

                improved = (current_conf > best_confidences) & (~solved_mask)

                if improved.any():
                    best_cf[improved] = cf[improved].detach()
                    best_confidences[improved] = current_conf[improved]

                solved_mask = best_confidences >= self.confidence_threshold

        return {
            "original_input": x.cpu(),
            "original_predicted_classes": original_classes.cpu().flatten().tolist(),
            "target_classes": target_classes.cpu().flatten().tolist(),
            "best_confidences": best_confidences.cpu().tolist(),
            "counterfactuals": best_cf.detach().cpu(),
            "diffusion_time": 0,
            "encoding_time": 0,
            "n_optimization_steps": i + 1,
        }


class RandomMutationBaseline(BaseExplainer):
    """Baseline: Random Mutation.

    Iteratively applies a single random mutation to the sequence tokens and checks
    if the target class probability improves. Since the predictor operates in embedding
    space, this baseline requires mapping discrete mutations back to embeddings.
    """

    def __init__(
        self,
        dataset: BaseDMSDataset,
        max_steps: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(dataset, **kwargs)
        self.max_steps = max_steps
        self.amino_acids = AMINO_ACIDS

    def mutate_sequence(self, seq: str) -> str:
        """Applies a single random point mutation to a sequence."""
        if not seq:
            return seq

        seq_list = list(seq)
        idx = random.randint(0, len(seq_list) - 1)
        original_aa = seq_list[idx]

        seq_list[idx] = random.choice(self.amino_acids)
        while seq_list[idx] == original_aa:
            seq_list[idx] = random.choice(self.amino_acids)
        return "".join(seq_list)

    def generate_batch(
        self,
        batch: dict,
        batch_idx: int,
    ) -> dict:
        """Generates counterfactuals by randomly mutating tokens."""
        x = batch["embedding"].to(self.device)
        seqs = batch["seq"]
        original_labels = batch["label"].to(self.device)
        batch_size = x.shape[0]
        encoding_time = 0

        target_classes = (1 - original_labels).float().to(self.device)

        best_embeddings = x.clone()
        best_seqs = deepcopy(seqs)
        best_confidences = torch.zeros(batch_size, device=self.device)

        solved_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            original_logits = self.predictor(x)
            original_classes = (original_logits > 0).int()

            best_confidences = self.target_confidence(original_logits, target_classes)

            solved_mask = best_confidences >= self.confidence_threshold

        current_seqs = deepcopy(seqs)

        for step in range(self.max_steps):
            if solved_mask.all():
                logger.info(f"Batch {batch_idx}: All samples solved at step {step}")
                break

            mutated_seqs_list = [
                self.mutate_sequence(s) if not solved_mask[i] else s for i, s in enumerate(current_seqs)
            ]

            with torch.no_grad():
                self._sync_cuda()
                t0 = time.perf_counter()
                new_embeddings = self.cheap_model.encode(mutated_seqs_list)
                self._sync_cuda()
                encoding_time += time.perf_counter() - t0
                new_embeddings = new_embeddings.to(self.device)

                pred_logits = self.predictor(new_embeddings)

                current_conf = self.target_confidence(pred_logits, target_classes)

                improved = (current_conf > best_confidences) & (~solved_mask)

                if improved.any():
                    best_confidences[improved] = current_conf[improved]
                    best_embeddings[improved] = new_embeddings[improved]

                    improved_indices = torch.nonzero(improved).flatten().tolist()
                    for idx in improved_indices:
                        best_seqs[idx] = mutated_seqs_list[idx]
                        current_seqs[idx] = mutated_seqs_list[idx]

                newly_solved = (best_confidences >= self.confidence_threshold) & (~solved_mask)
                solved_mask = best_confidences >= self.confidence_threshold

                if newly_solved.any():
                    logger.debug(f"Batch {batch_idx}: {newly_solved.sum()} new samples solved at step {step}")

                not_improved = (~improved) & (~solved_mask)
                not_improved_indices = torch.nonzero(not_improved).flatten().tolist()
                for idx in not_improved_indices:
                    current_seqs[idx] = best_seqs[idx]

            if step % 10 == 0:
                unsolved_conf = best_confidences[~solved_mask]
                avg_conf = unsolved_conf.mean().item() if len(unsolved_conf) > 0 else 1.0
                logger.info(
                    f"Batch {batch_idx} Step {step}: Avg Unsolved Conf {avg_conf:.4f} | "
                    f"Solved: {solved_mask.sum()}/{batch_size}"
                )

        return {
            "original_input": x.cpu(),
            "original_predicted_classes": original_classes.cpu().flatten().tolist(),
            "target_classes": target_classes.cpu().flatten().tolist(),
            "best_confidences": best_confidences.cpu().tolist(),
            "counterfactuals": best_embeddings.detach().cpu(),
            "counterfactual_sequences": best_seqs,
            "n_optimization_steps": step + 1,
            "encoding_time": encoding_time,
            "diffusion_time": 0,
        }


class GeneticAlgorithmBaseline(BaseExplainer):
    """Baseline: Genetic Algorithm (Evolutionary Strategy).

    Uses a population-based discrete evolution strategy.
    Fitness is defined by the predictor's probability of the target class,
    optionally penalized by the edit distance from the original sequence.
    """

    def __init__(
        self,
        dataset: BaseDMSDataset,
        pop_size: int = 40,
        generations: int = 30,
        crossover_rate: float = 0.5,
        edit_distance_penalty: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(dataset, **kwargs)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.edit_distance_penalty = edit_distance_penalty
        self.amino_acids = AMINO_ACIDS
        self.dataset.batch_size = min(self.dataset.batch_size, 8)  # Limit batch size for computational reasons

    def _calculate_fitness(
        self,
        probs: torch.Tensor,
        current_seqs: list[str],
        original_seqs: list[str],
        target_classes: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates fitness: confidence - penalty * edit_distance."""
        conf = torch.where(target_classes == 1, probs, 1.0 - probs).flatten()

        if self.edit_distance_penalty <= 0:
            return conf

        distances = []
        for curr, orig in zip(current_seqs, original_seqs, strict=True):
            dist = hamming_distance(curr, orig)
            distances.append(dist)

        distances = torch.tensor(distances, device=self.device, dtype=torch.float)
        return conf - (self.edit_distance_penalty * distances)

    def _crossover(self, parent1: str, parent2: str) -> str:
        """Single-point crossover."""
        if random.random() > self.crossover_rate:
            return parent1
        split = random.randint(1, len(parent1) - 1)
        return parent1[:split] + parent2[split:]

    def _mutate(self, seq: str) -> str:
        """Random point mutation based on mutation rate."""
        seq_list = list(seq)
        seq_len = len(seq_list)

        num_mutations = random.randint(1, 2)

        for _ in range(num_mutations):
            idx = random.randint(0, seq_len - 1)
            seq_list[idx] = random.choice(self.amino_acids)

        return "".join(seq_list)

    def _evolve_population(
        self,
        batch_size: int,
        fitness_matrix: torch.Tensor,
        population: list[list[str]],
    ) -> list[list[str]]:
        new_population = []

        for b in range(batch_size):
            pop_fitness = fitness_matrix[b]
            sample_pop = population[b]

            n_elites = max(1, int(self.pop_size * 0.2))
            top_indices = torch.topk(pop_fitness, k=n_elites).indices.tolist()
            new_sample_pop = [sample_pop[i] for i in top_indices]

            while len(new_sample_pop) < self.pop_size:
                parents = []
                for _ in range(2):
                    candidates = torch.randint(0, self.pop_size, (3,))
                    best_cand = candidates[torch.argmax(pop_fitness[candidates])]
                    parents.append(sample_pop[best_cand])

                child = self._crossover(parents[0], parents[1])
                child = self._mutate(child)
                new_sample_pop.append(child)

            new_population.append(new_sample_pop)
        return new_population

    def generate_batch(
        self,
        batch: dict,
        batch_idx: int,
    ) -> dict:
        """Generates counterfactuals using a genetic algorithm.

        Args:
            batch: The input batch containing embeddings, masks, and labels.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the following keys:
                - "original_input": The original input tensor.
                - "original_predicted_classes": The original predicted classes.
                - "target_classes": The target classes used for optimization.
                - "best_confidences": The best confidence scores achieved.
                - "counterfactuals": The generated counterfactual tensors.
        """
        encoding_time = 0
        x = batch["embedding"].to(self.device)
        original_seqs = batch["seq"]
        target_classes = (1 - batch["label"]).float().to(self.device)
        batch_size = x.shape[0]

        best_embeddings = x.clone()
        best_seqs = deepcopy(original_seqs)
        best_confidences = torch.zeros(batch_size, device=self.device)
        solved_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            original_classes = (self.predictor(x) > 0).int()

        population = []
        for s in original_seqs:
            sample_pop = [s] + [self._mutate(s) for _ in range(self.pop_size - 1)]
            population.append(sample_pop)

        for gen in range(self.generations):
            if solved_mask.all():
                logger.info(f"Batch {batch_idx}: All samples solved at gen {gen}")
                break

            flat_pop_seqs = [s for sample_pop in population for s in sample_pop]

            expanded_targets = target_classes.repeat_interleave(self.pop_size)
            expanded_orig_seqs = [s for s in original_seqs for _ in range(self.pop_size)]

            with torch.no_grad():
                self._sync_cuda()
                t0 = time.perf_counter()
                embeddings = self.cheap_model.encode(flat_pop_seqs)
                self._sync_cuda()
                encoding_time += time.perf_counter() - t0
                embeddings = embeddings.to(self.device)
                logits = self.predictor(embeddings)
                probs = torch.sigmoid(logits)

            fitness = self._calculate_fitness(probs, flat_pop_seqs, expanded_orig_seqs, expanded_targets)

            raw_conf = torch.where(expanded_targets == 1, probs, 1.0 - probs).flatten()

            fitness_matrix = fitness.view(batch_size, self.pop_size)
            raw_conf_matrix = raw_conf.view(batch_size, self.pop_size)

            _, max_fit_indices = fitness_matrix.max(dim=1)

            for b in range(batch_size):
                if solved_mask[b]:
                    continue

                idx_in_pop = max_fit_indices[b].item()
                current_best_conf = raw_conf_matrix[b, idx_in_pop]

                if current_best_conf > best_confidences[b]:
                    best_confidences[b] = current_best_conf
                    global_idx = b * self.pop_size + idx_in_pop
                    best_embeddings[b] = embeddings[global_idx]
                    best_seqs[b] = population[b][idx_in_pop]

            solved_mask = best_confidences >= self.confidence_threshold

            population = self._evolve_population(batch_size, fitness_matrix, population)

            if gen % 5 == 0:
                avg_conf = best_confidences.mean().item()
                logger.info(f"Batch {batch_idx} Gen {gen}: Avg Target Conf {avg_conf:.4f}")

        return {
            "original_input": x.cpu(),
            "original_predicted_classes": original_classes.cpu().flatten().tolist(),
            "target_classes": target_classes.cpu().flatten().tolist(),
            "best_confidences": best_confidences.cpu().tolist(),
            "counterfactuals": best_embeddings.detach().cpu(),
            "counterfactual_sequences": best_seqs,
            "encoding_time": encoding_time,
            "diffusion_time": 0,
            "n_optimization_steps": gen + 1,
        }
