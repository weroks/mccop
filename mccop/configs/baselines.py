from dima import get_stats_path
from hydra_zen import builds

from mccop.configs.mccop import editor_store
from mccop.counterfactual.baselines import GeneticAlgorithmBaseline, GradientAscentBaseline, RandomMutationBaseline

STATS_DIR = get_stats_path()

Baseline1Config = builds(
    RandomMutationBaseline,
    normalization_stats_path=STATS_DIR,
    max_steps = 50,
    original_class_filter = 0,
    populate_full_signature=True,
    zen_partial=True,
)

Baseline2Config = builds(
    GradientAscentBaseline,
    normalization_stats_path=STATS_DIR,
    original_class_filter = 0,
    populate_full_signature=True,
    zen_partial=True,
)

Baseline3Config = builds(
    GeneticAlgorithmBaseline,
    original_class_filter = 0,
    normalization_stats_path=STATS_DIR,
    edit_distance_penalty = 0.02,
    populate_full_signature=True,
    zen_partial=True,
)

editor_store(
    Baseline1Config,
    name="baseline1",
)

editor_store(
    Baseline2Config,
    name="baseline2",
)

editor_store(
    Baseline3Config,
    name="baseline3",
)
