from dima import get_stats_path
from hydra_zen import builds

from mccop.preprocessing.pipeline import PreprocessingPipeline
from mccop.preprocessing.steps.embedders import Embedder
from mccop.preprocessing.steps.loaders import (
    TapeFluorescenceLoader,
    TapeStabilityLoader,
    Ube4bLoader,
)
from mccop.preprocessing.steps.splitter import Splitter

STATS_DIR = get_stats_path()

TapeFluorescencePipelineConfig = builds(
    PreprocessingPipeline,
    steps=[
        builds(TapeFluorescenceLoader),
        builds(Splitter),
        builds(Embedder, stats_dir=STATS_DIR, embed_dim=1024, max_len=254),
    ],
)

TapeStabilityPipelineConfig = builds(
    PreprocessingPipeline,
    steps=[
        builds(TapeStabilityLoader),
        builds(Splitter),
        builds(Embedder, stats_dir=STATS_DIR, embed_dim=1024, max_len=254),
    ],
)

Ube4bPipelineConfig = builds(
    PreprocessingPipeline,
    steps=[
        builds(Ube4bLoader),
        builds(Splitter),
        builds(Embedder, stats_dir=STATS_DIR, embed_dim=1024, max_len=254),
    ],
)
