from hydra_zen import builds, store
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    F1Score,
    Precision,
    Recall,
)

from mccop.models.predictors import ResNet1D, ResNet2D, SimpleCNN, SimpleMLP

store(builds(nn.BCEWithLogitsLoss), name="bce", group="model/loss_fn")
store(builds(nn.MSELoss), name="mse", group="model/loss_fn")
store(builds(nn.L1Loss), name="mae", group="model/loss_fn")

RegressionMetrics = builds(
    MetricCollection,
    metrics={
        "rmse": builds(MeanSquaredError, squared=False),
        "mae": builds(MeanAbsoluteError),
        "r2": builds(R2Score),
    },
)

ClassificationMetrics = builds(
    MetricCollection,
    metrics={
        "accuracy": builds(Accuracy, task="binary"),
        "precision": builds(Precision, task="binary"),
        "recall": builds(Recall, task="binary"),
        "f1": builds(F1Score, task="binary"),
        "auroc": builds(AUROC, task="binary"),
    },
)

store(ClassificationMetrics, name="classification", group="model/metrics")
store(RegressionMetrics, name="regression", group="model/metrics")

SimpleMLPConfig = builds(
    SimpleMLP,
    loss_fn=builds(nn.BCEWithLogitsLoss),
    metrics=ClassificationMetrics,
    hidden_dims=[512, 256],
    dropout=0.5,
    populate_full_signature=True,
    zen_partial=True,
)

SimpleCNNConfig = builds(
    SimpleCNN,
    loss_fn=builds(nn.BCEWithLogitsLoss),
    metrics=ClassificationMetrics,
    num_filters=128,
    kernel_size=3,
    populate_full_signature=True,
    zen_partial=True,
)

ResNet1DConfig = builds(
    ResNet1D,
    loss_fn=builds(nn.BCEWithLogitsLoss),
    metrics=ClassificationMetrics,
    block_channels=[128, 256],
    kernel_size=3,
    dropout=0.5,
    populate_full_signature=True,
    zen_partial=True,
)

ResNet2DConfig = builds(
    ResNet2D,
    loss_fn=builds(nn.BCEWithLogitsLoss),
    metrics=ClassificationMetrics,
    block_channels=[64, 128],
    kernel_size=3,
    dropout=0.5,
    populate_full_signature=True,
    zen_partial=True,
)

store(SimpleMLPConfig, name="mlp", group="model")
store(SimpleCNNConfig, name="cnn", group="model")
store(ResNet1DConfig, name="resnet1d", group="model")
store(ResNet2DConfig, name="resnet2d", group="model")
