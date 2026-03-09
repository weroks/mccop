from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import MetricCollection


class BasePredictor(LightningModule):
    """A generic PyTorch Lightning base model for prediction tasks.

    This model is designed to be inherited by specific architectures. It
    provides the boilerplate for training, validation, and testing loops, as
    well as optimizer configuration.

    Attributes:
        loss_fn: The loss function.
        metrics: A collection of torchmetrics to be calculated.
        learning_rate: The learning rate for the optimizer.
    """

    def __init__(
        self,
        seq_len: int | None = None,
        embed_dim: int | None = None,
        loss_fn: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize the BasePredictor.

        Args:
            seq_len: The length of the input sequence.
            embed_dim: The embedding dimension of the input.
            loss_fn: The loss function.
            metrics: A collection of torchmetrics to be calculated for the
                validation and test sets.
            learning_rate: The learning rate for the optimizer.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.train_metrics = metrics.clone() if metrics else None
        self.val_metrics = metrics.clone() if metrics else None
        self.test_metrics = metrics.clone() if metrics else None

        if type(self) is BasePredictor:
            self.save_hyperparameters(ignore=["metrics", "loss_fn"])

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single step of computation.

        Args:
            batch: The input batch, containing features and labels.
            batch_idx: The index of the batch (just for compatibility).

        Returns:
            A tuple containing the loss, predictions, and true labels.
        """
        x, y = batch["embedding"], batch["label"]
        y_hat = self(x)
        y = y.float().view_as(y_hat)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            The loss for the training step.
        """
        loss, y_hat, y = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.train_metrics:
            self.train_metrics(torch.sigmoid(y_hat), y.int())
        return loss

    def on_train_epoch_end(self) -> None:
        """Log training metrics at the end of a training epoch."""
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict({f"train_{k}": v for k, v in metrics.items()})
            self.train_metrics.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
        """
        loss, y_hat, y = self._step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.val_metrics:
            self.val_metrics(torch.sigmoid(y_hat), y.int())

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at the end of a validation epoch."""
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict({f"val_{k}": v for k, v in metrics.items()})
            self.val_metrics.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
        """
        loss, y_hat, y = self._step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        if self.test_metrics:
            self.test_metrics(torch.sigmoid(y_hat), y.int())

    def on_test_epoch_end(self) -> None:
        """Log test metrics at the end of a test epoch."""
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict({f"test_{k}": v for k, v in metrics.items()})
            self.test_metrics.reset()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Explicitly inject the model's class path into the checkpoint.
        This is used for loading the model without knowing the class beforehand.

        Args:
            checkpoint: The checkpoint dictionary to be saved.
        """
        class_path = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        checkpoint["hyper_parameters"]["model_class_path"] = class_path
        checkpoint["hyper_parameters"]["_target_"] = class_path

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the model.

        Returns:
            The configured Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class SimpleMLP(BasePredictor):
    """A simple Multi-Layer Perceptron model that flattens the input."""

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        hidden_dims: list[int],
        loss_fn: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        learning_rate: float = 1e-3,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the SimpleMLP.

        Args:
            seq_len: The length of the input sequence.
            embed_dim: The embedding dimension of the input.
            hidden_dims: A list of integers specifying the size of each
                hidden layer.
            loss_fn: The loss function.
            metrics: A collection of torchmetrics for validation and test.
            learning_rate: The learning rate for the optimizer.
            dropout: The dropout rate to apply after each hidden layer.
        """
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()

        super().__init__(seq_len, embed_dim, loss_fn, metrics, learning_rate)
        self.save_hyperparameters(ignore=["metrics", "loss_fn"])

        input_dim = seq_len * embed_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.model = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the MLP.

        Args:
            x: The input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            The output tensor of shape (batch,).
        """
        return self.model(x).squeeze(-1)


class SimpleCNN(BasePredictor):
    """A simple 1D CNN model."""

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_filters: int,
        kernel_size: int,
        loss_fn: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize the SimpleCNN.

        Args:
            seq_len: The length of the input sequence.
            embed_dim: The embedding dimension, treated as input channels.
            num_filters: The number of output channels for the convolutional layer.
            kernel_size: The size of the convolutional kernel.
            loss_fn: The loss function.
            metrics: A collection of torchmetrics for validation and test.
            learning_rate: The learning rate for the optimizer.
        """
        super().__init__(seq_len, embed_dim, loss_fn, metrics, learning_rate)
        self.save_hyperparameters(ignore=["metrics", "loss_fn"])
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNN.

        Args:
            x: The input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            The output tensor of shape (batch,).
        """
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc1(x)
        return x.squeeze(-1)


class ResNet1DBlock(nn.Module):
    """A 1D ResNet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the ResNet1DBlock."""
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the ResNet block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1D(BasePredictor):
    """A ResNet-style architecture for 1D sequence data."""

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        block_channels: list[int],
        kernel_size: int,
        loss_fn: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        learning_rate: float = 1e-3,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the ResNet1D model."""
        super().__init__(seq_len, embed_dim, loss_fn, metrics, learning_rate)
        self.save_hyperparameters(ignore=["metrics", "loss_fn"])
        layers = []
        in_channels = embed_dim
        for out_channels in block_channels:
            layers.append(ResNet1DBlock(in_channels, out_channels, kernel_size, dropout=dropout))
            in_channels = out_channels
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the ResNet1D model."""
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


class ResNet2DBlock(nn.Module):
    """A 2D ResNet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the ResNet2DBlock."""
        super().__init__()
        padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the ResNet block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet2D(BasePredictor):
    """A ResNet-style architecture for 2D data."""

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        block_channels: list[int],
        loss_fn: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        learning_rate: float = 1e-3,
        kernel_size: int | tuple[int, int] = 3,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the ResNet2D model."""
        super().__init__(seq_len, embed_dim, loss_fn, metrics, learning_rate)
        self.save_hyperparameters(ignore=["metrics", "loss_fn"])
        layers = []
        in_channels = 1
        for out_channels in block_channels:
            layers.append(ResNet2DBlock(in_channels, out_channels, kernel_size, dropout=dropout))
            in_channels = out_channels
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the ResNet2D model."""
        x = x.unsqueeze(1)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(-1)
