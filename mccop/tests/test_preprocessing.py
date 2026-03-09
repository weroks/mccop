from pathlib import Path

import pytest
from hydra_zen import instantiate

from mccop.configs.data import store


@pytest.mark.skipif(
    not Path("datasets/tape_fluorescence").exists(), reason="Dataset files not available"
)
def test_tape_fluorescence_dataset() -> None:
    """Test the TapeFluorescenceDataset preprocessing pipeline."""
    # Retrieve the config from the store
    DatasetConfig = store[("dataset", "tape_fluorescence")]
    # Instantiate the dataset, providing the runtime parameter `base_path`
    dataset = instantiate(DatasetConfig, base_path="datasets/tape_fluorescence")()

    assert dataset.name == "tape_fluorescence"
    assert dataset.processed_path.exists()
    assert (dataset.processed_path / "tape_fluorescence.parquet").exists()


@pytest.mark.skipif(
    not Path("datasets/tape_stability").exists(), reason="Dataset files not available"
)
def test_tape_stability_dataset() -> None:
    """Test the TapeStabilityDataset preprocessing pipeline."""
    DatasetConfig = store[("dataset", "tape_stability")]
    dataset = instantiate(DatasetConfig, base_path="datasets/tape_stability")()

    assert dataset.name == "tape_stability"
    assert dataset.processed_path.exists()
    assert (dataset.processed_path / "tape_stability.parquet").exists()


@pytest.mark.skipif(not Path("datasets/ube4b").exists(), reason="Dataset files not available")
def test_ube4b_dataset() -> None:
    """Test the Ube4bDataset preprocessing pipeline."""
    DatasetConfig = store[("dataset", "ube4b")]
    dataset = instantiate(DatasetConfig, base_path="datasets/ube4b")()

    assert dataset.name == "ube4b"
    assert dataset.processed_path.exists()
    assert (dataset.processed_path / "ube4b.parquet").exists()


if __name__ == "__main__":
    pytest.main([__file__])
