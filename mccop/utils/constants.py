from enum import Enum
from typing import Final


class Splits(Enum):
    """Enum for dataset splits, containing both a string label and an integer value."""

    def __init__(self, label: str, value: int) -> None:
        self.label = label
        self._value_ = value

    TRAIN = ("train", 0)
    VAL = ("val", 1)
    TEST = ("test", 2)

    def __str__(self) -> str:
        """Allows casting the enum member to its string label directly."""
        return self.label


class ColumnNames:
    """DataFrame column names."""

    SEQ: Final[str] = "seq"
    LABEL: Final[str] = "label"
    BINARY_LABEL: Final[str] = "binary_label"
    SPLIT: Final[str] = "split"
    SEQ_LEN: Final[str] = "seq_len"
    IS_AUGMENTED: Final[str] = "is_augmented"


class Paths:
    """Commonly used file paths."""

    DATA_DIR: Final[str] = "datasets"
    OUTPUTS_DIR: Final[str] = "outputs"
    RESULTS_DIR: Final[str] = "results"
    PLOTS_DIR: Final[str] = "plots"
