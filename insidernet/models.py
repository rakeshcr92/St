"""Simplified model implementations without external dependencies."""
from __future__ import annotations

from typing import Iterable, List, Tuple


class PriceDirectionModel:
    """Naive classifier using label proportion as probability."""

    def __init__(self, method: str = "rf") -> None:
        self.prob = 0.5

    def fit(self, X: Iterable[dict], y: Iterable[int]) -> None:
        y_list = list(y)
        if y_list:
            self.prob = sum(y_list) / float(len(y_list))
        else:
            self.prob = 0.5

    def predict(self, X: Iterable[dict]) -> List[int]:
        thresh = 0.5
        return [1 if self.prob >= thresh else 0 for _ in X]

    def predict_proba(self, X: Iterable[dict]) -> List[float]:
        return [self.prob for _ in X]


def train_test_data(
    data: List[dict], label_column: str, test_size: float = 0.2
) -> Tuple[List[dict], List[dict], List[int], List[int]]:
    """Split list of dicts into train/test sets."""
    data_list = list(data)
    split = int(len(data_list) * (1 - test_size))
    train = data_list[:split]
    test = data_list[split:]
    X_train = [{k: row[k] for k in row if k != label_column} for row in train]
    y_train = [row[label_column] for row in train]
    X_test = [{k: row[k] for k in row if k != label_column} for row in test]
    y_test = [row[label_column] for row in test]
    return X_train, X_test, y_train, y_test
