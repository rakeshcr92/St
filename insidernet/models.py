"""Model utilities for InsiderNet v2.

This module keeps external dependencies optional. If ``scikit-learn`` is
installed the :class:`PriceDirectionModel` will make use of real classifiers
such as ``LogisticRegression`` or ``RandomForestClassifier``. When those
packages are missing a very naive probabilistic model is used instead so tests
can run in minimal environments.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple


class PriceDirectionModel:
    """Small wrapper around optional scikit-learn classifiers."""

    def __init__(self, method: str = "logit") -> None:
        self.method = method
        self.model = None
        self.prob = 0.5
        self._features: List[str] = []

    # ------------------------------------------------------------------
    def _setup_model(self) -> None:
        """Initialise the chosen sklearn model if possible."""
        if self.method == "logit":
            try:
                from sklearn.linear_model import LogisticRegression  # type: ignore

                self.model = LogisticRegression(max_iter=200)
            except Exception:
                self.method = "naive"
        elif self.method == "rf":
            try:
                from sklearn.ensemble import RandomForestClassifier  # type: ignore

                self.model = RandomForestClassifier(n_estimators=100)
            except Exception:
                self.method = "naive"
        else:
            self.method = "naive"

    def _to_matrix(self, X: Iterable[dict]) -> List[List[float]]:
        return [[row.get(k, 0.0) for k in self._features] for row in X]

    # ------------------------------------------------------------------
    def fit(self, X: Iterable[dict], y: Iterable[int]) -> None:
        self._features = list(X[0].keys()) if X else []
        y_list = list(y)
        self._setup_model()

        # fall back to naive model when data is missing or only a single class
        if self.method == "naive" or not X or len(set(y_list)) < 2:
            self.prob = sum(y_list) / float(len(y_list)) if y_list else 0.5
            return

        X_mat = self._to_matrix(X)
        self.model.fit(X_mat, y_list)

    def predict(self, X: Iterable[dict]) -> List[int]:
        if self.method == "naive" or self.model is None:
            thresh = 0.5
            return [1 if self.prob >= thresh else 0 for _ in X]
        X_mat = self._to_matrix(X)
        return [int(v) for v in self.model.predict(X_mat)]

    def predict_proba(self, X: Iterable[dict]) -> List[float]:
        if self.method == "naive" or self.model is None:
            return [self.prob for _ in X]
        X_mat = self._to_matrix(X)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_mat)
            return [float(p[1]) for p in probs]
        preds = self.model.predict(X_mat)
        return [float(p) for p in preds]


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
