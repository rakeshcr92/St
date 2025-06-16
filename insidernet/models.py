"""Model wrappers for InsiderNet v2."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class PriceDirectionModel:
    """Simple classifier for price direction using RF or logistic regression."""

    def __init__(self, method: str = "rf") -> None:
        if method == "logit":
            self.model = LogisticRegression(max_iter=200)
        else:
            self.model = RandomForestClassifier(n_estimators=200, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if hasattr(self.model, "predict_proba"):
            return pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)


def train_test_data(
    data: pd.DataFrame, label_column: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=[label_column])
    y = data[label_column]
    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
