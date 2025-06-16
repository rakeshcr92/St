"""Model wrappers for InsiderNet v2."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class PriceDirectionModel:
    """Simple random forest classifier for price direction."""

    def __init__(self) -> None:
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)


def train_test_data(
    data: pd.DataFrame, label_column: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=[label_column])
    y = data[label_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)
