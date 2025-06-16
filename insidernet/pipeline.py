"""Main pipeline entry point for InsiderNet v2."""

from __future__ import annotations

import pandas as pd

from .datasources import (
    RedditClient,
    TwitterClient,
    EdgarClient,
    GoogleTrendsClient,
    load_local_reddit,
)
from .features import compute_attention_vector
from .models import PriceDirectionModel, train_test_data


def run() -> None:
    """Execute a minimal data flow using sample local data."""
    posts = load_local_reddit("data/sample_reddit.json")

    by_ticker = {}
    for p in posts:
        by_ticker.setdefault(p.ticker, []).append(p.__dict__)

    rows = []
    for ticker, plist in by_ticker.items():
        feats = compute_attention_vector(plist)
        feats["ticker"] = ticker
        rows.append(feats)

    df = pd.DataFrame(rows)
    df["label"] = [1 if t == "AAPL" else 0 for t in df["ticker"]]

    X_train, X_test, y_train, y_test = train_test_data(df, "label")
    model = PriceDirectionModel(method="logit")
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    print("Predictions:", preds.tolist())


if __name__ == "__main__":
    run()
