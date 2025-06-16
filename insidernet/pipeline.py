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
    labels = []
    tickers = []
    for ticker, plist in by_ticker.items():
        feats = compute_attention_vector(plist)
        rows.append(feats)
        labels.append(1 if ticker == "AAPL" else 0)
        tickers.append(ticker)

    df = pd.DataFrame(rows)
    df["ticker"] = tickers
    df["label"] = labels

    # If the sample size is tiny, train on the entire dataset
    if len(df) < 4:
        X = df.drop(columns=["ticker", "label"])
        y = df["label"]
        model = PriceDirectionModel(method="logit")
        model.fit(X, y)
        preds = model.predict_proba(X)
    else:
        X_train, X_test, y_train, _ = train_test_data(df.drop(columns=["ticker"]), "label")
        model = PriceDirectionModel(method="logit")
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)

    print("Predictions:", preds.tolist())


if __name__ == "__main__":
    run()
