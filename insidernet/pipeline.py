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
from .models import PriceDirectionModel


def get_predictions() -> pd.DataFrame:
    """Return prediction scores for the sample data."""
    posts = load_local_reddit("data/sample_reddit.json")

    by_ticker: dict[str, list[dict]] = {}
    for p in posts:
        by_ticker.setdefault(p.ticker, []).append(p.__dict__)

    rows = []
    tickers = []
    labels = []
    for ticker, plist in by_ticker.items():
        feats = compute_attention_vector(plist)
        rows.append(feats)
        tickers.append(ticker)
        labels.append(1 if ticker == "AAPL" else 0)

    df = pd.DataFrame(rows)
    df["ticker"] = tickers
    df["label"] = labels

    X = df.drop(columns=["ticker", "label"])
    y = df["label"]
    model = PriceDirectionModel(method="logit")
    model.fit(X, y)
    scores = model.predict_proba(X)
    return pd.DataFrame({"ticker": tickers, "score": scores})


def run() -> None:
    """Execute a minimal data flow using sample local data."""
    df = get_predictions()
    print("Predictions:", df["score"].tolist())


if __name__ == "__main__":
    run()
