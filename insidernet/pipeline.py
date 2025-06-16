"""Main pipeline entry point for InsiderNet v2."""

from __future__ import annotations

import pandas as pd

from .datasources import RedditClient, TwitterClient, EdgarClient, GoogleTrendsClient
from .features import compute_attention_vector
from .models import PriceDirectionModel, train_test_data


def run() -> None:
    """Execute a minimal data flow using placeholder data."""
    # In a real implementation, fetch data from multiple sources here.
    reddit = RedditClient("id", "secret", "agent")
    posts = reddit.fetch_posts("stocks", limit=10)

    features = compute_attention_vector([p.__dict__ for p in posts])
    df = pd.DataFrame([features])
    df["label"] = 0  # placeholder label

    X_train, X_test, y_train, y_test = train_test_data(df, "label")
    model = PriceDirectionModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Predictions:", preds.tolist())


if __name__ == "__main__":
    run()
