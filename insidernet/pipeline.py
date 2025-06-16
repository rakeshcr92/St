"""Main pipeline entry point for InsiderNet v2."""

from __future__ import annotations

import os
from datetime import datetime


class SimpleDataFrame:
    """Minimal DataFrame-like container used in tests."""

    def __init__(self, records: list[dict]):
        self._records = records
        self.columns = list(records[0].keys()) if records else []

    @property
    def empty(self) -> bool:
        return not self._records

    def __getitem__(self, key: str):
        return [row.get(key) for row in self._records]

    def __setitem__(self, key: str, values):
        if len(self._records) != len(values):
            raise ValueError("length mismatch")
        for row, val in zip(self._records, values):
            row[key] = val
        if key not in self.columns:
            self.columns.append(key)

    def drop(self, columns: list[str]):
        result = []
        for row in self._records:
            result.append({k: v for k, v in row.items() if k not in columns})
        return SimpleDataFrame(result)

    def to_dict(self, orient: str = "records"):
        if orient == "records":
            return list(self._records)
        raise ValueError("unsupported orient")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # `python-dotenv` is optional; ignore if not available
    pass

from .datasources import (
    RedditClient,
    TwitterClient,
    PriceClient,
)
from .features import compute_attention_vector
from .models import PriceDirectionModel


def _gather_reddit_posts(subreddits: list[str], limit: int = 50):
    cid = os.getenv("REDDIT_CLIENT_ID")
    secret = os.getenv("REDDIT_CLIENT_SECRET")
    agent = os.getenv("REDDIT_USER_AGENT", "InsiderNet")
    if not cid or not secret:
        raise RuntimeError("Reddit credentials not configured")
    client = RedditClient(cid, secret, agent)
    posts = []
    for sub in subreddits:
        posts.extend(client.fetch_posts(sub, limit=limit))
    return posts


def _gather_twitter_posts(tickers: list[str], max_results: int = 50):
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        return []
    client = TwitterClient(token)
    tw_posts = []
    for tic in tickers:
        for t in client.search(f"${tic}", max_results=max_results):
            created_str = t.get("created_at")
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00")).timestamp()
            except Exception:
                created = 0.0
            tw_posts.append(
                {
                    "ticker": tic,
                    "text": t.get("text", ""),
                    "author": t.get("author_id", ""),
                    "created": created,
                }
            )
    return tw_posts


def _price_labels(tickers: list[str]) -> dict[str, int]:
    key = os.getenv("PRICE_API_KEY")
    if not key:
        raise RuntimeError("PRICE_API_KEY not configured")
    client = PriceClient(key)
    labels: dict[str, int] = {}
    for tic in tickers:
        prices = client.historical_prices(tic, days=4)
        if len(prices) >= 2:
            labels[tic] = 1 if prices[-1] > prices[0] else 0
    return labels


def get_predictions() -> SimpleDataFrame:
    """Fetch live data and return prediction scores."""
    reddit_posts = _gather_reddit_posts(["stocks", "wallstreetbets"])
    by_ticker: dict[str, list[dict]] = {}
    for p in reddit_posts:
        if not p.ticker:
            continue
        by_ticker.setdefault(p.ticker, []).append(p.__dict__)

    tickers = list(by_ticker.keys())
    twitter_posts = _gather_twitter_posts(tickers)
    for tpost in twitter_posts:
        by_ticker.setdefault(tpost["ticker"], []).append(tpost)

    labels = _price_labels(tickers)

    rows = []
    y = []
    valid_tickers = []
    for ticker, plist in by_ticker.items():
        feats = compute_attention_vector(plist)
        rows.append(feats)
        valid_tickers.append(ticker)
        y.append(labels.get(ticker, 0))

    df = SimpleDataFrame([dict(row) for row in rows])
    df["ticker"] = valid_tickers
    df["label"] = y
    X = df.drop(["ticker", "label"])
    model = PriceDirectionModel(method="logit")
    model.fit(X._records, df["label"])
    scores = model.predict_proba(X._records)
    result = SimpleDataFrame([
        {"ticker": t, "score": s} for t, s in zip(valid_tickers, scores)
    ])
    return result


def run() -> None:
    """Execute a minimal data flow using live data sources."""
    df = get_predictions()
    print("Predictions:", df["score"])


if __name__ == "__main__":
    run()
