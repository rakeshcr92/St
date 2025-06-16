"""Main pipeline entry point for InsiderNet v2."""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path


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
    EdgarClient,
    GoogleTrendsClient,
    PriceClient,
)
from .features import compute_attention_vector, anomaly_score
from .models import PriceDirectionModel

HISTORY_FILE = Path("post_history.json")
PRED_HISTORY_FILE = Path("prediction_history.json")


def _load_history() -> dict[str, list[int]]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_history(history: dict[str, list[int]]) -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(history))
    except Exception:
        pass


def _load_prediction_history() -> list[dict]:
    if PRED_HISTORY_FILE.exists():
        try:
            return json.loads(PRED_HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def _save_prediction_history(rows: list[dict]) -> None:
    hist = _load_prediction_history()
    timestamp = datetime.utcnow().isoformat()
    for r in rows:
        entry = dict(r)
        entry["timestamp"] = timestamp
        hist.append(entry)
    try:
        PRED_HISTORY_FILE.write_text(json.dumps(hist[-100:]))
    except Exception:
        pass


def get_prediction_history(limit: int = 20) -> SimpleDataFrame:
    """Return the most recent prediction entries."""
    hist = _load_prediction_history()
    hist = hist[-limit:]
    return SimpleDataFrame(hist)


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


def _event_proximity(ticker: str) -> float:
    client = EdgarClient()
    try:
        filings = client.recent_filings(ticker)
    except Exception:
        return 0.0
    if not filings:
        return 0.0
    # simple measure: recent filing exists -> 1.0
    return 1.0


def _trend_score(ticker: str) -> float:
    client = GoogleTrendsClient()
    try:
        series = client.interest_over_time(ticker)
    except Exception:
        return 0.0
    if not series:
        return 0.0
    return float(series[-1]) / (sum(series) / len(series)) if series else 0.0


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

    history = _load_history()
    rows = []
    y = []
    valid_tickers = []
    for ticker, plist in by_ticker.items():
        feats = compute_attention_vector(plist)
        hist_vals = history.get(ticker, [])
        feats["anomaly"] = anomaly_score(feats["post_count"], hist_vals)
        feats["event"] = _event_proximity(ticker)
        feats["trend"] = _trend_score(ticker)
        history.setdefault(ticker, []).append(feats["post_count"])
        history[ticker] = history[ticker][-30:]
        rows.append(feats)
        valid_tickers.append(ticker)
        y.append(labels.get(ticker, 0))

    _save_history(history)

    df = SimpleDataFrame([dict(row) for row in rows])
    df["ticker"] = valid_tickers
    df["label"] = y
    X = df.drop(["ticker", "label"])
    model = PriceDirectionModel(method="logit")
    model.fit(X._records, df["label"])
    probs = model.predict_proba(X._records)
    preds = model.predict(X._records)
    result_rows = []
    for t, p, pr, f in zip(valid_tickers, probs, preds, rows):
        vol = min(1.0, f.get("velocity", 0.0) / 10.0)
        result_rows.append(
            {
                "ticker": t,
                "direction": "\u2B06" if pr else "\u2B07",
                "confidence": round(p, 2),
                "volatility": vol,
                "anomaly": round(f.get("anomaly", 0.0), 2),
                "actual": labels.get(t, 0),
            }
        )

    result_rows.sort(key=lambda r: r["anomaly"], reverse=True)
    result_rows = result_rows[:5]
    _save_prediction_history(result_rows)
    return SimpleDataFrame(result_rows)


def run() -> None:
    """Execute a minimal data flow using live data sources."""
    df = get_predictions()
    for row in df.to_dict(orient="records"):
        print(
            f"{row['ticker']}: {row['direction']} conf={row['confidence']} "
            f"anom={row['anomaly']} vol={row['volatility']:.2f}"
        )


if __name__ == "__main__":
    run()
