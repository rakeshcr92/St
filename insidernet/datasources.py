"""Data source connectors for InsiderNet v2.

This module now includes simple implementations for fetching live data from
public APIs. The connectors rely on third‑party libraries such as ``praw`` and
``requests``; they are imported lazily so that the package can be imported even
when those dependencies are missing (e.g. in the test environment).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
import os
import re
import json
from pathlib import Path



@dataclass
class RedditPost:
    ticker: str
    text: str
    created: float
    author: str


class RedditClient:
    def __init__(self, client_id: str, client_secret: str, user_agent: str) -> None:
        """Store credentials for lazy initialization."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import praw  # type: ignore
            except ImportError as exc:
                raise RuntimeError("praw package is required for RedditClient") from exc
            self._client = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )

    def fetch_posts(self, subreddit: str, limit: int = 100) -> List[RedditPost]:
        """Fetch recent posts from a subreddit."""
        self._ensure_client()
        posts: List[RedditPost] = []
        ticker_pattern = re.compile(r"\$([A-Z]{1,5})")
        for submission in self._client.subreddit(subreddit).new(limit=limit):
            text = f"{submission.title} {getattr(submission, 'selftext', '')}"
            match = ticker_pattern.search(text)
            ticker = match.group(1) if match else ""
            posts.append(
                RedditPost(
                    ticker=ticker,
                    text=text,
                    created=float(submission.created_utc),
                    author=str(submission.author),
                )
            )
        return posts


class TwitterClient:
    def __init__(self, bearer_token: str) -> None:
        self.bearer_token = bearer_token

    def search(self, query: str, max_results: int = 100) -> List[dict[str, Any]]:
        """Search recent tweets using the Twitter v2 API."""
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError("requests package is required for TwitterClient") from exc
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {
            "query": query,
            "max_results": str(max_results),
            "tweet.fields": "author_id,created_at",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        return resp.json().get("data", [])


class EdgarClient:
    def __init__(self, user_agent: str = "InsiderNet") -> None:
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError("requests package is required for EdgarClient") from exc
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def recent_filings(self, ticker: str) -> List[str]:
        """Retrieve a list of recent filing accession numbers for a ticker."""
        url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
        resp = self.session.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("filings", {}).get("recent", {}).get("accessionNumber", [])


class GoogleTrendsClient:
    def __init__(self) -> None:
        self._trend = None

    def interest_over_time(self, query: str) -> List[float]:
        """Return normalized search interest for the query."""
        if self._trend is None:
            try:
                from pytrends.request import TrendReq  # type: ignore
            except ImportError as exc:
                raise RuntimeError("pytrends package is required for GoogleTrendClient") from exc
            self._trend = TrendReq()
        df = self._trend.get_interest_over_time(query)
        if query in df:
            return df[query].dropna().tolist()
        return []


class PriceClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def historical_prices(self, ticker: str, days: int = 30) -> List[float]:
        """Return historical closing prices for label generation."""
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError("requests package is required for PriceClient") from exc
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": self.api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json().get("Time Series (Daily)", {})
        prices = [float(v["4. close"]) for v in list(data.values())[:days]]
        prices.reverse()
        return prices


def load_local_reddit(path: str) -> List[RedditPost]:
    """Load reddit posts from a local JSON file for offline experiments."""
    data = json.loads(Path(path).read_text())
    posts = [RedditPost(**item) for item in data]
    return posts
