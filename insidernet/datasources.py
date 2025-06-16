"""Data source connectors for InsiderNet v2.

This module defines lightweight wrappers for external data sources such as Reddit,
X/Twitter, SEC EDGAR filings, Google Trends and price APIs. The implementations
are placeholders and are intended to be expanded with actual API calls and
authentication as needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
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
        """Initialize a Reddit API client using PRAW or another library."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        # TODO: instantiate actual API client

    def fetch_posts(self, subreddit: str, limit: int = 100) -> List[RedditPost]:
        """Fetch recent posts from a subreddit.

        This implementation returns an empty list as a placeholder.
        """
        # TODO: use API client to gather data
        return []


class TwitterClient:
    def __init__(self, bearer_token: str) -> None:
        self.bearer_token = bearer_token
        # TODO: instantiate tweepy client

    def search(self, query: str, max_results: int = 100) -> List[dict[str, Any]]:
        """Search recent tweets.

        Currently returns an empty list as a placeholder.
        """
        # TODO: use API client to gather data
        return []


class EdgarClient:
    def __init__(self) -> None:
        pass

    def recent_filings(self, ticker: str) -> List[str]:
        """Retrieve a list of recent filing URLs for a ticker."""
        # TODO: implement EDGAR scraping
        return []


class GoogleTrendsClient:
    def __init__(self) -> None:
        pass

    def interest_over_time(self, query: str) -> List[float]:
        """Return normalized interest for the query."""
        # TODO: implement Google Trends retrieval
        return []


class PriceClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def historical_prices(self, ticker: str, days: int = 30) -> List[float]:
        """Return historical prices for label generation."""
        # TODO: implement price retrieval
        return []


def load_local_reddit(path: str) -> List[RedditPost]:
    """Load reddit posts from a local JSON file for offline experiments."""
    data = json.loads(Path(path).read_text())
    posts = [RedditPost(**item) for item in data]
    return posts
