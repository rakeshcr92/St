from insidernet.pipeline import run, get_predictions
from insidernet import pipeline
from insidernet import datasources
from insidernet.datasources import RedditPost

SAMPLE_POST = RedditPost(
    ticker="AAPL",
    text="AAPL to the moon",
    created=0.0,
    author="user1",
)


def _stub_fetch_posts(self, subreddit, limit=50):
    return [SAMPLE_POST]


def _stub_search(self, query, max_results=50):
    return [{"text": "great", "author_id": "u", "created_at": "2020-01-01T00:00:00Z"}]


def _stub_prices(self, ticker, days=4):
    return [1.0, 1.5]


def _stub_event_proximity(ticker):
    return 0.0


def _stub_trend_score(ticker):
    return 0.0


def test_run(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("PRICE_API_KEY", "key")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "token")
    monkeypatch.setattr(datasources.RedditClient, "fetch_posts", _stub_fetch_posts)
    monkeypatch.setattr(datasources.TwitterClient, "search", _stub_search)
    monkeypatch.setattr(datasources.PriceClient, "historical_prices", _stub_prices)
    monkeypatch.setattr(pipeline, "_event_proximity", _stub_event_proximity)
    monkeypatch.setattr(pipeline, "_trend_score", _stub_trend_score)
    run()


def test_get_predictions(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("PRICE_API_KEY", "key")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "token")
    monkeypatch.setattr(datasources.RedditClient, "fetch_posts", _stub_fetch_posts)
    monkeypatch.setattr(datasources.TwitterClient, "search", _stub_search)
    monkeypatch.setattr(datasources.PriceClient, "historical_prices", _stub_prices)
    monkeypatch.setattr(pipeline, "_event_proximity", _stub_event_proximity)
    monkeypatch.setattr(pipeline, "_trend_score", _stub_trend_score)
    df = get_predictions()
    assert not df.empty
    assert set(df.columns) == {
        "ticker",
        "direction",
        "confidence",
        "volatility",
        "anomaly",
        "actual",
    }


def test_prediction_history(monkeypatch, tmp_path):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("PRICE_API_KEY", "key")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "token")
    monkeypatch.setattr(datasources.RedditClient, "fetch_posts", _stub_fetch_posts)
    monkeypatch.setattr(datasources.TwitterClient, "search", _stub_search)
    monkeypatch.setattr(datasources.PriceClient, "historical_prices", _stub_prices)
    monkeypatch.setattr(pipeline, "_event_proximity", _stub_event_proximity)
    monkeypatch.setattr(pipeline, "_trend_score", _stub_trend_score)
    monkeypatch.setattr(pipeline, "PRED_HISTORY_FILE", tmp_path / "hist.json")

    get_predictions()
    hist_df = pipeline.get_prediction_history()
    assert not hist_df.empty

