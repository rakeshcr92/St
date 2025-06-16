# InsiderNet v2

InsiderNet v2 is a human-behavior-driven market signal engine. It analyzes public
attention signals from a variety of sources to detect abnormal patterns prior
to market movements. The system focuses on modeling human behavior rather than
price history.

## Features
- **Data Sources**: Reddit, X/Twitter, SEC EDGAR filings, Google Trends, historical stock data for labels only.
- **Attention Vector Construction**: posts counts, comment length, sentiment, velocity of mentions, named entities, attention diversity and more.
- **Anomaly Detection**: daily abnormality scores using statistical techniques such as z-score or isolation forest.
- **Prediction Models**: RandomForest, XGBoost, Logistic Regression and optional sequence models.
- **Dashboard**: visualize predictions, rank tickers by anomaly strength and confidence.

This repository now fetches live data. The `insidernet` package includes:

- data source connectors for Reddit, X/Twitter, the SEC EDGAR system and
  Google Trends (API credentials required)
- feature engineering utilities for attention vectors
- a flexible modelling layer that falls back to a naive baseline when
  ``scikit-learn`` is unavailable
- a pipeline script that trains on freshly downloaded posts

Run the pipeline with your API credentials. The application automatically loads
variables from a `.env` file if present. Copy `/.env.example` to `.env` and
fill in your keys, or export them manually as environment variables:

```bash
export REDDIT_CLIENT_ID=...             # required
export REDDIT_CLIENT_SECRET=...         # required
export TWITTER_BEARER_TOKEN=...         # optional
export PRICE_API_KEY=...                # required for labels
python -m insidernet.pipeline
```

Launch the demo web app:

```bash
python -m insidernet.webapp
```

This starts a Flask server at `http://localhost:5000` showing prediction
results from the latest fetched data. The page lists the five tickers with the
highest anomaly scores along with the predicted direction (up or down), model
confidence and an estimated volatility metric.

Predictions are stored in ``prediction_history.json``. The web interface lists
recent predictions so you can compare results across runs.

The pipeline uses stratified train/test splitting. If only a few posts are
available for a ticker, it trains on all available data. When the resulting
labels contain only a single class the model automatically falls back to a
naïve probability-based approach so training never fails.

## Requirements

Install dependencies before running the pipeline or tests. ``scikit-learn`` is
optional; when installed the model will train a real logistic regression
classifier, otherwise a simple probabilistic baseline is used.

```bash
pip install -r requirements.txt  # core requirements
# install optional ML extras for better models
pip install -r requirements.txt[ml]
```

The demo relies on `python-dotenv` for loading environment variables. No heavy
scientific packages are required.


