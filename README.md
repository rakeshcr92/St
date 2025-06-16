# InsiderNet v2

InsiderNet v2 is a human-behavior-driven market signal engine. It analyzes public attention signals from a variety of sources to detect abnormal patterns prior to market movements. The system focuses on modeling human behavior rather than price history.

## Features
- **Data Sources**: Reddit, X/Twitter, SEC EDGAR filings, Google Trends, historical stock data for labels only.
- **Attention Vector Construction**: posts counts, comment length, sentiment, velocity of mentions, named entities, attention diversity and more.
- **Anomaly Detection**: daily abnormality scores using statistical techniques such as z-score or isolation forest.
- **Prediction Models**: RandomForest, XGBoost, Logistic Regression and optional sequence models.
- **Dashboard**: visualize predictions, rank tickers by anomaly strength and confidence.

This repository contains a lightweight prototype. The `insidernet` package
includes:

- basic data source connectors (with a local JSON loader for demo purposes)
- feature engineering functions for attention vectors
- simple ML models using scikit-learn
- a pipeline script demonstrating end-to-end training on the sample data in
  `data/sample_reddit.json`

Run the demo pipeline:

```bash
python -m insidernet.pipeline
```

## Requirements

Install dependencies before running the pipeline or tests:

```bash
pip install -r requirements.txt
```

The demo relies on optional packages such as pandas and numpy.


