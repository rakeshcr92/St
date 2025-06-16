# InsiderNet v2

InsiderNet v2 is a human-behavior-driven market signal engine. It analyzes public attention signals from a variety of sources to detect abnormal patterns prior to market movements. The system focuses on modeling human behavior rather than price history.

## Features
- **Data Sources**: Reddit, X/Twitter, SEC EDGAR filings, Google Trends, historical stock data for labels only.
- **Attention Vector Construction**: posts counts, comment length, sentiment, velocity of mentions, named entities, attention diversity and more.
- **Anomaly Detection**: daily abnormality scores using statistical techniques such as z-score or isolation forest.
- **Prediction Models**: RandomForest, XGBoost, Logistic Regression and optional sequence models.
- **Dashboard**: visualize predictions, rank tickers by anomaly strength and confidence.

This repository contains the initial project structure and placeholder modules.

