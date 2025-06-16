"""Visualization utilities for InsiderNet v2."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(tickers: Sequence[str], scores: Sequence[float]) -> None:
    """Display a simple bar chart of prediction scores."""
    df = pd.DataFrame({"ticker": tickers, "score": scores})
    df.sort_values("score", ascending=False, inplace=True)
    df.plot.bar(x="ticker", y="score", legend=False)
    plt.ylabel("Confidence")
    plt.title("InsiderNet Predictions")
    plt.tight_layout()
    plt.show()
