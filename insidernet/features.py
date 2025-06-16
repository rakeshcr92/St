"""Feature engineering utilities for InsiderNet v2."""
from __future__ import annotations

from typing import List, Dict

import numpy as np


def compute_attention_vector(posts: List[dict]) -> Dict[str, float]:
    """Compute basic attention features from a list of posts.

    Parameters
    ----------
    posts: List of post dictionaries with keys such as ``text`` and ``author``.

    Returns
    -------
    dict
        Feature dictionary containing placeholder values.
    """
    if not posts:
        return {
            "post_count": 0,
            "avg_comment_length": 0.0,
            "sentiment": 0.0,
        }
    post_count = len(posts)
    avg_length = float(np.mean([len(p.get("text", "")) for p in posts]))
    # TODO: integrate sentiment analyzer
    sentiment = 0.0
    return {
        "post_count": post_count,
        "avg_comment_length": avg_length,
        "sentiment": sentiment,
    }


def anomaly_score(current_value: float, history: List[float]) -> float:
    """Return a simple z-score anomaly value."""
    if not history:
        return 0.0
    mean = float(np.mean(history))
    std = float(np.std(history)) or 1.0
    return (current_value - mean) / std
