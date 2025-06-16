"""Feature engineering utilities for InsiderNet v2."""
from __future__ import annotations

from typing import List, Dict

POSITIVE = {"skyrocket", "buy", "bull", "long"}
NEGATIVE = {"crash", "sell", "bear", "short"}

import numpy as np


def compute_attention_vector(posts: List[dict]) -> Dict[str, float]:
    """Compute attention-based features from a list of posts."""
    if not posts:
        return {
            "post_count": 0,
            "avg_comment_length": 0.0,
            "sentiment": 0.0,
            "diversity": 0.0,
            "velocity": 0.0,
        }

    post_count = len(posts)
    lengths = [len(p.get("text", "")) for p in posts]
    avg_length = float(np.mean(lengths))

    unique_users = {p.get("author", "") for p in posts}
    diversity = len(unique_users) / float(post_count)

    created_times = sorted(p.get("created", 0) for p in posts)
    if len(created_times) > 1:
        velocity = post_count / float(created_times[-1] - created_times[0])
    else:
        velocity = 0.0

    words = " ".join(p.get("text", "").lower() for p in posts).split()
    score = sum(1 for w in words if w in POSITIVE) - sum(1 for w in words if w in NEGATIVE)
    sentiment = score / float(len(words) or 1)

    return {
        "post_count": post_count,
        "avg_comment_length": avg_length,
        "sentiment": sentiment,
        "diversity": diversity,
        "velocity": velocity,
    }


def anomaly_score(current_value: float, history: List[float]) -> float:
    """Return a simple z-score anomaly value."""
    if not history:
        return 0.0
    mean = float(np.mean(history))
    std = float(np.std(history)) or 1.0
    return (current_value - mean) / std
