from insidernet.features import compute_attention_vector

SAMPLE = [
    {"text": "I think $AAPL will skyrocket", "author": "u1", "created": 1},
    {"text": "Sell TSLA", "author": "u2", "created": 2},
]

def test_attention_vector_keys():
    vec = compute_attention_vector(SAMPLE)
    assert set(vec.keys()) == {
        "post_count",
        "avg_comment_length",
        "sentiment",
        "diversity",
        "velocity",
        "entity_count",
    }
    assert vec["post_count"] == 2
