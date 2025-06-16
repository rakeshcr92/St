from insidernet.pipeline import run, get_predictions


def test_run(tmp_path, monkeypatch):
    # Ensure pipeline runs without error and prints predictions
    run()


def test_get_predictions():
    df = get_predictions()
    assert not df.empty
    assert set(df.columns) == {"ticker", "score"}
