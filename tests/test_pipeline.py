from insidernet.pipeline import run


def test_run(tmp_path, monkeypatch):
    # Ensure pipeline runs without error and prints predictions
    run()
