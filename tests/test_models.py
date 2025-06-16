from insidernet.models import PriceDirectionModel


def test_fallback(monkeypatch):
    """Model should fall back to naive when sklearn is missing."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    model = PriceDirectionModel(method="logit")
    model.fit([{"a": 1}, {"a": 2}], [1, 1])
    preds = model.predict([{"a": 3}])
    assert preds == [1]

