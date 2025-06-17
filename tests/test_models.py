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


def test_single_class_fallback():
    """When only one class is present the model should use naive mode."""
    model = PriceDirectionModel(method="logit")
    # only class 0 present
    model.fit([{"a": 1}, {"a": 2}], [0, 0])
    assert model.method == "naive"
    preds = model.predict([{"a": 3}])
    assert preds == [0]

