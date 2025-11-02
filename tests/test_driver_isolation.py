"""Simple import test for nnnc_core.meta_cognitive_driver."""

def test_meta_driver_import():
    import importlib
    mod = importlib.import_module("nnnc_core.meta_cognitive_driver")
    assert hasattr(mod, 'drive')
