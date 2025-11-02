"""Simple import test for cosmic_laws.systemic_evolution_algorithm."""

def test_evolution_import():
    import importlib
    mod = importlib.import_module("cosmic_laws.systemic_evolution_algorithm")
    assert hasattr(mod, 'systemic_evolution_algorithm')
