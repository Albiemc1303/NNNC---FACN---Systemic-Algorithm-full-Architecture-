"""Simple import test for law_enforcer package."""

def test_law_enforcer_import():
    import importlib
    mod = importlib.import_module("law_enforcer.combined_cosmic_aspectual_integration_interface")
    assert hasattr(mod, 'check_compliance')
