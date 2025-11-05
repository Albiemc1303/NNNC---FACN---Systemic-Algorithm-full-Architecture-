import numpy as np

from reference_impl import NNNC


def test_facn_basic():
    facn = NNNC.FACN(input_size=16)
    inp = np.random.randn(16)
    out = facn.process_input(inp)
    # Basic shape and keys
    assert "output" in out
    assert isinstance(out["output"], np.ndarray)
    assert out["output"].shape == (10,)
    assert len(facn.hidden_layers[0]) == 32
    assert len(facn.hidden_layers[1]) == 16


def test_cogflux_efficiency_and_vector():
    cf = NNNC.CogFluxEngine(num_dimensions=6)
    eta = cf.calculate_global_efficiency()
    total = cf.calculate_total_capacity()
    vec = cf.get_capacity_vector()
    assert isinstance(eta, float)
    assert isinstance(total, float)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == 6

def test_cogflux_zero_dimensions():
    cf = NNNC.CogFluxEngine(num_dimensions=0)
    eta = cf.calculate_global_efficiency()
    total = cf.calculate_total_capacity()
    vec = cf.get_capacity_vector()
    assert isinstance(eta, float)
    assert isinstance(total, float)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == 0


def test_cogflux_edge_case_single_node():
    cf = NNNC.CogFluxEngine(num_dimensions=1)
    # With a single node, global efficiency should be zero
    assert cf.calculate_global_efficiency() == 0.0
import os
import sys
import numpy as np

# Make reference_impl importable from tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference_impl"))

import NNNC


def test_facn_basic():
    facn = NNNC.FACN(input_size=16)
    inp = np.random.randn(16)
    out = facn.process_input(inp)
    # Basic shape and keys
    assert "output" in out
    assert isinstance(out["output"], np.ndarray)
    assert out["output"].shape == (10,)
    assert len(facn.hidden_layers[0]) == 32
    assert len(facn.hidden_layers[1]) == 16


def test_cogflux_efficiency_and_vector():
    cf = NNNC.CogFluxEngine(num_dimensions=6)
    eta = cf.calculate_global_efficiency()
    total = cf.calculate_total_capacity()
    vec = cf.get_capacity_vector()
    assert isinstance(eta, float)
    assert isinstance(total, float)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == 6


def test_cogflux_edge_case_single_node():
    cf = NNNC.CogFluxEngine(num_dimensions=1)
    # With a single node, global efficiency should be zero
    assert cf.calculate_global_efficiency() == 0.0
