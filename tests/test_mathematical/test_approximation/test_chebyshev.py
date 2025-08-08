"""
Simple test cases for the "quevolutio.mathematical.approximation.chebyshev"
module.
"""

# Import external modules.
import numpy as np

# Import package modules.
import quevolutio.core.simulation as sim
import quevolutio.mathematical.approximation.chebyshev as ch


def test_gauss_nodes():
    """
    Test that the "gauss_nodes" function performs as expected.
    """

    # Generate Chebyshev-Gauss nodes.
    num_nodes: int = 20
    nodes: sim.RVector = ch.gauss_nodes(num_nodes)

    # Check that the nodes are in ascending order and are symmetric.
    assert np.all(np.diff(nodes) > 0)
    assert np.allclose(nodes, -nodes[::-1])

    # Check that the nodes are within the interval [-1, 1], excluding boundaries.
    assert np.all((-1.0 < nodes) & (nodes < 1.0))


def test_lobatto_nodes():
    """
    Test that the "lobatto_nodes" function performs as expected.
    """

    # Generate Chebyshev-Lobatto nodes.
    num_nodes: int = 20
    nodes: sim.RVector = ch.lobatto_nodes(num_nodes)

    # Check that the nodes are in ascending order and are symmetric.
    assert np.all(np.diff(nodes) > 0)
    assert np.allclose(nodes, -nodes[::-1])

    # Check that the nodes are within the interval [-1, 1], including boundaries.
    assert np.all((-1.0 <= nodes) & (nodes <= 1.0))
