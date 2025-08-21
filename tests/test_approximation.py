"""
Simple tests for the quevolutio.mathematical.approximation module.
"""

# Import external modules.
import numpy as np

# Import tested modules.
import quevolutio.mathematical.approximation as approx


def test_ch_gauss_nodes():
    """
    Tests for the approx.ch_gauss_nodes function.
    """

    # Generate the Chebyshev-Gauss nodes.
    num_nodes = 20
    nodes = approx.ch_gauss_nodes(num_nodes)

    # Check that the nodes are in the interval [-1, 1], excluding boundary points.
    assert np.all(nodes > -1.0) and np.all(nodes < 1.0)

    # Check that the nodes are monotonically increasing.
    assert np.all(np.diff(nodes) > 0.0)

    # Check that the nodes are symmetric.
    assert np.allclose(nodes, -nodes[::-1])


def test_ch_lobatto_nodes():
    """
    Tests for the approx.ch_lobatto_nodes function.
    """

    # Generate the Chebyshev-Lobatto nodes.
    num_nodes = 20
    nodes = approx.ch_lobatto_nodes(num_nodes)

    # Check that the nodes are in the interval [-1, 1], including boundary points.
    assert np.all(nodes[1:-1] > -1.0) and np.all(nodes[1:-1] < 1.0)
    assert np.isclose(nodes[0], -1.0)
    assert np.isclose(nodes[-1], 1.0)

    # Check that the nodes are monotonically increasing.
    assert np.all(np.diff(nodes) > 0.0)

    # Check that the nodes are symmetric.
    assert np.allclose(nodes, -nodes[::-1])
