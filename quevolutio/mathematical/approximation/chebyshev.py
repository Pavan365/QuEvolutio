"""
Mathematical functions for performing approximations of functions using
Chebyshev polynomials (first kind).

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import external modules.
import numpy as np

# Import local modules.
import quevolutio.core.simulation as sim


def gauss_nodes(num_nodes: int) -> sim.RVector:
    """
    Calculates the Chebyshev-Gauss nodes on the interval [-1, 1]. These are the
    root of a Chebyshev polynomial, excluding the endpoints -1 and 1. The nodes
    are calculated in ascending order.

    Parameters
    ----------
    num_nodes : int
        The number of Chebyshev-Gauss nodes to calculate.

    Returns
    -------
    nodes : simulation.RVector
        The Chebyshev-Gauss nodes.
    """

    # Generate the Chebyshev-Gauss nodes.
    nodes: sim.RVector = -np.cos(
        (np.pi * (np.arange(num_nodes, dtype=np.float64) + 0.5)) / num_nodes
    )

    return nodes


def lobatto_nodes(num_nodes: int) -> sim.RVector:
    """
    Calculates the Chebyshev-Lobatto nodes on the interval [-1, 1]. These are
    the extrema of a Chebyshev polynomial, including the endpoints -1 and 1. The
    nodes are calculated in ascending order.

    Parameters
    ----------
    num_nodes : int
        The number of Chebyshev-Lobatto nodes to calculate.

    Returns
    -------
    nodes : simulation.RVector
        The Chebyshev-Lobatto nodes.
    """

    # Generate the Chebyshev-Lobatto nodes.
    nodes: sim.RVector = -np.cos(
        (np.pi * np.arange(num_nodes, dtype=np.float64)) / (num_nodes - 1)
    )

    return nodes
