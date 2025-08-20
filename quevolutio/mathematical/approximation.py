"""
Functions for approximating mathematical functions and operators using
polynomials.

Abbreviations
-------------
+ ch : Chebyshev
+ ne : Newtonian
+ ta : Taylor

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import external modules.
import numpy as np

# Import local modules.
from quevolutio.core.aliases import RVector


def ch_gauss_nodes(num_nodes: int) -> RVector:
    """
    Generates the Chebyshev-Gauss nodes on the interval [-1, 1]. These are the
    roots of a Chebyshev polynomial (first kind), excluding the boundary points
    -1 and 1. The nodes are returned in ascending order.

    Parameters
    ----------
    num_nodes : int
        The number of Chebyshev-Gauss nodes to generate.

    Returns
    -------
    nodes : RVector
        The Chebyshev-Gauss nodes.
    """

    # Generate the Chebyshev-Gauss nodes.
    nodes: RVector = -np.cos(
        (np.pi * (np.arange(num_nodes, dtype=np.float64) + 0.5)) / num_nodes
    )

    return nodes


def ch_lobatto_nodes(num_nodes: int) -> RVector:
    """
    Generates the Chebyshev-Lobatto nodes on the interval [-1, 1]. These are
    the extrema of a Chebyshev polynomial (first kind), including the boundary
    points -1 and 1. The nodes are returned in ascending order.

    Parameters
    ----------
    num_nodes : int
        The number of Chebyshev-Lobatto nodes to generate.

    Returns
    -------
    nodes : RVector
        The Chebyshev-Lobatto nodes.
    """

    # Generate the Chebyshev-Lobatto nodes.
    nodes: RVector = -np.cos(
        (np.pi * np.arange(num_nodes, dtype=np.float64)) / (num_nodes - 1)
    )

    return nodes
