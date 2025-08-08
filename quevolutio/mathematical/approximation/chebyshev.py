"""
Mathematical functions for performing approximations of functions using
Chebyshev polynomials (first kind).

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import standard modules.
from typing import cast

# Import external modules.
import numpy as np
from scipy.fft import dct

# Import local modules.
import quevolutio.core.simulation as sim


def coefficients(
    function_values: sim.GTensor,
    dct_type: int,
) -> sim.GTensor:
    """
    Calculates the coefficients for a Chebyshev expansion of a function through
    the discrete cosine transform (DCT). The function being expanded should be
    evaluated on either Chebyshev-Gauss or Chebyshev-Lobatto nodes.

    + DCT-I     : Chebyshev-Lobatto
    + DCT-II    : Chebyshev-Gauss

    Parameters
    ----------
    function_values : sim.GTensor
        The values of the function evaluated on either Chebyshev-Gauss or
        Chebyshev-Lobatto nodes. If the function is multi-dimensional, the
        expansion is taken to be along the zeroth axis.
    type : int
        The type of discrete cosine transform (DCT) to use. DCT-I should be
        used for functions evaluated on Chebyshev-Lobatto nodes, and DCT-II
        for functions evaluated on Chebyshev-Gauss nodes.

    Returns
    -------
    coefficients : sim.GTensor
        The Chebyshev expansion coefficients.
    """

    if dct_type not in [1, 2]:
        raise ValueError("invalid DCT type")

    # Store the number of expansion terms.
    order: int = function_values.shape[0]

    # Perform the discrete cosine transform (DCT).
    coefficients: sim.GTensor = cast(
        sim.GTensor, dct(function_values, type=dct_type, axis=0, norm=None)
    )

    # Normalisation for DCT-I.
    if dct_type == 1:
        coefficients /= order - 1
        coefficients[0] /= 2
        coefficients[-1] /= 2

    # Normalisation for DCT-II.
    elif dct_type == 2:
        coefficients /= order
        coefficients[0] /= 2

    return coefficients


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
