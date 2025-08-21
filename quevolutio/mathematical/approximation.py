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

# Import standard modules.
from typing import Union, cast

# Import external modules.
import numpy as np
from scipy.fft import dct

# Import local modules.
from quevolutio.core.aliases import RVector, GTensor, GTensors


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


def ch_coefficients(
    function_values: Union[GTensor, GTensors], dct_type: int
) -> Union[GTensor, GTensors]:
    """
    Calculates the expansion coefficients of a function being approximated
    using Chebyshev polynomials (first kind) using the discrete cosine
    transform (DCT). The function being approximated should be evaluated on
    either Chebyshev-Gauss or Chebyshev-Lobatto quadrature nodes.


    + DCT-I     : Chebyshev-Lobatto
    + DCT-II    : Chebyshev-Gauss

    Parameters
    ----------
    function_values : Union[GTensor, GTensors]
        The function being approximated evaluated on either Chebyshev-Gauss or
        Chebyshev-Lobatto quadrature nodes. The expansion is taken to be along
        the zeroth axis.
    dct_type : int
        The type of discrete cosine transform (DCT) to use. DCT-I should be
        used for functions evaluated on Chebyshev-Lobatto nodes, and DCT-II
        for functions evaluated on Chebyshev-Gauss nodes.

    Returns
    -------
    coefficients : Union[GTensor, GTensors]
        The Chebyshev expansion coefficients.
    """

    # Store the number of expansion terms.
    order: int = function_values.shape[0]

    # Perform the discrete cosine transform (DCT).
    coefficients: Union[GTensor, GTensors] = cast(
        Union[GTensor, GTensors], dct(function_values, axis=0, type=dct_type, norm=None)
    )

    # Perform normalisation for DCT-I.
    if dct_type == 1:
        coefficients /= order - 1
        coefficients[0] = 2
        coefficients[-1] = 2

    # Perform normalisation for DCT-II.
    if dct_type == 2:
        coefficients /= order
        coefficients[0] /= 2

    return coefficients
