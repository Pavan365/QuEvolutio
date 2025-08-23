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
from typing import Optional, Union, cast

# Import external modules.
import numpy as np
from scipy.fft import dct
from scipy.special import jv

# Import local modules.
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RMatrix,
    GTensor,
    GTensors,
)
from quevolutio.core.tdse import Controls, Operator


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
    function_nodes: Union[GTensor, GTensors], dct_type: int
) -> Union[GTensor, GTensors]:
    """
    Calculates the expansion coefficients of a function being approximated
    using Chebyshev polynomials (first kind) using the discrete cosine
    transform (DCT). The function being approximated should be evaluated on
    either Chebyshev-Gauss or Chebyshev-Lobatto quadrature nodes, in descending
    order.

    + DCT-I     : Chebyshev-Lobatto
    + DCT-II    : Chebyshev-Gauss

    Parameters
    ----------
    function_nodes : Union[GTensor, GTensors]
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
    order: int = function_nodes.shape[0]

    # Perform the discrete cosine transform (DCT).
    coefficients: Union[GTensor, GTensors] = cast(
        Union[GTensor, GTensors], dct(function_nodes, axis=0, type=dct_type, norm=None)
    )

    # Perform normalisation for DCT-I.
    if dct_type == 1:
        coefficients /= order - 1
        coefficients[0] /= 2
        coefficients[-1] /= 2

    # Perform normalisation for DCT-II.
    if dct_type == 2:
        coefficients /= order
        coefficients[0] /= 2

    return coefficients


def ch_expansion(
    state: GTensor,
    operator: Operator,
    coefficients: GVector,
    controls: Optional[Controls] = None,
) -> GTensor:
    """
    Calculates the Chebyshev expansion of an operator acting on a state through
    the recursion relation of Chebyshev polynomials (first kind). The number of
    expansion terms is taken to be the number of coefficients.

    Parameters
    ----------
    state : GTensor
        The state being acted on by the operator.
    operator : Operator
        The operator being approximated. This should be a callable that returns
        the action of the operator on a state, rescaled to the domain [-1, 1].
    coefficients : GVector
        The Chebyshev expansion coefficients. The coefficients are expected to
        be the cosine transformed values of values generated from evaluating a
        function of the operator on Chebyshev-Gauss or Chebyshev-Lobatto nodes.
    controls : Optional[Controls]
        The controls that determine the structure of the operator. This should
        be passed if the operator has explicit time dependence.

    Returns
    -------
    expansion : GTensor
        The expansion term resulting from the Chebyshev expansion of the
        operator acting on the state.
    """

    # Store the number of expansion terms.
    order: int = coefficients.shape[0]

    # Calculate the first two Chebyshev expansion polynomials.
    polynomial_minus_2: GTensor = state
    polynomial_minus_1: GTensor = operator(state, controls)

    # Construct the starting expansion term.
    expansion: GTensor = (coefficients[0] * polynomial_minus_2) + (
        coefficients[1] * polynomial_minus_1
    )

    # Construct the complete expansion.
    for i in range(2, order):
        polynomial_n: GTensor = (
            2 * operator(polynomial_minus_1, controls)
        ) - polynomial_minus_2
        expansion += coefficients[i] * polynomial_n

        polynomial_minus_2: GTensor = polynomial_minus_1
        polynomial_minus_1: GTensor = polynomial_n

    return expansion


def ch_ta_conversion(order: int, time_min: float, time_max: float) -> RMatrix:
    """
    Calculates the square (lower triangular) conversion matrix for converting
    Chebyshev expansion coefficients to Taylor-like derivatives across a time
    interval. The matrix is intended for use with coefficients resulting from
    sampling a function on Chebyshev-Lobatto nodes.

    Parameters
    ----------
    order : int
        The size of the conversion matrix. This corresponds to the highest
        Taylor-like derivative produced from the matrix.
    time_min : float
        The lower bound of the time interval.
    time_max : float
        The upper bound of the time interval.

    Returns
    -------
    coefficients : RMatrix
        The conversion matrix.

    Notes
    -----
    The conversion matrix should be transposed when applied to the Chebyshev
    expansion coefficients.
    """

    # Calculate time interval information.
    time_sum: float = time_min + time_max
    time_dt: float = time_max - time_min

    # Calculate recurring coefficients.
    a: float = (2 * time_sum) / time_dt
    b: float = 4 / time_dt

    # Set up the conversion matrix.
    conversion: RMatrix = np.zeros((order, order), dtype=np.float64)
    conversion[0, 0] = 1.0

    conversion[1, 0] = -time_sum / time_dt
    conversion[1, 1] = 2 / time_dt

    # Construct the complete matrix.
    for i in range(2, order):
        # Calculate the m = 0 term (Semi-Global Appendix C.2).
        conversion[i, 0] = -(a * conversion[i - 1, 0]) - conversion[i - 2, 0]

        # Calculate the 1 <= m <= n - 2 terms (Semi-Global Appendix C.2).
        for j in range(1, i - 1):
            conversion[i, j] = (
                (b * conversion[i - 1, j - 1])
                - (a * conversion[i - 1, j])
                - conversion[i - 2, j]
            )

        # Calculate the m = n - 1 term (Semi-Global Appendix C.2).
        conversion[i, i - 1] = (b * conversion[i - 1, i - 2]) - (
            a * conversion[i - 1, i - 1]
        )

        # Calculate the m = n term (Semi-Global Appendix C.2).
        conversion[i, i] = b * conversion[i - 1, i - 1]

    return conversion


def ne_coefficients(nodes: RVector, function_nodes: GTensors) -> GTensor:
    """
    Calculates the expansion coefficients of a function being approximated
    using Newtonian interpolation polynomials using divided differences. The
    function being approximated should be evaluated on nodes in the target
    domain.

    Parameters
    ----------
    nodes : RVector
        The nodes that the function being approximated is evaluated on.
    function_nodes : GTensors
        The function being approximated evaluated on nodes in the target
        domain. This is expected to be at least 2 dimensional, where the
        expansion is taken to be along the zeroth axis.

    Returns
    -------
    coefficients : GTensor
        The Newtonian interpolation expansion coefficients.
    """

    # Store the number of expansion terms.
    order: int = nodes.shape[0]

    # Set up the divided differences tables.
    tables: GTensor = cast(
        GTensor,
        np.zeros((order, order, *function_nodes.shape[1:]), dtype=function_nodes.dtype),
    )
    tables[0] = function_nodes

    # Construct the divided differences tables (upper triangular).
    for i in range(1, order):
        for j in range(i, order):
            tables[i, j] = (tables[i - 1, j] - tables[i - 1, j - 1]) / (
                nodes[j] - nodes[j - i]
            )

    # Store the Newtonian interpolation coefficients.
    coefficients: GTensor = tables[
        np.arange(order, dtype=np.int32), np.arange(order, dtype=np.int32)
    ]

    return coefficients


def ne_ta_conversion(time_axis: RVector) -> RMatrix:
    """
    Calculates the square (lower triangular) conversion matrix for converting
    Newtonian interpolation expansion coefficients to Taylor-like derivatives
    across a time interval. In particular, the conversion matrix is defined to
    work with coefficients generated from nodes in a domain of length four. The
    number of expansion terms is taken to be the number of time points.

    Parameters
    ----------
    time_axis : RVector
        The time axis (time points) from which the Newtonian interpolation
        expansion coefficients were generated.

    Returns
    -------
    conversion : RMatrix
        The conversion matrix

    Notes
    -----
    The conversion matrix should be transposed when applied to the Newtonian
    interpolation expansion coefficients.
    """

    # Store the number of expansion terms.
    order: int = time_axis.shape[0]

    # Account for the length four domain.
    domain_factor: float = 4.0 / (time_axis[-1] - time_axis[0])

    # Set up the conversion matrix.
    conversion: RMatrix = np.zeros((order, order), dtype=np.float64)
    conversion[0, 0] = 1.0

    # Construct the complete matrix.
    for i in range(1, order):
        # Calculate the m = 0 term (Semi-Global Appendix C.1).
        conversion[i, 0] = -domain_factor * time_axis[i - 1] * conversion[i - 1, 0]

        # Calculate the 1 <= m <= n - 1 terms (Semi-Global Appendix C.1).
        for j in range(1, i):
            conversion[i, j] = domain_factor * (
                conversion[i - 1, j - 1] - (time_axis[i - 1] * conversion[i - 1, j])
            )

        # Calculate the m = n term (Semi-Global Appendix C.1).
        conversion[i, i] = domain_factor * conversion[i - 1, i - 1]

    return conversion
