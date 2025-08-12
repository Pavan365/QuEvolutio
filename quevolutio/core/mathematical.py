"""
Core mathematical functions for implementing the Semi-Global propagation scheme
for the time-dependent Schrödinger equation.

Abbreviations
-------------
+ ch : Chebyshev
+ ne : Newtonian

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import local modules.
from typing import Optional, Union, cast

# Import external modules.
import numpy as np
from scipy.fft import dct

# Import local modules.
import quevolutio.core.simulation as sim


def ch_coefficients(
    function_values: Union[sim.GTensor, sim.GTensors],
    dct_type: int,
) -> Union[sim.GTensor, sim.GTensors]:
    """
    Calculates the coefficients for a Chebyshev expansion of a function through
    the discrete cosine transform (DCT). The function being expanded should be
    evaluated on either Chebyshev-Gauss or Chebyshev-Lobatto nodes.

    + DCT-I     : Chebyshev-Lobatto
    + DCT-II    : Chebyshev-Gauss

    Parameters
    ----------
    function_values : Union[sim.GTensor, sim.GTensors]
        The values of the function evaluated on either Chebyshev-Gauss or
        Chebyshev-Lobatto nodes. The expansion is taken to be along the zeroth
        axis of the function values.
    type : int
        The type of discrete cosine transform (DCT) to use. DCT-I should be
        used for functions evaluated on Chebyshev-Lobatto nodes, and DCT-II
        for functions evaluated on Chebyshev-Gauss nodes.

    Returns
    -------
    coefficients : Union[sim.GTensor, sim.GTensors]
        The Chebyshev expansion coefficients.
    """

    # Check parameters.
    # TODO: Rewrite error messages.
    if dct_type not in [1, 2]:
        raise ValueError("invalid dct_type")

    # Store the number of expansion terms.
    order: int = function_values.shape[0]

    # Perform the discrete cosine transform (DCT).
    coefficients: Union[sim.GTensor, sim.GTensors] = cast(
        Union[sim.GTensor, sim.GTensors],
        dct(function_values, type=dct_type, axis=0, norm=None),
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


def ch_expansion(
    state: sim.GTensor,
    operator_rs: sim.Operator,
    controls: Optional[sim.Controls],
    coefficients: sim.GVector,
) -> sim.GTensor:
    """
    Calculates the Chebyshev expansion of an operator acting on a state through
    the recursion relation for the Chebyshev polynomials of the first kind. The
    number of expansion terms is taken to be the number of coefficients.

    Parameters
    ----------
    state : simulation.GVector
        The state being acted upon by the operator.
    operator_rs : sim.Operator
        The operator being expanded. This should be a function that returns the
        action of the operator on the state, rescaled to the domain [-1, 1].
    controls : Optional[sim.Controls]
        The controls that determine the structure of the operator. This should
        be passed if the operator has explicit time dependence.
    coefficients : simulation.GVector
        The Chebyshev expansion coefficients. The coefficients are expected to
        be the cosine transformed values of values generated from evaluating a
        function of the operator on Chebyshev-Gauss or Chebyshev-Lobatto nodes.

    Returns
    -------
    expansion : sim.GTensor
        The expansion term resulting from the Chebyshev expansion of the
        operator acting on the state.
    """

    # Store the number of expansion terms.
    order: int = coefficients.shape[0]

    # Calculate the first two Chebyshev expansion polynomials.
    polynomial_minus_2: sim.GTensor = state
    polynomial_minus_1: sim.GTensor = operator_rs(state, controls)

    # Construct the starting expansion term.
    expansion: sim.GTensor = (coefficients[0] * polynomial_minus_2) + (
        coefficients[1] * polynomial_minus_1
    )

    # Construct the complete expansion.
    for i in range(2, order):
        polynomial_n: sim.GTensor = (
            2 * operator_rs(polynomial_minus_1, controls)
        ) - polynomial_minus_2
        expansion += coefficients[i] * polynomial_n

        polynomial_minus_2: sim.GTensor = polynomial_minus_1
        polynomial_minus_1: sim.GTensor = polynomial_n

    return expansion


def ch_gauss_nodes(num_nodes: int) -> sim.RVector:
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
    nodes : sim.RVector
        The Chebyshev-Gauss nodes.
    """

    # Generate the Chebyshev-Gauss nodes.
    nodes: sim.RVector = -np.cos(
        (np.pi * (np.arange(num_nodes, dtype=np.float64) + 0.5)) / num_nodes
    )

    return nodes


def ch_lobatto_nodes(num_nodes: int) -> sim.RVector:
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
    nodes : sim.RVector
        The Chebyshev-Lobatto nodes.
    """

    # Generate the Chebyshev-Lobatto nodes.
    nodes: sim.RVector = -np.cos(
        (np.pi * np.arange(num_nodes, dtype=np.float64)) / (num_nodes - 1)
    )

    return nodes


def ch_ta_conversion(order: int, time_min: float, time_max: float) -> sim.RMatrix:
    """
    Calculates the square (lower triangular) conversion matrix for converting
    Chebyshev expansion coefficients to Taylor-like derivatives across a time
    interval. The matrix is intended for use with coefficients resulting from
    sampling a function on Chebyshev-Lobatto nodes.

    Parameters
    ----------
    order : int
        The size of the conversion matrix, which corresponds to the highest
        Taylor-like derivative produced from the matrix.
    time_min : float
        The lower bound of the time interval.
    time_max : float
        The upper bound of the time interval.

    Returns
    -------
    conversion : sim.RMatrix
        The conversion matrix.
    """

    # Calculate time interval information.
    time_sum: float = time_min + time_max
    time_dt: float = time_max - time_min

    # Calculate recurring coefficients.
    a: float = (2 * time_sum) / time_dt
    b: float = 4 / time_dt

    # Set up the conversion matrix.
    conversion: sim.RMatrix = np.zeros((order, order), dtype=np.float64)
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


def ne_coefficients(
    nodes: sim.RVector,
    function_values: sim.GTensors,
) -> sim.GTensor:
    """
    Calculates the coefficients for a Newtonian interpolation expansion of a
    function through building a divided differences table, which is upper
    triangular and contains the coefficients on the main diagonal. The function
    being expanded should be evaluated on nodes in the target domain.

    Parameters
    ----------
    nodes : sim.RVector
        The nodes in the target domain that the function being expanded is
        evaluated on.
    function_values : sim.GTensors
        The values of the function being expanded evaluated on the nodes in
        the target domain. This is expected to be at least two dimensional,
        where the expansion is taken to be along the zeroth axis.

    Returns
    -------
    coefficients : sim.GTensor
        The Newtonian interpolation coefficients.
    """

    # Store the number of expansion terms.
    order: int = nodes.shape[0]

    # Set up the divided differences tables.
    tables: sim.GTensor = cast(
        sim.GTensor,
        np.zeros(
            (order, order, *function_values.shape[1:]), dtype=function_values.dtype
        ),
    )
    tables[0] = function_values

    # Construct the divided differences tables (upper triangular).
    for i in range(1, order):
        for j in range(i, order):
            tables[i, j] = (tables[i - 1, j] - tables[i - 1, j - 1]) / (
                nodes[j] - nodes[j - i]
            )

    # Store the Newtonian interpolation coefficients.
    coefficients: sim.GTensor = tables[
        np.arange(order, dtype=np.int32), np.arange(order, dtype=np.int32)
    ]

    return coefficients


def rescale_tensor(
    tensor: sim.RTensor, a: float, b: float
) -> tuple[sim.RTensor, float, float]:
    """
    Rescales the domain of a real-valued tensor to the interval [a, b] using an
    affine transformation. This function also returns the factors used to
    perform the affine transformation.

    Returns
    -------
    tensor_rs : sim.RTensor
        The rescaled real-valued tensor.
    scale : float
        The scale factor used in the affine transformation.
    shift : float
        The shift factor used in the affine transformation.
    """

    # Get the domain of the tensor.
    tensor_min: float = np.min(tensor)
    tensor_max: float = np.max(tensor)

    # Calculate the scale and shift factors for the affine transformation.
    scale: float = (b - a) / (tensor_max - tensor_min)
    shift: float = ((a * tensor_max) - (b * tensor_min)) / (tensor_max - tensor_min)

    # Rescale the tensor.
    tensor_rs: sim.RTensor = (scale * tensor) + shift

    return tensor_rs, scale, shift
