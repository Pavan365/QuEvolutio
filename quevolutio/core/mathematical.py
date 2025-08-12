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

# Import external modules.
import numpy as np

# Import local modules.
import quevolutio.core.simulation as sim


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
