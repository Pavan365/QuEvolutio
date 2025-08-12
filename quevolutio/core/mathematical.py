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
