"""
Mathematical functions for transforming the domains of mathematical objects
using affine transformations.
"""

# Import external modules.
import numpy as np

# Import local modules.
from quevolutio.core.aliases import RTensor


def rescale_tensor(tensor: RTensor, a: float, b: float) -> tuple[RTensor, float, float]:
    """
    Rescales the domain of a real-valued tensor to the interval [a, b] using an
    affine transformation. This function also returns the scaling and shifting
    factors used to perform the affine transformation.

    Parameters
    ----------
    tensor : RTensor
        The real-valued tensor to rescale.
    a : float
        The lower bound of the target domain.
    b : float
        The upper bound of the target domain.

    Returns
    -------
    tensor_rs : RTensor
        The rescaled real-valued tensor.
    scale : float
        The scale factor used to perform the affine transformation.
    shift : float
        The shift factor used to perform the affine transformation.
    """

    # Store the domain of the tensor.
    tensor_min: float = np.min(tensor)
    tensor_max: float = np.max(tensor)

    # Calculate the scale and shift factors.
    scale: float = (b - a) / (tensor_max - tensor_min)
    shift: float = ((a * tensor_max) - (b * tensor_min)) / (tensor_max - tensor_min)

    # Rescale the tensor.
    tensor_rs: RTensor = (scale * tensor) + shift

    return tensor_rs, scale, shift
