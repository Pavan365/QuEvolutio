"""
Classes for setting up the domain of a quantum system.
"""

# Import standard modules.
from typing import Protocol

# Import external modules.
import numpy as np

# Import local modules.
from quevolutio.core.aliases import (
    IVector,
    RVector,
    RVectors,
    RVectorSeq,
    RTensorSeq,
)


class QuantumConstants(Protocol):
    """
    Interface for representing the physical constants of a quantum system. This
    class can be extended to contain system specific constants as required.

    Attributes
    ----------
    hbar : float
        The reduced Planck constant.
    """

    hbar: float


class HilbertSpace:
    """
    Represents a discretised Hilbert space. This class aims to represent an
    abstract Hilbert space, which is agnostic of a quantum system.

    Parameters
    ----------
    num_dimensions : int
        The number of dimensions.
    num_points : IVector
        The number of sampling points to use when discretising each dimension
        in position space. This should have shape (num_dimensions).
    position_bounds : RVectors
        The position space boundaries (lower & upper) of each dimension. This
        should have shape (num_dimensions, 2). The zeroth column should contain
        the lower bounds and the first column should contain the upper bounds.

    Attributes
    ----------
    num_dimensions : int
        The number of dimensions.
    num_points : IVector
        The number of sampling points used when discretising each dimension in
        position space. This has shape (num_dimensions).
    position_bounds : RVectors
        The position space boundaries (lower & upper) of each dimension. This
        has shape (num_dimensions, 2). The zeroth column has the lower bounds
        and the first column has the upper bounds.
    position_axes : RVectorSeq
        The position space axes. This is an immutable sequence of RVector with
        length (num_dimensions).
    position_meshes : RTensorSeq
        The position space mesh-grids. These store the combinations of position
        space points, generated from np.meshgrid (sparse). This is an immutable
        sequence of RTensor with length (num_dimensions).
    position_deltas : RVector
        The spacing between points in the position space axes. This has shape
        (num_dimensions).
    """

    def __init__(
        self, num_dimensions: int, num_points: IVector, position_bounds: RVectors
    ) -> None:
        # Assign attributes.
        self.num_dimensions: int = num_dimensions
        self.num_points: IVector = num_points
        self.position_bounds: RVectors = position_bounds

        # Construct the position space axes.
        self.position_axes: RVectorSeq = []
        for i in range(self.num_dimensions):
            self.position_axes.append(
                np.linspace(
                    self.position_bounds[i, 0],
                    self.position_bounds[i, 1],
                    self.num_points[i],
                    dtype=np.float64,
                )
            )
        self.position_axes: RVectorSeq = tuple(self.position_axes)

        # Construct the position space mesh-grids.
        self.position_meshes: RTensorSeq = tuple(
            np.meshgrid(*self.position_axes, indexing="ij", sparse=True)
        )

        # Calculate the position space deltas.
        self.position_deltas: RVector = np.asarray(
            [axis[1] - axis[0] for axis in self.position_axes]
        )
