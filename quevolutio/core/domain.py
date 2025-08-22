"""
Classes for setting up the domain of a quantum system.
"""

# Import standard modules.
from typing import Protocol, Union, cast

# Import external modules.
import numpy as np

# Import local modules.
from quevolutio.core.aliases import (  # isort: skip
    IVector,
    RVector,
    RVectors,
    RVectorSeq,
    CTensor,
    RTensor,
    GTensor,
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


class QuantumHilbertSpace(HilbertSpace):
    """
    Represents a discretised Hilbert space. This class aims to represent a
    Hilbert space in the context of a quantum system. It builds on the abstract
    HilbertSpace class through taking into account physical attributes.

    + All attributes from the HilbertSpace class are inherited.

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
    constants : QuantumConstants
        The physical constants of the quantum system.

    Attributes
    ----------
    constants : QuantumConstants
        The physical constants of the quantum system.
    momentum_axes : RVectorSeq
        The momentum space axes. This is an immutable sequence of RVector with
        length (num_dimensions).
    momentum_meshes : RTensorSeq
        The momentum space mesh-grids. These store the combinations of momentum
        space points, generated from np.meshgrid (sparse). This is an immutable
        sequence of RTensor with length (num_dimensions).
    momentum_deltas : RVector
        The spacing between points in the momentum space axes. This has shape
        (num_dimensions).

    Notes
    -----
    The momentum axes/meshes are stored unshifted (raw np.fft.fftfreq), which
    means that the momentum values are not centred.
    """

    def __init__(
        self,
        num_dimensions: int,
        num_points: IVector,
        position_bounds: RVectors,
        constants: QuantumConstants,
    ) -> None:
        # Construct the HilbertSpace class.
        super().__init__(num_dimensions, num_points, position_bounds)

        # Assign attributes.
        self.constants: QuantumConstants = constants

        # Construct the momentum space axes.
        self.momentum_axes: RVectorSeq = []
        for i in range(self.num_dimensions):
            self.momentum_axes.append(
                2
                * np.pi
                * self.constants.hbar
                * np.fft.fftfreq(self.num_points[i], self.position_deltas[i])
            )
        self.momentum_axes: RVectorSeq = tuple(self.momentum_axes)

        # Construct the momentum space mesh-grids.
        self.momentum_meshes: RVectorSeq = tuple(
            np.meshgrid(*self.momentum_axes, indexing="ij", sparse=True)
        )

        # Calculate the momentum space deltas.
        self.momentum_deltas: RVector = np.asarray(
            [axis[1] - axis[0] for axis in self.momentum_axes]
        )

    def inner_product(self, bra: GTensor, ket: GTensor) -> complex:
        """
        Calculates the inner product between two states, over the discretised
        Hilbert space. This function handles the complex conjugation of the bra
        state.

        Parameters
        ----------
        bra : GTensor
            The bra state. This should have shape (*num_points).
        ket : GTensor
            The ket state. This should have shape (*num_points).

        Returns
        -------
        complex
            The inner product of the two states.
        """

        # Calculate the inner product integral.
        integrand: Union[complex, CTensor] = (np.conjugate(bra) * ket).astype(
            np.complex128, copy=False
        )

        for i in range(self.num_dimensions):
            integrand = np.trapezoid(integrand, axis=0, dx=self.position_deltas[i])

        return cast(complex, integrand)

    def normalise_state(self, state: GTensor) -> GTensor:
        """
        Normalises a state over the discretised Hilbert space. This function
        uses the integral definition (as opposed to vector) of the norm.

        Attributes
        ----------
        state : GTensor
            The state to normalise. This should have shape (*num_points).

        Returns
        -------
        GTensor
            The normalised state.
        """

        # Calculate the norm integral.
        norm: Union[float, RTensor] = np.abs(state) ** 2

        for i in range(self.num_dimensions):
            norm = np.trapezoid(norm, axis=0, dx=self.position_deltas[i])

        return state / np.sqrt(norm)

    @staticmethod
    def position_space(state: GTensor) -> GTensor:
        """
        Converts a state from momentum space to position space. This function
        uses orthonormal normalisation with the inverse Fourier transform.

        Parameters
        ----------
        state : GTensor
            The state in momentum space.

        Returns
        -------
        GTensor
            The state in position space.
        """

        return np.fft.ifftn(state, norm="ortho")

    @staticmethod
    def momentum_space(state: GTensor) -> GTensor:
        """
        Converts a state from position space to momentum space. This function
        uses orthonormal normalisation with the Fourier transform.

        Parameters
        ----------
        state : GTensor
            The state in position space.

        Returns
        -------
        GTensor
            The state in momentum space.
        """

        return np.fft.fftn(state, norm="ortho")


class TimeGrid:
    """
    Represents a discretised time interval.

    Parameters
    ----------
    time_min : float
        The minimum time axis value.
    time_max : float
        The maximum time axis value.
    num_points : int
        The number of sampling points to use when discretising the time axis.

    Attributes
    ----------
    time_min : float
        The minimum time axis value.
    time_max : float
        The maximum time axis value.
    num_points : int
        The number of sampling points used when discretising the time axis.
    time_axis : RVector
        The time axis. This has shape (num_points).
    time_dt : float
        The spacing between points in the time axis.
    """

    def __init__(self, time_min: float, time_max: float, num_points: int) -> None:
        # Assign attributes.
        self.time_min: float = time_min
        self.time_max: float = time_max
        self.num_points: int = num_points

        # Construct the time axis.
        self.time_axis: RVector = np.linspace(
            time_min, time_max, num_points, dtype=np.float64
        )

        # Calculate the time axis spacing.
        self.time_dt: float = self.time_axis[1] - self.time_axis[0]
