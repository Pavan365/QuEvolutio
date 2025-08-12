"""
Core classes for setting up simulations.
"""

# Import standard modules.
from typing import Callable, Mapping, Protocol, Sequence, TypeAlias, Union, cast

# Import external modules.
import numpy as np
from numpy.typing import NDArray

# Type aliases for vectors.
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IVector: TypeAlias = NDArray[np.int64]
RVector: TypeAlias = NDArray[np.float64]
CVector: TypeAlias = NDArray[np.complex128]
GVector: TypeAlias = RVector | CVector

# Type aliases for collections of vectors (NumPy).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IVectors: TypeAlias = NDArray[np.int64]
RVectors: TypeAlias = NDArray[np.float64]
CVectors: TypeAlias = NDArray[np.complex128]
GVectors: TypeAlias = RVectors | CVectors

# Type aliases for collections of vectors (Python).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IVectorSeq: TypeAlias = Sequence[NDArray[np.int64]]
RVectorSeq: TypeAlias = Sequence[NDArray[np.float64]]
CVectorSeq: TypeAlias = Sequence[NDArray[np.complex128]]
GVectorSeq: TypeAlias = RVectorSeq | CVectorSeq

# Type aliases for matrices.
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IMatrix: TypeAlias = NDArray[np.int64]
RMatrix: TypeAlias = NDArray[np.float64]
CMatrix: TypeAlias = NDArray[np.complex128]
GMatrix: TypeAlias = RMatrix | CMatrix

# Type aliases for collections of matrices (NumPy).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IMatrices: TypeAlias = NDArray[np.int64]
RMatrices: TypeAlias = NDArray[np.float64]
CMatrices: TypeAlias = NDArray[np.complex128]
GMatrices: TypeAlias = RMatrices | CMatrices

# Type aliases for collections of matrices (Python).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IMatrixSeq: TypeAlias = Sequence[NDArray[np.int64]]
RMatrixSeq: TypeAlias = Sequence[NDArray[np.float64]]
CMatrixSeq: TypeAlias = Sequence[NDArray[np.complex128]]
GMatrixSeq: TypeAlias = RMatrixSeq | CMatrixSeq

# Type aliases for tensors.
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
ITensor: TypeAlias = NDArray[np.int64]
RTensor: TypeAlias = NDArray[np.float64]
CTensor: TypeAlias = NDArray[np.complex128]
GTensor: TypeAlias = RTensor | CTensor

# Type aliases for collections of tensors (NumPy).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
ITensors: TypeAlias = NDArray[np.int64]
RTensors: TypeAlias = NDArray[np.float64]
CTensors: TypeAlias = NDArray[np.complex128]
GTensors: TypeAlias = RTensors | CTensors

# Type aliases for collections of tensors (Python).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
ITensorSeq: TypeAlias = Sequence[NDArray[np.int64]]
RTensorSeq: TypeAlias = Sequence[NDArray[np.float64]]
CTensorSeq: TypeAlias = Sequence[NDArray[np.complex128]]
GTensorSeq: TypeAlias = RTensorSeq | CTensorSeq

# Type aliases for controls (time-dependent parameters).
Control: TypeAlias = float | complex | RTensor | CTensor
Controls: TypeAlias = Control | Sequence[Control] | Mapping[str, Control]

# Type alias for a callable that returns a set of controls, given a time.
ControlsGenerator: TypeAlias = Callable[[float], Controls]


class QuantumConstants(Protocol):
    """
    Interface class for representing the physical constants of a quantum
    system. This class can be extended to contain system specific constants as
    required.

    Attributes
    ----------
    hbar : float
        The reduced Planck constant.
    ...
    """

    hbar: float


class HilbertSpace:
    """
    Represents a discretised Hilbert space. This class aims to represent an
    abstract Hilbert space which is agnostic of a quantum system. It defines
    discretised axes/grids in position and wavevector space. The discretised
    wavevector axes/grids are stored unshifted (frequencies not centred).

    Parameters
    ----------
    num_dimensions : int
        The number of dimensions.
    num_points : IVector
        The number of sampling points for the discretised position space
        axes. This should have shape (num_dimensions).
    position_bounds : RVectors
        The boundaries of the discretised position space axes. This should have
        shape (num_dimensions, 2), where position_bounds[:, 0] are the minimum
        values and position_bounds[:, 1] are the maximum values.

    Attributes
    ----------
    num_dimensions : int
        The number of dimensions.
    num_points : IVector
        The number of sampling points for the discretised position space
        axes. This has shape (num_dimensions).
    position_bounds : RVectors
        The boundaries of the discretised position space axes. This has shape
        (num_dimensions, 2), where position_bounds[:, 0] are the minimum values
        and position_bounds[:, 1] are the maximum values.
    position_axes : RVectorSeq
        The discretised position space axes. This is an immutable sequence of
        RVector, which has length (num_dimensions).
    position_grids : RTensors
        The discretised position space grids. These store the combinations of
        discretised position space points. This is an array of RTensor, which
        has shape (num_dimensions, num_points[0], ..., num_points[-1]).
    position_deltas : RVector
        The spacing between points in the discretised position space axes. This
        has shape (num_dimensions).
    wavevector_axes : RVectorSeq
        The discretised wavevector space axes. This is an immutable sequence of
        RVector, which has length (num_dimensions).
    wavevector_grids : RTensors
        The discretised wavevector space grids. These store the combinations of
        discretised wavevector space points. This is an array of RTensor, which
        has shape (num_dimensions, num_points[0], ..., num_points[-1]).
    wavevector_deltas : RVector
        The spacing between points in the discretised wavevector space
        axes. This has shape (num_dimensions).
    """

    def __init__(
        self, num_dimensions: int, num_points: IVector, position_bounds: RVectors
    ) -> None:
        # Check parameters.
        # TODO: Rewrite error messages.
        if not (1 <= num_dimensions <= 3):
            raise ValueError("invalid num_dimensions")
        if num_points.size != num_dimensions:
            raise ValueError("invalid num_points")
        if position_bounds.shape != (num_dimensions, 2):
            raise ValueError("invalid position_bounds")

        # Assign attributes.
        self.num_dimensions: int = num_dimensions
        self.num_points: IVector = num_points
        self.position_bounds: RVectors = position_bounds

        # Define the position axes.
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

        # Define the position grids.
        self.position_grids: RTensors = np.asarray(
            np.meshgrid(*self.position_axes, indexing="ij"), dtype=np.float64
        )

        # Define the position deltas.
        self.position_deltas: RVector = np.asarray(
            [axis[1] - axis[0] for axis in self.position_axes]
        )

        # Define the wavevector axes.
        self.wavevector_axes: RVectorSeq = []
        for i in range(self.num_dimensions):
            self.wavevector_axes.append(
                2 * np.pi * np.fft.fftfreq(self.num_points[i], self.position_deltas[i])
            )
        self.wavevector_axes: RVectorSeq = tuple(self.wavevector_axes)

        # Define the wavevector grids.
        self.wavevector_grids: RTensors = np.asarray(
            np.meshgrid(*self.wavevector_axes, indexing="ij"), dtype=np.float64
        )

        # Define the wavevector deltas.
        self.wavevector_deltas: RVector = np.asarray(
            [axis[1] - axis[0] for axis in self.wavevector_axes]
        )


class QuantumHilbertSpace(HilbertSpace):
    """
    Represents a discretised Hilbert space. This class aims to represent a
    Hilbert space in the context of a quantum system. It builds on the abstract
    "HilbertSpace" class through defining discretised momentum space axes and
    grids, which are stored unshifted (frequencies not centred).

    + All attributes from the "HilbertSpace" class are inherited.

    Parameters
    ----------
    num_dimensions : int
        The number of dimensions.
    num_points : IVector
        The number of sampling points for the discretised position space
        axes. This should have shape (num_dimensions).
    position_bounds : RVectors
        The boundaries of the discretised position space axes. This should have
        shape (num_dimensions, 2), where position_bounds[:, 0] are the minimum
        values and position_bounds[:, 1] are the maximum values.
    constants : QuantumConstants
        The physical constants of the quantum system.

    Parameters
    ----------
    constants : QuantumConstants
        The physical constants of the quantum system.
    momentum_axes : RVectorSeq
        The discretised momentum space axes. This is an immutable sequence of
        RVector, which has length (num_dimensions).
    momentum_grids : RTensors
        The discretised momentum space grids. These store the combinations of
        discretised momentum space points. This is an array of RTensor, which
        has shape (num_dimensions, num_points[0], ..., num_points[-1]).
    momentum_deltas : RVector
        The spacing between points in the discretised momentum space axes. This
        has shape (num_dimensions).
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

        # Define the momentum axes.
        self.momentum_axes: RVectorSeq = [
            self.constants.hbar * axis for axis in self.wavevector_axes
        ]
        self.momentum_axes: RVectorSeq = tuple(self.momentum_axes)

        # Define the momentum grids & deltas.
        self.momentum_grids: RTensors = self.constants.hbar * self.wavevector_grids
        self.momentum_deltas: RVector = self.constants.hbar * self.wavevector_deltas

    def inner_product(self, bra: GTensor, ket: GTensor) -> complex:
        """
        Calculates the inner product between two states, over the discretised
        Hilbert space. This function handles the complex conjugation of the bra
        state.

        Parameters
        ----------
        bra : GTensor
            The bra state. This should have shape (num_points[0], ...,
            num_points[-1]).
        ket : GTensor
            The ket state. This should have shape (num_points[0], ...,
            num_points[-1]).

        Returns
        -------
        complex
            The inner product of the two states.
        """

        # Check parameters.
        # TODO: Rewrite error messages.
        if bra.ndim != self.num_dimensions:
            raise ValueError("invalid bra")
        if ket.ndim != self.num_dimensions:
            raise ValueError("invalid ket")
        if bra.shape != ket.shape:
            raise ValueError("invalid bra and ket")

        # Calculate the inner product integral.
        integrand: Union[complex, CTensor] = np.conjugate(
            bra.astype(np.complex128, copy=False)
        ) * ket.astype(np.complex128, copy=False)

        for i in range(self.num_dimensions):
            integrand = np.trapezoid(integrand, axis=0, dx=self.position_deltas[i])

        return cast(complex, integrand)

    def normalise_state(self, state: GTensor) -> GTensor:
        """
        Normalises a state over the discretised Hilbert space. This function
        uses the integral definition (not vector) of the norm.

        Attributes
        ----------
        state : GTensor
            The state to normalise. This should have shape (num_points[0], ...,
            num_points[-1]).

        Returns
        -------
        GTensor
            The normalised state.
        """

        # Check parameters.
        # TODO: Rewrite error messages.
        if state.ndim != self.num_dimensions:
            raise ValueError("invalid state")

        # Calculate the modulus squared of the state.
        norm: Union[float, RTensor] = np.abs(state) ** 2

        # Calculate the norm integral.
        for i in range(self.num_dimensions):
            norm = np.trapezoid(norm, axis=0, dx=self.position_deltas[i])

        return state / np.sqrt(norm)

    @staticmethod
    def position_basis(state: GTensor) -> GTensor:
        """
        Converts the basis of a state from momentum space to position space.

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
    def momentum_basis(state: GTensor) -> GTensor:
        """
        Converts the basis of a state from position space to momentum space.

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
