"""
Classes for setting up simulations.
"""

# Import standard modules.
from typing import cast, Sequence, TypeAlias, Union

# Import external modules.
import numpy as np
from numpy.typing import NDArray

# Generalised type aliases for vectors.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IVector: TypeAlias = NDArray[np.int64]
GVector: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RVector: TypeAlias = NDArray[np.float64]
CVector: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of vectors (NumPy).
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IVectors: TypeAlias = NDArray[np.int64]
GVectors: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RVectors: TypeAlias = NDArray[np.float64]
CVectors: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of vectors (Python).
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IVectorSeq: TypeAlias = Sequence[IVector]
GVectorSeq: TypeAlias = Sequence[RVector] | Sequence[CVector]
RVectorSeq: TypeAlias = Sequence[RVector]
CVectorSeq: TypeAlias = Sequence[CVector]

# Generalised type aliases for matrices.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IMatrix: TypeAlias = NDArray[np.int64]
GMatrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RMatrix: TypeAlias = NDArray[np.float64]
CMatrix: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of matrices (NumPy).
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IMatrices: TypeAlias = NDArray[np.int64]
GMatrices: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RMatrices: TypeAlias = NDArray[np.float64]
CMatrices: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of matrices (Python).
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IMatrixSeq: TypeAlias = Sequence[IMatrix]
GMatrixSeq: TypeAlias = Sequence[RMatrix] | Sequence[CMatrix]
RMatrixSeq: TypeAlias = Sequence[RMatrix]
CMatrixSeq: TypeAlias = Sequence[CMatrix]

# Generalised type aliases for tensors.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
ITensor: TypeAlias = NDArray[np.int64]
GTensor: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RTensor: TypeAlias = NDArray[np.float64]
CTensor: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of tensors (NumPy).
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
ITensors: TypeAlias = NDArray[np.int64]
GTensors: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RTensors: TypeAlias = NDArray[np.float64]
CTensors: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of tensors (Python).
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
ITensorSeq: TypeAlias = Sequence[ITensor]
GTensorSeq: TypeAlias = Sequence[RTensor] | Sequence[CTensor]
RTensorSeq: TypeAlias = Sequence[RTensor]
CTensorSeq: TypeAlias = Sequence[CTensor]


class HilbertSpace:
    """
    Represents a discretised Hilbert space. This class sets up discretised axes
    and grids in both position space and momentum space. The momentum space
    axes and grids are defined and stored unshifted (frequencies not centred).

    Attributes
    ----------
    num_dimensions : int
        The number of dimensions.
    hbar : float
        The reduced Planck constant.
    num_points : IVector
        The number of sampling points for the discretised position space
        axes. This should have shape (num_dimensions).
    position_bounds : RVectors
        The boundaries (minimum & maximum) of the position space. This
        should have shape (num_dimensions, 2), where position_bounds[:, 0]
        stores the minimum values and position_bounds[:, 1] stores the
        maximum values.
    position_axes : RVectorSeq
        The discretised position space axes. This is a sequence of RVector,
        with length (num_dimensions).
    position_grids : RTensors
        The discretised position space grids, which contains the combinations
        of coordinates in the position space. This is an array of RTensor, with
        shape (num_dimensions, num_points[0], ..., num_points[-1]).
    position_deltas : RVector
        The spacing between points in the discretised position space axes. This
        has shape (num_dimensions).
    momentum_axes : RVectorSeq
        The discretised momentum space axes (unshifted). This is a sequence of
        RVector, with length (num_dimensions).
    momentum_grids : RTensors
        The discretised momentum space grids, which contains the combinations
        of coordinates in the momentum space. This is an array of RTensor, with
        shape (num_dimensions, num_points[0], ..., num_points[-1]).
    momentum_deltas : RVector
        The spacing between points in the discretised momentum space axes. This
        has shape (num_dimensions).
    """

    def __init__(
        self,
        num_dimensions: int,
        hbar: float,
        num_points: IVector,
        position_bounds: RVectors,
    ) -> None:
        """
        Initialises an instance of the HilbertSpace class.

        Parameters
        ----------
        num_dimensions : int
            The number of dimensions.
        hbar : float
            The reduced Planck constant.
        num_points : IVector
            The number of sampling points for the discretised position space
            axes. This should have shape (num_dimensions).
        position_bounds : RVectors
            The boundaries (minimum & maximum) of the position space. This
            should have shape (num_dimensions, 2), where position_bounds[:, 0]
            stores the minimum values and position_bounds[:, 1] stores the
            maximum values.
        """

        # Assign attributes.
        self.num_dimensions: int = num_dimensions
        self.hbar: float = hbar

        self.num_points: IVector = num_points
        self.position_bounds: RVectors = position_bounds

        # Define the position axes.
        self.position_axes: RVectorSeq = []
        for i in range(num_dimensions):
            self.position_axes.append(
                np.linspace(
                    position_bounds[i, 0],
                    position_bounds[i, 1],
                    num_points[i],
                    dtype=np.float64,
                )
            )

        # Define the position grids.
        self.position_grids: RTensors = np.asarray(
            np.meshgrid(*self.position_axes, indexing="ij"), dtype=np.float64
        )

        # Define the position deltas.
        self.position_deltas: RVector = np.asarray(
            [axis[1] - axis[0] for axis in self.position_axes]
        )

        # Define the momentum axes.
        self.momentum_axes: RVectorSeq = []
        for i in range(num_dimensions):
            self.momentum_axes.append(
                self.hbar
                * 2
                * np.pi
                * np.fft.fftfreq(self.num_points[i], self.position_deltas[i])
            )

        # Define the momentum grids.
        self.momentum_grids: RTensors = np.asarray(
            np.meshgrid(*self.momentum_axes, indexing="ij"), dtype=np.float64
        )

        # Define the momentum deltas.
        self.momentum_deltas: RVector = np.asarray(
            [axis[1] - axis[0] for axis in self.momentum_axes]
        )

    def inner_product(self, bra: GTensor, ket: GTensor) -> complex:
        """
        Calculates the inner product between two states (bra & ket), over the
        Hilbert space. This function complex conjugates the bra state.

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

        # Calculate the inner product integral.
        integrand: Union[complex, CTensor] = np.conjugate(
            bra.astype(np.complex128, copy=False)
        ) * ket.astype(np.complex128, copy=False)

        for i in range(self.num_dimensions):
            integrand = np.trapezoid(integrand, axis=i, dx=self.position_deltas[i])

        return cast(complex, integrand)

    def normalise_state(self, state: GTensor) -> GTensor:
        """
        Normalises a state over the Hilbert space. This function uses the
        integral definition of the norm.

        Parameters
        ----------
        state : GTensor
            The state to normalise. This should have shape (num_points[0], ...,
            num_points[-1]).

        Returns
        -------
        GTensor
            The normalised state.
        """

        # Calculate the modulus squared of the state.
        norm: Union[float, RTensor] = np.abs(state) ** 2

        # Calculate the normalisation factor.
        for i in range(self.num_dimensions):
            norm = np.trapezoid(norm, axis=i, dx=self.position_deltas[i])

        return state / np.sqrt(norm)

    @staticmethod
    def position_basis(state: GTensor) -> GTensor:
        """
        Converts the basis of a state from momentum space (unshifted) to
        position space.

        Parameters
        ----------
        state : GTensor
            The state in momentum space (unshifted). This should have shape
            (num_points[0], ..., num_points[-1]).

        Returns
        -------
        GTensor
            The state in position space.
        """

        return np.fft.ifftn(state, norm="ortho")

    @staticmethod
    def momentum_basis(state: GTensor) -> GTensor:
        """
        Converts the basis of a state from position space to momentum space
        (unshifted).

        Parameters
        ----------
        state : GTensor
            The state in position space. This should have shape (num_points[0],
            ..., num_points[-1]).

        Returns
        -------
        GTensor
            The state in momentum space (unshifted).
        """

        return np.fft.fftn(state, norm="ortho")
