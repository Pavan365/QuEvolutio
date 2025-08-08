"""
Classes for setting up simulations.
"""

# Import standard modules.
from typing import Callable, Mapping, Protocol, Sequence, TypeAlias, Union, cast

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

# General type alias for controls (time-dependent parameters).
Control: TypeAlias = float | complex | GTensor
Controls: TypeAlias = Control | Sequence[Control] | Mapping[str, Control]

# General type alias for a callable that returns a set of controls, given a time.
HamiltonianControls: TypeAlias = Callable[[float], Controls]


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


class Hamiltonian(Protocol):
    """
    Represents a time-dependent Hamiltonian. This class should contain methods
    for calculating the action of the Hamiltonian, kinetic energy operator and
    potential energy operator on a state, with shape (domain.num_dimensions[0],
    ..., domain.num_dimensions[-1]).


    Attributes
    ----------
    mass : float
        The mass of the system.
    domain : HilbertSpace
        The discretised Hilbert space (domain) of the system.
    eigenvalue_min : float
        The minimum eigenvalue of the Hamiltonian (approximate).
    eigenvalue_max : float
        The maximum eigenvalue of the Hamiltonian (approximate).
    ...
    """

    mass: float
    domain: HilbertSpace
    eigenvalue_min: float
    eigenvalue_max: float

    def __call__(self, state: GTensor, controls: Controls) -> GTensor:
        """
        Calculates the action of the Hamiltonian on a state, given a set of
        time-dependent parameters (controls).

        Parameters
        ----------
        state : GTensor
            The state (e.g. wavefunction) to act on.
        controls : Controls
            The time-dependent parameters (controls) which determine the
            structure of the Hamiltonian.

        Returns
        -------
        GTensor
            The result of acting the Hamiltonian on the given state.
        """

        ...

    def ke_action(self, state: GTensor, controls: Controls) -> GTensor:
        """
        Calculates the action of the kinetic energy operator on a state, given
        a set of time-dependent parameters (controls).

        Parameters
        ----------
        state : GTensor
            The state (e.g. wavefunction) to act on.
        controls : Controls
            The time-dependent parameters (controls) which determine the
            structure of the Hamiltonian.

        Returns
        -------
        GTensor
            The result of acting the kinetic energy operator on the given
            state.
        """

        ...

    def pe_action(self, state: GTensor, controls: Controls) -> GTensor:
        """
        Calculates the action of the potential energy operator on a state,
        given a set of time-dependent parameters (controls).

        Parameters
        ----------
        state : GTensor
            The state (e.g. wavefunction) to act on.
        controls : Controls
            The time-dependent parameters (controls) which determine the
            structure of the Hamiltonian.

        Returns
        -------
        GTensor
            The result of acting the potential energy operator on the given
            state.
        """

        ...


class TimeGrid:
    """
    Represents a discretised time interval.

    Attributes
    ----------
    time_min : float
        The minimum time axis value.
    time_max : float
        The maximum time axis value.
    num_points : int
        The number of sampling points for the discretised time axis.
    time_axis : RVector
        The discretised time axis.
    time_dt : float
        The discretised time axis spacing.
    """

    def __init__(self, time_min: float, time_max: float, num_points: int) -> None:
        """
        Initialises an instance of the TimeGrid class.

        Parameters
        ----------
        time_min : float
            The minimum time axis value.
        time_max : float
            The maximum time axis value.
        num_points : int
            The number of sampling points for the discretised time axis.
        """

        # Assign attributes.
        self.time_min: float = time_min
        self.time_max: float = time_max
        self.num_points: int = num_points

        # Define the time axis.
        self.time_axis: RVector = np.linspace(
            time_min, time_max, num_points, dtype=np.float64
        )
        self.time_dt: float = self.time_axis[1] - self.time_axis[0]
