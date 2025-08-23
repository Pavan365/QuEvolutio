"""
Implementation of the Chebyshev propagation scheme for the time-dependent
Schrödinger equation (TDSE). This scheme is intended to be used with quantum
systems that do not have explicit time dependence.

References
----------
+ H. TalEzer, R. Kosloff (1984). Available at: https://doi.org/10.1063/1.448136
"""

# Import standard modules.
from typing import Any, Optional, cast

# Import external modules.
import numpy as np

# Import local modules.
import quevolutio.mathematical.approximation as approx
from quevolutio.core.aliases import CVector, CTensor, GTensor, CTensors  # isort: skip
from quevolutio.core.domain import TimeGrid
from quevolutio.core.tdse import Controls, Hamiltonian, TDSEControls


class ChebyshevHamiltonian(Hamiltonian):
    """
    Represents a Hamiltonian. This class builds on the base Hamiltonian
    interface class through defining methods specific to the Chebyshev
    propagation scheme.

    + All attributes from the Hamiltonian class are inherited.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian of the quantum system.

    Attributes
    ----------
    spectrum_centre : float
        The centre of the Hamiltonian's eigenvalue spectrum. This is the mean
        of largest and smallest eigenvalues.
    spectrum_half_span : float
        The half-span of the Hamiltonian's eigenvalue spectrum. This is half of
        the difference between the largest and smallest eigenvalues.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
    ) -> None:
        # Assign attributes.
        self._hamiltonian: Hamiltonian = hamiltonian

        # Calculate quantities related to the eigenvalue spectrum (used for rescaling).
        self.spectrum_centre: float = 0.5 * (self.eigenvalue_max + self.eigenvalue_min)
        self.spectrum_half_span: float = 0.5 * (
            self.eigenvalue_max - self.eigenvalue_min
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hamiltonian, name)

    def hamiltonian_rs(
        self, state: GTensor, controls: Optional[Controls] = None
    ) -> GTensor:
        """
        Calculates the action of the Hamiltonian on a state, using a rescaled
        Hamiltonian which has an eigenvalue spectrum that lies in the domain
        [-1, 1]. If the Hamiltonian has explicit time dependence, a set of
        controls should be passed.

        Parameters
        ----------
        state : GTensor
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the Hamiltonian has explicit time dependence.

        Returns
        -------
        GTensor
            The result of acting the Hamiltonian on the state, calculated using
            the rescaled Hamiltonian.
        """

        return (
            self(state, controls) - (self.spectrum_centre * state)
        ) / self.spectrum_half_span


class Chebyshev:
    """
    Represents the Chebyshev propagation scheme. This scheme is intended to be
    used for quantum systems that do not have explicit time-dependence.

    Parameters
    ----------
    system : TDSE
        The time-dependent Schrödinger equation (TDSE) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.
    order_k : int
        The order of the Chebyshev expansion used to approximate the
        time-evolution operator in the Chebyshev propagation scheme.

    Attributes
    ----------
    system : ChebyshevTDSE
        The time-dependent Schrödinger equation (TDSE) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.
    order_k : int
        The order of the Chebyshev expansion used to approximate the
        time-evolution operator in the Chebyshev propagation scheme.

    Internal Attributes
    -------------------
    _prefactor : complex
        The constant that multiplies the homogeneous term in the time-dependent
        Schrödinger equation (TDSE).
    _phase_factor : complex
        The global phase factor applied at the end of each propagation step in
        the Chebyshev propagation scheme.
    _coefficients : CVector
        The Chebyshev expansion coefficients of the time-evolution operator in
        the Chebyshev propagation scheme.
    """

    def __init__(
        self, hamiltonian: Hamiltonian, time_domain: TimeGrid, order_k: int
    ) -> None:
        # Assign attributes.
        self._hamiltonian: ChebyshevHamiltonian = ChebyshevHamiltonian(hamiltonian)  # type: ignore
        self._time_domain: TimeGrid = time_domain
        self._order_k: int = order_k

        # Calculate the homogeneous pre-factor (TDSE).
        self._prefactor: complex = -1j / self._hamiltonian.domain.constants.hbar

        # Pre-compute the global phase factor applied at each propagation step.
        self._phase_factor: complex = np.exp(
            self._prefactor
            * self.time_domain.time_dt
            * self._hamiltonian.spectrum_centre
        )

        # Pre-compute the Chebyshev expansion coefficients of the time-evolution operator.
        bessel_argument: complex = (
            self._time_domain.time_dt
            * self._hamiltonian.spectrum_half_span
            / self._hamiltonian.domain.constants.hbar
        )
        self._coefficients: CVector = approx.ch_bessel_coefficients(
            bessel_argument, self._order_k
        )

    ## NOTE: PROPERTIES --------------------------------------------------------

    @property
    def hamiltonian(self) -> ChebyshevHamiltonian:
        return self._hamiltonian

    @property
    def time_domain(self) -> TimeGrid:
        return self._time_domain

    @property
    def order_k(self) -> int:
        return self._order_k

    ## NOTE: PROPERTIES END ----------------------------------------------------

    def propagate(
        self, state: GTensor, controls_fn: Optional[TDSEControls]
    ) -> CTensors:
        """
        Propagates a state with respect to the time-dependent Schrödinger
        equation (TDSE) using the Chebyshev propagation scheme.

        Parameters
        ----------
        state : GTensor
            The state to propagate with respect to the TDSE.
        controls_fn : Optional[TDSEControls]
            A callable that generates the controls which determine the
            structure of the TDSE at a given time. This should be passed if the
            TDSE has explicit time dependence.

        Returns
        -------
        states : CTensors
            The propagated states.
        """

        # Ensure that a controls callable is passed for a time-dependent system.
        if self._hamiltonian.time_dependent and controls_fn is None:
            raise ValueError("invalid controls callable")
        assert controls_fn is not None

        # Create an array to store the propagated states.
        states: CTensors = np.zeros(
            (self._time_domain.num_points, *self._hamiltonian.domain.num_points),
            dtype=np.complex128,
        )
        states[0] = state.copy()

        # Propagate the state.
        for i in range(self._time_domain.num_points - 1):
            # Store the current state.
            state_curr: CTensor = states[i]

            # If the Hamiltonian has explicit time dependence.
            # Calculate the controls at the start of the time step.
            controls: Optional[Controls] = None
            if self._hamiltonian.time_dependent:
                controls: Optional[Controls] = controls_fn(
                    self._time_domain.time_axis[i]
                )

            # Calculate the Chebyshev expansion of the time-evolution operator.
            # Acting on the current state.
            state_next: CTensor = cast(
                CTensor,
                approx.ch_expansion(
                    state_curr,
                    self._hamiltonian.hamiltonian_rs,
                    self._coefficients,
                    controls,
                ),
            )

            # Calculate and store the propagated state.
            states[i + 1] = state_next * self._phase_factor

        return states
