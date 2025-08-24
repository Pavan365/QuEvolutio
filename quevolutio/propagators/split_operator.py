"""
Implementation of the 2nd-order Split-Operator (Split-Step) propagation scheme
for the time-dependent Schrödinger equation (TDSE). This scheme is intended to
be used for quantum systems with a separable Hamiltonian.

References
----------
+ M.D. Feit et al. (1982). Available at: https://doi.org/10.1016/0021-9991(82)90091-2
"""

# Import standard modules.
from typing import Optional, cast

# Import external modules.
import numpy as np

# Import local modules.
from quevolutio.core.aliases import (  # isort: skip
    GVectorSeq,
    GTensor,
    CTensor,
    CTensors,
)
from quevolutio.core.domain import TimeGrid
from quevolutio.core.tdse import HamiltonianSeparable, Controls, TDSEControls


class SplitOperator:
    """
    Represents the 2nd-order Split-Operator propagation scheme. This scheme
    is intended to be used for quantum systems with a separable Hamiltonian.

    Parameters
    ----------
    hamiltonian : HamiltonianSeparable
        The Hamiltonian (separable) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.

    Attributes
    ----------
    hamiltonian : HamiltonianSeparable
        The Hamiltonian (separable) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.

    Internal Attributes
    -------------------
    _time_dt_half : float
        The half spacing between points in the time axis.
    _homogeneous_factor : complex
        The constant that multiplies the homogeneous term in the time-dependent
        Schrödinger equation (TDSE).
    _full_step_factor : complex
        The constant used in the time-evolution operator when propagating a
        full-step.
    _half_step_factor : complex
        The constant used in the time-evolution operator when propagating a
        half-step.
    _ke_operator : Optional[GTensor]
        The kinetic energy operator of the quantum system. This is set to None
        if the kinetic energy operator has explicit time dependence.
    _pe_operator : Optional[GTensor]
        The potential energy operator of the quantum system. This is set to
        None if the potential energy operator has explicit time dependence.
    """

    def __init__(
        self, hamiltonian: HamiltonianSeparable, time_domain: TimeGrid
    ) -> None:
        # Assign attributes.
        self._hamiltonian: HamiltonianSeparable = hamiltonian
        self._time_domain: TimeGrid = time_domain

        # Calculate the half-step time value.
        self._time_dt_half: float = self._time_domain.time_dt / 2.0

        # Calculate the homogeneous pre-factor (TDSE).
        self._homogeneous_factor: complex = (
            -1j / self._hamiltonian.domain.constants.hbar
        )

        # Calculate the time-evolution operator pre-factors.
        self._full_step_factor: complex = (
            self._homogeneous_factor * self._time_domain.time_dt
        )
        self._half_step_factor: complex = self._homogeneous_factor * self._time_dt_half

        # Pre-compute the kinetic energy and potential energy operators if possible.
        self._ke_operator: Optional[GTensor] = None
        self._pe_operator: Optional[GTensor] = None

        if not self._hamiltonian.ke_time_dependent:
            ke_diagonals: GVectorSeq = self._hamiltonian.ke_diagonals()
            self._ke_operator: Optional[GTensor] = cast(
                GTensor, sum(np.meshgrid(*ke_diagonals, indexing="ij", sparse=True))
            )

        if not self._hamiltonian.pe_time_dependent:
            pe_diagonals: GVectorSeq = self._hamiltonian.pe_diagonals()
            self._pe_operator: Optional[GTensor] = cast(
                GTensor, sum(np.meshgrid(*pe_diagonals, indexing="ij", sparse=True))
            )

    ## NOTE: PROPERTIES --------------------------------------------------------

    @property
    def hamiltonian(self) -> HamiltonianSeparable:
        return self._hamiltonian

    @property
    def time_domain(self) -> TimeGrid:
        return self._time_domain

    ## NOTE: PROPERTIES END ----------------------------------------------------

    def propagate(
        self,
        state: GTensor,
        controls_fn: Optional[TDSEControls] = None,
        diagnostics: bool = False,
    ) -> CTensors:
        """
        Propagates a state with respect to the time-dependent Schrödinger
        equation (TDSE) using the 2nd-order Split-Operator propagation scheme.

        Parameters
        ----------
        state : GTensor
            The state to propagate with respect to the TDSE.
        controls_fn : Optional[TDSEControls]
            A callable that generates the controls which determine the
            structure of the TDSE at a given time. This should be passed if the
            TDSE has explicit time dependence.
        diagnostics : bool
            A boolean flag that indicates whether to output diagnostic
            information during propagation. In the current implementation this
            is just the time index.

        Returns
        -------
        states : CTensors
            The propagated states.
        """

        # Ensure that a controls callable is passed for a time-dependent system.
        if self._hamiltonian.time_dependent and controls_fn is None:
            raise ValueError("invalid controls callable")

        # Create an array to store the propagated states.
        states: CTensors = np.zeros(
            (self._time_domain.num_points, *self._hamiltonian.domain.num_points),
            dtype=np.complex128,
        )
        states[0] = state.copy()

        # Propagate the state.
        for i in range(self._time_domain.num_points - 1):
            # Print diagnostic information.
            if diagnostics:
                print(f"Time Index: {i}")

            # Store the current state.
            state_curr: CTensor = states[i]

            # If the Hamiltonian has explicit time dependence.
            # Calculate the kinetic energy and potential energy operators.
            if self.hamiltonian.time_dependent:
                # Calculate the controls at the start of the time step.
                assert controls_fn is not None
                controls: Controls = controls_fn(self._time_domain.time_axis[i])

                # Calculate the kinetic energy operator.
                if self._hamiltonian.ke_time_dependent:
                    ke_diagonals: GVectorSeq = self._hamiltonian.ke_diagonals(controls)
                    self._ke_operator: Optional[GTensor] = cast(
                        GTensor,
                        sum(np.meshgrid(*ke_diagonals, indexing="ij", sparse=True)),
                    )

                # Calculate the potential energy operator.
                if self._hamiltonian.pe_time_dependent:
                    pe_diagonals: GVectorSeq = self._hamiltonian.pe_diagonals(controls)
                    self._pe_operator: Optional[GTensor] = cast(
                        GTensor,
                        sum(np.meshgrid(*pe_diagonals, indexing="ij", sparse=True)),
                    )

            assert self._ke_operator is not None
            assert self._pe_operator is not None

            # Propagate a half-step in momentum space.
            state_next: CTensor = cast(
                CTensor, self._hamiltonian.domain.momentum_space(state_curr)
            )
            state_next: CTensor = (
                np.exp(self._half_step_factor * self._ke_operator) * state_next
            )

            # Propagate a full-step in position space.
            state_next: CTensor = cast(
                CTensor, self._hamiltonian.domain.position_space(state_next)
            )
            state_next: CTensor = (
                np.exp(self._full_step_factor * self._pe_operator) * state_next
            )

            # Propagate a half-step in momentum space.
            state_next: CTensor = cast(
                CTensor, self._hamiltonian.domain.momentum_space(state_next)
            )
            state_next: CTensor = (
                np.exp(self._half_step_factor * self._ke_operator) * state_next
            )

            # Convert to position space and store the propagated state.
            states[i + 1] = self._hamiltonian.domain.position_space(state_next)

        return states
