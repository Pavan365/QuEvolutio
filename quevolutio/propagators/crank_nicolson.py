"""
Implementation of the Crank-Nicolson propagation scheme for the time-dependent
Schrödinger equation (TDSE). This scheme is intended to be used for quantum
systems represented using finite-difference methods.

References
----------
+ J. Crank, P. Nicolson (1947). Available at: https://doi.org/10.1017/S0305004100023197
"""

# Import standard modules.
from typing import Optional, cast

# Import external modules.
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# Import local modules.
from quevolutio.core.aliases import CTensor, GTensor, CTensors, CSCMatrix  # isort: skip
from quevolutio.core.domain import TimeGrid
from quevolutio.core.tdse import Controls, HamiltonianMatrix, TDSEControls


class CrankNicolson1D:
    """
    Represents the one-dimensional (1D) Crank-Nicolson propagation scheme. This
    scheme is intended to be used for quantum systems represented using
    finite-difference methods.

    Parameters
    ----------
    hamiltonian : HamiltonianMatrix
        The Hamiltonian matrix (sparse) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.

    Attributes
    ----------
    hamiltonian : HamiltonianMatrix
        The Hamiltonian matrix (sparse) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.

    Internal Attributes
    -------------------
    _time_dt_half : float
        The half spacing between points in the time axis.
    _identity_matrix : CSCMatrix
        The identity matrix in sparse CSC format.
    _matrix_factor : complex
        The constant used to build the Crank-Nicolson matrices.
    _hamiltonian_matrix : Optional[CSCMatrix]
        The Hamiltonian matrix in sparse CSC format. This is set to None during
        initialisation if the Hamiltonian has explicit time dependence.
    _lhs_matrix : Optional[CSCMatrix]
        The left-hand side Crank-Nicolson matrix in sparse CSC format. This is
        set to None during initialisation if the Hamiltonian has explicit time
        dependence.
    _rhs_matrix : Optional[CSCMatrix]
        The right-hand side Crank-Nicolson matrix in sparse CSC format. This is
        set to None during initialisation if the Hamiltonian has explicit time
        dependence.
    _lhs_solver
        The sparse lower-upper (LU) decomposition of the left-hand side
        Crank-Nicolson matrix, which is used to solve the linear system in each
        time step. This is set to None during initialisation if the Hamiltonian
        has explicit time dependence.
    """

    def __init__(self, hamiltonian: HamiltonianMatrix, time_domain: TimeGrid) -> None:
        # Assign attributes.
        self._hamiltonian: HamiltonianMatrix = hamiltonian
        self._time_domain: TimeGrid = time_domain

        # Calculate the half-step time value.
        self._time_dt_half: float = self._time_domain.time_dt / 2.0

        # Calculate the identity matrix.
        self._identity_matrix: CSCMatrix = cast(
            CSCMatrix,
            sp.diags(np.ones(self._hamiltonian.domain.num_points[0]), format="csc"),
        )

        # Calculate the prefactor for the Crank-Nicolson matrices.
        self._matrix_factor: complex = (1j * self._time_domain.time_dt) / (
            2 * self.hamiltonian.domain.constants.hbar
        )

        # Pre-compute the Hamiltonian matrix if possible.
        self._hamiltonian_matrix: Optional[CSCMatrix] = None

        # Pre-compute the Crank-Nicolson matrices if possible.
        self._lhs_matrix: Optional[CSCMatrix] = None
        self._rhs_matrix: Optional[CSCMatrix] = None

        # Pre-compute the solver for the left-hand side if possible.
        self._lhs_solver = None

        if not self._hamiltonian.time_dependent:
            # Pre-compute the Hamiltonian matrix.
            self._hamiltonian_matrix: Optional[CSCMatrix] = (
                self._hamiltonian().tocsc()  # type:ignore
            )
            assert self._hamiltonian_matrix is not None

            # Pre-compute the Crank-Nicolson matrices.
            self._lhs_matrix: Optional[CSCMatrix] = self._identity_matrix + (
                self._matrix_factor * self._hamiltonian_matrix
            )
            self._rhs_matrix: Optional[CSCMatrix] = self._identity_matrix - (
                self._matrix_factor * self._hamiltonian_matrix
            )

            # Pre-compute the solver for the left-hand side.
            self._lhs_solver = splu(self._lhs_matrix)

    ## NOTE: PROPERTIES --------------------------------------------------------

    @property
    def hamiltonian(self) -> HamiltonianMatrix:
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
        equation (TDSE) using the 1D Crank-Nicolson propagation scheme.

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
            (self._time_domain.num_points, self._hamiltonian.domain.num_points[0]),
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
            # Calculate the Crank-Nicolson matrices and the left-hand side solver.
            if self._hamiltonian.time_dependent:
                # Calculate the controls at the start of the time step.
                assert controls_fn is not None
                controls: Controls = controls_fn(
                    self._time_domain.time_axis[i] + self._time_dt_half
                )

                # Calculate the Hamiltonian matrix.
                self._hamiltonian_matrix: Optional[CSCMatrix] = self._hamiltonian(
                    controls
                ).tocsc()  # type: ignore
                assert self._hamiltonian_matrix is not None

                # Pre-compute the Crank-Nicolson matrices.
                self._lhs_matrix: Optional[CSCMatrix] = self._identity_matrix + (
                    self._matrix_factor * self._hamiltonian_matrix
                )
                self._rhs_matrix: Optional[CSCMatrix] = self._identity_matrix - (
                    self._matrix_factor * self._hamiltonian_matrix
                )

                # Pre-compute the solver for the left-hand side.
                self._lhs_solver = splu(self._lhs_matrix)

            assert self._rhs_matrix is not None
            assert self._lhs_solver is not None

            # Calculate and store the propagated state.
            states[i + 1] = self._lhs_solver.solve(self._rhs_matrix @ state_curr)

        return states
