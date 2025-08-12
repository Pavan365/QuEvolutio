"""
Implementation of the Semi-Global propagation scheme for the time-dependent
Schrödinger equation.

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import standard modules.
from enum import Enum
from typing import Optional, cast

# Import local modules.
import quevolutio.core.simulation as sim


class ApproximationBasis(Enum):
    """
    Enumeration of the available approximation bases for the inhomogeneous
    term in the Semi-Global propagation scheme.

    Members
    -------
    CHEBYSHEV: str
        Represents a Chebyshev expansion of the inhomogeneous term.
    NEWTONIAN: str
        Represents a Newtonian interpolation expansion of the inhomogeneous
        term.
    """

    CHEBYSHEV = "ch"
    NEWTONIAN = "ne"


class SemiGlobalTDSE(sim.TDSE):
    """
    Represents a time-dependent Schrödinger equation. This class builds on the
    "sim.TDSE" class through defining methods that are specific to the
    Semi-Global propagator.

    + All attributes from the "sim.TDSE" class are inherited.

    Parameters
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
    hamiltonian : Hamiltonian
        The Hamiltonian of the quantum system.
    inhomogeneous : Optional[Inhomogeneous]
        The inhomogeneous term of the quantum system. This is an optional term
        that can be excluded.

    Attributes
    ----------
    eigenvalue_delta : float
        The difference between the maximum and minimum eigenvalues.
    eigenvalue_sum : float
        The sum of the maximum and minimum eigenvalues.
    """

    def __init__(
        self,
        domain: sim.QuantumHilbertSpace,
        hamiltonian: sim.Hamiltonian,
        inhomogeneous: Optional[sim.Inhomogeneous] = None,
    ) -> None:
        # Construct the TDSE class.
        super().__init__(domain, hamiltonian, inhomogeneous)

        # Define eigenvalue related constants (used for rescaling).
        self.eigenvalue_delta: float = (
            self.hamiltonian.eigenvalue_max - self.hamiltonian.eigenvalue_min
        )
        self.eigenvalue_sum: float = (
            self.hamiltonian.eigenvalue_max + self.hamiltonian.eigenvalue_min
        )

    def homogeneous_term_rs(
        self, state: sim.GTensor, controls: Optional[sim.Controls] = None
    ) -> sim.CTensor:
        """
        Calculates the homogeneous term of the time-dependent Schrödinger
        equation (TDSE), rescaled to the domain [-1, 1]. If the Hamiltonian has
        explicit time dependence, a set of controls should be passed.

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
        CTensor
            The homogeneous term of the time-dependent Schrödinger equation
            (TDSE), rescaled to the domain [-1, 1]. This has shape
            (*domain.num_points).
        """

        # Check parameters.
        # TODO: Rewrite error messages.
        if state.shape != self.domain.num_points:
            raise ValueError("invalid state")
        if self.hamiltonian.time_dependent and controls is None:
            raise ValueError("invalid controls")

        # Calculate the rescaled homogeneous term.
        homogeneous_rs: sim.GTensor = (
            (2 * self.hamiltonian(state, controls)) - (self.eigenvalue_sum * state)
        ) / self.eigenvalue_delta

        return cast(sim.CTensor, self.prefactor * homogeneous_rs)

    def homogeneous_term_dt(
        self,
        state: sim.GTensor,
        controls_1: sim.Controls,
        controls_2: sim.Controls,
    ) -> sim.CTensor:
        """
        Calculates the difference in the homogeneous term of the time-dependent
        Schrödinger equation (TDSE), for two different sets of controls. This
        function assumes that the Hamiltonian has explicit time dependence, as
        otherwise, the difference in the homogeneous term will be zero.

        Parameters
        ----------
        state : sim.GTensor
            The state being acted on. This should have shape
            (*domain.num_points).
        controls_1 : sim.Controls
            The first set of controls which determine the structure of the
            Hamiltonian.
        controls_2 : sim.Controls
            The second set of controls which determine the structure of the
            Hamiltonian.

        Returns
        -------
        sim.CTensor
            The difference in the homogeneous term of the time-dependent
            Schrödinger equation (TDSE), for two different sets of controls.
        """

        # Check parameters.
        # TODO: Rewrite error messages.
        if state.shape != self.domain.num_points:
            raise ValueError("invalid state")

        # If the Hamiltonian is time-independent, raise an error.
        # TODO: Rewrite error messages.
        if not self.hamiltonian.time_dependent:
            raise ValueError("invalid function call")

        # If the Hamiltonian has full time dependence.
        if self.hamiltonian.ke_time_dependent and self.hamiltonian.pe_time_dependent:
            action: sim.Operator = self.hamiltonian

        # If just the kinetic energy has time dependence.
        elif self.hamiltonian.ke_time_dependent:
            action: sim.Operator = self.hamiltonian.ke_action

        # If just the potential energy has time dependence.
        else:
            action: sim.Operator = self.hamiltonian.pe_action

        # Calculate the difference.
        difference: sim.GTensor = action(state, controls_1) - action(state, controls_2)

        return cast(sim.CTensor, self.prefactor * difference)
