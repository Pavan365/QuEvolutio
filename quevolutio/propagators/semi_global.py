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

# Import external modules.
import numpy as np
from scipy.special import factorial

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


def inhomogeneous_operator(
    nodes: sim.GVector, time: float, order: int, threshold: float, tolerance: float
) -> sim.CVector:
    """
    Evaluates the function of the inhomogeneous operator on a set of given
    nodes, at a specified time point. This function represents the residual of
    the time evolution operator, minus its truncated Taylor expansion.

    + Semi-Global Section 2.3.2

    Parameters
    ----------
    nodes : sim.GVector
        The nodes to evaluate the function of the inhomogeneous operator on. In
        general, these should be generated from either Chebyshev-Gauss or
        Chebyshev-Lobatto nodes.
    time : float
        The time point at which to evaluate the function of the inhomogeneous
        operator on.
    order : int
        The order of the truncated Taylor expansion of the inhomogeneous
        operator.
    threshold : float
        The threshold above which to use the exact definition of the function
        of the inhomogeneous operator. A stable definition is used if any of
        the (nodes x time) points fall below the threshold.
    tolerance : float
        The tolerance for the stable function definition of the inhomogeneous
        operator. This is used to truncate a Taylor series when convergence is
        reached.

    Returns
    -------
    sim.CVector
        The evaluated inhomogeneous operator function values.
    """

    # Stable definition (Taylor).
    if np.any(np.abs(nodes * time) < threshold):
        term_sta_1: float = time**order

        term_sta_2: sim.GVector = cast(
            sim.GVector, np.zeros(nodes.shape[0], dtype=nodes.dtype)
        )
        max_expansion: int = 100

        for i in range(max_expansion):
            term: sim.GVector = ((nodes * time) ** i) / factorial(i + order)
            term_sta_2 += term

            if np.all(np.abs(term) < tolerance):
                break

        return factorial(order) * term_sta_1 * term_sta_2

    # Standard definition.
    else:
        term_std_1: sim.GVector = nodes**-order
        term_std_2: sim.GVector = np.exp(nodes * time)

        term_std_3: sim.GVector = cast(
            sim.GVector, np.zeros(nodes.shape[0], dtype=nodes.dtype)
        )

        for i in range(order):
            term_std_3 += ((nodes * time) ** i) / factorial(i)

        return factorial(order) * term_std_1 * (term_std_2 - term_std_3)


def inhomogeneous_states(
    state: sim.GTensor,
    operator: sim.Operator,
    derivatives: sim.GTensors,
    controls: Optional[sim.Controls] = None,
) -> sim.CVectors:
    """
    Calculates the inhomogeneous states which represent time-evolved states
    with corrections from the time derivatives of an inhomogeneous term. The
    derivatives should be calculated through converting Chebyshev expansion
    coefficients or Newtonian interpolation coefficients to Taylor-like
    derivative terms. The number of states to calculate is taken to be the
    number of derivates plus one.

    + Semi-Global Section 2.4.2

    Parameters
    ----------
    state : sim.GTensor
        The state being acted upon by the operator.
    operator : sim.Operator
        The operator being expanded. This should be a function that returns the
        action of the operator on the state.
    derivatives: sim.GTensors
        The Taylor-like derivatives of an inhomogeneous term. The zeroth axis
        is taken to be the axis of increasing derivative order.
    controls : Optional[sim.Controls]
        The controls that determine the structure of the operator. This should
        be passed if the operator has explicit time dependence.

    Returns
    -------
    states: sim.CTensors
        The inhomogeneous states.
    """

    # Store the number of expansion terms.
    order: int = derivatives.shape[0] + 1

    # Set up the inhomogeneous states.
    states: sim.CTensor = np.zeros((order, *state.shape), dtype=np.complex128)
    states[0] = state.copy()

    # Calculate the inhomogeneous states.
    for i in range(1, order):
        states[i] = (operator(states[i - 1], controls) + derivatives[i - 1]) / i

    return states
