"""
Implementation of the Semi-Global propagation scheme for the time-dependent
Schrödinger equation.

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import standard modules.
from enum import Enum
from typing import Optional, Union, cast

# Import external modules.
import numpy as np
from scipy.special import factorial

# Import local modules.
import quevolutio.core.mathematical as math
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


def propagate(
    system: SemiGlobalTDSE,
    wavefunction: sim.GTensor,
    time_domain: sim.TimeGrid,
    order_m: int,
    order_f: int,
    tolerance: float,
    approximation: ApproximationBasis,
    controls_func: Optional[sim.TDSEControls] = None,
) -> sim.CVectors:
    """
    Propagates a wavefunction with respect to the time-dependent Schrödinger
    equation (TDSE) using the Semi-Global propagation scheme. This propagation
    scheme is intended to be used for systems with time-dependent Hamiltonians.

    Parameters
    ----------
    system : sim.SemiGlobalTDSE
        The time-dependent Schrödinger equation (TDSE) of the quantum system.
    wavefunction : simulation.GVector
        The wavefunction to propagate with respect to the time-dependent
        Schrödinger equation (TDSE).
    time_domain : simulation.TimeGrid
        The time domain (grid) over which to propagate the wavefunction.
    order_m : int
        The order of the truncated Taylor expansion, which approximates the
        homogeneous propagation term with corrections from the time derivatives
        of an inhomogeneous term.
    order_f : int
        The order of the Chebyshev expansion of the inhomogeneous operator
        function, which represents a correction (residual) to the Taylor
        expansion.
    tolerance : float
        The tolerance of the propagator. This is used to define the convergence
        criterion during iterative time ordering.
    approximation : ApproximationBasis
        The approximation basis to use for the inhomogeneous term. This can
        either be a Chebyshev expansion or a Newtonian interpolation expansion.
    controls_func : Optional[sim.TDSEControls]
        The function that generates the controls which determine the structure
        of the time-dependent Schrödinger equation (TDSE), at a given time.

    Returns
    -------
    wavefunctions: simulation.CVectors
        The propagated wavefunctions.
    """

    ## NOTE: GLOBAL PRE-COMPUTATIONS
    # Create a vector to store the wavefunctions.
    wavefunctions: sim.CTensors = np.zeros(
        (time_domain.num_points, *system.domain.num_points), dtype=np.complex128
    )
    wavefunctions[0] = wavefunction.copy()

    # Set up the Chebyshev-Gauss and Chebyshev-Lobatto nodes.
    cg_nodes: sim.RVector = math.ch_gauss_nodes(order_f)
    cl_nodes: sim.RVector = math.ch_lobatto_nodes(order_m)

    # Set up the entire time grid of Chebyshev-Lobatto nodes.
    cl_nodes_rs: sim.RVector = math.rescale_tensor(
        cl_nodes, time_domain.time_axis[0], time_domain.time_axis[1]
    )[0]

    t_nodes: sim.RVector = np.zeros(
        (((order_m - 1) * (time_domain.num_points - 1)) + 1), dtype=np.float64
    )
    t_nodes[0] = time_domain.time_axis[0]
    t_nodes[1:] = (
        cl_nodes_rs[1::]
        + (np.arange(time_domain.num_points - 1) * time_domain.time_dt).reshape(-1, 1)
    ).flatten(order="C")

    # Set up the Chebyshev-Gauss nodes for expanding the inhomogeneous operator.
    f_nodes: sim.CVector = cast(
        sim.CVector,
        (
            system.prefactor
            * math.rescale_tensor(
                cg_nodes,
                system.hamiltonian.eigenvalue_min,
                system.hamiltonian.eigenvalue_max,
            )[0]
        ),
    )

    # Evaluate the inhomogeneous operator on the first two time intervals.
    # These can be reused because the effective delta-time is constant.
    function_values: sim.CVectors = np.zeros((order_m, order_f), dtype=np.complex128)
    function_values_next: sim.CVectors = np.zeros(
        (order_m, order_f), dtype=np.complex128
    )

    for i in range(order_m):
        function_values[i] = inhomogeneous_operator(
            f_nodes, t_nodes[i], order_m, threshold=1e-2, tolerance=1e-16
        )
        function_values_next[i] = inhomogeneous_operator(
            f_nodes,
            t_nodes[i + (order_m - 1)],
            order_m,
            threshold=1e-2,
            tolerance=1e-16,
        )

    # Calculate the Chebyshev expansion coefficients of the inhomogeneous operator.
    function_coefficients: sim.CVectors = (
        math.ch_coefficients(function_values.T[::-1], dct_type=2)
        .astype(np.complex128, copy=False)
        .T
    )
    function_coefficients_next: sim.CVectors = (
        math.ch_coefficients(function_values_next.T[::-1], dct_type=2)
        .astype(np.complex128, copy=False)
        .T
    )

    # Generate the conversion matrix.
    # To convert expansion coefficients to Taylor-like derivatives.
    conversion_matrix: sim.RMatrix = np.zeros((order_m, order_m), dtype=np.float64)

    # Chebyshev expansion conversion.
    if approximation == ApproximationBasis.CHEBYSHEV:
        conversion_matrix = math.ch_ta_conversion(
            order_m, time_domain.time_axis[0], time_domain.time_axis[1]
        )

    # Newtonian expansion conversion.
    else:
        conversion_matrix = math.ne_ta_conversion(cl_nodes_rs)

    ## NOTE: STEP 1
    # Set the guess wavefunctions for the first time step.
    wf_guesses: sim.CVector = np.zeros(
        (order_m, *system.domain.num_points), dtype=np.complex128
    )
    wf_guesses[:] = wavefunction.copy()

    ## NOTE: STEP 2
    # Propagate the wavefunction.
    for i in range(time_domain.num_points - 1):
        ## NOTE: STEP 2.A
        # Store the time interval information.
        t_start_idx: int = i * (order_m - 1)
        t_final_idx: int = (order_m - 1) + (i * (order_m - 1))

        ## NOTE: LOCAL PRE-COMPUTES (TIME-INTERVAL SPECIFIC)
        # Store the Chebyshev-Lobatto nodes.
        t_nodes_current: sim.RVector = t_nodes[t_start_idx : (t_final_idx + 1)]
        t_mid_idx: int = order_m // 2
        t_mid: float = t_nodes_current[order_m // 2]

        ## NOTE NOTE CONTROLS.
        # Store controls.
        controls: Union[list[None], list[sim.Controls]] = [None] * len(t_nodes_current)
        if controls_func is not None:
            for j, time in enumerate(t_nodes_current):
                controls[j] = controls_func(time)  # type: ignore

        # Set up the inhomogeneous states (variable scoping).
        lambdas: sim.CVectors = np.zeros(1, dtype=np.complex128)

        # Set the starting convergence.
        convergence: float = np.inf

        # Define a counter and set the maximum number of iterations.
        count: int = 0
        max_iters: int = 20

        ## NOTE: STEP 2.C
        # Propagate the wavefunction until convergence is reached.
        while convergence > tolerance and count < max_iters:
            ## NOTE: STEP 2.C.I
            # Set up the inhomogeneous terms.
            inhomogeneous_values: sim.CVectors = np.zeros(
                (order_m, *system.domain.num_points), dtype=np.complex128
            )
            for j in range(order_m):
                # Time-dependent Hamiltonian.
                if system.hamiltonian.time_dependent:
                    inhomogeneous_values[j] = system.homogeneous_term_dt(
                        wf_guesses[j], controls[j], controls[t_mid_idx]  # type: ignore
                    )

                # Inhomogeneous term.
                if system.inhomogeneous is not None:
                    inhomogeneous_values[j] += system.inhomogeneous_term(controls[j])

            ## NOTE: STEP 2.C.II
            # Calculate the expansion coefficients ofs the inhomogeneous terms.
            inhomogeneous_coefficients: sim.CVectors = np.zeros(
                (order_m, *system.domain.num_points), dtype=np.complex128
            )

            # Chebyshev expansion coefficients.
            if approximation == ApproximationBasis.CHEBYSHEV:
                inhomogeneous_coefficients = math.ch_coefficients(
                    inhomogeneous_values[::-1], dct_type=1
                ).astype(np.complex128, copy=False)

            # Newtonian expansion coefficients.
            else:
                # Rescale to a length four domain.
                t_nodes_current_d4: sim.RVector = (
                    4.0 / time_domain.time_dt
                ) * t_nodes_current

                inhomogeneous_coefficients = math.ne_coefficients(
                    t_nodes_current_d4, inhomogeneous_values
                ).astype(np.complex128, copy=False)

            ## NOTE: STEP 2.C.III
            # Calculate the Taylor-like derivative terms.
            taylor_derivatives: sim.CVectors = np.einsum(
                "ij,jxy->ixy", conversion_matrix.T, inhomogeneous_coefficients
            )

            ## NOTE: STEP 2.C.IV
            # Calculate the inhomogeneous states (lambdas).
            lambdas: sim.CVectors = inhomogeneous_states(
                wf_guesses[0],
                system.homogeneous_term,
                taylor_derivatives,
                controls[t_mid_idx],
            ).astype(np.complex128, copy=False)

            ## NOTE: STEP 2.C.V
            # Store the previous guess wavefunction.
            wf_guess_old: sim.CVector = wf_guesses[-1].copy()

            ## NOTE: STEP 2.C.VI
            # Build the next set of guess wavefunctions.
            for j in range(1, order_m):
                # Store the time interval information.
                t_dt_m = t_nodes[j]

                operator_term: sim.CVector = math.ch_expansion(
                    lambdas[-1],
                    system.homogeneous_term_rs,
                    function_coefficients[j],
                    controls[t_mid_idx],
                ).astype(np.complex128, copy=False)

                # Calculate the truncated Taylor expansion.
                taylor_term: sim.CVector = np.zeros(
                    system.domain.num_points, dtype=np.complex128
                )
                for k in range(order_m):
                    taylor_term += (t_dt_m**k) * lambdas[k]

                # Store the new guess wavefunction.
                wf_guesses[j] = operator_term + taylor_term

            ## NOTE: STEP 2.C.VII
            # Calculate the convergence.
            wf_guess_new: sim.CVector = wf_guesses[-1].copy()
            convergence: float = cast(
                float,
                np.linalg.norm(wf_guess_new - wf_guess_old)
                / np.linalg.norm(wf_guess_old),
            )

            # Update the number of iterations.
            count += 1

            # NOTE: DIAGNOSTIC
            # Print diagnostic information.
            print(f"Time Index: {i} \t Iteration: {count}")
            print(f"Convergence: {convergence:.5e}")

        # If convergence failed, raise an error.
        if count >= max_iters:
            raise ValueError("convergence failed")

        ## NOTE: STEPS 2.D & 2.E
        # Store the propagated wavefunction.
        wavefunctions[i + 1] = wf_guesses[-1].copy()

        ## NOTE: STEP 2.F
        # Calculate the guess wavefunctions for the next time interval.
        if i < time_domain.num_points - 2:
            # Set the first guess wavefunction.
            wf_guesses[0] = wf_guesses[-1].copy()

            # Build the next set of guess wavefunctions.
            for j in range(1, order_m):
                # Store the time interval information.
                t_dt_next_m = t_nodes[j] + time_domain.time_dt

                operator_term_next: sim.CVector = math.ch_expansion(
                    lambdas[-1],
                    system.homogeneous_term_rs,
                    function_coefficients_next[j],
                    controls[t_mid_idx],
                ).astype(np.complex128, copy=False)

                # Calculate the truncated Taylor expansion.
                taylor_term_next: sim.CVector = np.zeros(
                    system.domain.num_points, dtype=np.complex128
                )
                for k in range(order_m):
                    taylor_term_next += (t_dt_next_m**k) * lambdas[k]

                # Store the new guess wavefunction.
                wf_guesses[j] = operator_term_next + taylor_term_next

    return wavefunctions
