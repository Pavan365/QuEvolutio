"""
Implementation of the Semi-Global propagation scheme for the time-dependent
Schrödinger equation (TDSE). This scheme is intended to be used for quantum
systems with explicit time-dependence.

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
import quevolutio.mathematical.affine as affine
import quevolutio.mathematical.approximation as approx
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    CVector,
    GVector,
    CVectors,
    RMatrix,
    CTensor,
    GTensor,
    CTensors,
    GTensors,
)
from quevolutio.core.domain import QuantumHilbertSpace, TimeGrid
from quevolutio.core.tdse import (
    TDSE,
    Controls,
    Hamiltonian,
    Operator,
    Source,
    TDSEControls,
)


class ApproximationBasis(Enum):
    """
    Enumeration of the available approximation bases for the correction term in
    the Semi-Global propagation scheme.

    Members
    -------
    CHEBYSHEV: str
        Represents a Chebyshev basis expansion of the correction term.
    NEWTONIAN: str
        Represents a Newtonian basis expansion of the correction term.
    """

    CHEBYSHEV = "ch"
    NEWTONIAN = "ne"


class SemiGlobalTDSE(TDSE):
    """
    Represents a time-dependent Schrödinger equation (TDSE). This class builds
    on the base TDSE class through defining methods specific to the Semi-Global
    propagation scheme.

    + All attributes from the TDSE class are inherited.

    Parameters
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
    hamiltonian : Hamiltonian
        The Hamiltonian of the quantum system.
    source : Optional[Source]
        The source term of the quantum system. This is an optional term that
        can be excluded.

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
        domain: QuantumHilbertSpace,
        hamiltonian: Hamiltonian,
        source: Optional[Source] = None,
    ) -> None:
        # Construct the TDSE class.
        super().__init__(domain, hamiltonian, source)

        # Calculate quantities related to the eigenvalue spectrum (used for rescaling).
        self.spectrum_centre: float = 0.5 * (
            self.hamiltonian.eigenvalue_max + self.hamiltonian.eigenvalue_min
        )
        self.spectrum_half_span: float = 0.5 * (
            self.hamiltonian.eigenvalue_max - self.hamiltonian.eigenvalue_min
        )

    def homogeneous_term_rs(
        self, state: GTensor, controls: Optional[Controls] = None
    ) -> CTensor:
        """
        Calculates the homogeneous term of the time-dependent Schrödinger
        (TDSE), using a rescaled Hamiltonian which has an eigenvalue spectrum
        that lies in the domain [-1, 1]. If the Hamiltonian has explicit time
        dependence, a set of controls should be passed.

        Parameters
        ----------
        state : GTensor
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the TDSE. This should
            be passed if the Hamiltonian has explicit time dependence.

        Returns
        -------
        CTensor
            The homogeneous term of the TDSE, calculated using the rescaled
            Hamiltonian.
        """

        # Calculate the homogeneous term.
        homogeneous_rs: GTensor = (
            self.hamiltonian(state, controls) - (self.spectrum_centre * state)
        ) / self.spectrum_half_span

        return cast(CTensor, self.homogeneous_factor * homogeneous_rs)

    def homogeneous_term_dt(
        self, state: GTensor, controls_1: Controls, controls_2: Controls
    ) -> CTensor:
        """
        Calculates the difference in the homogeneous term of the time-dependent
        Schrödinger equation (TDSE), for two different sets of controls.

        Parameters
        ----------
        state : GTensor
            The state being acted on. This should have shape
            (*domain.num_points).
        controls_1 : Controls
            The 1st set of controls which determine the structure of the TDSE.
        controls_2 : Controls
            The 2nd set of controls which determine the structure of the TDSE.

        Returns
        -------
        CTensor
            The difference in the homogeneous term of the TDSE, for two
            different sets of controls.
        """

        # If the Hamiltonian is time-independent, raise an error.
        if not self.hamiltonian.time_dependent:
            raise ValueError("invalid function call")

        # If the Hamiltonian has explicit time dependence.
        if self.hamiltonian.ke_time_dependent and self.hamiltonian.pe_time_dependent:
            action: Operator = self.hamiltonian

        # If the KE operator has explicit time dependence.
        elif self.hamiltonian.ke_time_dependent:
            action: Operator = self.hamiltonian.ke_action

        # If the PE operator has explicit time dependence.
        else:
            action: Operator = self.hamiltonian.pe_action

        # Calculate the difference.
        difference: GTensor = action(state, controls_1) - action(state, controls_2)

        return cast(CTensor, self.homogeneous_factor * difference)


class SemiGlobal:
    """
    Represents the Semi-Global propagation scheme. This scheme is intended to
    be used for quantum systems with explicit time-dependence.

    Parameters
    ----------
    system : TDSE
        The time-dependent Schrödinger equation (TDSE) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.
    order_m : int
        The order of the Taylor-like expansion of the time-evolution operator
        in the Semi-Global propagation scheme.
    order_k : int
        The order of the Chebyshev expansion used to approximate the correction
        operator in the Semi-Global propagation scheme.
    tolerance : float
        The tolerance of the propagator. This is used to define the convergence
        criterion during iterative time ordering.
    approximation : ApproximationBasis
        The approximation basis to use for the time-dependent inhomogeneous
        term in the Semi-Global propagation scheme. This can either be a
        Chebyshev basis approximation or a Newtonian basis approximation.

    Attributes
    ----------
    system : SemiGlobalTDSE
        The time-dependent Schrödinger equation (TDSE) of the quantum system.
    time_domain : TimeGrid
        The time domain (grid) over which to propagate.
    order_m : int
        The order of the Taylor-like expansion of the time-evolution operator
        in the Semi-Global propagation scheme.
    order_k : int
        The order of the Chebyshev expansion used to approximate the correction
        operator in the Semi-Global propagation scheme.
    tolerance : float
        The tolerance of the propagator. This is used to define the convergence
        criterion during iterative time ordering.
    approximation : ApproximationBasis
        The approximation basis to use for the time-dependent inhomogeneous
        term in the Semi-Global propagation scheme.

    Internal Attributes
    -------------------
    _time_nodes : RVector
        The complete propagation time grid constructed from Chebyshev-Lobatto
        nodes. These contain the interior points on which terms are expanded in
        time.
    _correction_coefficients_curr : CVectors
        The Chebyshev expansion coefficients of the correction operator for the
        current time interval.
    _correction_coefficients_next : CVectors
        The Chebyshev expansion coefficients of the correction operator for the
        next time interval.
    _conversion_matrix : RMatrix
        The conversion matrix used to convert the expansion coefficients of the
        inhomogeneous term in the TDSE to Taylor-like derivative terms.
    """

    def __init__(
        self,
        system: TDSE,
        time_domain: TimeGrid,
        order_m: int,
        order_k: int,
        tolerance: float,
        approximation: ApproximationBasis,
    ) -> None:
        # Assign attributes.
        self._system: SemiGlobalTDSE = SemiGlobalTDSE(
            system.domain, system.hamiltonian, system.source
        )
        self._time_domain: TimeGrid = time_domain

        self._order_m: int = order_m
        self._order_k: int = order_k
        self._tolerance: float = tolerance
        self._approximation: ApproximationBasis = approximation

        # Perform pre-computations.
        self.pre_computations()

    ## NOTE: PROPERTIES --------------------------------------------------------

    @property
    def system(self) -> SemiGlobalTDSE:
        return self._system

    @property
    def time_domain(self) -> TimeGrid:
        return self._time_domain

    @property
    def order_m(self) -> int:
        return self._order_m

    @property
    def order_k(self) -> int:
        return self._order_k

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def approximation(self) -> ApproximationBasis:
        return self._approximation

    ## NOTE: PROPERTIES END ----------------------------------------------------

    def pre_computations(self) -> None:
        """
        Performs pre-computations of constant quantities in the Semi-Global
        propagation scheme.
        """

        # Set up Chebyshev-Gauss and Chebyshev-Lobatto nodes.
        cg_nodes: RVector = approx.ch_gauss_nodes(self._order_k)
        cl_nodes: RVector = approx.ch_lobatto_nodes(self._order_m)

        # Rescale the Chebyshev-Lobatto nodes to the first time interval.
        cl_nodes_rs: RVector = affine.rescale_tensor(
            cl_nodes, self._time_domain.time_min, self._time_domain.time_max
        )[0]

        # Set up the entire propagation time grid of Chebyshev-Lobatto nodes.
        self._time_nodes: RVector = np.zeros(
            (((self._order_m - 1) * (self._time_domain.num_points - 1)) + 1),
            dtype=np.float64,
        )
        self._time_nodes[0] = self.time_domain.time_axis[0]
        self._time_nodes[1:] = (
            cl_nodes_rs[1::]
            + (
                np.arange(self._time_domain.num_points - 1) * self._time_domain.time_dt
            ).reshape(-1, 1)
        ).flatten(order="C")

        # Set up the Chebyshev-Gauss nodes.
        # For calculating the Chebyshev expansion coefficients of the correction operator.
        correction_nodes: CVector = cast(
            CTensor,
            self._system.homogeneous_factor
            * affine.rescale_tensor(
                cg_nodes,
                self._system.hamiltonian.eigenvalue_min,
                self._system.hamiltonian.eigenvalue_max,
            )[0],
        )

        # Evaluate the correction operator function on the first two time intervals.
        # These can be reused because the effective time steps are constant.
        correction_values_curr: CVectors = np.zeros(
            (self._order_m, self._order_k), dtype=np.complex128
        )
        correction_values_next: CVectors = np.zeros(
            (self._order_m, self._order_k), dtype=np.complex128
        )

        for i in range(self._order_m):
            correction_values_curr[i] = self.correction_operator(
                correction_nodes, self._time_nodes[i], threshold=1e-2
            )
            correction_values_next[i] = self.correction_operator(
                correction_nodes,
                self._time_nodes[i + (self._order_m - 1)],
                threshold=1e-2,
            )

        # Calculate the Chebyshev expansion coefficients of the correction operator.
        self._correction_coefficients_curr: CVectors = cast(
            CVectors,
            approx.ch_coefficients(correction_values_curr.T[::-1], dct_type=2).T,
        )
        self._correction_coefficients_next: CVectors = cast(
            CVectors,
            approx.ch_coefficients(correction_values_next.T[::-1], dct_type=2).T,
        )

        # Generate the conversion matrix.
        # To convert expansion coefficients to Taylor-like derivatives.
        # Chebyshev basis approximation.
        if self._approximation == ApproximationBasis.CHEBYSHEV:
            self._conversion_matrix: RMatrix = approx.ch_ta_conversion(
                self._order_m,
                self.time_domain.time_axis[0],
                self.time_domain.time_axis[1],
            ).T

        # Newtonian basis approximation.
        else:
            self._conversion_matrix: RMatrix = approx.ne_ta_conversion(cl_nodes_rs).T

    def correction_operator(
        self, nodes: GVector, time: float, threshold: float
    ) -> GVector:
        """
        Evaluates the correction operator function, which corrects the
        Taylor-like expansion of the time-evolution operator in the Semi-Global
        propagation scheme. This function calculates the residual between the
        time-evolution operator and its Taylor expansion, scaled for the m-th
        derivative.

        + Semi-Global Paper Section 2.3.2

        Parameters
        ----------
        nodes : GVector
            The nodes on which to evaluate the correction operator function.
        time : float
            The time point at which to evaluate the correction operator
            function. This can also represent a time step.
        threshold : float
            The threshold below which to use a stable Taylor expansion of the
            correction operator function instead of the exact definition. The
            stable definition is used if any |nodes * time| points fall below
            threshold.

        Returns
        -------
        GVector
            The evaluated correction operator function.
        """

        # Pre-calculate reused quantities.
        nodes_x_time: GVector = nodes * time

        # Stable definition.
        if np.any(np.abs(nodes_x_time) < threshold):
            term_sta_1: float = time**self._order_m
            term_sta_2: GVector = cast(
                GVector, np.zeros(nodes.shape[0], dtype=nodes.dtype)
            )

            max_expansion: int = 100
            for i in range(max_expansion):
                term: GVector = (nodes_x_time**i) / factorial(i + self._order_m)
                term_sta_2 += term

                if np.all(np.abs(term) < np.finfo(np.float64).eps):
                    break

            correction_nodes: GVector = term_sta_1 * term_sta_2

        # Standard definition.
        else:
            term_std_1: GVector = nodes**-self._order_m
            term_std_2: GVector = np.exp(nodes_x_time)
            term_std_3: GVector = cast(
                GVector, np.zeros(nodes.shape[0], dtype=nodes.dtype)
            )

            for i in range(self._order_m):
                term_std_3 += (nodes_x_time**i) / factorial(i)

            correction_nodes: GVector = term_std_1 * (term_std_2 - term_std_3)

        return factorial(self._order_m) * correction_nodes

    def expansion_states(
        self, state: GTensor, derivatives: GTensors, controls: Optional[Controls]
    ) -> CTensors:
        """
        Constructs the expansion states used to construct a propagated state in
        the Semi-Global propagation scheme. These states are used to construct
        the Taylor-like expansion of the time-evolution operator propagating a
        state, and the corresponding correction term. The states are made from
        the homogeneous term of the time-dependent Schrödinger equation (TDSE)
        and the time derivatives of the inhomogeneous term of the TDSE.

        + Semi-Global Paper Section 2.4.2

        Parameters
        ----------
        state : GTensor
            The state being propagated by the time-evolution operator.
        derivatives : GTensors
            The Taylor-like time derivatives of the inhomogeneous term of the
            TDSE. The zeroth axis is taken to be the axis of increasing order
            derivatives.
        controls : Optional[Controls]
            The controls which determine the structure of the TDSE. This should
            be passed if the Hamiltonian has explicit time dependence.

        Returns
        -------
        CTensors
            The expansion states used to construct a propagated state.
        """

        # Set the number of expansion terms.
        order: int = self._order_m + 1

        # Set up the expansion states.
        states: CTensors = np.zeros((order, *state.shape), dtype=np.complex128)
        states[0] = state.copy()

        # Calculate the expansion states.
        for i in range(1, order):
            states[i] = (
                self._system.homogeneous_term(states[i - 1], controls)
                + derivatives[i - 1]
            ) / i

        return states

    def propagate(
        self,
        state: GTensor,
        controls_fn: Optional[TDSEControls] = None,
        diagnostics: bool = False,
    ) -> CTensors:
        """
        Propagates a state with respect to the time-dependent Schrödinger
        equation (TDSE) using the Semi-Global propagation scheme.

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
            information regarding convergence during propagation.

        Returns
        -------
        states : CTensors
            The propagated states.
        """

        # Ensure that a controls callable is passed for a time-dependent system.
        if self._system.time_dependent and controls_fn is None:
            raise ValueError("invalid controls callable")
        assert controls_fn is not None

        # Create an array to store the propagated states.
        states: CTensors = np.zeros(
            (self._time_domain.num_points, *self._system.domain.num_points),
            dtype=np.complex128,
        )
        states[0] = state.copy()

        # Set the guess states for the first time interval.
        guess_states_curr: CTensors = np.zeros(
            (self._order_m, *self._system.domain.num_points), dtype=np.complex128
        )
        guess_states_curr[:] = state.copy()

        ## NOTE: SEMI-GLOBAL STEP 2
        # Propagate the state.
        for i in range(self._time_domain.num_points - 1):
            ## NOTE: SEMI-GLOBAL STEP 2.A
            # Store the time interval information.
            time_start_idx: int = i * (self._order_m - 1)
            time_final_idx: int = (self._order_m - 1) + time_start_idx

            ## NOTE: LOCAL PRE-COMPUTES (TIME-INTERVAL SPECIFIC)
            # Store the Chebyshev-Lobatto nodes.
            time_nodes_curr: RVector = self._time_nodes[
                time_start_idx : (time_final_idx + 1)
            ]
            time_mid_idx: int = self._order_m // 2

            # Store the controls for the time interval.
            controls: list[Optional[Controls]] = [None] * self._order_m
            if self._system.time_dependent:
                for j, time in enumerate(time_nodes_curr):
                    controls[j] = controls_fn(time)

            # Set up the expansion states (variable scoping).
            expansion_states_curr: Optional[CTensors] = None

            # Set the starting convergence.
            convergence: float = np.inf

            # Define a counter and set the maximum number of iterations.
            count: int = 0
            max_count: int = 20

            ## NOTE: SEMI-GLOBAL STEP 2.C
            # Propagate the wavefunction until convergence is reached.
            while convergence > self._tolerance and count < max_count:
                ## NOTE: STEP 2.C.I
                # Set up the inhomogeneous terms.
                inhomogeneous_values: CTensors = np.zeros(
                    (self._order_m, *self.system.domain.num_points), dtype=np.complex128
                )
                for j in range(self._order_m):
                    # Time-dependent Hamiltonian term.
                    if self._system.hamiltonian.time_dependent:
                        inhomogeneous_values[j] = self._system.homogeneous_term_dt(
                            guess_states_curr[j],
                            cast(Controls, controls[j]),
                            cast(Controls, controls[time_mid_idx]),
                        )

                    # Source term.
                    if self._system.source is not None:
                        inhomogeneous_values[j] += self._system.source_term(controls[j])

                ## NOTE: SEMI-GLOBAL STEP 2.C.II
                # Calculate the expansion coefficients of the inhomogeneous terms.
                inhomogeneous_coefficients: CTensors = np.zeros(
                    (self._order_m, *self._system.domain.num_points),
                    dtype=np.complex128,
                )
                # Chebyshev expansion coefficients.
                if self._approximation == ApproximationBasis.CHEBYSHEV:
                    inhomogeneous_coefficients = cast(
                        CTensors,
                        approx.ch_coefficients(inhomogeneous_values[::-1], dct_type=1),
                    )

                # Newtonian interpolation expansion coefficients.
                else:
                    # Rescale the current time nodes to domain of length four.
                    nodes_range_four: RVector = (
                        4.0 / self._time_domain.time_dt
                    ) * time_nodes_curr

                    inhomogeneous_coefficients: CTensors = cast(
                        CTensors,
                        approx.ne_coefficients(nodes_range_four, inhomogeneous_values),
                    )

                ## NOTE: SEMI-GLOBAL STEP 2.C.III
                # Calculate the Taylor-like derivative terms.
                inhomogeneous_derivatives: CTensors = cast(
                    CTensors,
                    np.tensordot(
                        self._conversion_matrix, inhomogeneous_coefficients, axes=(1, 0)
                    ),
                )

                ## NOTE: SEMI-GLOBAL STEP 2.C.IV
                # Calculate the expansion states.
                expansion_states_curr: Optional[CTensors] = self.expansion_states(
                    guess_states_curr[0],
                    inhomogeneous_derivatives,
                    controls[time_mid_idx],
                )

                ## NOTE: SEMI-GLOBAL STEP 2.C.V
                # Store the previous guess propagated state.
                guess_state_old: CVector = guess_states_curr[-1].copy()

                ## NOTE: SEMI-GLOBAL STEP 2.C.VI
                # Build the next set of guess states.
                for j in range(1, self._order_m):
                    # Store the time interval information.
                    time_dt_m = self._time_nodes[j]

                    # Calculate the Taylor-like expansion term.
                    expansion_term: CTensor = np.zeros(
                        *self._system.domain.num_points, dtype=np.complex128
                    )
                    for k in range(self._order_m):
                        expansion_term += (time_dt_m**k) * expansion_states_curr[k]

                    # Calculate the correction term.
                    correction_term: CTensor = cast(
                        CTensor,
                        approx.ch_expansion(
                            expansion_states_curr[-1],
                            self._system.homogeneous_term_rs,
                            self._correction_coefficients_curr[j],
                            controls[time_mid_idx],
                        ),
                    )

                    # Store the new guess state.
                    guess_states_curr[j] = expansion_term + correction_term

                ## NOTE: SEMI-GLOBAL STEP 2.C.VII
                # Calculate the convergence.
                guess_state_new: CTensor = guess_states_curr[-1].copy()
                convergence: float = cast(
                    float,
                    np.linalg.norm(guess_state_new - guess_state_old)
                    / np.linalg.norm(guess_state_old),
                )

                # Update the number of iterations.
                count += 1

                # NOTE: DIAGNOSTIC
                # Print diagnostic information.
                if diagnostics:
                    print(f"Time Index: {i} \t Iteration: {count}")
                    print(f"Convergence: {convergence:.5e}")

            # If convergence failed, raise an error.
            if count >= max_count:
                raise ValueError("convergence failed")

            ## NOTE: SEMI-GLOBAL STEPS 2.D & 2.E
            # Store the propagated state.
            states[i + 1] = guess_states_curr[-1].copy()

            ## NOTE: SEMI-GLOBAL STEP 2.F
            # Calculate the guess states for the next time interval.
            if i < self._time_domain.num_points - 2:
                # Assert for expansion states (variable scoping).
                assert expansion_states_curr is not None

                # Set the first guess state.
                guess_states_curr[0] = guess_states_curr[-1].copy()

                # Build the next set of guess states.
                for j in range(1, self._order_m):
                    # Store the time interval information.
                    time_dt_m = self._time_nodes[j] + self._time_domain.time_dt

                    # Calculate the Taylor-like expansion term.
                    expansion_term: CTensor = np.zeros(
                        *self._system.domain.num_points, dtype=np.complex128
                    )
                    for k in range(self._order_m):
                        expansion_term += (time_dt_m**k) * expansion_states_curr[k]

                    # Calculate the correction term.
                    correction_term: CTensor = cast(
                        CTensor,
                        approx.ch_expansion(
                            expansion_states_curr[-1],
                            self._system.homogeneous_term_rs,
                            self._correction_coefficients_next[j],
                            controls[time_mid_idx],
                        ),
                    )

                    # Store the new guess state.
                    guess_states_curr[j] = expansion_term + correction_term

        return states
