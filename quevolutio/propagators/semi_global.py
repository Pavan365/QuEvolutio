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
from quevolutio.core.aliases import CTensor, GTensor
from quevolutio.core.domain import QuantumHilbertSpace
from quevolutio.core.tdse import TDSE, Controls, Hamiltonian, Operator, Source


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
    ):
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

        return cast(CTensor, self.prefactor * homogeneous_rs)

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

        return cast(CTensor, self.prefactor * difference)
