"""
Classes for setting up the time-dependent Schrödinger equation of a quantum
system.
"""

# Import standard modules.
from typing import Mapping, Optional, Protocol, Sequence, TypeAlias

# Import local modules.
from quevolutio.core.aliases import GTensor
from quevolutio.core.domain import QuantumHilbertSpace

# Type aliases for controls (time-dependent parameters).
Control: TypeAlias = float | complex | GTensor
Controls: TypeAlias = Control | Sequence[Control] | Mapping[str, Control]


class Hamiltonian(Protocol):
    """
    Interface for representing a Hamiltonian. This class can be extended to
    contain system specific attributes, pre-computed operators and methods as
    required.

    Attributes
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
    eigenvalue_min : float
        The minimum eigenvalue of the Hamiltonian.
    eigenvalue_max : float
        The maximum eigenvalue of the Hamiltonian.
    time_dependent : bool
        A boolean flag that indicates whether the Hamiltonian has explicit time
        dependence.
    ke_time_dependent : bool
        A boolean flag that indicates whether the KE operator has explicit time
        dependence.
    pe_time_dependent : bool
        A boolean flag that indicates whether the PE operator has explicit time
        dependence.
    """

    domain: QuantumHilbertSpace
    eigenvalue_min: float
    eigenvalue_max: float
    time_dependent: bool
    ke_time_dependent: bool
    pe_time_dependent: bool

    def __call__(self, state: GTensor, controls: Optional[Controls] = None) -> GTensor:
        """
        Calculates the action of the Hamiltonian on a state. If the Hamiltonian
        has explicit time dependence, a set of controls should be passed.

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
            The result of acting the Hamiltonian on the sate.
        """

        ...

    def ke_action(self, state: GTensor, controls: Optional[Controls] = None) -> GTensor:
        """
        Calculates the action of the kinetic energy operator on a state. If the
        kinetic energy operator has explicit time dependence, a set of controls
        should be passed.

        Parameters
        ----------
        state : GTensor
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the kinetic energy operator has explicit time
            dependence.

        Returns
        -------
        GTensor
            The result of acting the kinetic energy operator on the state.
        """

        ...

    def pe_action(self, state: GTensor, controls: Optional[Controls] = None) -> GTensor:
        """
        Calculates the action of the potential energy operator on a state. If
        the potential energy operator has explicit time dependence, a set of
        controls should be passed.

        Parameters
        ----------
        state : GTensor
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the potential energy operator has explicit time
            dependence.

        Returns
        -------
        GTensor
            The result of acting the potential energy operator on the state.
        """

        ...
