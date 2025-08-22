"""
Classes for setting up the time-dependent Schrödinger equation of a quantum
system.
"""

# Import standard modules.
from typing import Callable, Mapping, Optional, Protocol, Sequence, TypeAlias, cast

# Import local modules.
from quevolutio.core.aliases import CTensor, GTensor, GVectorSeq
from quevolutio.core.domain import QuantumHilbertSpace

# Type aliases for controls (time-dependent parameters).
Control: TypeAlias = float | complex | GTensor
Controls: TypeAlias = Control | Sequence[Control] | Mapping[str, Control]

# Type alias for a callable that returns a set of controls, given a time.
TDSEControls: TypeAlias = Callable[[float], Controls]

# Type alias for a callable that returns the action of an operator on a state.
Operator: TypeAlias = Callable[[GTensor, Optional[Controls]], GTensor]


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


class HamiltonianSeparable(Protocol):
    """
    Interface for representing a separable Hamiltonian. This class can be
    extended to contain system specific attributes, pre-computed terms and
    methods as required.

    Attributes
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
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
    time_dependent: bool
    ke_time_dependent: bool
    pe_time_dependent: bool

    def ke_operator(self, controls: Optional[Controls] = None) -> GVectorSeq:
        """
        Calculates the kinetic energy diagonal(s) in momentum space. A sequence
        of vectors is returned, where each vector corresponds to each dimension
        in the domain. If the kinetic energy operator has explicit time
        dependence, a set of controls should be passed.

        Parameters
        ----------
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the kinetic energy operator has explicit time
            dependence.

        Returns
        -------
        GVectorSeq
            The kinetic energy diagonal(s) in momentum space. This is a
            sequence of GVector, with length (domain.num_dimensions).
        """

        ...

    def pe_operator(self, controls: Optional[Controls] = None) -> GVectorSeq:
        """
        Calculates the potential energy diagonal(s) in position space. A
        sequence of vectors is returned, where each vector corresponds to each
        dimension in the domain. If the potential energy operator has explicit
        time dependence, a set of controls should be passed.

        Parameters
        ----------
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the potential energy operator has explicit time
            dependence.

        Returns
        -------
        GVectorSeq
            The potential energy diagonal(s) in position space. This is a
            sequence of GVector, with length (domain.num_dimensions).
        """

        ...


class Source(Protocol):
    """
    Interface for representing a source term. This class can be extended to
    contain system specific attributes, pre-computed terms and methods as
    required.

    Attributes
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
    time_dependent : bool
        A boolean flag that indicates whether the source term has explicit time
        dependence.
    """

    domain: QuantumHilbertSpace
    time_dependent: bool

    def __call__(self, controls: Optional[Controls] = None) -> GTensor:
        """
        Calculates the source term. If the source term has explicit time
        dependence, a set of controls should be passed.

        Attributes
        ----------
        controls : Optional[Controls]
            The controls which determine the structure of the source term. This
            should be passed if the source term has explicit time dependence.

        Returns
        -------
        GTensor
            The source term. This has shape (*domain.num_points)
        """

        ...


class TDSE:
    """
    Represents a time-dependent Schrödinger equation (TDSE). This class aims to
    encapsulate the right-hand side of the TDSE. This class is designed for use
    in the time evolution of quantum states.

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
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
    hamiltonian : Hamiltonian
        The Hamiltonian of the quantum system.
    source : Optional[Source]
        The source term of the quantum system. This is an optional term that
        can be excluded.
    prefactor : complex
        The constant that multiplies the homogeneous term in the TDSE.
    time_dependent : bool
        A boolean flag that indicates whether the TDSE has explicit time
        dependence.
    """

    def __init__(
        self,
        domain: QuantumHilbertSpace,
        hamiltonian: Hamiltonian,
        source: Optional[Source] = None,
    ) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain
        self.hamiltonian: Hamiltonian = hamiltonian
        self.source: Optional[Source] = source

        # Calculate the homogeneous pre-factor.
        self.prefactor: complex = -1j / self.domain.constants.hbar

        # Determine the time-dependence of the TDSE.
        self.time_dependent: bool = self.hamiltonian.time_dependent or (
            self.source is not None and self.source.time_dependent
        )

    def __call__(self, state: GTensor, controls: Optional[Controls] = None) -> CTensor:
        """
        Calculates the right-hand side of the time-dependent Schrödinger
        equation (TDSE). This is the combination of the homogeneous term and an
        optional source term.

        Parameters
        ----------
        state : GTensor
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the TDSE. This should
            be passed if the TDSE has explicit time dependence.

        Returns
        -------
        CTensor
            The right-hand side of the TDSE.
        """

        # Calculate the homogeneous term.
        homogeneous: CTensor = self.homogeneous_term(state, controls)

        # Calculate the source term if it is defined.
        return (
            homogeneous if self.source is None else homogeneous + self.source(controls)
        )

    def homogeneous_term(
        self, state: GTensor, controls: Optional[Controls] = None
    ) -> CTensor:
        """
        Calculates the homogeneous term of the time-dependent Schrödinger
        equation (TDSE). If the Hamiltonian has explicit time dependence, a set
        of controls should be passed.

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
            The homogeneous term of the TDSE.
        """

        return cast(CTensor, self.prefactor * self.hamiltonian(state, controls))

    def source_term(self, controls: Optional[Controls] = None) -> Optional[GTensor]:
        """
        Calculates the source term of the time-dependent Schrödinger equation
        (TDSE). If the source term has explicit time dependence, a set of
        controls should be passed.

        Parameters
        ----------
        controls : Optional[Controls]
            The controls which determine the structure of the TDSE. This should
            be passed if the source term has explicit time dependence.

        Returns
        -------
        Optional[GTensor]
            The source term of the TDSE. None is returned if the source term is
            not defined.
        """

        # If the source term is not defined, return None.
        if self.source is None:
            return None

        return self.source(controls)
