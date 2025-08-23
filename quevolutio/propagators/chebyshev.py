"""
Implementation of the Chebyshev propagation scheme for the time-dependent
Schrödinger equation (TDSE). This scheme is intended to be used with quantum
systems that do not have explicit time dependence.

References
----------
+ H. TalEzer, R. Kosloff (1984). Available at: https://doi.org/10.1063/1.448136
"""

# Import standard modules.
from typing import Any, Optional

# Import local modules.
from quevolutio.core.aliases import GTensor
from quevolutio.core.tdse import Controls, Hamiltonian


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
