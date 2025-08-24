"""
Example simulation of a standard harmonic oscillator in one dimension (1D)
using the Split-Operator propagation scheme.
"""

# Import standard modules.
import sys
import time
from pathlib import Path
from typing import Optional, cast

# Import external modules.
import numpy as np
from scipy.sparse.linalg import eigsh

# Import local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.numerical as numerical
import utils.standard_ho as sho
import utils.visualisation as vis

# Import QuEvolutio modules.
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    RVectorSeq,
    CTensors,
    CSRMatrix,
)
from quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from quevolutio.core.tdse import Controls, HamiltonianSeparable
from quevolutio.propagators.split_operator import SplitOperator

## NOTE: SIMULATION SET UP -----------------------------------------------------


class SHOConstants(QuantumConstants):
    """
    Represents the constants of a 1D standard harmonic oscillator in natural
    units through the QuantumConstants interface.

    Attributes
    ----------
    hbar : float
        The reduced Planck constant.
    mass : float
        The mass of the system.
    omega : float
        The angular frequency of the system.
    """

    hbar: float = 1.0
    mass: float = 1.0
    omega: float = 1.0


class SHOHamiltonian(HamiltonianSeparable):
    """
    Represents the Hamiltonian of a 1D standard harmonic oscillator through the
    HamiltonianSeparable interface.

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

    Internal Attributes
    -------------------
    _ke_diagonal : RVector
        The pre-computed kinetic energy diagonal for the 1D standard harmonic
        oscillator system.
    _pe_diagonal : RVector
        The pre-computed potential energy diagonal for the 1D standard harmonic
        oscillator system.
    """

    time_dependent: bool = False
    ke_time_dependent: bool = False
    pe_time_dependent: bool = False

    def __init__(self, domain: QuantumHilbertSpace) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain

        # For static type checking (not runtime required).
        self.domain.constants = cast(SHOConstants, self.domain.constants)

        # Pre-compute the kinetic energy diagonal.
        self._ke_diagonal: RVector = (domain.momentum_axes[0] ** 2) / (
            2.0 * self.domain.constants.mass
        )

        # Pre-compute the potential energy diagonal.
        self._pe_diagonal: RVector = (
            0.5
            * self.domain.constants.mass
            * (self.domain.constants.omega**2)
            * self.domain.position_axes[0] ** 2
        )

    def ke_diagonals(self, controls: Optional[Controls] = None) -> RVectorSeq:
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
        RVectorSeq
            The kinetic energy diagonal(s) in momentum space. This is a
            sequence of GVector, with length (domain.num_dimensions).
        """

        return (self._ke_diagonal,)

    def pe_diagonals(self, controls: Optional[Controls] = None) -> RVectorSeq:
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
        RVectorSeq
            The potential energy diagonal(s) in position space. This is a
            sequence of GVector, with length (domain.num_dimensions).
        """

        return (self._pe_diagonal,)


## NOTE: SIMULATION SET UP END -------------------------------------------------


def main():
    # Set up the domain.
    constants: SHOConstants = SHOConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=1,
        num_points=np.array([512]),
        position_bounds=np.array([[-10.0, 10.0]]),
        constants=constants,
    )

    # Set up the initial state.
    # Here we use the eigenstates of the exact Hamiltonian.
    h_matrix: CSRMatrix = sho.hamiltonian_matrix(
        domain, constants.mass, [constants.omega]
    )
    eigenvalues, eigenvectors = eigsh(h_matrix, k=5, which="SA")

    state_idx: int = 0
    state_initial: RVector = cast(
        RVector, domain.normalise_state(eigenvectors[:, state_idx])
    )

    # Set up the Hamiltonian.
    hamiltonian: SHOHamiltonian = SHOHamiltonian(domain)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)

    # Set up the propagator.
    propagator = SplitOperator(hamiltonian, time_domain)

    # Propagate the initial state (timed).
    start_time: float = time.time()
    states: CTensors = propagator.propagate(state_initial)
    final_time: float = time.time()

    # Calculate the norms of the states.
    norms: RVector = numerical.states_norms(states, domain)

    # Calculate the errors from the exact solutions.
    states_exact: CTensors = sho.eigenstate_solutions(
        state_initial, eigenvalues[state_idx], domain.constants.hbar, time_domain
    )
    errors: RVector = numerical.states_norms((states_exact - states), domain)

    # Print simulation information.
    print(f"Runtime \t\t: {(final_time - start_time):.5f} seconds")
    print(f"Max Error \t\t: {np.max(errors):.5e}")
    print(f"Max Norm Deviation \t: {np.max(np.abs(norms - norms[0])):.5e}")

    # Set a common filename.
    filename: str = "standard_ho_1d"

    # Create directories for saving data if they do not exist.
    for folder in (Path("data"), Path("figures"), Path("anims")):
        folder.mkdir(parents=True, exist_ok=True)

    # Save the propagated states.
    np.save(f"data/{filename}.npy", states)

    # Plot the propagated states.
    vis.plot_state_1D(
        states[0],
        domain,
        r"$\text{State} \; (T = 0.00)$",
        f"figures/{filename}_start.png",
    )
    vis.plot_state_1D(
        states[-1],
        domain,
        rf"$\text{{State}} \; (T = {time_domain.time_axis[-1]:.2f})$",
        f"figures/{filename}_final.png",
    )

    # Animate the propagated states.
    vis.animate_states_1D(states, domain, time_domain, f"anims/{filename}.mp4")


if __name__ == "__main__":
    main()
