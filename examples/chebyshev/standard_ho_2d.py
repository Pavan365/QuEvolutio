"""
Example simulation of a standard harmonic oscillator in two dimensions (2D)
using the Chebyshev propagation scheme.
"""

# Import standard modules.
import gc
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, cast

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
    CVector,
    RTensor,
    GTensor,
    CTensors,
    CSRMatrix,
)
from quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from quevolutio.core.tdse import Controls, Hamiltonian
from quevolutio.propagators.chebyshev import Chebyshev

## NOTE: SIMULATION SET UP -----------------------------------------------------


class SHOConstants(QuantumConstants):
    """
    Represents the constants of a 2D standard harmonic oscillator in natural
    units through the QuantumConstants interface.

    Attributes
    ----------
    hbar : float
        The reduced Planck constant.
    mass : float
        The mass of the system.
    omega : Sequence[float]
        The angular frequency of the system in each dimension.
    """

    hbar: float = 1.0
    mass: float = 1.0
    omega: Sequence[float] = (1.0, 1.0)


class SHOHamiltonian(Hamiltonian):
    """
    Represents the Hamiltonian of a 2D standard harmonic oscillator through the
    "Hamiltonian" interface.

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

    Internal Attributes
    -------------------
    _ke_operator : RVector
        The pre-computed kinetic energy operator for the 2D standard harmonic
        oscillator system.
    _pe_operator : RVector
        The pre-computed potential energy operator for the 2D standard harmonic
        oscillator system.
    """

    time_dependent: bool = False
    ke_time_dependent: bool = False
    pe_time_dependent: bool = False

    def __init__(
        self, domain: QuantumHilbertSpace, eigenvalue_min: float, eigenvalue_max: float
    ) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain
        self.eigenvalue_min: float = eigenvalue_min
        self.eigenvalue_max: float = eigenvalue_max

        # For static type checking (not runtime required).
        self.domain.constants = cast(SHOConstants, self.domain.constants)

        # Pre-compute the kinetic energy operator.
        self._ke_operator: RTensor = (domain.momentum_meshes[0] ** 2) + (
            domain.momentum_meshes[1] ** 2
        )
        self._ke_operator /= 2.0 * self.domain.constants.mass

        # Pre-compute the potential energy operator.
        self._pe_operator: RTensor = (
            (self.domain.constants.omega[0] ** 2)
            * (self.domain.position_meshes[0] ** 2)
        ) + (
            (self.domain.constants.omega[1] ** 2)
            * (self.domain.position_meshes[1] ** 2)
        )
        self._pe_operator *= 0.5 * self.domain.constants.mass

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
            The result of acting the Hamiltonian on the state.
        """

        return self.ke_action(state) + self.pe_action(state)

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

        return self.domain.position_space(
            self._ke_operator * self.domain.momentum_space(state)
        )

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

        return self._pe_operator * state


## NOTE: SIMULATION SET UP END -------------------------------------------------


def main():
    # Set up the domain.
    constants: SHOConstants = SHOConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=2,
        num_points=np.array([128, 128]),
        position_bounds=np.array([[-10.0, 10.0], [-10.0, 10.0]]),
        constants=constants,
    )

    # Set up the initial state.
    # Here we use the eigenstates of the exact Hamiltonian.
    h_matrix: CSRMatrix = sho.hamiltonian_matrix(
        domain, constants.mass, constants.omega
    )
    eigenvalues, eigenvectors = eigsh(h_matrix, k=5, which="SA")

    state_idx: int = 0
    state_initial: RTensor = cast(
        RTensor,
        domain.normalise_state(
            eigenvectors[:, state_idx].reshape(
                domain.num_points[0], domain.num_points[1]
            )
        ),
    )

    # Store the minimum and maximum eigenvalues.
    eigenvalue_min = eigsh(h_matrix, k=1, which="SA")[0][0]
    eigenvalue_max = eigsh(h_matrix, k=1, which="LA")[0][0]

    # Delete the Hamiltonian matrix and reclaim memory (garbage collection).
    del h_matrix
    gc.collect()

    # Set up the Hamiltonian.
    hamiltonian: SHOHamiltonian = SHOHamiltonian(domain, eigenvalue_min, eigenvalue_max)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)

    # Set up the propagator.
    propagator = Chebyshev(hamiltonian, time_domain, order_k=20)

    # Propagate the initial state (timed).
    print("Propagation Start")
    start_time: float = time.time()
    states: CTensors = propagator.propagate(
        state_initial, controls_fn=None, diagnostics=True
    )
    final_time: float = time.time()
    print("Propagation Done")

    # Calculate the norms and energies of the states.
    norms: RVector = np.empty(time_domain.num_points, dtype=np.float64)
    energies: RVector = numerical.states_energies(states, hamiltonian, time_domain)

    # Calculate the error from the exact solutions.
    errors: RVector = np.empty(time_domain.num_points, dtype=np.float64)

    # This is done in batches due to memory constraints.
    batch: int = 1000

    for i in range(0, time_domain.num_points, batch):
        # Calculate the current batch index and times.
        idx: int = min((i + batch), time_domain.num_points)
        times: RVector = time_domain.time_axis[i:idx]

        # Calculate the norms of the states.
        norms[i:idx] = numerical.states_norms(states[i:idx], domain)

        # Calculate the analytical solutions.
        phase_factors: CVector = np.exp(
            -1j * eigenvalues[state_idx] * times / constants.hbar
        )
        states_exact: CTensors = (
            state_initial[np.newaxis, ...]
            * phase_factors[(...,) + ((np.newaxis,) * domain.num_dimensions)]
        )

        # Store the errors.
        errors[i:idx] = numerical.states_norms((states_exact - states[i:idx]), domain)

    # Print simulation information.
    print(f"Runtime \t\t: {(final_time - start_time):.5f} seconds")
    print(f"Max Error \t\t: {np.max(errors):.5e}")
    print(f"Max Norm Deviation \t: {np.max(np.abs(norms - norms[0])):.5e}")
    print(f"Max Energy Deviation \t: {np.max(np.abs(energies - energies[0])):.5e}")

    # Set a common filename.
    filename: str = "standard_ho_2d"

    # Create directories for saving data if they do not exist.
    for folder in (Path("data"), Path("figures"), Path("anims")):
        folder.mkdir(parents=True, exist_ok=True)

    # Save the propagated states.
    np.save(f"data/{filename}.npy", states)

    # Plot the propagated states.
    vis.plot_state_2D(
        states[0],
        domain,
        r"$\text{State} \; (T = 0.00)$",
        f"figures/{filename}_start.png",
    )
    vis.plot_state_2D(
        states[-1],
        domain,
        rf"$\text{{State}} \; (T = {time_domain.time_axis[-1]:.2f})$",
        f"figures/{filename}_final.png",
    )

    # Animate the propagated states.
    vis.animate_states_2D(states, domain, time_domain, f"anims/{filename}.mp4")


if __name__ == "__main__":
    main()
