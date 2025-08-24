"""
Example simulation of a driven harmonic oscillator in two dimensions (2D) using
the Semi-Global propagation scheme.
"""

# Import standard modules.
import gc
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, cast

# Import external modules.
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# Import local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.numerical as numerical
import utils.standard_ho as sho
import utils.visualisation as vis

# Import QuEvolutio modules.
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RTensor,
    CTensors,
    CSRMatrix,
)
from quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from quevolutio.core.tdse import TDSE, Controls, Hamiltonian
from quevolutio.propagators.semi_global import ApproximationBasis, SemiGlobal

## NOTE: SIMULATION SET UP -----------------------------------------------------


class DHOConstants(QuantumConstants):
    """
    Represents the constants of a 2D driven harmonic oscillator in natural
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
    omega: Sequence[float] = (1.0, 1.0)


class DHOHamiltonian(Hamiltonian):
    """
    Represents the Hamiltonian of a 2D driven harmonic oscillator through the
    "Hamiltonian" interface.

    + Driving Term: f(x, t) = x sin(t)
    + Driving Term: f(y, t) = y sin(t)

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
    _ke_operator : RTensor
        The pre-computed kinetic energy operator for the 2D driven harmonic
        oscillator system.
    _pe_operator : RTensor
        The pre-computed standard potential energy operator for the 2D driven
        harmonic oscillator system.
    """

    time_dependent: bool = True
    ke_time_dependent: bool = False
    pe_time_dependent: bool = True

    def __init__(
        self, domain: QuantumHilbertSpace, eigenvalue_min: float, eigenvalue_max: float
    ) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain
        self.eigenvalue_min: float = eigenvalue_min
        self.eigenvalue_max: float = eigenvalue_max

        # For static type checking (not runtime required).
        self.domain.constants = cast(DHOConstants, self.domain.constants)

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

    def __call__(self, state: GVector, controls: Optional[Controls] = None) -> GVector:
        """
        Calculates the action of the Hamiltonian on a state. If the Hamiltonian
        has explicit time dependence, a set of controls should be passed.

        Parameters
        ----------
        state : GVector
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the Hamiltonian has explicit time dependence.

        Returns
        -------
        GVector
            The result of acting the Hamiltonian on the state.
        """

        return self.ke_action(state) + self.pe_action(state, controls)

    def ke_action(self, state: GVector, controls: Optional[Controls] = None) -> GVector:
        """
        Calculates the action of the kinetic energy operator on a state. If the
        kinetic energy operator has explicit time dependence, a set of controls
        should be passed.

        Parameters
        ----------
        state : GVector
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the kinetic energy operator has explicit time
            dependence.

        Returns
        -------
        GVector
            The result of acting the kinetic energy operator on the state.
        """

        return self.domain.position_space(
            self._ke_operator * self.domain.momentum_space(state)
        )

    def pe_action(self, state: GVector, controls: Optional[Controls] = None) -> GVector:
        """
        Calculates the action of the potential energy operator on a state. If
        the potential energy operator has explicit time dependence, a set of
        controls should be passed.

        Parameters
        ----------
        state : GVector
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the potential energy operator has explicit time
            dependence.

        Returns
        -------
        GVector
            The result of acting the potential energy operator on the state.
        """

        pe_operator: RTensor = self._pe_operator + (
            (self.domain.position_meshes[0] * np.sin(controls))  # type: ignore
            + (self.domain.position_meshes[1] * np.sin(controls))  # type: ignore
        )

        return pe_operator * state


def controls_fn(time: float) -> Controls:
    """
    Evaluates the controls which determine the structure of the time-dependent
    Schrödinger equation (TDSE) for a 2D driven harmonic oscillator system. In
    this case, this is just the time.

    Parameters
    ----------
    time : float
        The time at which to evaluate the controls.

    Returns
    -------
    Controls
        The controls for the TDSE at the given time.
    """

    return time


## NOTE: SIMULATION SET UP END -------------------------------------------------


def main():
    # Set up the domain.
    constants: DHOConstants = DHOConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=2,
        num_points=np.array([128, 128]),
        position_bounds=np.array([[-10.0, 10.0], [-10.0, 10.0]]),
        constants=constants,
    )

    # Set up the initial state.
    # Here we use the eigenstates of the exact Hamiltonian (T = 0.00).
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

    # Store the minimum and maximum eigenvalues, accounting for the driving term.
    driving_term: CSRMatrix = cast(
        CSRMatrix,
        sp.diags(
            (domain.position_meshes[0] + domain.position_meshes[1]).flatten(),
            format="csr",
        ),
    )

    # Account for the minimum driving term and restore.
    h_matrix += -driving_term
    eigenvalue_min = eigsh(h_matrix, k=1, which="SA")[0][0]
    h_matrix -= -driving_term

    # Account for the maximum driving term and restore.
    h_matrix += driving_term
    eigenvalue_max = eigsh(h_matrix, k=1, which="LA")[0][0]
    h_matrix -= driving_term

    # Delete objects and reclaim memory (garbage collection).
    del h_matrix
    del driving_term
    gc.collect()

    # Set up the TDSE.
    hamiltonian: DHOHamiltonian = DHOHamiltonian(domain, eigenvalue_min, eigenvalue_max)
    tdse: TDSE = TDSE(domain, hamiltonian)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=1.0, num_points=1001)

    # Set up the propagator.
    propagator = SemiGlobal(
        tdse,
        time_domain,
        order_m=10,
        order_k=10,
        tolerance=1e-5,
        approximation=ApproximationBasis.NEWTONIAN,
    )

    # Propagate the initial state (timed).
    print("Propagation Start")
    start_time: float = time.time()
    states: CTensors = propagator.propagate(
        state_initial, controls_fn, diagnostics=True
    )
    final_time: float = time.time()
    print("Propagation Done")

    # Calculate the norms of the states.
    norms: RVector = np.empty(time_domain.num_points, dtype=np.float64)

    # Calculate the position expectation values in of the states.
    states_expectation_x: RVector = np.empty(time_domain.num_points, dtype=np.float64)
    states_expectation_y: RVector = np.empty(time_domain.num_points, dtype=np.float64)

    # This is done in batches due to memory constraints.
    batch: int = 1000

    for i in range(0, time_domain.num_points, batch):
        # Calculate the current batch index.
        idx: int = min((i + batch), time_domain.num_points)

        # Calculate the norms of the states.
        norms[i:idx] = numerical.states_norms(states[i:idx], domain)

        # Calculate the position expectation values in x of the states.
        states_expectation_x[i:idx] = np.trapezoid(
            np.trapezoid(
                ((np.abs(states[i:idx]) ** 2) * domain.position_meshes[0]),
                dx=domain.position_deltas[0],
                axis=1,
            ),
            axis=1,
            dx=domain.position_deltas[1],
        )

        # Calculate the position expectation values in y of the states.
        states_expectation_y[i:idx] = np.trapezoid(
            np.trapezoid(
                ((np.abs(states[i:idx]) ** 2) * domain.position_meshes[1]),
                dx=domain.position_deltas[0],
                axis=1,
            ),
            axis=1,
            dx=domain.position_deltas[1],
        )

    # Calculate the error from the exact position expectation values in x.
    exact_expectation_x: RVector = 0.5 * (
        (time_domain.time_axis * np.cos(time_domain.time_axis))
        - np.sin(time_domain.time_axis)
    )
    errors_x: RVector = np.abs(exact_expectation_x - states_expectation_x)

    # Calculate the error from the exact position expectation values in y.
    exact_expectation_y: RVector = 0.5 * (
        (time_domain.time_axis * np.cos(time_domain.time_axis))
        - np.sin(time_domain.time_axis)
    )
    errors_y: RVector = np.abs(exact_expectation_y - states_expectation_y)

    # Calculate the total magnitude error.
    errors: RVector = np.sqrt((errors_x**2) + (errors_y**2))

    # Print simulation information.
    print(f"Runtime \t\t: {(final_time - start_time):.5f} seconds")
    print(f"Max Error \t\t: {np.max(errors):.5e}")
    print(f"Max Norm Deviation \t: {np.max(np.abs(norms - norms[0])):.5e}")

    # Set a common filename.
    filename: str = "driven_ho_2d"

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
