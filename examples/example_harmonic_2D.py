"""
Example use case of the Semi-Global propagator. Driven Harmonic Oscillator 1D.
"""

# Import standard libraries.
import time
from typing import Optional, cast

# Import external libraries.
import numpy as np

# Import local libraries.
import quevolutio.core.simulation as sim
import quevolutio.propagators.semi_global as sg
import utils_2D as utils


class ConstantsDHO:
    hbar: float = 1.0
    mass: float = 1.0


class HamiltonianDHO:
    time_dependent: bool = True
    ke_time_dependent: bool = False
    pe_time_dependent: bool = True

    def __init__(
        self,
        domain: sim.QuantumHilbertSpace,
        eigenvalue_min: float,
        eigenvalue_max: float,
    ):
        # Assign attributes.
        self.domain = domain
        self.eigenvalue_min = eigenvalue_min
        self.eigenvalue_max = eigenvalue_max

        # Type checker candy.
        self.domain.constants = cast(ConstantsDHO, self.domain.constants)

        # Store kinetic energy operator.
        self.ke_operator = (
            (self.domain.momentum_grids[0] ** 2) / (2 * self.domain.constants.mass)
        ) + ((self.domain.momentum_grids[1] ** 2) / (2 * self.domain.constants.mass))

        # Store the standard potential operator.
        self.pe_operator_std = 0.5 * (
            (self.domain.position_grids[0] ** 2) + (self.domain.position_grids[1] ** 2)
        )

    def __call__(
        self, state: sim.GTensor, controls: Optional[sim.Controls] = None
    ) -> sim.GTensor:
        return self.ke_action(state, controls) + self.pe_action(state, controls)

    def ke_action(
        self, state: sim.GTensor, controls: Optional[sim.Controls] = None
    ) -> sim.GTensor:
        # Calculate the action of the kinetic energy operator.
        return self.domain.position_basis(
            self.ke_operator * self.domain.momentum_basis(state)
        )

    def pe_action(
        self, state: sim.GTensor, controls: Optional[sim.Controls] = None
    ) -> sim.GTensor:
        # Calculate the action of the potential energy operator.
        return (
            self.pe_operator_std + (self.domain.position_grids[0] * np.cos(controls))  # type: ignore
        ) * state


def controls_func(time: float) -> sim.Controls:
    return time


def main():
    # Define the constants.
    constants: ConstantsDHO = ConstantsDHO()

    # Define the domain.
    domain: sim.QuantumHilbertSpace = sim.QuantumHilbertSpace(
        2, (128, 128), np.array([[-10.0, 10.0], [-10.0, 10.0]]), constants
    )

    # Set up the wavefunction.
    wavefunction = np.exp(
        -((domain.position_grids[0]) ** 2 + (domain.position_grids[1]) ** 2) / 2
    )
    wavefunction = domain.normalise_state(wavefunction)

    # Set up the system.
    eigenvalue_min: float = 0.5
    eigenvalue_max: float = 150.0

    # Set up the Hamiltonian.
    hamiltonian: HamiltonianDHO = HamiltonianDHO(domain, eigenvalue_min, eigenvalue_max)
    system: sg.SemiGlobalTDSE = sg.SemiGlobalTDSE(domain, hamiltonian)

    # Set up the time domain.
    t_min: float = 0.0
    t_max: float = 10.0

    t_num_points: int = 10000
    t_num_points += 1

    time_domain: sim.TimeGrid = sim.TimeGrid(t_min, t_max, t_num_points)

    # Set up the simulation.
    order_m: int = 10
    order_f: int = 10

    tolerance: float = 1e-5
    approximation: sg.ApproximationBasis = sg.ApproximationBasis.NEWTONIAN

    # Propagate the wavefunction (timed).
    time_start: float = time.time()
    wavefunctions: sim.CVectors = sg.propagate(
        system,
        wavefunction,
        time_domain,
        order_m,
        order_f,
        tolerance,
        approximation,
        controls_func,
    )
    time_final: float = time.time()

    # Print the runtime.
    runtime: float = time_final - time_start
    print(f"Runtime: {runtime:.2f} seconds")

    filename = "test2D"
    np.save(f"{filename}.npy", wavefunctions)

    # Calculate the norms and energies.
    norms: sim.RVector = utils.wavefunctions_norms_2d(wavefunctions, domain)
    print(f"Max Norm Deviation: {np.max(np.abs(norms[0] - norms)):.2e}")

    # Generate figures and animation.
    filename = "test"
    utils.plot_wavefunctions_2d(
        wavefunctions[[0, -1]], domain, ["Initial", "Final"], f"{filename}.png"
    )
    utils.animate_wavefunctions_2d(
        wavefunctions, domain, time_domain, f"{filename}.mp4"
    )


if __name__ == "__main__":
    main()
