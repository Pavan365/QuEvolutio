# Import external modules.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

# Import local modules.
import quevolutio.core.simulation as sim


def hamiltonian_standard(domain: sim.QuantumHilbertSpace) -> sim.RMatrix:
    """
    Generates the Hamiltonian of a standard harmonic oscillator in natural
    units. This function uses a fourth-order central difference approximation
    to construct the kinetic energy operator in position space. This function
    also enforces periodic boundaries.

    Parameters
    ----------
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.

    Returns
    -------
    sim.RMatrix
        The Hamiltonian matrix.
    """

    # Construct the kinetic energy operator.
    ke_coeff: float = -1 / (2 * 12 * (domain.position_deltas[0] ** 2))

    ke_coeff_diag_0: float = -30 * ke_coeff
    ke_coeff_diag_1: float = 16 * ke_coeff
    ke_coeff_diag_2: float = -ke_coeff

    ke_diag_0: sim.RVector = np.full(
        domain.num_points, ke_coeff_diag_0, dtype=np.float64
    )
    ke_diag_1: sim.RVector = np.full(
        (domain.num_points[0] - 1), ke_coeff_diag_1, dtype=np.float64
    )
    ke_diag_2: sim.RVector = np.full(
        (domain.num_points[0] - 2), ke_coeff_diag_2, dtype=np.float64
    )

    ke_operator: sim.RMatrix = (
        np.diag(ke_diag_0, k=0)
        + np.diag(ke_diag_1, k=1)
        + np.diag(ke_diag_2, k=2)
        + np.diag(ke_diag_1, k=-1)
        + np.diag(ke_diag_2, k=-2)
    )

    # Enforce periodic boundaries.
    ke_operator[0, -1] = ke_coeff_diag_1
    ke_operator[-1, 0] = ke_coeff_diag_1

    ke_operator[0, -2] = ke_coeff_diag_2
    ke_operator[-2, 0] = ke_coeff_diag_2
    ke_operator[-1, 1] = ke_coeff_diag_2
    ke_operator[1, -1] = ke_coeff_diag_2

    # Construct the potential energy operator.
    pe_diag: sim.RVector = 0.5 * (domain.position_axes[0] ** 2)
    pe_operator: sim.RMatrix = np.diag(pe_diag, k=0)

    return ke_operator + pe_operator


def wavefunctions_norms(
    wavefunctions: sim.GVectors, domain: sim.QuantumHilbertSpace
) -> sim.RVector:
    """
    Calculates the norms of a set of wavefunctions over a given domain. This
    function uses the integral definition of the norm.

    Parameters
    ----------
    wavefunctions: simulation.GVectors
        The wavefunctions to calculate the norms of. These should be passed
        with shape (num_wavefunctions, num_points).
    domain: simulation.GVector
        The discretised Hilbert space (domain) of the system.

    Returns
    -------
    simulation.RVector
        The norms of the wavefunctions.
    """

    return np.sqrt(
        np.trapezoid((np.abs(wavefunctions) ** 2), dx=domain.position_deltas[0], axis=1)
    )


"""
Functions for plotting and animating wavefunctions.
"""


def plot_wavefunctions(
    wavefunctions: sim.GVectors,
    domain: sim.QuantumHilbertSpace,
    labels: list[str],
    filename: str,
) -> None:
    """
    Plots the probability densities of a set of wavefunctions.

    Parameters
    ----------
    wavefunctions: simulation.GVectors
        The wavefunctions to plot.
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    labels: list[str]
        The labels for the wavefunctions.
    filename: str
        The filename to save the plot to.
    """

    # Calculate the probability densities.
    prob_densities: sim.RVectors = np.abs(wavefunctions) ** 2

    # Plot the probability densities.
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(domain.position_bounds[0, 0], domain.position_bounds[0, 1])
    ax.set_ylim(0, np.max(prob_densities) * 1.05)

    for prob_density, label in zip(prob_densities, labels):
        ax.plot(domain.position_axes[0], prob_density, label=label)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\psi(x)|^{2}$")
    ax.legend(loc="upper right")

    fig.savefig(filename, dpi=300)
    plt.close(fig)


def animate_wavefunctions(
    wavefunctions: sim.GVectors,
    domain: sim.QuantumHilbertSpace,
    time_domain: sim.TimeGrid,
    filename: str,
) -> None:
    """
    Animates a set of wavefunctions over time. This function creates an
    animation that contains the probability densities of the wavefunctions
    along with the real and complex components.

    Parameters
    ----------
    wavefunctions: simulation.GVectors
        The wavefunctions to animate.
    domain: simulation.HilbertSpace1D
        The discretised Hilbert space (domain) of the system.
    time_domain: simulation.TimeGrid
        The time domain (grid) over which the wavefunctions are defined.
    filename: str
        The filename to save the animation to.
    """

    # Calculate the probability densities.
    prob_densities: sim.RVectors = np.abs(wavefunctions) ** 2

    # Store the real and complex components.
    wavefunctions_real: sim.RVectors = np.real(wavefunctions)
    wavefunctions_imag: sim.RVectors = np.imag(wavefunctions)

    # Animate the wavefunctions.
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(domain.position_bounds[0, 0], domain.position_bounds[0, 1])
    ax.set_ylim(-1.0, 1.0)

    (line_prob_density,) = ax.plot(
        domain.position_axes[0],
        prob_densities[0],
        label=r"$|\psi(x)|^{2}$",
        color="rebeccapurple",
    )
    (line_real,) = ax.plot(
        domain.position_axes[0],
        wavefunctions_real[0],
        "--",
        label=r"$\mathrm{Re}[\psi(x)]$",
        color="royalblue",
        alpha=0.75,
    )
    (line_imag,) = ax.plot(
        domain.position_axes[0],
        wavefunctions_imag[0],
        "--",
        label=r"$\mathrm{Im}[\psi(x)]$",
        color="crimson",
        alpha=0.75,
    )

    ax.set_title(r"$\text{Wavefunction} \; (T = 0.00)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\psi(x)|^2, \; \psi(x)$")
    ax.legend(loc="upper right")

    def animate(frame):
        line_prob_density.set_ydata(prob_densities[frame])
        line_real.set_ydata(wavefunctions_real[frame])
        line_imag.set_ydata(wavefunctions_imag[frame])

        ax.set_title(
            r"$\text{Wavefunction} \;$" + f"$(T = {time_domain.time_axis[frame]:.2f})$"
        )

        return line_prob_density, line_real, line_imag

    frames = range(0, wavefunctions.shape[0], int(wavefunctions.shape[0] * 0.01))
    fps, bitrate, dpi = 30, 2000, 200

    ani = FuncAnimation(fig, animate, frames, blit=True, interval=(1000 / fps))
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)

    ani.save(filename, writer, dpi=dpi)
    plt.close(fig)
