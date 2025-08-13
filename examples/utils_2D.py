import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import quevolutio.core.simulation as sim


def wavefunctions_norms_2d(
    wavefunctions: sim.GVectors, domain: sim.QuantumHilbertSpace
) -> sim.RVector:
    """
    Calculates the norms of a set of 2D wavefunctions over a given domain.
    Uses the integral definition of the norm.
    """
    dx, dy = domain.position_deltas
    # Sum over both spatial axes
    return np.sqrt(np.sum(np.abs(wavefunctions) ** 2, axis=(1, 2)) * dx * dy)


def plot_wavefunctions_2d(
    wavefunctions: sim.GVectors,
    domain: sim.QuantumHilbertSpace,
    labels: list[str],
    filename: str,
):
    """
    Plots the probability densities of a set of 2D wavefunctions as heatmaps.
    """
    x_grid = domain.position_axes[0]
    y_grid = domain.position_axes[1]

    for psi, label in zip(wavefunctions, labels):
        plt.figure(figsize=(6, 5))
        plt.imshow(
            np.abs(psi) ** 2,
            origin="lower",
            extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            aspect="auto",
            cmap="viridis",
        )
        plt.colorbar(label=r"$|\psi(x,y)|^2$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(f"{filename}_{label}.png", dpi=300)
        plt.close()


def animate_wavefunctions_2d(
    wavefunctions: sim.GVectors,
    domain: sim.QuantumHilbertSpace,
    time_domain: sim.TimeGrid,
    filename: str,
):
    """
    Animates 2D wavefunctions over time as a probability density heatmap.
    """
    x_grid = domain.position_axes[0]
    y_grid = domain.position_axes[1]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        np.abs(wavefunctions[0]) ** 2,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=np.max(np.abs(wavefunctions) ** 2),
    )
    cbar = fig.colorbar(im, ax=ax, label=r"$|\psi(x,y)|^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"T = {time_domain.time_axis[0]:.2f}")

    def animate(frame):
        im.set_data(np.abs(wavefunctions[frame]) ** 2)
        ax.set_title(f"T = {time_domain.time_axis[frame]:.2f}")
        return (im,)

    frames = range(0, wavefunctions.shape[0], max(1, wavefunctions.shape[0] // 100))
    fps, bitrate, dpi = 30, 2000, 200

    ani = FuncAnimation(fig, animate, frames, blit=True, interval=(1000 / fps))
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save(filename, writer, dpi=dpi)
    plt.close(fig)
