"""
Simple visual tests for modules within "quevolutio.mathematical".
"""

# Import external modules.
import matplotlib.pyplot as plt

# Import package modules.
import quevolutio.core.simulation as sim
import quevolutio.mathematical.approximation.chebyshev as ch


def plot_ch_gauss_nodes():
    """
    Test that the "gauss_nodes" function performs as expected.
    """

    # Generate Chebyshev-Gauss nodes.
    num_nodes: int = 20
    nodes: sim.RVector = ch.gauss_nodes(num_nodes)

    # Plot the Chebyshev-Gauss nodes.
    plt.plot(nodes)
    plt.title("Chebyshev-Gauss Nodes")
    plt.show()


def plot_ch_lobatto_nodes():
    """
    Test that the "lobatto_nodes" function performs as expected.
    """

    # Generate Chebyshev-Lobatto nodes.
    num_nodes: int = 20
    nodes: sim.RVector = ch.lobatto_nodes(num_nodes)

    # Plot the Chebyshev-Lobatto nodes.
    plt.plot(nodes)
    plt.title("Chebyshev-Lobatto Nodes")
    plt.show()


if __name__ == "__main__":
    plot_ch_gauss_nodes()
    plot_ch_lobatto_nodes()
