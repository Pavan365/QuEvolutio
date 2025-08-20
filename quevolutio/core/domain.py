"""
Classes for setting up the domain of a quantum system.
"""

# Import standard modules.
from typing import Protocol


class QuantumConstants(Protocol):
    """
    Interface for representing the physical constants of a quantum system. This
    class can be extended to contain system specific constants as required.

    Attributes
    ----------
    hbar : float
        The reduced Planck constant.
    """

    hbar: float
