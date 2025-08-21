"""
Implementation of the Semi-Global propagation scheme for the time-dependent
Schrödinger equation.

References
----------
+ I. Schaefer et al. (2017). Available at: https://doi.org/10.1016/j.jcp.2017.04.017.
"""

# Import standard modules.
from enum import Enum


class ApproximationBasis(Enum):
    """
    Enumeration of the available approximation bases for the correction term in
    the Semi-Global propagation scheme.

    Members
    -------
    CHEBYSHEV: str
        Represents a Chebyshev basis expansion of the correction term.
    NEWTONIAN: str
        Represents a Newtonian basis expansion of the correction term.
    """

    CHEBYSHEV = "ch"
    NEWTONIAN = "ne"
