"""
Simple tests for the quevolutio.mathematical.approximation module.
"""

# Import external modules.
import numpy as np

# Import tested modules.
import quevolutio.mathematical.affine as affine
import quevolutio.mathematical.approximation as approx
from quevolutio.core.domain import HilbertSpace


def test_ch_gauss_nodes():
    """
    Tests for the approx.ch_gauss_nodes function.
    """

    # Generate the Chebyshev-Gauss nodes.
    num_nodes = 20
    nodes = approx.ch_gauss_nodes(num_nodes)

    # Check that the nodes are in the interval [-1, 1], excluding boundary points.
    assert np.all(nodes > -1.0) and np.all(nodes < 1.0)

    # Check that the nodes are monotonically increasing.
    assert np.all(np.diff(nodes) > 0.0)

    # Check that the nodes are symmetric.
    assert np.allclose(nodes, -nodes[::-1])


def test_ch_lobatto_nodes():
    """
    Tests for the approx.ch_lobatto_nodes function.
    """

    # Generate the Chebyshev-Lobatto nodes.
    num_nodes = 20
    nodes = approx.ch_lobatto_nodes(num_nodes)

    # Check that the nodes are in the interval [-1, 1], including boundary points.
    assert np.all(nodes[1:-1] > -1.0) and np.all(nodes[1:-1] < 1.0)
    assert np.isclose(nodes[0], -1.0)
    assert np.isclose(nodes[-1], 1.0)

    # Check that the nodes are monotonically increasing.
    assert np.all(np.diff(nodes) > 0.0)

    # Check that the nodes are symmetric.
    assert np.allclose(nodes, -nodes[::-1])


def test_ch_coefficients():
    """
    Tests for the approx.ch_coefficients function.
    """

    # Define a known function.
    def function(x):
        return x**3

    # Define the Chebyshev expansion.
    def ch_expansion(x, coefficients):
        # Store the number of expansion terms.
        order = coefficients.shape[0]

        # Calculate the first two Chebyshev expansion polynomials.
        polynomial_minus_2 = np.ones(len(x), dtype=np.float64)
        polynomial_minus_1 = x.copy()

        # Construct the starting expansion term.
        expansion = (coefficients[0] * polynomial_minus_2) + (
            coefficients[1] * polynomial_minus_1
        )

        # Construct the complete expansion.
        for i in range(2, order):
            polynomial_n = (2 * x * polynomial_minus_1) - polynomial_minus_2
            expansion += coefficients[i] * polynomial_n

            polynomial_minus_2 = polynomial_minus_1
            polynomial_minus_1 = polynomial_n

        return expansion

    # Construct the exact solution.
    x_axis = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
    function_exact = function(x_axis)

    # Construct the approximate solution using Chebyshev-Gauss nodes.
    order = 20
    nodes = approx.ch_gauss_nodes(order)

    function_nodes = function(nodes)
    function_coefficients = approx.ch_coefficients(function_nodes[::-1], dct_type=2)
    function_approx = ch_expansion(x_axis, function_coefficients)

    # Check that the coefficients match analytical values.
    assert np.isclose(function_coefficients[1], 0.75)
    assert np.isclose(function_coefficients[3], 0.25)

    mask = np.ones(order, dtype=np.bool_)
    mask[[1, 3]] = False
    assert np.allclose(function_coefficients[mask], 0.0)

    # Check that the approximated solution is similar to the exact solution.
    assert np.allclose(function_exact, function_approx)

    # Construct the approximate solution using Chebyshev-Lobatto nodes.
    order = 20
    nodes = approx.ch_lobatto_nodes(order)

    function_nodes = function(nodes)
    function_coefficients = approx.ch_coefficients(function_nodes[::-1], dct_type=1)
    function_approx = ch_expansion(x_axis, function_coefficients)

    # Check that the coefficients match analytical values.
    assert np.isclose(function_coefficients[1], 0.75)
    assert np.isclose(function_coefficients[3], 0.25)

    mask = np.ones(order, dtype=np.bool_)
    mask[[1, 3]] = False
    assert np.allclose(function_coefficients[mask], 0.0)

    # Check that the approximated solution is similar to the exact solution.
    assert np.allclose(function_exact, function_approx)


def test_ch_expansion():
    """
    Tests for the approx.ch_expansion function.
    """

    # Define the position operator.
    domain = HilbertSpace(
        1, np.array([100], dtype=np.int64), np.array([[-5.0, 5.0]], dtype=np.float64)
    )
    position = np.diag(domain.position_axes[0])

    # Define a wavefunction.
    wavefunction = np.exp(-domain.position_axes[0] ** 2)

    # Define an operator function.
    def function(x):
        return x**2

    # Calculate the exact solution.
    exact = position @ (position @ wavefunction)

    # Rescale the eigenvalue domain of the position operator to [-1, 1].
    position_rs, scale, shift = affine.rescale_tensor(position, -1, 1)

    # Define a function that returns the action of the position operator on a state.
    def operator(state, controls=None):
        return position_rs @ state

    # Calculate the approximate solution.
    order = 10
    nodes = (approx.ch_gauss_nodes(order) - shift) / scale

    function_nodes = function(nodes)
    function_coefficients = approx.ch_coefficients(function_nodes[::-1], dct_type=2)
    approximation = approx.ch_expansion(wavefunction, operator, function_coefficients)

    # Check that the approximated solution is similar to the exact solution.
    assert np.allclose(exact, approximation)


def test_ne_coefficients():
    """
    Tests for the approx.ne_coefficients function.
    """

    # Define a known function.
    def function(x):
        return x**3

    # Define the Newtonian interpolation expansion.
    def ne_expansion(x, nodes, coefficients):
        order = coefficients.shape[0]

        polynomial_n = np.ones(len(x), dtype=np.float64)
        expansion = coefficients[0] * polynomial_n

        for i in range(1, order):
            polynomial_n *= x - nodes[i - 1]
            expansion += coefficients[i] * polynomial_n

        return expansion

    # Construct the exact solution.
    x_axis = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
    function_exact = function(x_axis)

    # Construct the approximate solution using Chebyshev-Lobatto nodes.
    order = 20
    nodes = approx.ch_lobatto_nodes(order)

    function_nodes = function(nodes)
    function_coefficients = approx.ne_coefficients(nodes, function_nodes.reshape(-1, 1))
    function_approx = ne_expansion(x_axis, nodes, function_coefficients)

    # Check that the approximated solution is similar to the exact solution.
    assert np.allclose(function_exact, function_approx)
