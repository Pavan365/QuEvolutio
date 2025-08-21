"""
Simple tests for the quevolutio.mathematical.affine module.
"""

# Import external modules.
import numpy as np

# Import tested modules.
import quevolutio.mathematical.affine as affine


def test_rescale_tensor():
    """
    Tests for the affine.rescale_tensor function.
    """

    # Define a starting domain and vector.
    lower, upper = -5.0, 5.0
    vector = np.linspace(lower, upper, 1000, dtype=np.float64)

    # Define a target domain and rescale the vector.
    a, b = -1.0, 1.0
    vector_rs, scale, shift = affine.rescale_tensor(vector, a, b)

    # Check that the boundary values are correct.
    assert np.isclose(np.min(vector_rs), a)
    assert np.isclose(np.max(vector_rs), b)

    # Check that the rescaled vector is monotonically increasing (assumes sorted).
    assert np.all(np.diff(vector_rs) > 0.0)

    # Check that the affine transformation factors are correct.
    assert np.allclose(vector_rs, (scale * vector) + shift)
    assert np.allclose(vector, (vector_rs - shift) / scale)


def test_rescale_matrix():
    """
    Tests for the affine.rescale_matrix function.
    """

    # Construct a matrix.
    matrix = np.diag(np.linspace(-200, 200, 20, dtype=np.float64))

    # Define a target domain and rescale the eigenvalue domain of the matrix.
    a, b = -1.0, 1.0
    matrix_rs, scale, shift = affine.rescale_matrix(matrix, a, b)

    # Get the rescaled eigenvalue domain.
    eigenvalues_rs = np.linalg.eigvalsh(matrix_rs)

    # Check that the boundary values are correct.
    assert np.isclose(np.min(eigenvalues_rs), a)
    assert np.isclose(np.max(eigenvalues_rs), b)

    # Check that the affine transformation factors are correct.
    identity = np.identity(matrix.shape[0], dtype=matrix.dtype)

    assert np.allclose(matrix_rs, (scale * matrix) + (shift * identity))
    assert np.allclose(matrix, (matrix_rs - (shift * identity)) / scale)
