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
    assert np.all(np.diff(vector_rs) >= 0.0)

    # Check that the affine transformation factors are correct.
    assert np.allclose(vector_rs, (scale * vector) + shift)
    assert np.allclose(vector, (vector_rs - shift) / scale)
