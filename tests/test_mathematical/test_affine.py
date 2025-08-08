"""
Simple test cases for the "quevolutio.mathematical.affine" module.
"""

# Import external modules.
import numpy as np

# Import package modules.
import quevolutio.core.simulation as sim
import quevolutio.mathematical.affine as affine


def test_rescale_tensor() -> None:
    """
    Test that the "rescale_tensor" function performs as expected.
    """

    # Define a target interval.
    a: float = 0.0
    b: float = 1.0

    # Construct a simple vector, matrix and tensor.
    limit: float = 200.0
    num_points: int = 100

    vector: sim.RVector = np.linspace(-limit, limit, num_points, dtype=np.float64)
    matrix: sim.RMatrix = np.tile(vector, (2, 1))
    tensor: sim.RTensor = np.tile(vector, (2, 2, 1))

    # Rescale the vector, matrix and tensor.
    vector_rs, v_scale, v_shift = affine.rescale_tensor(vector, a, b)
    matrix_rs, m_scale, m_shift = affine.rescale_tensor(matrix, a, b)
    tensor_rs, t_scale, t_shift = affine.rescale_tensor(tensor, a, b)

    # Check that the rescaled objects are as expected.
    assert np.isclose(np.min(vector_rs), a)
    assert np.isclose(np.max(vector_rs), b)

    assert np.isclose(np.min(matrix_rs), a)
    assert np.isclose(np.max(matrix_rs), b)

    assert np.isclose(np.min(tensor_rs), a)
    assert np.isclose(np.max(tensor_rs), b)

    # Check that the scale and shift factors perform the correct inverse transformation.
    vector_original: sim.RVector = (vector_rs - v_shift) / v_scale
    matrix_original: sim.RMatrix = (matrix_rs - m_shift) / m_scale
    tensor_original: sim.RTensor = (tensor_rs - t_shift) / t_scale

    assert np.allclose(vector, vector_original)
    assert np.allclose(matrix, matrix_original)
    assert np.allclose(tensor, tensor_original)
