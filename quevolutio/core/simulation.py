"""
Classes for setting up simulations.
"""

# Import standard modules.
from typing import TypeAlias

# Import external modules.
import numpy as np
from numpy.typing import NDArray

# Generalised type aliases for vectors.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IVector: TypeAlias = NDArray[np.int64]
GVector: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RVector: TypeAlias = NDArray[np.float64]
CVector: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of vectors.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IVectors: TypeAlias = NDArray[np.int64]
GVectors: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RVectors: TypeAlias = NDArray[np.float64]
CVectors: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for matrices.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IMatrix: TypeAlias = NDArray[np.int64]
GMatrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RMatrix: TypeAlias = NDArray[np.float64]
CMatrix: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of matrices.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
IMatrices: TypeAlias = NDArray[np.int64]
GMatrices: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RMatrices: TypeAlias = NDArray[np.float64]
CMatrices: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for tensors.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
ITensor: TypeAlias = NDArray[np.int64]
GTensor: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RTensor: TypeAlias = NDArray[np.float64]
CTensor: TypeAlias = NDArray[np.complex128]

# Generalised type aliases for collections of tensors.
# I -> Integer, G -> General (floating), R -> Real, C -> Complex.
ITensors: TypeAlias = NDArray[np.int64]
GTensors: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RTensors: TypeAlias = NDArray[np.float64]
CTensors: TypeAlias = NDArray[np.complex128]
