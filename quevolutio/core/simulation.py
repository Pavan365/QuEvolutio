"""
Core classes for setting up simulations.
"""

# Import standard modules.
from typing import Sequence, TypeAlias

# Import external modules.
import numpy as np
from numpy.typing import NDArray

# Type aliases for vectors.
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IVector: TypeAlias = NDArray[np.int64]
RVector: TypeAlias = NDArray[np.float64]
CVector: TypeAlias = NDArray[np.complex128]
GVector: TypeAlias = RVector | CVector

# Type aliases for collections of vectors (NumPy).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IVectors: TypeAlias = NDArray[np.int64]
RVectors: TypeAlias = NDArray[np.float64]
CVectors: TypeAlias = NDArray[np.complex128]
GVectors: TypeAlias = RVectors | CVectors

# Type aliases for collections of vectors (Python).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IVectorSeq: TypeAlias = Sequence[NDArray[np.int64]]
RVectorSeq: TypeAlias = Sequence[NDArray[np.float64]]
CVectorSeq: TypeAlias = Sequence[NDArray[np.complex128]]
GVectorSeq: TypeAlias = RVectorSeq | CVectorSeq

# Type aliases for matrices.
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IMatrix: TypeAlias = NDArray[np.int64]
RMatrix: TypeAlias = NDArray[np.float64]
CMatrix: TypeAlias = NDArray[np.complex128]
GMatrix: TypeAlias = RMatrix | CMatrix

# Type aliases for collections of matrices (NumPy).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IMatrices: TypeAlias = NDArray[np.int64]
RMatrices: TypeAlias = NDArray[np.float64]
CMatrices: TypeAlias = NDArray[np.complex128]
GMatrices: TypeAlias = RMatrices | CMatrices

# Type aliases for collections of matrices (Python).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
IMatrixSeq: TypeAlias = Sequence[NDArray[np.int64]]
RMatrixSeq: TypeAlias = Sequence[NDArray[np.float64]]
CMatrixSeq: TypeAlias = Sequence[NDArray[np.complex128]]
GMatrixSeq: TypeAlias = RMatrixSeq | CMatrixSeq

# Type aliases for tensors.
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
ITensor: TypeAlias = NDArray[np.int64]
RTensor: TypeAlias = NDArray[np.float64]
CTensor: TypeAlias = NDArray[np.complex128]
GTensor: TypeAlias = RTensor | CTensor

# Type aliases for collections of tensors (NumPy).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
ITensors: TypeAlias = NDArray[np.int64]
RTensors: TypeAlias = NDArray[np.float64]
CTensors: TypeAlias = NDArray[np.complex128]
GTensors: TypeAlias = RTensors | CTensors

# Type aliases for collections of tensors (Python).
# I -> Integer, R -> Real, C -> Complex, G -> General (floating).
ITensorSeq: TypeAlias = Sequence[NDArray[np.int64]]
RTensorSeq: TypeAlias = Sequence[NDArray[np.float64]]
CTensorSeq: TypeAlias = Sequence[NDArray[np.complex128]]
GTensorSeq: TypeAlias = RTensorSeq | CTensorSeq
