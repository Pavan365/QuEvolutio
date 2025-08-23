"""
Type aliases for the QuEvolutio framework.
"""

# Import standard modules.
from typing import TypeAlias, Sequence

# Import external modules.
import scipy.sparse as sp
from numpy import int64, float64, complex128
from numpy.typing import NDArray

## Semantic type aliases for mathematical objects (dense representations).
## I -> Integer, R -> Real, C -> Complex, G -> General (floating).

# Type aliases for vectors.
IVector: TypeAlias = NDArray[int64]
RVector: TypeAlias = NDArray[float64]
CVector: TypeAlias = NDArray[complex128]
GVector: TypeAlias = RVector | CVector

# Type aliases for collections of vectors (NumPy).
IVectors: TypeAlias = NDArray[int64]
RVectors: TypeAlias = NDArray[float64]
CVectors: TypeAlias = NDArray[complex128]
GVectors: TypeAlias = RVectors | CVectors

# Type aliases for collections of vectors (Python).
IVectorSeq: TypeAlias = Sequence[IVector]
RVectorSeq: TypeAlias = Sequence[RVector]
CVectorSeq: TypeAlias = Sequence[CVector]
GVectorSeq: TypeAlias = RVectorSeq | CVectorSeq

# Type aliases for matrices.
IMatrix: TypeAlias = NDArray[int64]
RMatrix: TypeAlias = NDArray[float64]
CMatrix: TypeAlias = NDArray[complex128]
GMatrix: TypeAlias = RMatrix | CMatrix

# Type aliases for collections of matrices (NumPy).
IMatrices: TypeAlias = NDArray[int64]
RMatrices: TypeAlias = NDArray[float64]
CMatrices: TypeAlias = NDArray[complex128]
GMatrices: TypeAlias = RMatrices | CMatrices

# Type aliases for collections of matrices (Python).
IMatrixSeq: TypeAlias = Sequence[IMatrix]
RMatrixSeq: TypeAlias = Sequence[RMatrix]
CMatrixSeq: TypeAlias = Sequence[CMatrix]
GMatrixSeq: TypeAlias = RMatrixSeq | CMatrixSeq

# Type aliases for tensors.
ITensor: TypeAlias = NDArray[int64]
RTensor: TypeAlias = NDArray[float64]
CTensor: TypeAlias = NDArray[complex128]
GTensor: TypeAlias = RTensor | CTensor

# Type aliases for collections of tensors (NumPy).
ITensors: TypeAlias = NDArray[int64]
RTensors: TypeAlias = NDArray[float64]
CTensors: TypeAlias = NDArray[complex128]
GTensors: TypeAlias = RTensors | CTensors

# Type aliases for collections of tensors (Python).
ITensorSeq: TypeAlias = Sequence[ITensor]
RTensorSeq: TypeAlias = Sequence[RTensor]
CTensorSeq: TypeAlias = Sequence[CTensor]
GTensorSeq: TypeAlias = RTensorSeq | CTensorSeq

## Semantic type aliases for mathematical objects (sparse representations).
## SP -> GENERAL SPARSE MATRIX
## BSR -> Block Sparse Row
## COO -> Coordinate
## CSC -> Compressed Sparse Column
## CSR -> Compressed Sparse Row
## DIA -> Diagonal
## DOK -> Dictionary Of Keys
## LIL -> List Of Lists

# Type aliases for sparse matrices.
SPMatrix: TypeAlias = sp.spmatrix
BSRMatrix: TypeAlias = sp.bsr_matrix
COOMatrix: TypeAlias = sp.coo_matrix
CSCMatrix: TypeAlias = sp.csc_matrix
CSRMatrix: TypeAlias = sp.csr_matrix
DIAMatrix: TypeAlias = sp.dia_matrix
DOKMatrix: TypeAlias = sp.dok_matrix
LILMatrix: TypeAlias = sp.lil_matrix

# Type aliases for collections of sparse matrices (Python).
SPMatrixSeq: TypeAlias = Sequence[SPMatrix]
BSRMatrixSeq: TypeAlias = Sequence[BSRMatrix]
COOMatrixSeq: TypeAlias = Sequence[COOMatrix]
CSCMatrixSeq: TypeAlias = Sequence[CSCMatrix]
CSRMatrixSeq: TypeAlias = Sequence[CSRMatrix]
DIAMatrixSeq: TypeAlias = Sequence[DIAMatrix]
DOKMatrixSeq: TypeAlias = Sequence[DOKMatrix]
LILMatrixSeq: TypeAlias = Sequence[LILMatrix]
