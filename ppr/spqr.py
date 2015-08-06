#!/usr/bin/env python
__author__ = 'Rich Li'
""" SuiteSparseQR Python wrapper """

import os.path
import ctypes
from ctypes import c_double, c_size_t, byref, pointer, POINTER
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy import sparse

# Assume spqr_wrapper.so (or a link to it) is in the same directory as this file
print __file__
if __file__.startswith('/'):
    spqrlib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + os.path.sep + "spqr_wrapper.so")
else:
    spqrlib = ctypes.cdll.LoadLibrary("./spqr_wrapper.so")

# Function prototypes for spqr_wrapper.so
# void qr_solve(double const *A_data, double const *A_row, double const *A_col, size_t A_nnz, size_t A_m, size_t A_n, double const *b_data, double *x_data)
spqrlib.qr_solve.argtypes = [
        ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # A_data
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # A_row
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # A_col
        c_size_t,  # A_nnz
        c_size_t,  # A_m
        c_size_t,  # A_n
        ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # b_data
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # b_row
        c_size_t,  # b_nnz
        ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # Z_data
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # Z_row
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # Z_col
        ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # R_data
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS'), # R_row
        ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS') # R_col
        ]
spqrlib.qr_solve.restype = None

def qr_solve(A, b):
    """ Python wrapper to qr_solve """

    Z = sparse.coo_matrix(A.shape, dtype=np.float64)
    R = sparse.coo_matrix(A.shape, dtype=np.float64)



    spqrlib.qr_solve(
            np.require(A.data, np.float64, 'C'),
            np.require(A.row, np.int, 'C'),
            np.require(A.col, np.int, 'C'),
            A.nnz, A.shape[0], A.shape[1],
            np.require(b.data, np.float64, 'C'),
            np.require(b.row, np.int, 'C'),
            b.nnz,
            np.require(Z.data, np.float64, 'C'),
            np.require(Z.row, np.int, 'C'),
            np.require(Z.col, np.int, 'C'),
            np.require(R.data, np.float64, 'C'),
            np.require(R.row, np.int, 'C'),
            np.require(R.col, np.int, 'C')
            )
    print 'Done'
    #return x_data

def main():
    print("Testing qr_solve")
    '''
    A_data = np.array([5, 1, 5, 9, 1, 2, 4], dtype=np.float64)
    A_row = np.array([0, 0, 1, 1, 2, 2, 2])
    A_col = np.array([0, 1, 1, 2, 0, 1, 2])
    b_data = np.array([4, 2, 1], dtype=np.float64)
    b_row = np.array([0, 1, 2])
    '''
    A = sparse.coo_matrix([[5,1,0],[0,5,9],[1,2,4]], dtype=np.float64)
    b = sparse.coo_matrix([[4],[2],[1]], dtype=np.float64)

    fuck = qr_solve(A, b);
    print 'return',fuck
    #print(x_data)

if __name__ == "__main__":
    main()
