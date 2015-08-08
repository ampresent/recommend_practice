import spqr_wrapper
import os
import sys
from scipy.sparse import *
import numpy as np

A = coo_matrix([[0,4,2],[1,0,3],[4,0,4]], dtype=np.float64)
b = coo_matrix([[5],[3],[0]], dtype=np.float64)

Z_data, Z_row, Z_col,\
R_data, R_row, R_col =\
        spqr_wrapper.qr(\
        A.data.tolist(),\
        A.row.tolist(),\
        A.col.tolist(),\
        A.shape[0], A.shape[1],\
        b.data.tolist(), \
        b.row.tolist())
print Z_col
print 'done'
#print type(E)

