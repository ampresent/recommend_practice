// C wrapper to SparseSuiteQR library et al. for Python

// We pass in the sparse matrix data in a COO sparse matrix format. Cholmod
// refers to this as a "cholmod_triplet" format. This is then converted to its
// "cholmod_sparse" format, which is a CSC matrix.

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "SuiteSparseQR_C.h"

void qr_solve(double const *A_data, long const *A_row, long const *A_col, size_t A_nnz, size_t A_m, size_t A_n, double const *b_data, long const * b_row, size_t b_nnz){
    // Solves the matrix equation Ax=b where A is a sparse matrix and x and b
    // are dense column vectors. A and b are inputs, x is solved for in the
    // least squares sense using a rank-revealing QR factorization.
    //
    // Inputs
    //
    // A_data, A_row, A_col: the COO data
    // A_nnz: number of non-zero entries, ie the length of the arrays A_data, etc
    // A_m: number of rows in A
    // A_n: number of cols in A
    // b_data: the data in b. It is A_m entries long.
    //
    // Outputs
    //
    // x_data: the data in x. It is A_n entries long
    //
    // MAKE SURE x_data is allocated to the right size before calling this function
    //
    cholmod_common Common, *cc;
	// Why must it compress ??????
    cholmod_sparse *A_csc, *b_csc, *Z, *R, *P;
    cholmod_triplet *A_coo, *b_coo;
    size_t k;
    // Helper pointers
    long *Ai, *Aj, *bi, *bj;
    double *Ax, *bx;

    /* start CHOLMOD */
    cc = &Common ;
    cholmod_l_start (cc) ;

    // Create A, first as a COO matrix, then convert to CSC
    A_coo = cholmod_l_allocate_triplet(A_m, A_n, A_nnz, 0, CHOLMOD_REAL, cc);
    if (A_coo == NULL) {
        fprintf(stderr, "ERROR: cannot allocate triplet");
        return;
    }
    // Copy in data
    Ai = A_coo->i;
    Aj = A_coo->j;
    Ax = A_coo->x;
    for (k=0; k<A_nnz; k++) {
        Ai[k] = A_row[k];
        Aj[k] = A_col[k];
        Ax[k] = A_data[k];
    }
    A_coo->nnz = A_nnz;
    // Make sure the matrix is valid
    if (cholmod_l_check_triplet(A_coo, cc) != 1) {
        fprintf(stderr, "ERROR: triplet matrix is not valid");
        return;
    }
    // Convert to CSC
    A_csc = cholmod_l_triplet_to_sparse(A_coo, A_nnz, cc);

    // Create b as a dense matrix
    b_coo = cholmod_l_allocate_triplet(A_m, 1, b_nnz, 0, CHOLMOD_REAL, cc);
	if (b_coo == NULL) {
		fprintf(stderr, "ERROR: cannot allocate triplet");
		return;
	}
	bi = b_coo->i;
	bj = b_coo->j;
    bx = b_coo->x;
    for (k=0; k<b_nnz; k++) {
		bi[k] = b_row[k];
		bj[k] = 0; 
        bx[k] = b_data[k];
    }
	b_coo->nnz = b_nnz;
    // Make sure the matrix is valid
    if (cholmod_l_check_triplet(b_coo, cc) != 1) {
        fprintf(stderr, "ERROR: b vector is not valid");
        return;
    }
	b_csc = cholmod_l_triplet_to_sparse(b_coo, b_nnz, cc);
    //x = SuiteSparseQR_C_backslash_default(A_csc, b, cc);
	// What About econ???????????
	
	for (k=0; k<b_nnz;k++){
		fprintf(stderr, "%d -> %d\n", k, bx[k]);
	}
	int rank = SuiteSparseQR_C(SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, 0, 0, A_csc, b_csc, NULL, &Z, NULL, &R, &P, NULL, NULL, NULL, cc);


	/*
    // Return values of x
    xx = x->x;
    for (k=0; k<A_n; k++) {
        x_data[k] = xx[k];
    }
	*/

    /* free everything and finish CHOLMOD */
    cholmod_l_free_triplet(&A_coo, cc);
    cholmod_l_free_sparse(&A_csc, cc);
    cholmod_l_free_triplet(&b_coo, cc);
    cholmod_l_free_sparse(&b_csc, cc);
    cholmod_l_free_sparse(&Z, cc);
    //cholmod_l_free_sparse(&R, cc);???????
	// How to free E , or P ????????????? Refer to SPQR/Tcov/qrtest.cpp ??????
    // cholmod_l_free_sparse(&P, cc);
    cholmod_l_finish(cc);
    return;
}
PyObject* wrap_fact(PyObject* self, PyObject* args)
{
  int n, result;

  if (! PyArg_ParseTuple(args, "i:fact", &n))
    return NULL;
  result = fact(n);
  return Py_BuildValue("i", result);
}

static PyMethodDef exampleMethods[] =
{
  {"fact", wrap_fact, METH_VARARGS, "Caculate N!"},
  {NULL, NULL}
};

void initexample()
{
  PyObject* m;
  m = Py_InitModule("example", exampleMethods);
}

