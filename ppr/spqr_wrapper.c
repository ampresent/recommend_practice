// C wrapper to SparseSuiteQR library et al. for Python

// We pass in the sparse matrix data in a COO sparse matrix format. Cholmod
// refers to this as a "cholmod_triplet" format. This is then converted to its
// "cholmod_sparse" format, which is a CSC matrix.

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "SuiteSparseQR_C.h"
PyObject *pModule = NULL;
PyObject *pFunc = NULL;
PyObject *pDict = NULL;
PyObject *pClass = NULL;
PyObject *pGetnnz = NULL;

void qr(double const *A_data, long const *A_row, long const *A_col, size_t A_nnz, size_t A_m, size_t A_n, double const *b_data, long const * b_row, size_t b_nnz, PyListObject **Z_data, PyListObject **Z_row, PyListObject **Z_col, PyListObject **R_data, PyListObject **R_row, PyListObject **R_col){
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
	cholmod_sparse *A_csc, *b_csc, *Z_csc, *R_csc;
	cholmod_triplet *A_coo, *b_coo, *Z_coo, *R_coo;
	//SuiteSparse_long *P;
	size_t k, R_nnz, Z_nnz;
	// Helper pointers
	long *Ai, *Aj, *bi, *bj, *Zi, *Zj, *Ri, *Rj;
	double *Ax, *bx, *Zx, *Rx;

	/*
	   fprintf(stderr, "A_nnz = %d\n",A_nnz);
	   for (k=0;k<A_nnz;k++){
	   fprintf(stderr, "(%ld,%ld)\t%lf\n",A_row[k],A_col[k],A_data[k]);
	   }

	   fprintf(stderr, "b_nnz = %d\n",b_nnz);
	   for (k=0;k<b_nnz;k++){
	   fprintf(stderr, "(%ld,1)\t%lf\n",b_row[k],b_data[k]);
	   }

	   fprintf(stderr, "m = %zu, n = %zu, A_nnz = %zu\n", A_m, A_n, A_nnz);
	   */

	/* start CHOLMOD */
	cc = &Common ;
	cholmod_l_start (cc) ;

	// Create A, first as a COO matrix, then convert to CSC
	A_coo = cholmod_l_allocate_triplet(A_m, A_n, A_nnz, 0, CHOLMOD_REAL, cc);
	if (A_coo == NULL) {
		fprintf(stderr, "ERROR: cannot allocate triplet A");
		return;
	}
	// Copy in data
	Ai = A_coo->i;
	Aj = A_coo->j;
	Ax = A_coo->x;
	//fprintf(stderr, "sizeof(Ai[0]) = %zu\n", sizeof(Ai[0]));
	for (k=0; k<(size_t)A_nnz; k++) {
		Ai[k] = A_row[k];
		Aj[k] = A_col[k];
		Ax[k] = A_data[k];
		//fprintf(stderr, "Ai[k]=%ld,Aj[k]=%ld,Ax[kk]=%lf\n",A_row[k],A_col[k],A_data[k]);
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
		fprintf(stderr, "ERROR: cannot allocate triplet b");
		return;
	}
	bi = b_coo->i;
	bj = b_coo->j;
	bx = b_coo->x;
	for (k=0; k<(size_t)b_nnz; k++) {
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

	// TODO ORDER???
	int rank = SuiteSparseQR_C(0/*SPQR_ORDERING_DEFAULT*/, SPQR_DEFAULT_TOL, 0, 0, A_csc, b_csc, NULL, &Z_csc, NULL, &R_csc, NULL/*&P*/, NULL, NULL, NULL, cc);
	Z_coo = cholmod_l_sparse_to_triplet(Z_csc, cc);
	R_coo = cholmod_l_sparse_to_triplet(R_csc, cc);

	/*
	for (k=0;k<(size_t)rank;k++)
		fprintf(stderr, "%d,", P[k]);
	fprintf(stderr, "\n");
	*/

	Zi = Z_coo->i;
	Zj = Z_coo->j;
	Zx = Z_coo->x;
	Z_nnz = Z_coo->nnz;

	*Z_row = PyList_New(Z_nnz);
	*Z_col = PyList_New(Z_nnz);
	*Z_data = PyList_New(Z_nnz);
	for (k=0;k<Z_nnz;k++)
	{
		PyList_SetItem(*Z_row, k, Py_BuildValue("l", Zi[k]));
		PyList_SetItem(*Z_col, k, Py_BuildValue("l", Zj[k]));
		PyList_SetItem(*Z_data, k, Py_BuildValue("d", Zx[k]));
	}

	
	Py_INCREF(*Z_row);
	Py_INCREF(*Z_col);
	Py_INCREF(*Z_data);
	

	Ri = R_coo->i;
	Rj = R_coo->j;
	Rx = R_coo->x;
	R_nnz= R_coo->nnz;

	*R_row = PyList_New(R_nnz);
	*R_col = PyList_New(R_nnz);
	*R_data = PyList_New(R_nnz);
	for (k=0;k<R_nnz;k++)
	{
		PyList_SetItem(*R_row, k, Py_BuildValue("l", Ri[k]));
		PyList_SetItem(*R_col, k, Py_BuildValue("l", Rj[k]));
		PyList_SetItem(*R_data, k, Py_BuildValue("d", Rx[k]));
	}

	Py_INCREF(*R_row);
	Py_INCREF(*R_col);
	Py_INCREF(*R_data);
	
	/*
	Zi = malloc(sizeof(long) * rank);
	Zj = malloc(sizeof(long) * rank);
	Zx = malloc(sizeof(double) * rank);

	Zi = malloc(sizeof(long) * rank);
	Zj = malloc(sizeof(long) * rank);
	Zx = malloc(sizeof(double) * rank);

	for (k=0; k<Z_nnz;k++){
		Z_data[k] = Zx[k];
		Z_row[k] = Zi[k];
		Z_col[k] = Zj[k];
	}

	for (k=0; k<R_nnz;k++){
		R_data[k] = Rx[k];
		R_row[k] = Ri[k];
		R_col[k] = Rj[k];
	}
	*/
	/* free everything and finish CHOLMOD */
	cholmod_l_free_triplet(&A_coo, cc);
	cholmod_l_free_sparse(&A_csc, cc);
	cholmod_l_free_triplet(&b_coo, cc);
	cholmod_l_free_sparse(&b_csc, cc);
	cholmod_l_free_triplet(&Z_coo, cc);
	cholmod_l_free_sparse(&Z_csc, cc);
	cholmod_l_free_triplet(&R_coo, cc);
	cholmod_l_free_sparse(&R_csc, cc);
	//free(P);
	cholmod_l_finish(cc);
}
PyObject* wrap_qr(PyObject* self, PyObject* args) 
{
	PyListObject *pyA_data, *pyA_row, *pyA_col,
				 *pyb_data, *pyb_row,
				 *result;
	long A_nnz, b_nnz, A_n, A_m;
	size_t k;
	long *A_row, *A_col, *b_row;
	double *A_data, *b_data;
	PyObject *res;
	PyListObject *R_row, *R_col, *R_data,
				 *Z_row, *Z_col, *Z_data;

	if (!PyArg_ParseTuple(args, "OOOllOO",
				&pyA_data, &pyA_row, &pyA_col,
				&A_m, &A_n,
				&pyb_data, &pyb_row))
		return NULL;

	//result = PyObject_GetAttrString(A, "shape");
	//PyArg_Parse(result, "(ii)", &A_m, &A_n);

	A_nnz = pyA_data->ob_size;
	b_nnz = pyb_data->ob_size;

	A_row = malloc(sizeof(long) * A_nnz);
	A_col = malloc(sizeof(long) * A_nnz);
	A_data = malloc(sizeof(double) * A_nnz);

	for (k=0;k<(size_t)A_nnz;k++)
	{
		PyArg_Parse(pyA_data->ob_item[k], "d", A_data+k);
		PyArg_Parse(pyA_col->ob_item[k], "l", A_col+k);
		PyArg_Parse(pyA_row->ob_item[k], "l", A_row+k);
	}

	b_row = malloc(sizeof(long) * b_nnz);
	b_data = malloc(sizeof(double) * b_nnz);

	for (k=0;k<(size_t)b_nnz;k++)
	{
		PyArg_Parse(pyb_data->ob_item[k], "d", b_data+k);
		PyArg_Parse(pyb_row->ob_item[k], "l", b_row+k);
	}

	//PyArg_Parse(tmp, "i", &nnz); tmp->ob_item[0]

	qr(A_data, A_row, A_col, (size_t)A_nnz, (size_t)A_m, (size_t)A_n, b_data, b_row, (size_t)b_nnz, &Z_data, &Z_row, &Z_col, &R_data, &R_row, &R_col);

	//fprintf(stderr, "%p\n", R_col);

	res = Py_BuildValue("OOOOOO", Z_data, Z_row, Z_col,
								R_data, R_row, R_col);
	//Py_DECREF(Z_row);
	//Py_DECREF(Z_col);
	free(A_row);
	free(A_col);
	free(A_data);
	free(b_row);
	free(b_data);

	return res;
}
static PyMethodDef exampleMethods[] = 
{
	{"qr", wrap_qr, METH_VARARGS, "Sparse Matrix QR decomposition"},
	{NULL, NULL}
};
void initspqr_wrapper() 
{
	PyObject* m;
	/*
	pModule = PyImport_ImportModule("scipy.sparse");
	if ( !pModule ) 
	{ 
		fprintf(stderr, "Can't find scipy.sparse");
		return;
	} 
	pDict = PyModule_GetDict(pModule);
	if ( !pDict ) 
	{ 
		fprintf(stderr, "Can't read module scipy.sparse");
		return;
	} 
	pClass = PyDict_GetItemString(pDict, "coo_matrix"); 
	if ( !pClass)
	{ 
		fprintf(stderr, "Can't find class coo_matrix");
		return;
	} 
	*/
	/*
	   pDict = PyModule_GetDict(pClass);
	   if ( !pDict ) 
	   { 
	   fprintf(stderr, "Can't read class scipy.sparse.coo.coo_matrix");
	   return;
	   } 
	   pGetnnz = PyDict_GetItemString(pDict, "getnnz"); 
	   if ( !pGetnnz)
	   { 
	   fprintf(stderr, "Can't find method getnnz");
	   return;
	   } 
	   */
	//pInstance = PyInstance_New(pClass, Py_BuildValue("i", 1), Py_BuildValue("i",0));
	m = Py_InitModule("spqr_wrapper", exampleMethods);
	return;
}
