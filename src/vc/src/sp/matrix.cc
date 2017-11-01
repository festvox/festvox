/*********************************************************************/
/*                                                                   */
/*            Nagoya Institute of Technology, Aichi, Japan,          */
/*       Nara Institute of Science and Technology, Nara, Japan       */
/*                                and                                */
/*             Carnegie Mellon University, Pittsburgh, PA            */
/*                      Copyright (c) 2003-2004                      */
/*                        All Rights Reserved.                       */
/*                                                                   */
/*  Permission is hereby granted, free of charge, to use and         */
/*  distribute this software and its documentation without           */
/*  restriction, including without limitation the rights to use,     */
/*  copy, modify, merge, publish, distribute, sublicense, and/or     */
/*  sell copies of this work, and to permit persons to whom this     */
/*  work is furnished to do so, subject to the following conditions: */
/*                                                                   */
/*    1. The code must retain the above copyright notice, this list  */
/*       of conditions and the following disclaimer.                 */
/*    2. Any modifications must be clearly marked as such.           */
/*    3. Original authors' names are not deleted.                    */
/*                                                                   */    
/*  NAGOYA INSTITUTE OF TECHNOLOGY, NARA INSTITUTE OF SCIENCE AND    */
/*  TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, AND THE CONTRIBUTORS TO  */
/*  THIS WORK DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,  */
/*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, */
/*  IN NO EVENT SHALL NAGOYA INSTITUTE OF TECHNOLOGY, NARA           */
/*  INSTITUTE OF SCIENCE AND TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, */
/*  NOR THE CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR      */
/*  CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM   */
/*  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,  */
/*  NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN        */
/*  CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.         */
/*                                                                   */
/*********************************************************************/
/*                                                                   */
/*          Author :  Hideki Banno                                   */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  Slightly modified by Tomoki Toda (tomoki@ics.nitech.ac.jp)       */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/defs.h"
#include "../include/memory.h"
#include "../include/matrix.h"

/*
 *	allocate and free memory
 */
LMATRIX xlmalloc(long row, long col)
{
    LMATRIX matrix;

    matrix = xalloc(1, struct LMATRIX_STRUCT);
    matrix->data = xlmatalloc(row, col);
    matrix->imag = NULL;
    matrix->row = row;
    matrix->col = col;

    return matrix;
}

DMATRIX xdmalloc(long row, long col)
{
    DMATRIX matrix;

    matrix = xalloc(1, struct DMATRIX_STRUCT);
    matrix->data = xdmatalloc(row, col);
    matrix->imag = NULL;
    matrix->row = row;
    matrix->col = col;

    return matrix;
}

void xlmfree(LMATRIX matrix)
{
    if (matrix != NULL) {
	if (matrix->data != NULL) {
	    xlmatfree(matrix->data, matrix->row);
	}
	if (matrix->imag != NULL) {
	    xlmatfree(matrix->imag, matrix->row);
	}
	xfree(matrix);
    }

    return;
}

void xdmfree(DMATRIX matrix)
{
    if (matrix != NULL) {
	if (matrix->data != NULL) {
	    xdmatfree(matrix->data, matrix->row);
	}
	if (matrix->imag != NULL) {
	    xdmatfree(matrix->imag, matrix->row);
	}
	xfree(matrix);
    }

    return;
}

void lmialloc(LMATRIX x)
{
    if (x->imag != NULL) {
	xlmatfree(x->imag, x->row);
    }
    x->imag = xlmatalloc(x->row, x->col);

    return;
}

void dmialloc(DMATRIX x)
{
    if (x->imag != NULL) {
	xdmatfree(x->imag, x->row);
    }
    x->imag = xdmatalloc(x->row, x->col);

    return;
}

void lmifree(LMATRIX x)
{
    if (x->imag != NULL) {
	xlmatfree(x->imag, x->row);
    }
    
    return;
}

void dmifree(DMATRIX x)
{
    if (x->imag != NULL) {
	xdmatfree(x->imag, x->row);
    }
    
    return;
}

LMATRIX xlmrialloc(long row, long col)
{
    LMATRIX matrix;

    matrix = xlmalloc(row, col);
    lmialloc(matrix);

    return matrix;
}

DMATRIX xdmrialloc(long row, long col)
{
    DMATRIX matrix;

    matrix = xdmalloc(row, col);
    dmialloc(matrix);

    return matrix;
}

LMATRICES xlmsalloc(long num)
{
    long k;
    LMATRICES xs;

    xs = xalloc(1, struct LMATRICES_STRUCT);
    xs->matrix = xalloc(MAX(num, 1), LMATRIX);
    xs->num_matrix = num;
    
    for (k = 0; k < xs->num_matrix; k++) {
	xs->matrix[k] = NODATA;
    }

    return xs;
}

DMATRICES xdmsalloc(long num)
{
    long k;
    DMATRICES xs;

    xs = xalloc(1, struct DMATRICES_STRUCT);
    xs->matrix = xalloc(MAX(num, 1), DMATRIX);
    xs->num_matrix = num;
    
    for (k = 0; k < xs->num_matrix; k++) {
	xs->matrix[k] = NODATA;
    }

    return xs;
}

void xlmsfree(LMATRICES xs)
{
    long k;

    if (xs != NULL) {
	if (xs->matrix != NULL) {
	    for (k = 0; k < xs->num_matrix; k++) {
		if (xs->matrix[k] != NODATA) {
		    xlmfree(xs->matrix[k]);
		}
	    }
	    xfree(xs->matrix);
	}
	xfree(xs);
    }

    return;
}

void xdmsfree(DMATRICES xs)
{
    long k;

    if (xs != NULL) {
	if (xs->matrix != NULL) {
	    for (k = 0; k < xs->num_matrix; k++) {
		if (xs->matrix[k] != NODATA) {
		    xdmfree(xs->matrix[k]);
		}
	    }
	    xfree(xs->matrix);
	}
	xfree(xs);
    }

    return;
}

void lmreal(LMATRIX x)
{
    if (x->imag != NULL) {
	xlmatfree(x->imag, x->row);
    }
    
    return;
}

void dmreal(DMATRIX x)
{
    if (x->imag != NULL) {
	xdmatfree(x->imag, x->row);
    }
    
    return;
}

void lmimag(LMATRIX x)
{
    if (x->imag == NULL) {
	lmzeros(x, 0, 0);
	return;
    }

    xfree(x->data);
    x->data = x->imag;
    x->imag = NULL;
    
    return;
}

void dmimag(DMATRIX x)
{
    if (x->imag == NULL) {
	dmzeros(x, 0, 0);
	return;
    }

    xfree(x->data);
    x->data = x->imag;
    x->imag = NULL;
    
    return;
}

LMATRIX xlmnums(long row, long col, long value)
{
    long k, l;
    LMATRIX mat;

    if (row <= 0 || col <= 0) {
	fprintf(stderr, "wrong value\n");
#if 0
	mat = xlmnull();
	return mat;
#else
	return NODATA;
#endif
    }

    /* memory allocate */
    mat = xlmalloc(row, col);

    /* initailize data */
    for (k = 0; k < mat->row; k++) {
	for (l = 0; l < mat->col; l++) {
	    mat->data[k][l] = value;
	}
    }

    return mat;
}

DMATRIX xdmnums(long row, long col, double value)
{
    long k, l;
    DMATRIX mat;

    if (row <= 0 || col <= 0) {
	fprintf(stderr, "wrong value\n");
#if 0
	mat = xdmnull();
	return mat;
#else
	return NODATA;
#endif
    }

    /* memory allocate */
    mat = xdmalloc(row, col);

    /* initailize data */
    for (k = 0; k < mat->row; k++) {
	for (l = 0; l < mat->col; l++) {
	    mat->data[k][l] = value;
	}
    }

    return mat;
}

void lmnums(LMATRIX mat, long row, long col, long value)
{
    long k, l;

    if (row <= 0 || row > mat->row) {
	row = mat->row;
    }
    if (col <= 0 || col > mat->col) {
	col = mat->col;
    }

    /* initailize data */
    for (k = 0; k < row; k++) {
	for (l = 0; l < col; l++) {
	    mat->data[k][l] = value;
	}
    }

    return;
}

void dmnums(DMATRIX mat, long row, long col, double value)
{
    long k, l;

    if (row <= 0 || row > mat->row) {
	row = mat->row;
    }
    if (col <= 0 || col > mat->col) {
	col = mat->col;
    }

    /* initailize data */
    for (k = 0; k < row; k++) {
	for (l = 0; l < col; l++) {
	    mat->data[k][l] = value;
	}
    }

    return;
}

void lminums(LMATRIX mat, long row, long col, long value)
{
    long k, l;

    if (row <= 0 || row > mat->row) {
	row = mat->row;
    }
    if (col <= 0 || col > mat->col) {
	col = mat->col;
    }

    if (mat->imag == NULL) {
	/* memory allocate */
	lmizeros(mat, 0, 0);
    }

    /* initailize data */
    for (k = 0; k < row; k++) {
	for (l = 0; l < col; l++) {
	    mat->imag[k][l] = value;
	}
    }

    return;
}

void dminums(DMATRIX mat, long row, long col, double value)
{
    long k, l;

    if (row <= 0 || row > mat->row) {
	row = mat->row;
    }
    if (col <= 0 || col > mat->col) {
	col = mat->col;
    }

    if (mat->imag == NULL) {
	/* memory allocate */
	dmizeros(mat, 0, 0);
    }

    /* initailize data */
    for (k = 0; k < row; k++) {
	for (l = 0; l < col; l++) {
	    mat->imag[k][l] = value;
	}
    }

    return;
}

LMATRIX xlmrinums(long row, long col, long value)
{
    LMATRIX mat;

    mat = xlmnums(row, col, value);
    lmialloc(mat);
    lminums(mat, row, col, value);

    return mat;
}

DMATRIX xdmrinums(long row, long col, double value)
{
    DMATRIX mat;

    mat = xdmnums(row, col, value);
    dmialloc(mat);
    dminums(mat, row, col, value);

    return mat;
}

/*
 *	initialize each rows
 */
DMATRIX xdminitrow(long nrow, double j, double incr, double n)
{
    long k, l;
    long num;
    DMATRIX mat;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
#if 0
	mat = xdmnull();
	return mat;
#else
	return NODATA;
#endif
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    fprintf(stderr, "wrong value\n");
#if 0
	    mat = xdmnull();
	    return mat;
#else
	    return NODATA;
#endif
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    
    /* memory allocate */
    mat = xdmalloc(nrow, num);

    /* initailize data */
    for (k = 0; k < mat->row; k++) {
	for (l = 0; l < mat->col; l++) {
	    mat->data[k][l] = j + (l * incr);
	}
    }

    return mat;
}

/*
 *	initialize each columns
 */
DMATRIX xdminitcol(long ncol, double j, double incr, double n)
{
    long k, l;
    long num;
    DMATRIX mat;

    if ((incr > 0.0 && j > n) || (incr < 0.0 && j < n)) {
	fprintf(stderr, "bad increment value\n");
#if 0
	mat = xdmnull();
	return mat;
#else
	return NODATA;
#endif
    }
    if (incr == 0.0) {
	num = (long)n;
	if (num <= 0) {
	    fprintf(stderr, "wrong value\n");
#if 0
	    mat = xdmnull();
	    return mat;
#else
	    return NODATA;
#endif
	}
    } else {
	num = labs((long)((n - j) / incr)) + 1;
    }
    
    /* memory allocate */
    mat = xdmalloc(num, ncol);

    /* initailize data */
    for (l = 0; l < mat->col; l++) {
	for (k = 0; k < mat->row; k++) {
	    mat->data[k][l] = j + (k * incr);
	}
    }

    return mat;
}

/*
 *	cut one row of matrix
 */
LVECTOR xlmcutrow(LMATRIX mat, long row, long offset, long length)
{
    long k;
    long pos;
    LVECTOR vec;

    if (row < 0 || row >= mat->row) {
#if 0
	vec = xlvnull();
	return vec;
#else
	return NODATA;
#endif
    }

    if (mat->imag != NULL) {
	vec = xlvrizeros(length);
    } else {
	vec = xlvzeros(length);
    }

    for (k = 0; k < vec->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < mat->col) {
	    vec->data[k] = mat->data[row][pos];
	    if (vec->imag != NULL) {
		vec->imag[k] = mat->imag[row][pos];
	    }
	}
    }

    return vec;
}

DVECTOR xdmcutrow(DMATRIX mat, long row, long offset, long length)
{
    long k;
    long pos;
    DVECTOR vec;

    if (row < 0 || row >= mat->row) {
#if 0
	vec = xdvnull();
	return vec;
#else
	return NODATA;
#endif
    }

    if (mat->imag != NULL) {
	vec = xdvrizeros(length);
    } else {
	vec = xdvzeros(length);
    }

    for (k = 0; k < vec->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < mat->col) {
	    vec->data[k] = mat->data[row][pos];
	    if (vec->imag != NULL) {
		vec->imag[k] = mat->imag[row][pos];
	    }
	}
    }

    return vec;
}

/*
 *	cut one column of matrix
 */
LVECTOR xlmcutcol(LMATRIX mat, long col, long offset, long length)
{
    long k;
    long pos;
    LVECTOR vec;

    if (col < 0 || col >= mat->col) {
#if 0
	vec = xlvnull();
	return vec;
#else
	return NODATA;
#endif
    }

    if (mat->imag != NULL) {
	vec = xlvrizeros(length);
    } else {
	vec = xlvzeros(length);
    }

    for (k = 0; k < vec->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < mat->row) {
	    vec->data[k] = mat->data[pos][col];
	    if (vec->imag != NULL) {
		vec->imag[k] = mat->imag[pos][col];
	    }
	}
    }

    return vec;
}

DVECTOR xdmcutcol(DMATRIX mat, long col, long offset, long length)
{
    long k;
    long pos;
    DVECTOR vec;

    if (col < 0 || col >= mat->col) {
#if 0
	vec = xdvnull();
	return vec;
#else
	return NODATA;
#endif
    }

    if (mat->imag != NULL) {
	vec = xdvrizeros(length);
    } else {
	vec = xdvzeros(length);
    }

    for (k = 0; k < vec->length; k++) {
	pos = k + offset;
	if (pos >= 0 && pos < mat->row) {
	    vec->data[k] = mat->data[pos][col];
	    if (vec->imag != NULL) {
		vec->imag[k] = mat->imag[pos][col];
	    }
	}
    }

    return vec;
}

/*
 *	paste vector on the row of matrix 
 */
void lmpasterow(LMATRIX mat, long row, LVECTOR vec,
		long offset, long length, int overlap)
{
    long k;
    long pos;

    if (row < 0 || row >= mat->row) {
	return;
    }
    if (length <= 0 || length > vec->length) {
	length = vec->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->col) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[row][pos] += vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[row][pos] += vec->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->col) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[row][pos] = vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[row][pos] = vec->imag[k];
		}
	    }
	}
    }
	
    return;
}

void dmpasterow(DMATRIX mat, long row, DVECTOR vec,
		long offset, long length, int overlap)
{
    long k;
    long pos;

    if (row < 0 || row >= mat->row) {
	return;
    }
    if (length <= 0 || length > vec->length) {
	length = vec->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->col) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[row][pos] += vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[row][pos] += vec->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->col) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[row][pos] = vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[row][pos] = vec->imag[k];
		}
	    }
	}
    }
	
    return;
}

/*
 *	paste vector on the column of matrix 
 */
void lmpastecol(LMATRIX mat, long col, LVECTOR vec,
		long offset, long length, int overlap)
{
    long k;
    long pos;

    if (col < 0 || col >= mat->col) {
	return;
    }
    if (length <= 0 || length > vec->length) {
	length = vec->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->row) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[pos][col] += vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[pos][col] += vec->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->row) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[pos][col] = vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[pos][col] = vec->imag[k];
		}
	    }
	}
    }
	
    return;
}

void dmpastecol(DMATRIX mat, long col, DVECTOR vec,
		long offset, long length, int overlap)
{
    long k;
    long pos;

    if (col < 0 || col >= mat->col) {
	return;
    }
    if (length <= 0 || length > vec->length) {
	length = vec->length;
    }

    if (overlap) {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->row) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[pos][col] += vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[pos][col] += vec->imag[k];
		}
	    }
	}
    } else {
	for (k = 0; k < length; k++) {
	    pos = k + offset;
	    if (pos >= mat->row) {
		break;
	    }
	    if (pos >= 0) {
		mat->data[pos][col] = vec->data[k];
		if (vec->imag != NULL && mat->imag != NULL) {
		    mat->imag[pos][col] = vec->imag[k];
		}
	    }
	}
    }
	
    return;
}

LVECTOR xlmrmax(LMATRIX mat)
{
    long k, l;
    long index;
    long max;
    LVECTOR x;

    x = xlvalloc(mat->row);

    for (k = 0; k < mat->row; k++) {
	max = mat->data[k][0];
	index = 0;
	for (l = 1; l < mat->col; l++) {
	    if (max < mat->data[k][l]) {
		max = mat->data[k][l];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xdmrmax(DMATRIX mat)
{
    long k, l;
    long index;
    double max;
    LVECTOR x;

    x = xlvalloc(mat->row);

    for (k = 0; k < mat->row; k++) {
	max = mat->data[k][0];
	index = 0;
	for (l = 1; l < mat->col; l++) {
	    if (max < mat->data[k][l]) {
		max = mat->data[k][l];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xlmrmin(LMATRIX mat)
{
    long k, l;
    long index;
    long min;
    LVECTOR x;

    x = xlvalloc(mat->row);

    for (k = 0; k < mat->row; k++) {
	min = mat->data[k][0];
	index = 0;
	for (l = 1; l < mat->col; l++) {
	    if (min > mat->data[k][l]) {
		min = mat->data[k][l];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xdmrmin(DMATRIX mat)
{
    long k, l;
    long index;
    double min;
    LVECTOR x;

    x = xlvalloc(mat->row);

    for (k = 0; k < mat->row; k++) {
	min = mat->data[k][0];
	index = 0;
	for (l = 1; l < mat->col; l++) {
	    if (min > mat->data[k][l]) {
		min = mat->data[k][l];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xlmrextract(LMATRIX mat, LVECTOR idx)
{
    long k;
    LVECTOR x;

    x = xlvalloc(idx->length);
    if (mat->imag != NULL) {
	lvialloc(x);
    }

    for (k = 0; k < x->length; k++) {
	if (k < mat->row && 
	    idx->data[k] >= 0 && idx->data[k] < mat->col) {
	    x->data[k] = mat->data[k][idx->data[k]];
	    if (x->imag != NULL) {
		x->imag[k] = mat->imag[k][idx->data[k]];
	    }
	} else {
	    x->data[k] = 0;
	    if (x->imag != NULL) {
		x->imag[k] = 0;
	    }
	}
    }

    return x;
}

DVECTOR xdmrextract(DMATRIX mat, LVECTOR idx)
{
    long k;
    DVECTOR x;

    x = xdvalloc(idx->length);
    if (mat->imag != NULL) {
	dvialloc(x);
    }

    for (k = 0; k < x->length; k++) {
	if (k < mat->row && 
	    idx->data[k] >= 0 && idx->data[k] < mat->col) {
	    x->data[k] = mat->data[k][idx->data[k]];
	    if (x->imag != NULL) {
		x->imag[k] = mat->imag[k][idx->data[k]];
	    }
	} else {
	    x->data[k] = 0.0;
	    if (x->imag != NULL) {
		x->imag[k] = 0.0;
	    }
	}
    }

    return x;
}

LVECTOR xlmcmax(LMATRIX mat)
{
    long k, l;
    long index;
    long max;
    LVECTOR x;

    x = xlvalloc(mat->col);

    for (k = 0; k < mat->col; k++) {
	max = mat->data[0][k];
	index = 0;
	for (l = 1; l < mat->row; l++) {
	    if (max < mat->data[l][k]) {
		max = mat->data[l][k];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xdmcmax(DMATRIX mat)
{
    long k, l;
    long index;
    double max;
    LVECTOR x;

    x = xlvalloc(mat->col);

    for (k = 0; k < mat->col; k++) {
	max = mat->data[0][k];
	index = 0;
	for (l = 1; l < mat->row; l++) {
	    if (max < mat->data[l][k]) {
		max = mat->data[l][k];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xlmcmin(LMATRIX mat)
{
    long k, l;
    long index;
    long min;
    LVECTOR x;

    x = xlvalloc(mat->col);

    for (k = 0; k < mat->col; k++) {
	min = mat->data[0][k];
	index = 0;
	for (l = 1; l < mat->row; l++) {
	    if (min > mat->data[l][k]) {
		min = mat->data[l][k];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xdmcmin(DMATRIX mat)
{
    long k, l;
    long index;
    double min;
    LVECTOR x;

    x = xlvalloc(mat->col);

    for (k = 0; k < mat->col; k++) {
	min = mat->data[0][k];
	index = 0;
	for (l = 1; l < mat->row; l++) {
	    if (min > mat->data[l][k]) {
		min = mat->data[l][k];
		index = l;
	    }
	}
	x->data[k] = index;
    }

    return x;
}

LVECTOR xlmcextract(LMATRIX mat, LVECTOR idx)
{
    long k;
    LVECTOR x;

    x = xlvalloc(idx->length);
    if (mat->imag != NULL) {
	lvialloc(x);
    }

    for (k = 0; k < x->length; k++) {
	if (k < mat->col && 
	    idx->data[k] >= 0 && idx->data[k] < mat->row) {
	    x->data[k] = mat->data[idx->data[k]][k];
	    if (x->imag != NULL) {
		x->imag[k] = mat->imag[idx->data[k]][k];
	    }
	} else {
	    x->data[k] = 0;
	    if (x->imag != NULL) {
		x->imag[k] = 0;
	    }
	}
    }

    return x;
}

DVECTOR xdmcextract(DMATRIX mat, LVECTOR idx)
{
    long k;
    DVECTOR x;

    x = xdvalloc(idx->length);
    if (mat->imag != NULL) {
	dvialloc(x);
    }

    for (k = 0; k < x->length; k++) {
	if (k < mat->col && 
	    idx->data[k] >= 0 && idx->data[k] < mat->row) {
	    x->data[k] = mat->data[idx->data[k]][k];
	    if (x->imag != NULL) {
		x->imag[k] = mat->imag[idx->data[k]][k];
	    }
	} else {
	    x->data[k] = 0.0;
	    if (x->imag != NULL) {
		x->imag[k] = 0.0;
	    }
	}
    }

    return x;
}
#if 0
#endif
