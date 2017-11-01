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
/*          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          */
/*          Date   :  June 2004                                      */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  Matrix Operation                                                 */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/memory.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "matope_sub.h"

void CovInvert(double **cov, int dim)
{
    int i, j;
    double *fx = NULL, *bx = NULL;
    double **Lmat = NULL;

    // memory allocation
    Lmat = xalloc(dim, double *);
    for (i = 0; i < dim; i++) Lmat[i] = xalloc(dim, double);

    // Choleski decomposition
    if (Choleski(cov, Lmat, (long)dim) == XFALSE) {
	fprintf(stderr, "Can't perform Choleski decomposition\n");
	exit(1);
    }

    // memory allocation
    fx = xalloc(dim, double);
    bx = xalloc(dim, double);

    // forward and backward substitutions
    for (i = 0; i < dim; i++) {
	Choleski_fsub(Lmat, fx, dim, i);
	Choleski_bsub(Lmat, fx, bx, dim, i);
	for (j = i + 1, cov[i][i] = bx[i]; j < dim; j++) {
	    cov[i][j] = bx[j];
	    cov[j][i] = bx[j];
	}
    }

    // memory free
    delete [] fx;	fx = NULL;
    delete [] bx;	bx = NULL;
    for (i = 0; i < dim; i++) {delete [] Lmat[i];	Lmat[i] = NULL;}
    delete [] Lmat;	Lmat = NULL;

    return;
}

// Choleski decomposition
XBOOL Choleski(double **mat, double **Lmat, long dim)
{
    long k, l, m;

    if (mat[0][0] < 0.0) return XFALSE;
    Lmat[0][0] = sqrt(mat[0][0]);
    for (l = 1; l < dim; l++) {
	Lmat[0][l] = mat[0][l] / Lmat[0][0];
	Lmat[l][0] = Lmat[0][l];
    }

    for (k = 1; k < dim; k++) {
	Lmat[k][k] = mat[k][k];
	for (l = k - 1; l >= 0; l--)
	    Lmat[k][k] -= Lmat[l][k] * Lmat[l][k];
	if (Lmat[k][k] < 0.0) return XFALSE;
	Lmat[k][k] = sqrt(Lmat[k][k]);
	for (l = k + 1; l < dim; l++) {
	    Lmat[k][l] = mat[k][l];
	    for (m = k - 1; m >= 0; m--)
		Lmat[k][l] -= Lmat[m][k] * Lmat[m][l];
	    Lmat[k][l] /= Lmat[k][k];
	    Lmat[l][k] = Lmat[k][l];
	}
    }

    return XTRUE;
}

void Choleski_fsub(double **Lmat, double *fx, long dim, long didx)
{
    long k, l;
    double sum;

    for (k = 0; k < didx; k++) fx[k] = 0.0;

    fx[didx] = 1.0 / Lmat[didx][didx];
    for (k = didx + 1; k < dim; k++) {
	for (l = didx, sum = 0.0; l < k; l++)
	    sum += Lmat[k][l] * fx[l];
	fx[k] = -sum / Lmat[k][k];
    }

    return;
}

void Choleski_bsub(double **Lmat, double *fx, double *bx, long dim, long didx)
{
    long k, l;
    double sum;

    bx[dim - 1] = fx[dim - 1] / Lmat[dim - 1][dim - 1];
    for (k = dim - 2; k >= didx; k--) {
	for (l = dim - 1, sum = 0.0; l > k; l--)
	    sum += Lmat[k][l] * bx[l];
	bx[k] = (fx[k] - sum) / Lmat[k][k];
    }

    return;
}

double get_det_CovInvert(DMATRIX c)
{
    DMATRIX l = NODATA;
    DVECTOR fx = NODATA, bx = NODATA;
    double ldet = 1.0;
    long i, j;

    // memory allocation
    l = xdmalloc(c->row, c->row);
    if (Choleski(c->data, l->data, c->row) == XTRUE){
	// memory allocation
	fx = xdvalloc(c->row);
	bx = xdvalloc(c->row);

	for (i = 0; i < c->row; i++) {
	    ldet *= l->data[i][i];
	    Choleski_fsub(l->data, fx->data, c->row, i);
	    Choleski_bsub(l->data, fx->data, bx->data, c->row, i);
	    for (j = i + 1, c->data[i][i] = bx->data[i]; j < c->row; j++) {
		c->data[i][j] = bx->data[j];
		c->data[j][i] = bx->data[j];
	    }
	}
	ldet = ldet * ldet;
	// memory free
	xdvfree(fx);
	xdvfree(bx);
    } else {
	fprintf(stderr, "Can't perform Choleski decomposition\n");
	exit(1);
    }
    // memory free
    xdmfree(l);

    return ldet;
}

// based on JDE
void get_diamat_jde(
		    DMATRIX covmat) // [dim*(number of class)][dim]
{
    long i, ri, ci;
    long clsnum;
    long dim2;

    clsnum = covmat->row / covmat->col;
    dim2 = covmat->col / 2;
    
    for (i = 0; i < clsnum; i++) {
	for (ri = 0; ri < covmat->col; ri++) {
	    for (ci = 0; ci < covmat->col; ci++) {
		if (ri != ci && (ri + dim2 != ci && ri - dim2 != ci)) {
		    covmat->data[ri + (i * covmat->col)][ci] = 0.0;
		}
	    }
	}
    }
}

DMATRIX xdmclone(
		 DMATRIX mat)
{
    long ri;
    DVECTOR vec;
    DMATRIX clonemat;

    clonemat = xdmzeros(mat->row, mat->col);

    for (ri = 0; ri < mat->row; ri++) {
	vec = xdmextractrow(mat, ri);
	dmcopyrow(clonemat, ri, vec);
	xdvfree(vec);
    }

    return clonemat;
}

// flooring
void floor_diamat(DMATRIX covmat, // [dim*(number of class)][dim]
		  DMATRIX bcovmat)
{
    long i, ri;
    long clsnum;

    clsnum = covmat->row / covmat->col;

    for (i = 0; i < clsnum; i++)
      for (ri = 0; ri < covmat->col; ri++)
	if (covmat->data[ri + (i * covmat->col)][ri] < bcovmat->data[ri][ri])
	  covmat->data[ri + (i * covmat->col)][ri] = bcovmat->data[ri][ri];

    return;
}

// joint matrixes row [A B]
DMATRIX xjoint_matrow(
		      DMATRIX mat1,
		      DMATRIX mat2)
{
    long ri, ci;
    DMATRIX jmat;
    
    // error check
    if (mat1->row != mat2->row) {
	fprintf(stderr, "Can't joint matrixes\n");
	return NODATA;
    }
    // memory allocation
    jmat = xdmalloc(mat1->row, (mat1->col+mat2->col));

    for (ri = 0; ri < mat1->row; ri++) {
	for (ci = 0; ci < mat1->col; ci++) {
	    jmat->data[ri][ci] = mat1->data[ri][ci];
	}
	for (ci = 0; ci < mat2->col; ci++) {
	    jmat->data[ri][ci + mat1->col] = mat2->data[ri][ci];
	}
    }

    return jmat;
}
// joint matrixes row [A B] using file
DMATRIX xjoint_matrow_file(
			   char *mat1file,
			   long mat1ncol,
			   char *mat2file,
			   long mat2ncol)
{
    long ri, ci;
    long length;
    long mat1nrow;
    long mat2nrow;
    char *basicname1;
    char *basicname2;
    DMATRIX jmat;
    DMATRIX mat1vmat;
    DMATRIX mat2vmat;
    FILE *fp1, *fp2;
    
    // get data length
    if ((length = getsiglen(mat1file, 0, double)) <= 0) {
	return NODATA;
    }
    if (length % mat1ncol != 0) {
	fprintf(stderr, "wrong data format: %s\n", mat1file);
	return NODATA;
    }
    mat1nrow = length / mat1ncol;
    if ((length = getsiglen(mat2file, 0, double)) <= 0) {
	return NODATA;
    }
    if (length % mat2ncol != 0) {
	fprintf(stderr, "wrong data format: %s\n", mat2file);
	return NODATA;
    }
    mat2nrow = length / mat2ncol;
    // error check
    if (mat1nrow != mat2nrow) {
	fprintf(stderr, "Can't joint matrixes\n");
	return NODATA;
    }
    // memory allocation
    jmat = xdmalloc(mat1nrow, (mat1ncol+mat2ncol));
    mat1vmat = xdmalloc(1, mat1ncol);
    mat2vmat = xdmalloc(1, mat2ncol);
    // get basic name
    basicname1 = xgetbasicname(mat1file);
    basicname2 = xgetbasicname(mat2file);

    if (streq(basicname1, "-") || streq(basicname1, "stdin")) {
	fp1 = stdin;
    } else {
	// open file
	if (NULL == (fp1 = fopen(mat1file, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", mat1file);
	    return NODATA;
	}
    }
    if (streq(basicname2, "-") || streq(basicname2, "stdin")) {
	fp2 = stdin;
    } else {
	// open file
	if (NULL == (fp2 = fopen(mat2file, "rb"))) {
	    fprintf(stderr, "can't open file: %s\n", mat2file);
	    return NODATA;
	}
    }

    for (ri = 0; ri < mat1nrow; ri++) {
	freaddouble(mat1vmat->data[0], mat1ncol, 0, fp1);
	for (ci = 0; ci < mat1ncol; ci++) {
	    jmat->data[ri][ci] = mat1vmat->data[0][ci];
	}
	freaddouble(mat2vmat->data[0], mat2ncol, 0, fp2);
	for (ci = 0; ci < mat2ncol; ci++) {
	    jmat->data[ri][ci + mat1ncol] = mat2vmat->data[0][ci];
	}
    }
    // close file
    if (fp1 != stdin) fclose(fp1);
    if (fp2 != stdin) fclose(fp2);
    // memory free
    xdmfree(mat1vmat);
    xdmfree(mat2vmat);

    return jmat;
}
