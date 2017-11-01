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
/*  GMM subroutine                                                   */
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
#include "gmm_sub.h"

DVECTOR xget_paramvec(DMATRIX weightmat,	// [num class][1]
		      DMATRIX meanmat,		// [num class][dim]
		      DMATRIX covmat,		// [num class * dim][dim]
		      long ydim,
		      XBOOL dia_flag)
{
    long p, b, i, ri, ci, clsnum, dim, xdim;
    DVECTOR param = NODATA;

    clsnum = weightmat->row;
    dim = meanmat->col;
    xdim = dim - ydim;

    // memory allocation
    if (xdim == ydim && dia_flag == XTRUE)	// all diagonal cov
	param = xdvalloc(clsnum * (1 + dim + dim + ydim));
    else if (dia_flag == XTRUE)	// XX diagonal cov
	param = xdvalloc(clsnum * (1 + dim + xdim + ydim * xdim
				   + ydim * ydim));
    else	// all full cov
	param = xdvalloc(clsnum * (1 + dim + xdim * xdim + ydim * xdim
				   + ydim * ydim));
    p = 0;
    // weight
    for (ri = 0; ri < clsnum; ri++, p++)
	param->data[p] = weightmat->data[ri][0];
    // X mean
    for (ri = 0; ri < clsnum; ri++)
	for (ci = 0; ci < xdim; ci++, p++)
	    param->data[p] = meanmat->data[ri][ci];
    // Y mean
    for (ri = 0; ri < clsnum; ri++)
	for (ci = 0; ci < ydim; ci++, p++)
	    param->data[p] = meanmat->data[ri][ci + xdim];
    // XX cov
    if (dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++, p++)
		param->data[p] = covmat->data[ri + b][ri];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++)
		for (ci = 0; ci < xdim; ci++, p++)
		    param->data[p] = covmat->data[ri + b][ci];
    // YX cov
    if (xdim == ydim && dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++, p++)
		param->data[p] = covmat->data[ri + b + xdim][ri];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < ydim; ri++)
		for (ci = 0; ci < xdim; ci++, p++)
		    param->data[p] = covmat->data[ri + b + xdim][ci];
    // YY cov
    if (xdim == ydim && dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < ydim; ri++, p++)
		param->data[p] = covmat->data[ri + b + xdim][ri + xdim];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < ydim; ri++)
		for (ci = 0; ci < ydim; ci++, p++)
		    param->data[p] = covmat->data[ri + b + xdim][ci + xdim];

    if (p != param->length) {
	fprintf(stderr, "Error xget_paramvec\n");
	exit(1);
    }

    return param;
}

DVECTOR xget_paramvec_yx(DMATRIX weightmat,	// [num class][1]
			 DMATRIX meanmat,	// [num class][dim]
			 DMATRIX covmat,	// [num class * dim][dim]
			 long ydim,
			 XBOOL dia_flag)
{
    long p, b, i, ri, ci, clsnum, dim, xdim;
    DVECTOR param = NODATA;

    clsnum = weightmat->row;
    dim = meanmat->col;
    xdim = dim - ydim;

    // memory allocation
    if (xdim == ydim && dia_flag == XTRUE)	// all diagonal cov
	param = xdvalloc(clsnum * (1 + dim + dim + ydim));
    else if (dia_flag == XTRUE)	// YY diagonal cov
	param = xdvalloc(clsnum * (1 + dim + ydim + ydim * xdim
				   + xdim * xdim));
    else	// all full cov
	param = xdvalloc(clsnum * (1 + dim + ydim * ydim + ydim * xdim
				   + xdim * xdim));
    p = 0;
    // weight
    for (ri = 0; ri < clsnum; ri++, p++)
	param->data[p] = weightmat->data[ri][0];
    // Y mean
    for (ri = 0; ri < clsnum; ri++)
	for (ci = 0; ci < ydim; ci++, p++)
	    param->data[p] = meanmat->data[ri][ci + xdim];
    // X mean
    for (ri = 0; ri < clsnum; ri++)
	for (ci = 0; ci < xdim; ci++, p++)
	    param->data[p] = meanmat->data[ri][ci];
    // YY cov
    if (dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < ydim; ri++, p++)
		param->data[p] = covmat->data[ri + b + xdim][ri + xdim];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < ydim; ri++)
		for (ci = 0; ci < ydim; ci++, p++)
		    param->data[p] = covmat->data[ri + b + xdim][ci + xdim];
    // XY cov
    if (xdim == ydim && dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++, p++)
		param->data[p] = covmat->data[ri + b][ri + xdim];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++)
		for (ci = 0; ci < ydim; ci++, p++)
		    param->data[p] = covmat->data[ri + b][ci + xdim];
    // XX cov
    if (xdim == ydim && dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++, p++)
		param->data[p] = covmat->data[ri + b][ri];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++)
		for (ci = 0; ci < xdim; ci++, p++)
		    param->data[p] = covmat->data[ri + b][ci];

    if (p != param->length) {
	fprintf(stderr, "Error xget_paramvec\n");
	exit(1);
    }

    return param;
}

void get_paramvec(DVECTOR param, DMATRIX weight, DMATRIX xmean,
		  DMATRIX ymean, DMATRIX xxcov, DMATRIX yxcov, DMATRIX yycov,
		  XBOOL dia_flag, XBOOL msg_flag)
{
    long p, b, i, ri, ci, clsnum, dim, xdim, ydim;

    clsnum = weight->row;
    xdim = xmean->col;
    ydim = ymean->col;
    dim = xdim + ydim;

    if (xdim == ydim && dia_flag == XTRUE) {	// all diagonal cov
	if (param->length != clsnum * (1 + dim + dim + ydim)) {
	    fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld], all diag covs\n", param->length, clsnum, xdim, ydim);
	    exit(1);
	}
    } else if (dia_flag == XTRUE) {	// XX diagonal cov
	if (param->length != clsnum * (1 + dim + xdim + ydim * xdim
				       + ydim * ydim)) {
	    fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld], XX diag cov\n", param->length, clsnum, xdim, ydim);
	    exit(1);
	}
    } else {	// all full cov
	if (param->length != clsnum * (1 + dim + xdim * xdim + ydim * xdim
				       + ydim * ydim)) {
	    fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld]\n", param->length, clsnum, xdim, ydim);
	    exit(1);
	}
    }

    p = 0;
    // weight
    for (ri = 0; ri < clsnum; ri++, p++)
	weight->data[ri][0] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "weight vector [%ld][%ld]\n",
		weight->row, weight->col);
    // X mean
    for (ri = 0; ri < clsnum; ri++)
	for (ci = 0; ci < xdim; ci++, p++)
	    xmean->data[ri][ci] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "X mean vectors [%ld][%ld]\n", xmean->row, xmean->col);
    // Y mean
    for (ri = 0; ri < clsnum; ri++)
	for (ci = 0; ci < ydim; ci++, p++)
	     ymean->data[ri][ci] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "Y mean vectors [%ld][%ld]\n", ymean->row, ymean->col);
    // XX cov
    if (dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * xdim; ri < xdim; ri++, p++)
		    xxcov->data[ri + b][ri] = param->data[p];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * xdim; ri < xdim; ri++)
		for (ci = 0; ci < xdim; ci++, p++)
		    xxcov->data[ri + b][ci] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "XX cov matrices [%ld][%ld]\n",
		xxcov->row, xxcov->col);
    // YX cov
    if (xdim == ydim && dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0; ri < xdim; ri++, p++)
		yxcov->data[i][ri] = param->data[p];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * ydim; ri < ydim; ri++)
		for (ci = 0; ci < xdim; ci++, p++)
		    yxcov->data[ri + b][ci] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "YX cov matrices [%ld][%ld]\n",
		yxcov->row, yxcov->col);
    // YY cov
    if (xdim == ydim && dia_flag == XTRUE)
	for (i = 0; i < clsnum; i++)
	    for (ri = 0; ri < ydim; ri++, p++)
		yycov->data[i][ri] = param->data[p];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * ydim; ri < ydim; ri++)
		for (ci = 0; ci < ydim; ci++, p++)
		    yycov->data[ri + b][ci] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "YY cov matrices [%ld][%ld]\n",
		yycov->row, yycov->col);

    if (p != param->length) {
	fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld]\n", param->length, clsnum, xdim, ydim);
	exit(1);
    }

    return;
}

void get_paramvec(DVECTOR param, long xdim, long ydim, DMATRIX weight,
		  DMATRIX mean, DMATRIX cov, XBOOL dia_flag, XBOOL msg_flag)
{
    long p, b, i, ri, ci, clsnum, dim;

    clsnum = weight->row;
    dim = xdim + ydim;

    if (xdim == ydim && dia_flag == XTRUE) {	// all diagonal cov
	if (param->length != clsnum * (1 + dim + dim + ydim)) {
	    fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld], all diag covs\n", param->length, clsnum, xdim, ydim);
	    exit(1);
	}
    } else if (dia_flag == XTRUE) {	// XX diagonal cov
	if (param->length != clsnum * (1 + dim + xdim + ydim * xdim
				       + ydim * ydim)) {
	    fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld], XX diag cov\n", param->length, clsnum, xdim, ydim);
	    exit(1);
	}
    } else {	// all full cov
	if (param->length != clsnum * (1 + dim + xdim * xdim + ydim * xdim
				       + ydim * ydim)) {
	    fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld]\n", param->length, clsnum, xdim, ydim);
	    exit(1);
	}
    }

    p = 0;
    // weight
    for (ri = 0; ri < clsnum; ri++, p++)
	weight->data[ri][0] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "weight vector [%ld][%ld]\n",
		weight->row, weight->col);
    // mean
    for (ri = 0; ri < clsnum; ri++)	// X mean
	for (ci = 0; ci < xdim; ci++, p++)
	    mean->data[ri][ci] = param->data[p];
    for (ri = 0; ri < clsnum; ri++)	// Y mean
	for (ci = 0; ci < ydim; ci++, p++)
	     mean->data[ri][ci + xdim] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "XY mean vectors [%ld][%ld]\n", mean->row, mean->col);
    // covariance
    if (dia_flag == XTRUE)	// XX cov
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++, p++)
		    cov->data[ri + b][ri] = param->data[p];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < xdim; ri++)
		for (ci = 0; ci < xdim; ci++, p++)
		    cov->data[ri + b][ci] = param->data[p];
    if (xdim == ydim && dia_flag == XTRUE) {    // YX cov
	for (i = 0; i < clsnum; i++) {
	    for (ri = 0, b = i * dim; ri < xdim; ri++, p++) {
		cov->data[ri + b][ri + xdim] = param->data[p];
		cov->data[ri + xdim + b][ri] = param->data[p];
	    }
	}
    } else {
	for (i = 0; i < clsnum; i++) {
	    for (ri = 0, b = i * dim; ri < ydim; ri++) {
		for (ci = 0; ci < xdim; ci++, p++) {
		    cov->data[ri + xdim + b][ci] = param->data[p];
		    cov->data[ci + b][ri + xdim] = param->data[p];
		}
	    }
	}
    }
    if (xdim == ydim && dia_flag == XTRUE)	// YY cov
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < ydim; ri++, p++)
		cov->data[ri + xdim + b][ri + xdim] = param->data[p];
    else
	for (i = 0; i < clsnum; i++)
	    for (ri = 0, b = i * dim; ri < ydim; ri++)
		for (ci = 0; ci < ydim; ci++, p++)
		    cov->data[ri + xdim + b][ci + xdim] = param->data[p];
    if (msg_flag == XTRUE)
	fprintf(stderr, "XYXY cov matrices [%ld][%ld]\n", cov->row, cov->col);

    if (p != param->length) {
	fprintf(stderr, "Error get_paramvec [%ld], class [%ld], dimension X[%ld] Y[%ld]\n", param->length, clsnum, xdim, ydim);
	exit(1);
    }

    return;
}

long get_clsnum(DVECTOR param, long xdim, long ydim, XBOOL dia_flag)
{
    long clsnum;

    if (xdim == ydim && dia_flag == XTRUE) {	// all diagonal cov
	clsnum = param->length / (1 + xdim + ydim + xdim + ydim + ydim);
    } else if (dia_flag == XTRUE) {	// XX diagonal cov
	clsnum = param->length
	    / (1 + xdim + ydim + xdim + ydim * xdim + ydim * ydim);
    } else {	// all full cov
	clsnum = param->length
	    / (1 + xdim + ydim + xdim * xdim + ydim * xdim + ydim * ydim);
    }

    return clsnum;
}

// diagonal covariance
DVECTOR xget_detvec_diamat2inv(DMATRIX covmat)	// [num class][dim]
{
    long dim, clsnum;
    long i, j;
    double det;
    DVECTOR detvec = NODATA;

    clsnum = covmat->row;
    dim = covmat->col;
    // memory allocation
    detvec = xdvalloc(clsnum);
    for (i = 0; i < clsnum; i++) {
	for (j = 0, det = 1.0; j < dim; j++) {
	    det *= covmat->data[i][j];
	    if (det > 0.0) {
		covmat->data[i][j] = 1.0 / covmat->data[i][j];
	    } else {
		fprintf(stderr, "error:(class %ld) determinant <= 0, det = %f\n", i, det);
		return NODATA;
	    }
	}
	detvec->data[i] = det;
    }

    return detvec;
}

// calculate determinant(detvec) and inverse of covariance(covmat)
DVECTOR xget_detvec_mat2inv(long clsnum,	   // number of class
			    DMATRIX covmat, // [dim*(number of class)] [dim]
			    XBOOL dia_flag) // diagonal covariance matrix
{
    long i, ri, ci;
    long dim;
    double det;
    DVECTOR detvec = NODATA;
    DMATRIX clscovmat = NODATA;

    // dimension
    dim = covmat->col;
    // memory allocation
    detvec = xdvalloc(clsnum);
    // -dia option: use diagonal covariance matrix
    if (dia_flag == XTRUE) {
	for (i = 0; i < clsnum; i++) {
	    for (ri = 0, det = 1.0; ri < dim; ri++) {
		det *= covmat->data[ri + (i*dim)][ri];
		if (det > 0.0) {
		    covmat->data[ri + (i*dim)][ri] =
			1.0 / covmat->data[ri+(i*dim)][ri];
		} else {
		    fprintf(stderr, "error:(class %ld) determinant <= 0, det = %f\n", i, det);
		    return NODATA;
		}
	    }
	    detvec->data[i] = det;
	}
	return detvec;
    }
    // memory allocation
    clscovmat = xdmalloc(dim, dim);

    for (i = 0; i < clsnum; i++) {
	// get i-th class covariance
	for (ri = 0; ri < dim; ri++)
	    for (ci = 0; ci < dim; ci++)
		clscovmat->data[ri][ci] = covmat->data[ri+(i*dim)][ci];
	// get i-th determinant and inverse of covariance
	detvec->data[i] = get_det_CovInvert(clscovmat);
	if (detvec->data[i] <= 0.0) {
	    printf("#error: determinant <= 0\n");
	    return NODATA;
	}
	// substitute clscov for cov
	for (ri = 0; ri < dim; ri++)
	    for (ci = 0; ci < dim; ci++)
		covmat->data[ri+(i*dim)][ci] = clscovmat->data[ri][ci];
    }
    // memory free
    xdmfree(clscovmat);

    return detvec;
}

// calculate determinant(detvec) and inverse of covariance(covmat)
DVECTOR xget_detvec_mat2inv_jde(
				long clsnum,	   // number of class
				DMATRIX covmat,//[dim*(number of class)][dim]
				XBOOL dia_flag) //diagonal covariance matrix
{
    long i, ri, ci;
    long dim;
    long dim2;
    double adet;
    double edet;
    DVECTOR detvec = NODATA;
    DVECTOR avec = NODATA;
    DVECTOR bvec = NODATA;
    DVECTOR ibvec = NODATA;
    DVECTOR cvec = NODATA;
    DVECTOR evec = NODATA;
    DVECTOR fvec = NODATA;
    DMATRIX clscovmat = NODATA;

    // dimension
    dim = covmat->col;
    dim2 = covmat->col / 2;
    // memory allocation
    detvec = xdvalloc(clsnum);
    // -dia option: use diagonal covariance matrix
    if (dia_flag == XTRUE) {
	// memory allocation
	avec = xdvalloc(dim2);
	bvec = xdvalloc(dim2);
	cvec = xdvalloc(dim2);

	for (i = 0; i < clsnum; i++) {
	    for (ri = 0; ri < dim2; ri++) {
		avec->data[ri] = covmat->data[ri + (i * dim)][ri];
		bvec->data[ri] = covmat->data[ri + (i * dim)][ri + dim2];
		cvec->data[ri] = covmat->data[ri + (i * dim)+ dim2][ri+dim2];
	    }

	    for (ci = 0, adet = 1.0; ci < dim2; ci++) {
		if ((adet *= avec->data[ci]) > 0.0) {
		    // avec inverse
		    avec->data[ci] = 1.0 / avec->data[ci];
		} else {
		    printf("#error:(class %ld) [A]determinant <= 0, det = %e\n", i, adet);
		    return NODATA;
		}
	    }
	    evec = xdvoper(bvec, "*", avec);
	    fvec = xdvclone(evec);
	    dvoper(evec, "*", bvec);
	    dvoper(cvec, "-", evec);

	    // memoty free
	    xdvfree(evec);

	    evec = xdvclone(cvec);

	    for (ci = 0, edet = 1.0; ci < dim2; ci++) {
		if ((edet *= evec->data[ci]) > 0.0) {
		    // evec inverse
		    evec->data[ci] = 1.0 / evec->data[ci];
		} else {
		    printf("#error:(class %ld) [E]determinant <= 0, det = %e\n", i, edet);
		    return NODATA;
		}
	    }
	    detvec->data[i] = adet * edet;
	    if (detvec->data[i] <= 0.0) {
		printf("#error:(class %ld) determinant <= 0, det = %e\n",
		       i, detvec->data[i]);
		return NODATA;
	    }

	    ibvec = xdvoper(evec, "*", fvec);
	    dvoper(fvec, "*", ibvec);
	    dvoper(avec, "+", fvec);
	    dvscoper(ibvec, "*", -1.0);

	    for (ri = 0; ri < dim2; ri++) {
		covmat->data[ri + (i * dim)][ri] = avec->data[ri];
		covmat->data[ri + (i * dim)][ri + dim2] = ibvec->data[ri];
		covmat->data[ri + (i * dim)+dim2][ri] = ibvec->data[ri];
		covmat->data[ri + (i * dim)+dim2][ri+dim2] = evec->data[ri];
	    }

	    // memory free
	    xdvfree(ibvec);
	    xdvfree(evec);
	    xdvfree(fvec);
	}
	// memory free
	xdvfree(avec);
	xdvfree(bvec);
	xdvfree(cvec);

	return detvec;
    }
    // memory allocation
    clscovmat = xdmalloc(dim, dim);

    for (i = 0; i < clsnum; i++) {
	// get i-th class covariance
	for (ri = 0; ri < dim; ri++) {
	    for (ci = 0; ci < dim; ci++) {
		clscovmat->data[ri][ci] = covmat->data[ri+(i*dim)][ci];
	    }
	}
	// get i-th determinant and inverse of covariance
	detvec->data[i] = get_det_CovInvert(clscovmat);
	if (detvec->data[i] <= 0.0) {
	    printf("#error: determinant <= 0\n");
	    return NODATA;
	}
	// substitute clscov for cov
	for (ri = 0; ri < dim; ri++) {
	    for (ci = 0; ci < dim; ci++) {
		covmat->data[ri+(i*dim)][ci] = clscovmat->data[ri][ci];
	    }
	}
    }
    // memory free
    xdmfree(clscovmat);

    return detvec;
}

double cal_xmcxmc(DVECTOR x,
		  DVECTOR m,
		  DMATRIX c)
{
    long k, l, dim;
    double *vec = NULL;
    double td, d;

    dim = x->length;
    if (m->length != dim || c->row != dim || c->col != dim) {
	fprintf(stderr, "Error cal_xmcxmc: different dimension\n");
	exit(1);
    }

    // memory allocation
    vec = xalloc((int)dim, double);
    for (k = 0; k < dim; k++) vec[k] = x->data[k] - m->data[k];
    for (k = 0, d = 0.0; k < dim; k++) {
	for (l = 0, td = 0.0; l < dim; l++) td += vec[l] * c->data[l][k];
	d += td * vec[k];
    }
    // memory free
    delete [] vec;	vec = NULL;

    return d;
}

double cal_xmcxmc(long clsidx,
		  DVECTOR x,
		  DMATRIX mm,	// [num class][dim]
		  DMATRIX cm)	// [num class * dim][dim]
{
    long clsnum, k, l, b, dim;
    double *vec = NULL;
    double td, d;

    dim = x->length;
    clsnum = mm->row;
    b = clsidx * dim;
    if (mm->col != dim || cm->col != dim || clsnum * dim != cm->row) {
	fprintf(stderr, "Error cal_xmcxmc: different dimension\n");
	exit(1);
    }

    // memory allocation
    vec = xalloc((int)dim, double);
    for (k = 0; k < dim; k++) vec[k] = x->data[k] - mm->data[clsidx][k];
    for (k = 0, d = 0.0; k < dim; k++) {
	for (l = 0, td = 0.0; l < dim; l++) td += vec[l] * cm->data[l + b][k];
	d += td * vec[k];
    }
    // memory free
    delete [] vec;	vec = NULL;

    return d;
}

// calculate gauss
double get_gauss_full(long clsidx,
		      DVECTOR vec,		// [dim]
		      DVECTOR detvec,		// [clsnum]
		      DMATRIX weightmat,	// [clsnum][1]
		      DMATRIX meanvec,		// [clsnum][dim]
		      DMATRIX invcovmat)	// [clsnum * dim][dim]
{
    double gauss;

    if (detvec->data[clsidx] <= 0.0) {
	printf("#error: det <= 0.0\n");
	exit(1);
    }

    gauss = weightmat->data[clsidx][0]
	/ sqrt(pow(2.0 * PI, (double)vec->length) * detvec->data[clsidx])
	* exp(-1.0 * cal_xmcxmc(clsidx, vec, meanvec, invcovmat) / 2.0);
    
    return gauss;
}

// diagonal covariance
double get_gauss_dia(double det,
		     double weight,
		     DVECTOR vec,		// dim
		     DVECTOR meanvec,		// dim
		     DVECTOR invcovvec)		// dim
{
    double gauss, sb;
    long k;

    if (det <= 0.0) {
	printf("#error: det <= 0.0\n");
	exit(1);
    }

    for (k = 0, gauss = 0.0; k < vec->length; k++) {
	sb = vec->data[k] - meanvec->data[k];
	gauss += sb * invcovvec->data[k] * sb;
    }

    gauss = weight / sqrt(pow(2.0 * PI, (double)vec->length) * det)
	* exp(-gauss / 2.0);

    return gauss;
}

// diagonal covariance
double get_gauss_dia(long clsidx,
		     DVECTOR vec,		// [dim]
		     DVECTOR detvec,		// [clsnum]
		     DMATRIX weightmat,		// [clsnum][1]
		     DMATRIX meanmat,		// [clsnum][dim]
		     DMATRIX invcovmat)		// [clsnum][dim]
{
    double gauss, sb;
    long k;

    if (detvec->data[clsidx] <= 0.0) {
	printf("#error: det <= 0.0\n");
	exit(1);
    }

    for (k = 0, gauss = 0.0; k < vec->length; k++) {
	sb = vec->data[k] - meanmat->data[clsidx][k];
	gauss += sb * invcovmat->data[clsidx][k] * sb;
    }

    gauss = weightmat->data[clsidx][0]
	/ sqrt(pow(2.0 * PI, (double)vec->length) * detvec->data[clsidx])
	* exp(-gauss / 2.0);



    return gauss;
}

// calculate gauss for joint density matrix
double get_gauss_jde_dia(long clsidx,
			 DVECTOR vec,	       	// [dim]
			 DVECTOR detvec,	// [clsnum]
			 DMATRIX weightmat,	// [clsnum][1]
			 DMATRIX meanmat,	// [clsnum][dim]
			 DMATRIX invcovmat)	// [clsnum * dim][dim]
{
    long k, b, clsnum, dim, dim2;
    double gauss;
    DVECTOR subvec = NODATA;

    dim = vec->length;
    clsnum = detvec->length;
    dim2 = dim / 2;
    b = clsidx * dim;

    if (detvec->data[clsidx] <= 0.0) {
	printf("#error: det <= 0.0\n");
	exit(1);
    }
    // memory allocation
    subvec = xdvalloc(dim);
    for (k = 0; k < dim; k++)
	subvec->data[k] = vec->data[k] - meanmat->data[clsidx][k];
    // [x y][a b][x] = (xa + yb) * x + (xb + yc) * y
    //      [b c][y]
    for (k = 0, gauss = 0.0; k < dim2; k++) {
	gauss += (subvec->data[k] * invcovmat->data[k + b][k]
		  + subvec->data[k + dim2] * invcovmat->data[k + b + dim2][k])
	    * subvec->data[k]
	    + (subvec->data[k] * invcovmat->data[k + b][k + dim2]
	       + subvec->data[k + dim2]
	       * invcovmat->data[k + b + dim2][k + dim2])
	    * subvec->data[k + dim2];
    }
    gauss = weightmat->data[clsidx][0]
	/ sqrt(pow(2.0 * PI, (double)vec->length) * detvec->data[clsidx])
	* exp(-1.0 * gauss / 2.0);
    // memory free
    xdvfree(subvec);

    return gauss;
}

// JDE
DVECTOR xget_gaussvec_jde(DVECTOR vec,
			  DVECTOR detvec,	// num class
			  DMATRIX weightmat,	// [num class][1]
			  DMATRIX meanmat,	// [num class][dim]
			  DMATRIX invcovmat,	// [num class * dim][dim]
			  XBOOL dia_flag)
{			
    long i, clsnum;
    DVECTOR gaussvec = NODATA;
    
    clsnum = weightmat->row;

    gaussvec = xdvalloc(clsnum);
    // calculate gaussmat
    for (i = 0; i < clsnum; i++) {
	if (dia_flag == XTRUE)
	    gaussvec->data[i] = get_gauss_jde_dia(i, vec, detvec, weightmat,
						  meanmat, invcovmat);
	else
	    gaussvec->data[i] = get_gauss_full(i, vec, detvec, weightmat,
					       meanmat, invcovmat);
    }

    return gaussvec;
}

// diagonal covariance
DVECTOR xget_gaussvec_dia(DVECTOR vec,
			  DVECTOR detvec,	// num class
			  DMATRIX weightmat,	// [num class][1]
			  DMATRIX meanmat,	// [num class][dim]
			  DMATRIX invcovmat)	// [num class][dim]
{			
    long i, clsnum;
    DVECTOR gaussvec = NODATA;
    
    clsnum = weightmat->row;

    gaussvec = xdvalloc(clsnum);
   // calculate gaussmat
    for (i = 0; i < clsnum; i++)
	gaussvec->data[i] =
	    get_gauss_dia(i, vec, detvec, weightmat, meanmat, invcovmat);

    return gaussvec;
}

// full covariance
DVECTOR xget_gaussvec_full(DVECTOR vec,
			   DVECTOR detvec,	// num class
			   DMATRIX weightmat,	// [num class][1]
			   DMATRIX meanmat,	// [num class][dim]
			   DMATRIX invcovmat)	// [num class * dim][dim]
{			
    long i, clsnum;
    DVECTOR gaussvec = NODATA;
    
    clsnum = weightmat->row;

    gaussvec = xdvalloc(clsnum);
    // calculate gaussmat
    for (i = 0; i < clsnum; i++)
	gaussvec->data[i] = get_gauss_full(i, vec, detvec, weightmat,
					   meanmat, invcovmat);

    return gaussvec;
}

// calculate gauss matrix with full covariance on joint density
void get_gaussmat_jde_file(DVECTOR detvec,	// num of class
			   DMATRIX gaussm,	// output[num of data][clsnum]
			   char *vecmatfile,	// [num of data][dim]
			   long dnum,		// [num of data]
			   long dim,		// [dim]
			   DMATRIX weightmat,	// [num class][1]
			   DMATRIX meanmat,	// [num class][dim]
			   DMATRIX invcovmat,	// [dim * num class][dim]
			   XBOOL dia_flag)
{			
    long t;
    DVECTOR vec = NODATA;
    DVECTOR gvec = NODATA;
    FILE *ifp;

    // memory allocation
    vec = xdvalloc(dim);
    if (gaussm->row != dnum || gaussm->col != detvec->length) {
	fprintf(stderr, "Error get_gaussmat_jde_file\n");
	exit(1);
    }

    // open file
    if ((ifp = fopen(vecmatfile, "rb")) == NULL) {
	fprintf(stderr, "can't open file: %s\n", vecmatfile);
	exit(1);
    }
    // calculate gaussmat
    for (t = 0; t < dnum; t++) {
	freaddouble(vec->data, dim, 0, ifp);
	gvec = xget_gaussvec_jde(vec, detvec, weightmat, meanmat,
				 invcovmat, dia_flag);
	dmcopyrow(gaussm, t, gvec);
	xdvfree(gvec);
    }
    // close file
    fclose(ifp);
    // memory free
    xdvfree(vec);

    return;
}

// calculate gauss matrix with full covariance on joint density
void get_gaussmat_jde_file(DVECTOR detvec,	// num of class
			   char *gaussmatfile,	// output
			   char *vecmatfile,	// [num of data][dim]
			   long dnum,		// [num of data]
			   long dim,		// [dim]
			   DMATRIX weightmat,	// [num class][1]
			   DMATRIX meanmat,	// [num class][dim]
			   DMATRIX invcovmat,	// [dim * num class][dim]
			   XBOOL dia_flag)
{			
    long t;
    DVECTOR vec = NODATA;
    DVECTOR gvec = NODATA;
    FILE *ifp, *ofp;

    // memory allocation
    vec = xdvalloc(dim);

    // open file
    if ((ifp = fopen(vecmatfile, "rb")) == NULL) {
	fprintf(stderr, "can't open file: %s\n", vecmatfile);
	exit(1);
    }
    if ((ofp = fopen(gaussmatfile, "wb")) == NULL) {
	fprintf(stderr, "can't open file: %s\n", gaussmatfile);
	exit(1);
    }
    // calculate gaussmat
    for (t = 0; t < dnum; t++) {
	freaddouble(vec->data, dim, 0, ifp);
	gvec = xget_gaussvec_jde(vec, detvec, weightmat, meanmat,
				 invcovmat, dia_flag);
	fwrite(gvec->data, sizeof(double), (size_t)gvec->length, ofp);
	xdvfree(gvec);
    }
    // close file
    fclose(ifp);
    fclose(ofp);
    // memory free
    xdvfree(vec);

    return;
}

DVECTOR xget_gammavec(DVECTOR gaussvec)		// [number of class]
{
    long i;
    double sumgauss;
    DVECTOR gammavec = NODATA;

    gammavec = xdvalloc(gaussvec->length);
    sumgauss = dvsum(gaussvec);
    // TEMPORARY
    if (sumgauss == 0.0) {
	for (i = 0; i < gaussvec->length; i++)
	    gammavec->data[i] = 1.0 / (double)gaussvec->length;
    } else {
	for (i = 0; i < gaussvec->length; i++)
	    gammavec->data[i] = gaussvec->data[i] / sumgauss;
    }

    return gammavec;
}

// calculate gamma matrix
DVECTOR xget_sumgvec_gammamat(DMATRIX gaussm,
			      long row,
			      long col,
			      DMATRIX gammam)
{
    long t, i;
    double sumg;
    DVECTOR sumgvec = NODATA;

    // memory allocation
    sumgvec = xdvzeros(col);

    for (t = 0; t < row; t++) {
	for (i = 0, sumg = 0.0; i < col; i++) sumg += gaussm->data[t][i];
	// TEMPORARY
	if (sumg == 0.0) {
	    fprintf(stderr, "#Warning: outlier frame [%ld]\n", t);
	    for (i = 0; i < col; i++) {
		gammam->data[t][i] = 1.0 / (double)col;
		sumgvec->data[i] += gammam->data[t][i];
	    }
	} else {
	    for (i = 0; i < col; i++) {
		gammam->data[t][i] = gaussm->data[t][i] / sumg;
		sumgvec->data[i] += gammam->data[t][i];
	    }
	}
    }

    return sumgvec;
}

// calculate gamma matrix
DVECTOR xget_sumgvec_gammamat_file(char *gaussmatfile,
				   long row,
				   long col,
				   char *gammamatfile)
{
    long t, i;
    double sumg, gamma;
    double *gvec = NULL;
    DVECTOR sumgvec = NODATA;
    FILE *ifp, *ofp;

    // memory allocation
    gvec = xalloc((int)col, double);
    sumgvec = xdvzeros(col);

    // open file
    if ((ifp = fopen(gaussmatfile, "rb")) == NUL) {
	fprintf(stderr, "can't open file: %s\n", gaussmatfile);
	exit(1);
    }
    if ((ofp = fopen(gammamatfile, "wb")) == NUL) {
	fprintf(stderr, "can't open file: %s\n", gammamatfile);
	exit(1);
    }
    
    for (t = 0; t < row; t++) {
	freaddouble(gvec, col, 0, ifp);
	for (i = 0, sumg = 0.0; i < col; i++) sumg += gvec[i];
	// TEMPORARY
	if (sumg == 0.0) {
	    fprintf(stderr, "#Warning: outlier frame [%ld]\n", t);
	    for (i = 0; i < col; i++) {
		gamma = 1.0 / (double)col;
		fwrite(&gamma, sizeof(double), (size_t)1, ofp);
		sumgvec->data[i] += gamma;
	    }
	} else {
	    for (i = 0; i < col; i++) {
		gamma = gvec[i] / sumg;
		fwrite(&gamma, sizeof(double), (size_t)1, ofp);
		sumgvec->data[i] += gamma;
	    }
	}
    }
    // close file
    fclose(ifp);
    fclose(ofp);
    // memory free
    delete [] gvec;	gvec = NULL;

    return sumgvec;
}

// estimate weight
void estimate_weight(DMATRIX weightmat,	// [num class][1]
		     DVECTOR sumgvec)	// [num class]
{
    long i;
    double sumg;

    // estimate weight
    sumg = dvsum(sumgvec);
    for (i = 0; i < weightmat->row; i++) 
	weightmat->data[i][0] = sumgvec->data[i] / sumg;

    fprintf(stderr, "estimating weight done\n");

    return;
}

// estimate mean
void estimate_mean_file(char *vecmatfile,	// [num data][dim]
			long dim,
			DMATRIX meanmat,	// [num class][dim]
			DMATRIX gammam,		// [num data][num class]
			DVECTOR sumgvec,	// [num class]
			long clsnum)
{
    long t, i, d;
    long dnum;
    DVECTOR vec;
    FILE *fp1;

    dnum = gammam->row;
    // memory allocarion
    vec = xdvalloc(dim);
    // initialization
    for (i = 0; i < meanmat->row; i++)
	for (d = 0; d < meanmat->col; d++)
	    meanmat->data[i][d] = 0.0;
    // open file
    if (NULL == (fp1 = fopen(vecmatfile, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", vecmatfile);
	exit(1);
    }
    // estimate mean vectors
    for (t = 0; t < dnum; t++) {
	freaddouble(vec->data, dim, 0, fp1);
	for (i = 0; i < clsnum; i++)
	    for (d = 0; d < dim; d++)
		meanmat->data[i][d] += vec->data[d] * gammam->data[t][i];
    }
    for (i = 0; i < clsnum; i++)
	for (d = 0; d < dim; d++) meanmat->data[i][d] /= sumgvec->data[i];
    // close file
    fclose(fp1);
    // memory free
    xdvfree(vec);

    fprintf(stderr, "estimating mean done\n");

    return;
}

// estimate mean
void estimate_mean_file(char *vecmatfile,	// [num data][dim]
			long dim,
			DMATRIX meanmat,	// [num class][dim]
			char *gammamatfile,	// [num data][num class]
			DVECTOR sumgvec,	// [num class]
			long clsnum)
{
    long t, i, d;
    long length, dnum;
    DVECTOR vec;
    DVECTOR gammavec;
    FILE *fp1, *fp2;

    // get data length
    if ((length = getsiglen(gammamatfile, 0, double)) <= 0) {
	fprintf(stderr, "wrong data length: %s\n", gammamatfile);
	exit(1);
    }
    if (length % clsnum != 0) {
	fprintf(stderr, "wrong data format: %s\n", gammamatfile);
	exit(1);
    }
    dnum = length / clsnum;
    // memory allocarion
    vec = xdvalloc(dim);
    gammavec = xdvalloc(clsnum);
    // initialization
    for (i = 0; i < meanmat->row; i++)
	for (d = 0; d < meanmat->col; d++)
	    meanmat->data[i][d] = 0.0;
    // open file
    if (NULL == (fp1 = fopen(vecmatfile, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", vecmatfile);
	exit(1);
    }
    if (NULL == (fp2 = fopen(gammamatfile, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", gammamatfile);
	exit(1);
    }
    // estimate mean vectors
    for (t = 0; t < dnum; t++) {
	freaddouble(vec->data, dim, 0, fp1);
	freaddouble(gammavec->data, clsnum, 0, fp2);
	for (i = 0; i < clsnum; i++)
	    for (d = 0; d < dim; d++)
		meanmat->data[i][d] += vec->data[d] * gammavec->data[i];
    }
    for (i = 0; i < clsnum; i++)
	for (d = 0; d < dim; d++) meanmat->data[i][d] /= sumgvec->data[i];
    // close file
    fclose(fp1);
    fclose(fp2);
    // memory free
    xdvfree(vec);
    xdvfree(gammavec);

    fprintf(stderr, "estimating mean done\n");

    return;
}

// estimate covariance on joint density
void estimate_cov_jde_file(char *vecmatfile,	// [num data][dim]
			   long dim,
			   DMATRIX meanmat,	// [num class][dim]
			   DMATRIX covmat,	// [dim * num class][dim]
			   DMATRIX gammam,	// [num data][num class]
			   DVECTOR sumgvec,	// [num class]
			   long clsnum,
			   XBOOL dia_flag)	// diagonal covariance
{
    long i, t, b, ri, ci, dim2;
    long dnum;
    DVECTOR vec = NODATA;
    DVECTOR sbvec = NODATA;
    FILE *fp1;

    dim2 = dim / 2;
    // get data length
    dnum = gammam->row;
    // memory allocarion
    vec = xdvalloc(dim);
    sbvec = xdvalloc(dim);
    // initialization
    for (ri = 0; ri < covmat->row; ri++)
	for (ci = 0; ci < covmat->col; ci++) covmat->data[ri][ci] = 0.0;

    // open file
    if (NULL == (fp1 = fopen(vecmatfile, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", vecmatfile);
	exit(1);
    }
    for (t = 0; t < dnum; t++) {
	freaddouble(vec->data, dim, 0, fp1);
	for (i = 0; i < clsnum; i++) {
	    b = i * dim;
	    for (ri = 0; ri < dim; ri++)
		sbvec->data[ri] = vec->data[ri] - meanmat->data[i][ri];
	    if (dia_flag == XTRUE) {
		for (ri = 0; ri < dim2; ri++) {
		    covmat->data[ri + b][ri] += sbvec->data[ri]
			* sbvec->data[ri] * gammam->data[t][i];
		    covmat->data[ri + b][ri + dim2] +=
			sbvec->data[ri] * sbvec->data[ri + dim2]
			* gammam->data[t][i];
		    covmat->data[ri + b + dim2][ri + dim2] +=
			sbvec->data[ri + dim2] * sbvec->data[ri + dim2]
			* gammam->data[t][i];
		}
	    } else {
		for (ri = 0; ri < dim; ri++)
		    for (ci = ri; ci < dim; ci++)
			covmat->data[ri + b][ci] += sbvec->data[ri]
			    * sbvec->data[ci] * gammam->data[t][i];
	    }
	}
    }
    for (i = 0; i < clsnum; i++) {
	b = i * dim;
	if (dia_flag == XTRUE) {
	    for (ri = 0; ri < dim2; ri++) {
		covmat->data[ri + b][ri] /= sumgvec->data[i];
		covmat->data[ri + b][ri + dim2] /= sumgvec->data[i];
		covmat->data[ri + b + dim2][ri] =
		    covmat->data[ri + b][ri + dim2];
		covmat->data[ri + b + dim2][ri + dim2] /= sumgvec->data[i];
	    }
	} else {
	    for (ri = 0; ri < covmat->col; ri++) {
		for (ci = ri; ci < covmat->col; ci++) {
		    covmat->data[ri + b][ci] /= sumgvec->data[i];
		    if (ri != ci)
			covmat->data[ci + b][ri] = covmat->data[ri + b][ci];
		}
	    }
	}
    }
    // close file
    fclose(fp1);
	
    // memory free
    xdvfree(vec);
    xdvfree(sbvec);

    fprintf(stderr, "estimating covariance done\n");

    return;
}

// estimate covariance on joint density
void estimate_cov_jde_file(char *vecmatfile,	// [num data][dim]
			   long dim,
			   DMATRIX meanmat,	// [num class][dim]
			   DMATRIX covmat,	// [dim * num class][dim]
			   char *gammamatfile,	// [num data][num class]
			   DVECTOR sumgvec,	// [num class]
			   long clsnum,
			   XBOOL dia_flag)	// diagonal covariance
{
    long i, t, b, ri, ci, dim2;
    long length, dnum;
    DVECTOR vec = NODATA;
    DVECTOR sbvec = NODATA;
    DVECTOR gammavec = NODATA;
    FILE *fp1, *fp2;

    dim2 = dim / 2;
    // get data length
    if ((length = getsiglen(gammamatfile, 0, double)) <= 0) {
	fprintf(stderr, "wrong data length: %s\n", gammamatfile);
	exit(1);
    }
    if (length % clsnum != 0) {
	fprintf(stderr, "wrong data format: %s\n", gammamatfile);
	exit(1);
    }
    dnum = length / clsnum;
    // memory allocarion
    vec = xdvalloc(dim);
    sbvec = xdvalloc(dim);
    gammavec = xdvalloc(clsnum);
    // initialization
    for (ri = 0; ri < covmat->row; ri++)
	for (ci = 0; ci < covmat->col; ci++) covmat->data[ri][ci] = 0.0;

    // open file
    if (NULL == (fp1 = fopen(vecmatfile, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", vecmatfile);
	exit(1);
    }
    if (NULL == (fp2 = fopen(gammamatfile, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", gammamatfile);
	exit(1);
    }
    for (t = 0; t < dnum; t++) {
	freaddouble(vec->data, dim, 0, fp1);
	freaddouble(gammavec->data, clsnum, 0, fp2);
	for (i = 0; i < clsnum; i++) {
	    b = i * dim;
	    for (ri = 0; ri < dim; ri++)
		sbvec->data[ri] = vec->data[ri] - meanmat->data[i][ri];
	    if (dia_flag == XTRUE) {
		for (ri = 0; ri < dim2; ri++) {
		    covmat->data[ri + b][ri] += sbvec->data[ri]
			* sbvec->data[ri] * gammavec->data[i];
		    covmat->data[ri + b][ri + dim2] +=
			sbvec->data[ri] * sbvec->data[ri + dim2]
			* gammavec->data[i];
		    covmat->data[ri + b + dim2][ri + dim2] +=
			sbvec->data[ri + dim2] * sbvec->data[ri + dim2]
			* gammavec->data[i];
		}
	    } else {
		for (ri = 0; ri < dim; ri++)
		    for (ci = ri; ci < dim; ci++)
			covmat->data[ri + b][ci] += sbvec->data[ri]
			    * sbvec->data[ci] * gammavec->data[i];
	    }
	}
    }
    for (i = 0; i < clsnum; i++) {
	b = i * dim;
	if (dia_flag == XTRUE) {
	    for (ri = 0; ri < dim2; ri++) {
		covmat->data[ri + b][ri] /= sumgvec->data[i];
		covmat->data[ri + b][ri + dim2] /= sumgvec->data[i];
		covmat->data[ri + b + dim2][ri] =
		    covmat->data[ri + b][ri + dim2];
		covmat->data[ri + b + dim2][ri + dim2] /= sumgvec->data[i];
	    }
	} else {
	    for (ri = 0; ri < covmat->col; ri++) {
		for (ci = ri; ci < covmat->col; ci++) {
		    covmat->data[ri + b][ci] /= sumgvec->data[i];
		    if (ri != ci)
			covmat->data[ci + b][ri] = covmat->data[ri + b][ci];
		}
	    }
	}
    }
    // close file
    fclose(fp1);
    fclose(fp2);
	
    // memory free
    xdvfree(vec);
    xdvfree(sbvec);
    xdvfree(gammavec);

    fprintf(stderr, "estimating covariance done\n");

    return;
}

// calculate likelihood
double get_likelihood(long datanum,	// number of data
		      long clsnum,	// number of class
		      DMATRIX gaussmat)// [number of data][number of class]
{
    long i, t;
    double sumgauss;
    double like;

    // error check
    if (datanum != gaussmat->row || clsnum != gaussmat->col) {
	fprintf(stderr, "error: get_likelihood (datanum, clsnum)\n");
	exit(1);
    }
    // calculate likelihood
    for (t = 0, like = 0.0; t < datanum; t++) {
	for (i = 0, sumgauss = 0.0; i < clsnum; i++) {
	    sumgauss += gaussmat->data[t][i];
	}
	// TEMPORARY
	if (sumgauss > 0.0) like += log(sumgauss);
	else like += -1.0e+38;
    }
    like /= (double)datanum;

    return like;
}

// calculate likelihood
double get_likelihood_file(long datanum,	// num of data
			   long clsnum,		// num of class
			   char *gaussmatfile) 	// [num of data][num of class]
{
    long i, t;
    double sumgauss;
    double like;
    DVECTOR gauss;
    FILE *fp;

    // memory allocarion
    gauss = xdvalloc(clsnum);

    // open file
    if (NULL == (fp = fopen(gaussmatfile, "rb"))) {
	fprintf(stderr, "can't open file: %s\n", gaussmatfile);
	exit(1);
    }
    // calculate likelihood
    for (t = 0, like = 0.0; t < datanum; t++) {
	freaddouble(gauss->data, clsnum, 0, fp);
	for (i = 0, sumgauss = 0.0; i < clsnum; i++) {
	    sumgauss += gauss->data[i];
	}
	// TEMPORARY
	if (sumgauss > 0.0) like += log(sumgauss);
	else like += -1.0e+38;
    }
    like /= (double)datanum;
    // close file
    fclose(fp);
    // memory free
    xdvfree(gauss);

    return like;
}
