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
/*  Subroutine for GMM Mapping                                       */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"
#include "../include/memory.h"

#include "../sub/gmm_sub.h"
#include "../sub/matope_sub.h"

#include "gmmmap_sub.h"

GMMPARA xgmmpara(char *gmmfile, long xdim, long ydim, char *xcovfile,
		 XBOOL dia_flag, XBOOL msg_flag)
{
    long k, l;
    GMMPARA gmmpara;
    DVECTOR paramvec = NODATA;
    DMATRIX ymean = NODATA;
    DMATRIX yxcov = NODATA;
    DMATRIX yycov = NODATA;
    DMATRIX tmpcov = NODATA;

    // memory allocation
    gmmpara = xalloc(1, struct GMMPARA_STRUCT);

    gmmpara->xdim = xdim;
    gmmpara->ydim = ydim;
    gmmpara->dia_flag = dia_flag;
    // read parameter for GMM mapping
    if ((paramvec = xreaddvector(gmmfile, 0)) == NODATA) {
	fprintf(stderr, "Error GMM parameter; %s\n", gmmfile);
	exit(1);
    }
    gmmpara->clsnum = get_clsnum(paramvec, xdim, ydim, dia_flag);
    if (msg_flag == XTRUE)
	fprintf(stderr, "Number of classes: %ld\n", gmmpara->clsnum);

    // memory allocation
    gmmpara->wght = xdmalloc(gmmpara->clsnum, 1);
    gmmpara->xmean = xdmalloc(gmmpara->clsnum, xdim);
    ymean = xdmalloc(gmmpara->clsnum, ydim);
    gmmpara->xxcov = xdmalloc(gmmpara->clsnum * xdim, xdim);
    if (xdim == ydim && dia_flag == XTRUE) {
	yxcov = xdmalloc(gmmpara->clsnum, xdim);
	yycov = xdmalloc(gmmpara->clsnum, ydim);
	gmmpara->lm = xdmalloc(gmmpara->clsnum * 2, xdim);
	gmmpara->dm = xdmalloc(gmmpara->clsnum, xdim);
    } else {
	yxcov = xdmalloc(gmmpara->clsnum * ydim, xdim);
	yycov = xdmalloc(gmmpara->clsnum * ydim, ydim);
	gmmpara->lm = xdmalloc(gmmpara->clsnum * ydim, xdim + 1);
	gmmpara->dm = xdmalloc(gmmpara->clsnum * ydim, ydim);
    }

    // parameters for GMM
    get_paramvec(paramvec, gmmpara->wght, gmmpara->xmean, ymean,
		 gmmpara->xxcov, yxcov, yycov, dia_flag, msg_flag);
    // XX covariance
    if (!strnone(xcovfile)) {
	writedmatrix(xcovfile, gmmpara->xxcov, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write X covariance matrices [%ld][%ld]\n",
		    gmmpara->xxcov->row, gmmpara->xxcov->col);
    }
    // inverse XX cov
    if ((gmmpara->detvec = xget_detvec_mat2inv(gmmpara->clsnum, gmmpara->xxcov,
					       dia_flag)) == NODATA) {
	fprintf(stderr, "Can't calculate inverse XX cov\n");
	exit(1);
    } else if (dia_flag == XTRUE) {
	tmpcov = xdmalloc(gmmpara->clsnum, xdim);
	for (k = 0; k < gmmpara->clsnum; k++)
	    for (l = 0; l < xdim; l++)
		tmpcov->data[k][l] = gmmpara->xxcov->data[l + k * xdim][l];
	// memory free
	xdmfree(gmmpara->xxcov);
	gmmpara->xxcov = xdmclone(tmpcov);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "XX diagonal cov matrices [%ld][%ld]\n",
		    gmmpara->xxcov->row, gmmpara->xxcov->col);
	// memory free
	xdmfree(tmpcov);
    }
    // parameters for linear conversion
    get_gmmmappara(gmmpara->lm, gmmpara->dm, gmmpara->xmean, ymean,
		   gmmpara->xxcov, yxcov, yycov, dia_flag, msg_flag);

    // memory free
    xdvfree(paramvec);
    xdmfree(ymean);
    xdmfree(yxcov);
    xdmfree(yycov);

    return gmmpara;
}

void xgmmparafree(GMMPARA gmmpara)
{
    if (gmmpara != NODATA) {
	if (gmmpara->detvec != NODATA) xdvfree(gmmpara->detvec);
	if (gmmpara->wght != NODATA) xdmfree(gmmpara->wght);
	if (gmmpara->xmean != NODATA) xdmfree(gmmpara->xmean);
	if (gmmpara->xxcov != NODATA) xdmfree(gmmpara->xxcov);
	if (gmmpara->lm != NODATA) xdmfree(gmmpara->lm);
	if (gmmpara->dm != NODATA) xdmfree(gmmpara->dm);
	xfree(gmmpara);
    }

    return;
}

void get_gmmmappara(DMATRIX lm,		// [clsnum * 2 or * ydim][xdim or + 1]
		    DMATRIX dm,		// [clsnum or * ydim][ydim]
		    DMATRIX xmean,	// [clsnum][xdim]
		    DMATRIX ymean,	// [clsnum][ydim]
		    DMATRIX xxicov,	// [clsnum or * xdim][xdim]
		    DMATRIX yxcov,	// [clsnum or * ydim][xdim]
		    DMATRIX yycov,	// [clsnum or * ydim][ydim]
		    XBOOL dia_flag,
		    XBOOL msg_flag)
{
    long i, by, bx, ri, ci, k, clsnum, xdim, ydim;
    double yx, sum;

    clsnum = xmean->row;
    xdim = xmean->col;
    ydim = ymean->col;

    // E = yxcov * xxicov	Eb = ymean - yxcov * xxicov * xmean
    // D = yycov - yxcov * xxicov * xycov
    if (dia_flag == XTRUE && xdim == ydim) {
	for (i = 0; i < clsnum; i++) {
	    for (ci = 0; ci < xdim; ci++) {
		// E = yxcov * xxicov
		yx = yxcov->data[i][ci] * xxicov->data[i][ci];
		lm->data[i * 2][ci] = yx;
		// Eb = ymean - (yxcov * xxicov) * xmean
		lm->data[i * 2 + 1][ci] =
		    ymean->data[i][ci] - yx * xmean->data[i][ci];
		// D = yycov - (yxcov * xxicov) * xycov
		dm->data[i][ci] = yycov->data[i][ci] - yx * yxcov->data[i][ci];
	    }
	}
    } else {
	if (dia_flag == XTRUE) {	// XX diagonal cov 
	    for (i = 0; i < clsnum; i++) {
		by = i * ydim;
		for (ri = 0; ri < ydim; ri++) {
		    for (ci = 0, sum = 0.0; ci < xdim; ci++) {
			// E = yxcov * xxicov
			yx = yxcov->data[ri + by][ci] * xxicov->data[i][ci];
			lm->data[ri + by][ci] = yx;
			// (yxcov * xxicov) * xmean
			sum += yx * xmean->data[i][ci];
		    }
		    // Eb = ymean - (yxcov * xxicov * xmean)
		    lm->data[ri + by][xdim] = ymean->data[i][ri] - sum;
		}
		for (ri = 0; ri < ydim; ri++) {
		    for (ci = 0; ci < ydim; ci++) {
			for (k = 0, sum = 0.0; k < xdim; k++)
			    // (yxcov * xxicov) * xycov
			    sum += lm->data[ri + by][k] * yxcov->data[ci][k];
			// D = yycov - (yxcov * xxicov * xycov)
			dm->data[ri + by][ci] = yycov->data[ri + by][ci] - sum;
		    }
		}
	    }
	} else {
	    for (i = 0; i < clsnum; i++) {
		by = i * ydim;	bx = i * xdim;
		for (ri = 0; ri < ydim; ri++) {
		    for (ci = 0; ci < xdim; ci++) {
			// E = yxcov * xxicov
			for (k = 0, sum = 0.0; k < xdim; k++)
			    sum += yxcov->data[ri + by][k]
				* xxicov->data[k + bx][ci];
			lm->data[ri + by][ci] = sum;
		    }
		    // (yxcov * xxicov) * xmean
		    for (ci = 0, sum = 0.0; ci < xdim; ci++)
			sum += lm->data[ri + by][ci] * xmean->data[i][ci];
		    // Eb = ymean - (yxcov * xxicov * xmean)
		    lm->data[ri + by][xdim] = ymean->data[i][ri] - sum;
		}
		for (ri = 0; ri < ydim; ri++) {
		    for (ci = 0; ci < ydim; ci++) {
			for (k = 0, sum = 0.0; k < xdim; k++)
			    // (yxcov * xxicov) * xycov
			    sum +=
				lm->data[ri + by][k] * yxcov->data[ci + by][k];
			// D = yycov - (yxcov * xxicov * xycov)
			dm->data[ri + by][ci] = yycov->data[ri + by][ci] - sum;
		    }
		}
	    }
	}
    }
    if (msg_flag == XTRUE) {
	fprintf(stderr, "Linear mapping matrix [%ld][%ld]\n",
		lm->row, lm->col);
	fprintf(stderr, "Mapping cov matrix [%ld][%ld]\n", dm->row, dm->col);
    }

    return;
}

void gmmmap(char *inf, char *outf, char *wseqf, char *mseqf, char *covf,
	    char *xmseqf, GMMPARA gmmpara, XBOOL msg_flag)
{
    long t, k, l;
    DMATRIX xm = NODATA;
    DMATRIX mm = NODATA;
    DMATRIX wseq = NODATA;
    DMATRIX mseq = NODATA;

    // read input file
    if ((xm = xreaddmatrix(inf, gmmpara->xdim, 0)) == NODATA) {
	fprintf(stderr, "Can't read file: %s\n", inf);
	exit(1);
    } else if (msg_flag == XTRUE)
	fprintf(stderr, "read input vectors [%ld][%ld]\n", xm->row, xm->col);
    // mapping
    mm = xget_gmmmapmat(xm, gmmpara->detvec, gmmpara->wght, gmmpara->xmean,
			gmmpara->xxcov, gmmpara->lm, gmmpara->ydim,
			gmmpara->dia_flag);
    // write output file
    writedmatrix(outf, mm, 0);
    if (msg_flag == XTRUE)
	fprintf(stderr, "Write mapping vectors [%ld][%ld]\n",
		mm->row, mm->col);
    // weight vector sequence
    if (!strnone(wseqf)) {
	wseq = xget_gmmmap_wghtseq(xm, gmmpara->detvec, gmmpara->wght,
				   gmmpara->xmean, gmmpara->xxcov,
				   gmmpara->dia_flag);
	writedmatrix(wseqf, wseq, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write weight vector sequence [%ld][%ld]\n",
		    wseq->row, wseq->col);
	// memory free
	xdmfree(wseq);
    }
    // mean vector sequence
    if (!strnone(mseqf)) {
	mseq = xget_gmmmap_meanseq(xm, gmmpara->lm, gmmpara->ydim,
				   gmmpara->clsnum, gmmpara->dia_flag);
	writedmatrix(mseqf, mseq, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write mean vector sequence [%ld][%ld]\n",
		    mseq->row, mseq->col);
	// memory free
	xdmfree(mseq);
    }
    // original mean vector sequence
    if (!strnone(xmseqf)) {
	mseq = xdmalloc(xm->row * gmmpara->clsnum, gmmpara->xdim);
	for (t = 0; t < xm->row; t++)
	    for (k = 0; k < gmmpara->clsnum; k++)
		for (l = 0; l < gmmpara->xdim; l++)
		    mseq->data[t * gmmpara->clsnum + k][l] =
			gmmpara->xmean->data[k][l];
	writedmatrix(xmseqf, mseq, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write X mean vector sequence [%ld][%ld]\n",
		    mseq->row, mseq->col);
	// memory free
	xdmfree(mseq);
    }
    // covariance matrices
    if (!strnone(covf)) {
	writedmatrix(covf, gmmpara->dm, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write covariance matrices [%ld][%ld]\n",
		    gmmpara->dm->row, gmmpara->dm->col);
    }

    // memory free
    xdmfree(xm);
    xdmfree(mm);

    return;
}

void gmmmap_file(char *inf, char *outf, char *wseqf, char *mseqf, char *covf,
		 GMMPARA gmmpara, XBOOL msg_flag)
{
    // mapping
    if (msg_flag == XTRUE)
	fprintf(stderr, "Input vectors: %s\n", inf);
    get_gmmmapmat_file(inf, gmmpara->detvec, gmmpara->wght, gmmpara->xmean,
		       gmmpara->xxcov, gmmpara->lm, gmmpara->ydim, outf,
		       gmmpara->dia_flag, msg_flag);
    // weight vector sequence
    if (!strnone(wseqf))
	get_gmmmap_wghtseq_file(inf, gmmpara->detvec, gmmpara->wght,
				gmmpara->xmean, gmmpara->xxcov, wseqf,
				gmmpara->dia_flag, msg_flag);
    // mean vector sequence
    if (!strnone(mseqf))
	get_gmmmap_meanseq_file(inf, gmmpara->lm, gmmpara->xdim,
				gmmpara->ydim, gmmpara->clsnum, mseqf,
				gmmpara->dia_flag, msg_flag);
    // covariance matrices
    if (!strnone(covf)) {
	writedmatrix(covf, gmmpara->dm, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write covariance matrices [%ld][%ld]\n",
		    gmmpara->dm->row, gmmpara->dm->col);
    }

    return;
}

void gmmmap_vit(char *inf, char *outf, char *wseqf, char *covf, char *clsseqf,
		char *xmseqf, GMMPARA gmmpara, XBOOL msg_flag)
{
    long dnum, t, l;
    LVECTOR clsidxv = NODATA;
    DMATRIX xm = NODATA;
    DMATRIX mm = NODATA;
    DMATRIX wseq = NODATA;
    DMATRIX mseq = NODATA;
    FILE *fp;

    dnum = get_dnum_file(inf, gmmpara->xdim);
    // memory allocation
    clsidxv = xlvalloc(dnum);

    // read input file
    if ((xm = xreaddmatrix(inf, gmmpara->xdim, 0)) == NODATA) {
	fprintf(stderr, "Can't read file: %s\n", inf);
	exit(1);
    } else if (msg_flag == XTRUE)
	fprintf(stderr, "read input vectors [%ld][%ld]\n", xm->row, xm->col);

    // weight vector sequence
    wseq = xget_gmmmap_wghtseq_vit(xm, gmmpara->detvec, gmmpara->wght,
				   gmmpara->xmean, gmmpara->xxcov,
				   gmmpara->dia_flag, clsidxv);
    // mean vector sequence
    mm = xget_gmmmap_meanseq_vit(xm, gmmpara->lm, gmmpara->ydim,
				 gmmpara->clsnum, gmmpara->dia_flag, clsidxv);
    // write mean vector sequence
    writedmatrix(outf, mm, 0);
    if (msg_flag == XTRUE)
	fprintf(stderr, "Write mapping vectors [%ld][%ld]\n",
		mm->row, mm->col);
    // write weight sequence
    if (!strnone(wseqf)) {
	writedmatrix(wseqf, wseq, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write weight vector sequence [%ld][%ld]\n",
		    wseq->row, wseq->col);
    }
    // original mean vector sequence
    if (!strnone(xmseqf)) {
	mseq = xdmalloc(xm->row, gmmpara->xdim);
	for (t = 0; t < xm->row; t++)
	    for (l = 0; l < gmmpara->xdim; l++)
		mseq->data[t][l] = gmmpara->xmean->data[clsidxv->data[t]][l];
	writedmatrix(xmseqf, mseq, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write X mean vector sequence [%ld][%ld]\n",
		    mseq->row, mseq->col);
	// memory free
	xdmfree(mseq);
    }

    // covariance matrices
    if (!strnone(covf)) {
	writedmatrix(covf, gmmpara->dm, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write covariance matrices [%ld][%ld]\n",
		    gmmpara->dm->row, gmmpara->dm->col);
    }
    // class index sequence
    if (!strnone(clsseqf)) {
	if ((fp = fopen(clsseqf, "wb")) == NULL) {
	    fprintf(stderr, "Can't open file: %s\n", clsseqf);
	    exit(1);
	}
	fwritelong(clsidxv->data, clsidxv->length, 0, fp);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write class index vector [%ld]\n",
		    clsidxv->length);
	fclose(fp);
    }

    // memory free
    xlvfree(clsidxv);
    xdmfree(xm);
    xdmfree(mm);
    xdmfree(wseq);

    return;
}

void gmmmap_vit_file(char *inf, char *outf, char *wseqf, char *covf,
		     char *clsseqf, GMMPARA gmmpara, XBOOL msg_flag)
{
    long dnum;
    LVECTOR clsidxv = NODATA;
    FILE *fp;

    dnum = get_dnum_file(inf, gmmpara->xdim);
    // memory allocation
    clsidxv = xlvalloc(dnum);

    // mapping
    if (msg_flag == XTRUE) fprintf(stderr, "Input vectors: %s\n", inf);
    if (!strnone(wseqf) && msg_flag == XTRUE)
	fprintf(stderr, "Weight vector sequence: %s\n", wseqf);
    get_gmmmap_wghtseq_vit_file(inf, gmmpara->detvec, gmmpara->wght,
				gmmpara->xmean, gmmpara->xxcov, wseqf,
				gmmpara->dia_flag, clsidxv, msg_flag);
    get_gmmmap_meanseq_vit_file(inf, gmmpara->lm, gmmpara->xdim,
				gmmpara->ydim, gmmpara->clsnum, outf,
				gmmpara->dia_flag, clsidxv, msg_flag);
    // covariance matrices
    if (!strnone(covf)) {
	writedmatrix(covf, gmmpara->dm, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write covariance matrices [%ld][%ld]\n",
		    gmmpara->dm->row, gmmpara->dm->col);
    }
    // class index sequence
    if (!strnone(clsseqf)) {
	if ((fp = fopen(clsseqf, "wb")) == NULL) {
	    fprintf(stderr, "Can't open file: %s\n", clsseqf);
	    exit(1);
	}
	fwritelong(clsidxv->data, clsidxv->length, 0, fp);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Write class index vector [%ld]\n",
		    clsidxv->length);
	fclose(fp);
    }

    return;
}

DVECTOR xget_gmmmap_clsvec(DVECTOR x,	// [xdim]
			   long clsidx,
			   long ydim,
			   DMATRIX lm,	// [clsnum * 2 or * ydim][xdim or + 1]
			   XBOOL dia_flag)
{
    long k, l, b, xdim;
    DVECTOR mapx = NODATA;

    xdim = x->length;

    mapx = xdvalloc(ydim);
    if (xdim == ydim && dia_flag == XTRUE) {
	for (k = 0; k < xdim; k++)
	    mapx->data[k] = lm->data[clsidx * 2][k] * x->data[k]
		+ lm->data[clsidx * 2 + 1][k];
    } else {
	b = clsidx * ydim;
	for (k = 0; k < ydim; k++) {
	    mapx->data[k] = lm->data[k + b][xdim];
	    for (l = 0; l < xdim; l++)
		mapx->data[k] += lm->data[k + b][l] * x->data[l];
	}
    }

    return mapx;
}

DVECTOR xget_gmmmapvec(DVECTOR x,	// [xdim]
		       DVECTOR detvec,	// [clsnum]
		       DMATRIX wghtmat,	// [clsnum][1]
		       DMATRIX xmean,	// [clsnum][xdim]
		       DMATRIX xxicov,	// [clsnum or * xdim][xdim]
		       DMATRIX lm,	// [clsnum * 2 or * ydim][xdim or + 1]
		       long ydim,
		       XBOOL dia_flag)
{
    long k, l, clsnum;
    DVECTOR gv = NODATA;
    DVECTOR wv = NODATA;
    DVECTOR cy = NODATA;
    DVECTOR y = NODATA;

    clsnum = wghtmat->row;
    y = xdvzeros(ydim);

    // weight: p(cls|x)
    if (dia_flag == XTRUE)
	gv = xget_gaussvec_dia(x, detvec, wghtmat, xmean, xxicov);
    else
	gv = xget_gaussvec_full(x, detvec, wghtmat, xmean, xxicov);
    wv = xget_gammavec(gv);
    // mean: E(y|x)
    for (k = 0; k < clsnum; k++) {
	cy = xget_gmmmap_clsvec(x, k, ydim, lm, dia_flag);
	for (l = 0; l < ydim; l++) y->data[l] += wv->data[k] * cy->data[l];
	xdvfree(cy);
    }
    // memory free
    xdvfree(gv);
    xdvfree(wv);
    
    return y;
}

DMATRIX xget_gmmmapmat(DMATRIX xm,	// [dnum][xdim]
		       DVECTOR detvec,	// [clsnum]
		       DMATRIX wghtmat,	// [clsnum][1]
		       DMATRIX xmean,	// [clsnum][xdim]
		       DMATRIX xxicov,	// [clsnum or * xdim][xdim]
		       DMATRIX lm,	// [clsnum * 2 or * ydim][xdim or + 1]
		       long ydim,
		       XBOOL dia_flag)
{
    long k;
    DVECTOR xv = NODATA;
    DVECTOR mv = NODATA;
    DMATRIX mm = NODATA;

    mm = xdmalloc(xm->row, ydim);
    for (k = 0; k < xm->row; k++) {
	xv = xdmextractrow(xm, k);
	// mean: E(y|x) weighted sum of mapping vectors
	mv = xget_gmmmapvec(xv, detvec, wghtmat, xmean, xxicov, lm, ydim,
			    dia_flag);
	dmcopyrow(mm, k, mv);
	// memory free
	xdvfree(mv);
	xdvfree(xv);
    }

    return mm;
}

void get_gmmmapmat_file(char *xfile,
			DVECTOR detvec,	// [clsnum]
			DMATRIX wghtmat,// [clsnum][1]
			DMATRIX xmean,	// [clsnum][xdim]
			DMATRIX xxicov,	// [clsnum or * xdim][xdim]
			DMATRIX lm,	// [clsnum * 2 or * ydim][xdim or + 1]
			long ydim,
			char *mfile,
			XBOOL dia_flag,
			XBOOL msg_flag)
{
    long k, xdim, dnum;
    DVECTOR xv = NODATA;
    DVECTOR mv = NODATA;
    FILE *xfp, *mfp;

    xdim = xmean->col;
    dnum = get_dnum_file(xfile, xdim);

    // open file
    if ((xfp = fopen(xfile, "rb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", xfile);
	exit(1);
    }
    if ((mfp = fopen(mfile, "wb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", mfile);
	exit(1);
    }

    // memory allocation
    xv = xdvalloc(xdim);
    for (k = 0; k < dnum; k++) {
	fread(xv->data, sizeof(double), (size_t)xdim, xfp);
	// mean: E(y|x) weighted sum of mapping vectors
	mv = xget_gmmmapvec(xv, detvec, wghtmat, xmean, xxicov, lm, ydim,
			    dia_flag);
	fwrite(mv->data, sizeof(double), (size_t)mv->length, mfp);
	// memory free
	xdvfree(mv);
    }
    if (msg_flag == XTRUE)
	fprintf(stderr, "write mapping vectors [%ld][%ld]\n", dnum, ydim);

    // close file
    fclose(xfp);
    fclose(mfp);
    // memory free
    xdvfree(xv);

    return;
}

DMATRIX xget_gmmmap_wghtseq(DMATRIX xm,		// [dnum][xdim]
			    DVECTOR detvec,	// [clsnum]
			    DMATRIX wghtmat,	// [clsnum][1]
			    DMATRIX xmean,	// [clsnum][xdim]
			    DMATRIX xxicov,	// [clsnum or * xdim][xdim]
			    XBOOL dia_flag)
{
    long k;
    DVECTOR xv = NODATA;
    DVECTOR gv = NODATA;
    DVECTOR wv = NODATA;
    DMATRIX wm = NODATA;

    wm = xdmalloc(xm->row, detvec->length);
    for (k = 0; k < xm->row; k++) {
	xv = xdmextractrow(xm, k);
	if (dia_flag == XTRUE)
	    gv = xget_gaussvec_dia(xv, detvec, wghtmat, xmean, xxicov);
	else
	    gv = xget_gaussvec_full(xv, detvec, wghtmat, xmean, xxicov);
	wv = xget_gammavec(gv);
	dmcopyrow(wm, k, wv);
	// memory free
	xdvfree(xv);
	xdvfree(gv);
	xdvfree(wv);
    }

    return wm;
}

DMATRIX xget_gmmmap_wghtseq_vit(DMATRIX xm,		// [dnum][xdim]
				DVECTOR detvec,	// [clsnum]
				DMATRIX wghtmat,	// [clsnum][1]
				DMATRIX xmean,	// [clsnum][xdim]
				DMATRIX xxicov,	// [clsnum or * xdim][xdim]
				XBOOL dia_flag,
				LVECTOR clsidxv)
{
    long k, l;
    DVECTOR xv = NODATA;
    DVECTOR gv = NODATA;
    DVECTOR wv = NODATA;
    DMATRIX wm = NODATA;

    if (clsidxv->length != xm->row) {
	fprintf(stderr, "Error: xget_gmmmap_wghtseq_vit\n");
	exit(1);
    }
    wm = xdmalloc(xm->row, 1);
    for (k = 0; k < xm->row; k++) {
	xv = xdmextractrow(xm, k);
	if (dia_flag == XTRUE)
	    gv = xget_gaussvec_dia(xv, detvec, wghtmat, xmean, xxicov);
	else
	    gv = xget_gaussvec_full(xv, detvec, wghtmat, xmean, xxicov);
	wv = xget_gammavec(gv);
	// selecting class having maximum conditional probability
	wm->data[k][0] = wv->data[0];	clsidxv->data[k] = 0;
	for (l = 1; l < wv->length; l++) {
	    if (wm->data[k][0] < wv->data[l]) {
		wm->data[k][0] = wv->data[l];	clsidxv->data[k] = l;
	    }
	}
	// memory free
	xdvfree(xv);
	xdvfree(gv);
	xdvfree(wv);
    }

    return wm;
}

void get_gmmmap_wghtseq_file(char *xfile,
			     DVECTOR detvec,	// [clsnum]
			     DMATRIX wghtmat,	// [clsnum][1]
			     DMATRIX xmean,	// [clsnum][xdim]
			     DMATRIX xxicov,	// [clsnum or * xdim][xdim]
			     char *mfile,
			     XBOOL dia_flag,
			     XBOOL msg_flag)
{
    long k, xdim, dnum;
    DVECTOR xv = NODATA;
    DVECTOR gv = NODATA;
    DVECTOR wv = NODATA;
    FILE *xfp, *mfp;

    xdim = xmean->col;
    dnum = get_dnum_file(xfile, xdim);

    // open file
    if ((xfp = fopen(xfile, "rb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", xfile);
	exit(1);
    }
    if ((mfp = fopen(mfile, "wb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", mfile);
	exit(1);
    }

    // memory allocation
    xv = xdvalloc(xdim);
    for (k = 0; k < dnum; k++) {
	fread(xv->data, sizeof(double), (size_t)xdim, xfp);
	if (dia_flag == XTRUE)
	    gv = xget_gaussvec_dia(xv, detvec, wghtmat, xmean, xxicov);
	else
	    gv = xget_gaussvec_full(xv, detvec, wghtmat, xmean, xxicov);
	wv = xget_gammavec(gv);
	fwrite(wv->data, sizeof(double), (size_t)wv->length, mfp);
	// memory free
	xdvfree(gv);
	xdvfree(wv);
    }
    if (msg_flag == XTRUE)
	fprintf(stderr, "write weight vector sequence [%ld][%ld]\n",
		dnum, detvec->length);

    // close file
    fclose(xfp);
    fclose(mfp);
    // memory free
    xdvfree(xv);

    return;
}

void get_gmmmap_wghtseq_vit_file(char *xfile,
				 DVECTOR detvec,	// [clsnum]
				 DMATRIX wghtmat,	// [clsnum][1]
				 DMATRIX xmean,	// [clsnum][xdim]
				 DMATRIX xxicov,// [clsnum or * xdim][xdim]
				 char *mfile,
				 XBOOL dia_flag,
				 LVECTOR clsidxv,
				 XBOOL msg_flag)
{
    long k, l, xdim, dnum;
    DVECTOR xv = NODATA;
    DVECTOR gv = NODATA;
    DVECTOR wv = NODATA;
    FILE *xfp, *mfp = NULL;

    xdim = xmean->col;
    dnum = get_dnum_file(xfile, xdim);
    if (clsidxv->length != dnum) {
	fprintf(stderr, "Error: get_gmmmap_wghtseq_vit_file\n");
	exit(1);
    }

    // open file
    if ((xfp = fopen(xfile, "rb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", xfile);
	exit(1);
    }
    if (!strnone(mfile)) {
	if ((mfp = fopen(mfile, "wb")) == NULL) {
	    fprintf(stderr, "Can't open file: %s\n", mfile);
	    exit(1);
	}
    }

    // memory allocation
    xv = xdvalloc(xdim);
    for (k = 0; k < dnum; k++) {
	fread(xv->data, sizeof(double), (size_t)xdim, xfp);
	if (dia_flag == XTRUE)
	    gv = xget_gaussvec_dia(xv, detvec, wghtmat, xmean, xxicov);
	else
	    gv = xget_gaussvec_full(xv, detvec, wghtmat, xmean, xxicov);
	wv = xget_gammavec(gv);
	// selecting class having maximum conditional probability
	clsidxv->data[k] = 0;
	for (l = 1; l < wv->length; l++) {
	    if (wv->data[0] < wv->data[l]) {
		wv->data[0] = wv->data[l];	clsidxv->data[k] = l;
	    }
	}
	if (mfp != NULL) fwrite(wv->data, sizeof(double), (size_t)1, mfp);
	// memory free
	xdvfree(gv);
	xdvfree(wv);
    }
    if (msg_flag == XTRUE)
	fprintf(stderr, "write matrix [%ld][1]\n", dnum);

    // close file
    fclose(xfp);
    if (mfp != NULL) fclose(mfp);
    // memory free
    xdvfree(xv);

    return;
}

DMATRIX xget_gmmmap_meanseq(DMATRIX xm,	// [dnum][xdim]
			    DMATRIX lm,	// [clsnum * 2 or * ydim][xdim or + 1]
			    long ydim,
			    long clsnum,
			    XBOOL dia_flag)
{
    long i, k, b;
    DVECTOR xv = NODATA;
    DVECTOR mv = NODATA;
    DMATRIX mm = NODATA;

    mm = xdmalloc(xm->row * clsnum, ydim);
    for (k = 0; k < xm->row; k++) {
	xv = xdmextractrow(xm, k);
	// means: E(y|x)
	for (i = 0, b = k * clsnum; i < clsnum; i++) {
	    mv = xget_gmmmap_clsvec(xv, i, ydim, lm, dia_flag);
	    dmcopyrow(mm, i + b, mv);
	    // memory free
	    xdvfree(mv);
	}
	// memory free
	xdvfree(xv);
    }

    return mm;
}

DMATRIX xget_gmmmap_meanseq_vit(DMATRIX xm,	// [dnum][xdim]
				DMATRIX lm,//[clsnum*2 or *ydim][xdim or + 1]
				long ydim,
				long clsnum,
				XBOOL dia_flag,
				LVECTOR clsidxv)
{
    long k;
    DVECTOR xv = NODATA;
    DVECTOR mv = NODATA;
    DMATRIX mm = NODATA;

    if (clsidxv->length != xm->row) {
	fprintf(stderr, "Error: xget_gmmmap_meanseq_vit\n");
	exit(1);
    }
    mm = xdmalloc(xm->row, ydim);
    for (k = 0; k < xm->row; k++) {
	xv = xdmextractrow(xm, k);
	// mean: E(y|x)
	mv = xget_gmmmap_clsvec(xv, clsidxv->data[k], ydim, lm, dia_flag);
	dmcopyrow(mm, k, mv);
	// memory free
	xdvfree(mv);
	xdvfree(xv);
    }

    return mm;
}

void get_gmmmap_meanseq_file(char *xfile,
			     DMATRIX lm,// [clsnum * 2 or * ydim][xdim or + 1]
			     long xdim,
			     long ydim,
			     long clsnum,
			     char *mfile,
			     XBOOL dia_flag,
			     XBOOL msg_flag)
{
    long i, k, dnum;
    DVECTOR xv = NODATA;
    DVECTOR mv = NODATA;
    FILE *xfp, *mfp;

    dnum = get_dnum_file(xfile, xdim);

    // open file
    if ((xfp = fopen(xfile, "rb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", xfile);
	exit(1);
    }
    if ((mfp = fopen(mfile, "wb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", mfile);
	exit(1);
    }

    // memory allocation
    xv = xdvalloc(xdim);
    for (k = 0; k < dnum; k++) {
	fread(xv->data, sizeof(double), (size_t)xdim, xfp);
	// means: E(y|x)
	for (i = 0; i < clsnum; i++) {
	    mv = xget_gmmmap_clsvec(xv, i, ydim, lm, dia_flag);
	    fwrite(mv->data, sizeof(double), (size_t)mv->length, mfp);
	    // memory free
	    xdvfree(mv);
	}
    }
    if (msg_flag == XTRUE)
	fprintf(stderr, "write mean vector sequence [%ld][%ld]\n",
		dnum * clsnum, ydim);

    // close file
    fclose(xfp);
    fclose(mfp);
    // memory free
    xdvfree(xv);

    return;
}

void get_gmmmap_meanseq_vit_file(char *xfile,
				 DMATRIX lm,// [clsnum*2 or *ydim][xdim or + 1]
				 long xdim,
				 long ydim,
				 long clsnum,
				 char *mfile,
				 XBOOL dia_flag,
				 LVECTOR clsidxv,
				 XBOOL msg_flag)
{
    long k, dnum;
    DVECTOR xv = NODATA;
    DVECTOR mv = NODATA;
    FILE *xfp, *mfp;

    dnum = get_dnum_file(xfile, xdim);
    if (clsidxv->length != dnum) {
	fprintf(stderr, "Error: get_gmmmap_meanseq_vit_file\n");
	exit(1);
    }

    // open file
    if ((xfp = fopen(xfile, "rb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", xfile);
	exit(1);
    }
    if ((mfp = fopen(mfile, "wb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", mfile);
	exit(1);
    }

    // memory allocation
    xv = xdvalloc(xdim);
    for (k = 0; k < dnum; k++) {
	fread(xv->data, sizeof(double), (size_t)xdim, xfp);
	// means: E(y|x)
	mv = xget_gmmmap_clsvec(xv, clsidxv->data[k], ydim, lm, dia_flag);
	fwrite(mv->data, sizeof(double), (size_t)mv->length, mfp);
	// memory free
	xdvfree(mv);
    }
    if (msg_flag == XTRUE)
	fprintf(stderr, "write matrix [%ld][%ld]\n", dnum, ydim);

    // close file
    fclose(xfp);
    fclose(mfp);
    // memory free
    xdvfree(xv);

    return;
}
