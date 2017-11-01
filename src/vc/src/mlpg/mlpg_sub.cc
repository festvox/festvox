/*  ---------------------------------------------------------------  */
/*      The HMM-Based Speech Synthesis System (HTS): version 1.1.1   */
/*                        HTS Working Group                          */
/*                                                                   */
/*                   Department of Computer Science                  */
/*                   Nagoya Institute of Technology                  */
/*                                and                                */
/*    Interdisciplinary Graduate School of Science and Engineering   */
/*                   Tokyo Institute of Technology                   */
/*                      Copyright (c) 2001-2003                      */
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
/*                                                                   */
/*    2. Any modifications must be clearly marked as such.           */
/*                                                                   */
/*  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,  */
/*  HTS WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM    */
/*  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL       */
/*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF        */
/*  TECHNOLOGY, HTS WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE    */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY        */
/*  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  */
/*  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS   */
/*  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR          */
/*  PERFORMANCE OF THIS SOFTWARE.                                    */
/*                                                                   */
/*  ---------------------------------------------------------------  */
/*    mlpg.c : speech parameter generation from pdf sequence         */
/*                                                                   */
/*                                    2003/12/26 by Heiga Zen        */
/*  ---------------------------------------------------------------  */

/*********************************************************************/
/*                                                                   */
/*            Nagoya Institute of Technology, Aichi, Japan,          */
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
/*  NAGOYA INSTITUTE OF TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, AND  */
/*  THE CONTRIBUTORS TO THIS WORK DISCLAIM ALL WARRANTIES WITH       */
/*  REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF     */
/*  MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL NAGOYA INSTITUTE  */
/*  OF TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, NOR THE CONTRIBUTORS  */
/*  BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR  */
/*  ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR       */
/*  PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER   */
/*  TORTUOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE    */
/*  OR PERFORMANCE OF THIS SOFTWARE.                                 */
/*                                                                   */
/*********************************************************************/
/*                                                                   */
/*  ML-Based Parameter Generation                                    */
/*                                    2003/12/26 by Heiga Zen        */
/*                                                                   */
/*  Basic functions are extracted from HTS and                       */
/*   modified by Tomoki Toda (tomoki@ics.nitech.ac.jp)               */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          */
/*          Date   :  June 2004                                      */
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

#include "../sub/matope_sub.h"
#include "../sub/gmm_sub.h"
#include "mlpg_sub.h"

MLPGPARA xmlpgpara(long dim, long dim2, long dnum, long clsnum,
		   char *dynwinf, char *wseqf, char *mseqf, char *covf,
		   char *stf, char *vmfile, char *vvfile,
		   PStreamChol *pst, XBOOL dia_flag, XBOOL msg_flag)
{
    MLPGPARA param;
    
    // memory allocation
    param = xalloc(1, struct MLPGPARA_STRUCT);
    param->ov = xdvalloc(dim);
    if (dia_flag == XTRUE) param->iuv = xdvalloc(dim);
    else param->iuv = xdvalloc(dim * dim);
    param->iumv = xdvalloc(dim);
    param->flkv = xdvalloc(dnum);
    param->stm = NODATA;
    param->dltm = xdmalloc(dnum, dim2);
    param->pdf = NODATA;
    param->detvec = NODATA;
    param->wght = xdmalloc(clsnum, 1);
    param->mean = xdmalloc(clsnum, dim);
    param->cov = NODATA;
    param->clsidxv = NODATA;
    param->clsdetv = NODATA;
    param->clscov = NODATA;
    param->vdet = 1.0;
    param->vm = NODATA;
    param->vv = NODATA;
    param->var = NODATA;

    // initial static feature sequence
    get_stm_mlpgpara(param, stf, wseqf, mseqf, dim, dim2, dnum, clsnum,
		     msg_flag);
 
   // GMM parameters
    get_gmm_mlpgpara(param, pst, dynwinf, covf, dim, dim2, dnum, clsnum,
		     dia_flag);

    // global variance parameters
    get_gv_mlpgpara(param, vmfile, vvfile, dim2, msg_flag);

    return param;
}

MLPGPARA xmlpgpara_vit(long dim, long dim2, long dnum, long clsnum,
		       char *dynwinf, char *cseqf, char *mseqf, char *covf,
		       char *vmfile, char *vvfile,
		       PStreamChol *pst, XBOOL dia_flag, XBOOL msg_flag)
{
    MLPGPARA param;

    // memory allocation
    param = xalloc(1, struct MLPGPARA_STRUCT);
    param->ov = xdvalloc(dim);
    param->iuv = NODATA;
    param->iumv = NODATA;
    param->flkv = xdvalloc(dnum);
    param->stm = NODATA;
    param->dltm = xdmalloc(dnum, dim2);
    param->pdf = NODATA;
    param->detvec = NODATA;
    param->wght = xdmalloc(1, 1);
    param->mean = xdmalloc(1, dim);
    param->cov = NODATA;
    param->clsidxv = NODATA;
    if (dia_flag == XTRUE) {
	param->clsdetv = xdvalloc(1);
	param->clscov = xdmalloc(1, dim);
    } else {
	param->clsdetv = xdvalloc(1);
	param->clscov = xdmalloc(dim, dim);
    }
    param->vdet = 1.0;
    param->vm = NODATA;
    param->vv = NODATA;
    param->var = NODATA;

    // mixture-index sequence
    get_cseq_mlpgpara(param, cseqf, dnum);

    // initial static feature sequence
    get_stm_mlpgpara(param, mseqf, dim, dim2, dnum);

    // GMM parameters
    get_gmm_mlpgpara(param, pst, dynwinf, covf, dim, dim2, dnum, clsnum,
		     dia_flag);

    // global variance parameters
    get_gv_mlpgpara(param, vmfile, vvfile, dim2, msg_flag);

    return param;
}

void xmlpgparafree(MLPGPARA param)
{
    if (param != NODATA) {
	if (param->ov != NODATA) xdvfree(param->ov);
	if (param->iuv != NODATA) xdvfree(param->iuv);
	if (param->iumv != NODATA) xdvfree(param->iumv);
	if (param->flkv != NODATA) xdvfree(param->flkv);
	if (param->stm != NODATA) xdmfree(param->stm);
	if (param->dltm != NODATA) xdmfree(param->dltm);
	if (param->pdf != NODATA) xdmfree(param->pdf);
	if (param->detvec != NODATA) xdvfree(param->detvec);
	if (param->wght != NODATA) xdmfree(param->wght);
	if (param->mean != NODATA) xdmfree(param->mean);
	if (param->cov != NODATA) xdmfree(param->cov);
	if (param->clsidxv != NODATA) xlvfree(param->clsidxv);
	if (param->clsdetv != NODATA) xdvfree(param->clsdetv);
	if (param->clscov != NODATA) xdmfree(param->clscov);
	if (param->vm != NODATA) xdvfree(param->vm);
	if (param->vv != NODATA) xdvfree(param->vv);
	if (param->var != NODATA) xdvfree(param->var);
	xfree(param);
    }

    return;
}

void get_cseq_mlpgpara(MLPGPARA param, char *cseqf, long dnum)
{
    FILE *fp;

    param->clsidxv = xlvalloc(dnum);
    if ((fp = fopen(cseqf, "rb")) == NULL) {
	fprintf(stderr, "Can't read file: %s\n", cseqf);
	exit(1);
    }
    freadlong(param->clsidxv->data, dnum, 0, fp);
    fclose(fp);

    return;
}

void get_stm_mlpgpara(MLPGPARA param, char *mseqf,
		      long dim, long dim2, long dnum)
{
    long d, k;
    FILE *fp;

    param->stm = xdmalloc(dnum, dim2);
    if ((fp = fopen(mseqf, "rb")) == NULL) {
	fprintf(stderr, "Can't read file: %s\n", mseqf);
	exit(1);
    }
    for (d = 0; d < dnum; d++) {
	fread(param->mean->data[0], sizeof(double), (size_t)dim, fp);
	for (k = 0; k < dim2; k++)
	    param->stm->data[d][k] = param->mean->data[0][k];
    }
    fclose(fp);

    return;
}

void get_stm_mlpgpara(MLPGPARA param, char *stf, char *wseqf, char *mseqf,
		      long dim, long dim2, long dnum, long clsnum,
		      XBOOL msg_flag)
{
    long d, c, k;
    FILE *wfp = NULL, *mfp = NULL;

    if (!strnone(stf)) {
	// static [dnum][dim]
	if ((param->stm = xreaddmatrix(stf, dim2, 0)) == NODATA) {
	    fprintf(stderr, "Can't read file: %s\n", stf);
	    exit(1);
	} else if (dnum != param->stm->row) {
	    fprintf(stderr, "Error file format: %s\n", stf);
	    exit(1);
	}
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Initial static file: %s\n", stf);
    } else if (!strnone(wseqf) && !strnone(mseqf)) {
	// open files
	if ((wfp = fopen(wseqf, "rb")) == NULL) {
	    fprintf(stderr, "Can't read file: %s\n", wseqf);
	    exit(1);
	}
	if ((mfp = fopen(mseqf, "rb")) == NULL) {
	    fprintf(stderr, "Can't read file: %s\n", mseqf);
	    exit(1);
	}
	param->stm = xdmalloc(dnum, dim2);
	for (d = 0; d < dnum; d++) {
	    for (c = 0; c < clsnum; c++) {
		fread(param->wght->data[c], sizeof(double), (size_t)1, wfp);
		fread(param->mean->data[c], sizeof(double), (size_t)dim, mfp);
	    }
	    for (k = 0; k < dim2; k++)
		for (c = 0, param->stm->data[d][k] = 0.0; c < clsnum; c++)
		    param->stm->data[d][k] +=
			param->wght->data[c][0] * param->mean->data[c][k];
	}
	// close file
	fclose(wfp);
	fclose(mfp);
    } else {
	fprintf(stderr, "Error: get_stm_mlpgpara: Need parameter files\n");
	exit(1);
    }

    return;
}

void get_gmm_mlpgpara(MLPGPARA param, PStreamChol *pst,
		      char *dynwinf, char *covf,
		      long dim, long dim2, long dnum, long clsnum,
		      XBOOL dia_flag)
{
    long k, l;
    DMATRIX fcov = NODATA;

    if (dia_flag == XTRUE) {	// diagonal covariance
	InitPStreamChol(pst, dynwinf, NULL, (int)(dim2 - 1), (int)dnum);
	param->pdf = xdmalloc(dnum, dim * 2);
	// input covariances [clsnum * dim][dim]
	if ((fcov = xreaddmatrix(covf, dim, 0)) == NODATA) {
	    fprintf(stderr, "Can't read covariance file: %s\n", covf);
	    exit(1);
	}
	if (fcov->row == clsnum * dim) {
	    // covariances [clsnum][dim]
	    param->cov = xdmalloc(clsnum, dim);
	    for (k = 0; k < clsnum; k++)
		for (l = 0; l < dim; l++)
		    param->cov->data[k][l] = fcov->data[k * dim + l][l];
	} else if (fcov->row == clsnum) {
	    param->cov = xdmclone(fcov);
	} else {
	    fprintf(stderr, "Error file format: %s\n", covf);
	    exit(1);
	}
	// memory free
	xdmfree(fcov);
	// inverse covariance matrices
	param->detvec = xget_detvec_diamat2inv(param->cov);
    } else {				// full covariance
	InitPStreamCholFC(pst, dynwinf, NULL, (int)(dim2 - 1), (int)dnum);
	param->pdf = xdmalloc(dnum, dim * (dim + 1));
	// input covariances [clsnum * dim][dim]
	if ((param->cov = xreaddmatrix(covf, dim, 0)) == NODATA) {
	    fprintf(stderr, "Can't read covariance file: %s\n", covf);
	    exit(1);
	}
	if (param->cov->row != clsnum * dim) {
	    fprintf(stderr, "Error file format: %s\n", covf);
	    exit(1);
	}
	// inverse covariance matrices
	param->detvec = xget_detvec_mat2inv(clsnum, param->cov, XFALSE);
    }

    return;
}

void get_gv_mlpgpara(MLPGPARA param, char *vmfile, char *vvfile,
		     long dim2, XBOOL msg_flag)
{
    long k;

    if (!strnone(vmfile) && !strnone(vvfile)) {
	param->vm = xreaddvector(vmfile, 0);
	param->vv = xreaddvector(vvfile, 0);
	if (msg_flag == XTRUE)
	    fprintf(stderr, "Read variance parameter files\n");
	if (param->vm->length == dim2 && param->vv->length == dim2) {
	    if (msg_flag == XTRUE)
		fprintf(stderr, "variance of static feature\n");
	} else {
	    fprintf(stderr,
		    "Error: xmlpgpara: dimension of variance features\n");
	    exit(1);
	}
	// inverse covariance matrices
	for (k = 0, param->vdet = 1.0; k < param->vv->length; k++) {
	    param->vdet *= param->vv->data[k];
	    if (param->vdet <= 0.0) {
		fprintf(stderr,
			"Error: xmlpgpara: determinant of variance vector\n");
		exit(1);
	    }
	    param->vv->data[k] = 1.0 / param->vv->data[k];
	}
	param->var = xdvzeros(param->vv->length);
    }

    return;
}

double get_like_pdfseq(long dim, long dim2, long dnum, long clsnum,
		       MLPGPARA param, FILE *wfp, FILE *mfp,
		       XBOOL dia_flag, XBOOL vit_flag)
{
    long d, c, k, l;
    double sumgauss;
    double like = 0.0;
    DVECTOR gaussv = NODATA;
    DVECTOR gammav = NODATA;

    // frame-by-frame processing
    fseek(wfp, 0, SEEK_SET);
    fseek(mfp, 0, SEEK_SET);
    for (d = 0, like = 0.0; d < dnum; d++) {
	// read weight and mean sequences
	for (c = 0; c < clsnum; c++) {
	    fread(param->wght->data[c], sizeof(double), (size_t)1, wfp);
	    fread(param->mean->data[c], sizeof(double), (size_t)dim, mfp);
	}

	// observation vector
	for (k = 0; k < dim2; k++) {
	    param->ov->data[k] = param->stm->data[d][k];
	    param->ov->data[k + dim2] = param->dltm->data[d][k];
	}

	// calculating gamma
	if (dia_flag == XTRUE) {
	    gaussv = xget_gaussvec_dia(param->ov, param->detvec, param->wght,
				       param->mean, param->cov);
	} else {
	    gaussv = xget_gaussvec_full(param->ov, param->detvec, param->wght,
					param->mean, param->cov);
	}
	if (vit_flag == XTRUE) {
	    for (k = 1, c = 0; k < gaussv->length; k++)
		if (gaussv->data[c] < gaussv->data[k]) c = k;
	    for (k = 0; k < gaussv->length; k++)
		if (c != k) gaussv->data[k] = 0.0;
	}
	gammav = xget_gammavec(gaussv);
	if ((sumgauss = dvsum(gaussv)) <= 0.0)
	    param->flkv->data[d] = -1.0 * INFTY2;
	else param->flkv->data[d] = log(sumgauss);
	like += param->flkv->data[d];

	// estimating U', U'*M
	if (dia_flag == XTRUE) {
	    for (k = 0; k < dim; k++) {
		param->iuv->data[k] = 0.0;	param->iumv->data[k] = 0.0;
		for (c = 0; c < clsnum; c++) {
		    param->iuv->data[k] +=
			gammav->data[c] * param->cov->data[c][k];
		    param->iumv->data[k] += gammav->data[c]
			* param->cov->data[c][k] * param->mean->data[c][k];
		}
	    }
	    // PDF [U'*M U']
	    for (k = 0; k < dim; k++) {
		param->pdf->data[d][k] = param->iumv->data[k];
		param->pdf->data[d][k + dim] = param->iuv->data[k];
	    }
	} else {
	    for (k = 0; k < dim; k++) {
		param->iumv->data[k] = 0.0;
		for (l = 0; l < dim; l++) {
		    param->iuv->data[k * dim + l] = 0.0;
		    for (c = 0; c < clsnum; c++) {
			param->iuv->data[k * dim + l] += gammav->data[c]
			    * param->cov->data[c * dim + k][l];
			param->iumv->data[k] += gammav->data[c]
			    * param->cov->data[c * dim + k][l]
			    * param->mean->data[c][l];
		    }
		}
	    }
	    // PDF [U'*M U']
	    for (k = 0; k < dim; k++) {
		param->pdf->data[d][k] = param->iumv->data[k];
		for (l = 0; l < dim; l++)
		    param->pdf->data[d][k * dim + dim + l] =
			param->iuv->data[k * dim + l];
	    }
	}
	// memory free
	xdvfree(gaussv);
	xdvfree(gammav);
    }
    like /= (double)dnum;

    return like;
}

double get_like_pdfseq_vit(long dim, long dim2, long dnum, long clsnum,
			   MLPGPARA param, FILE *wfp, FILE *mfp,
			   XBOOL dia_flag)
{
    long d, c, k, l;
    double sumgauss;
    double like = 0.0;

    for (d = 0, like = 0.0; d < dnum; d++) {
	// read weight and mean sequences
	fread(param->wght->data[0], sizeof(double), (size_t)1, wfp);
	fread(param->mean->data[0], sizeof(double), (size_t)dim, mfp);

	// observation vector
	for (k = 0; k < dim2; k++) {
	    param->ov->data[k] = param->stm->data[d][k];
	    param->ov->data[k + dim2] = param->dltm->data[d][k];
	}

	// mixture index
	if ((c = param->clsidxv->data[d]) >= clsnum) {
	    fprintf(stderr,
		    "Error: get_like_pdfseq_vit: class index sequence file\n");
	    exit(1);
	}
	param->clsdetv->data[0] = param->detvec->data[c];

	// calculating likelihood
	if (dia_flag == XTRUE) {
	    for (k = 0; k < param->clscov->col; k++)
		param->clscov->data[0][k] = param->cov->data[c][k];
	    sumgauss = get_gauss_dia(0, param->ov, param->clsdetv,
				     param->wght, param->mean, param->clscov);
	} else {
	    for (k = 0; k < param->clscov->row; k++)
		for (l = 0; l < param->clscov->col; l++)
		    param->clscov->data[k][l] =
			param->cov->data[k + param->clscov->row * c][l];
	    sumgauss = get_gauss_full(0, param->ov, param->clsdetv,
				      param->wght, param->mean, param->clscov);
	}
	if (sumgauss <= 0.0) param->flkv->data[d] = -1.0 * INFTY2;
	else param->flkv->data[d] = log(sumgauss);
	like += param->flkv->data[d];

	// estimating U', U'*M
	if (dia_flag == XTRUE) {
	    // PDF [U'*M U']
	    for (k = 0; k < dim; k++) {
		param->pdf->data[d][k] =
		    param->clscov->data[0][k] * param->mean->data[0][k];
		param->pdf->data[d][k + dim] = param->clscov->data[0][k];
	    }
	} else {
	    // PDF [U'*M U']
	    for (k = 0; k < dim; k++) {
		param->pdf->data[d][k] = 0.0;
		for (l = 0; l < dim; l++) {
		    param->pdf->data[d][k * dim + dim + l] =
			param->clscov->data[k][l];
		    param->pdf->data[d][k] +=
			param->clscov->data[k][l] * param->mean->data[0][l];
		}
	    }
	}
    }

    like /= (double)dnum;

    return like;
}

double get_like_gv(long dim2, long dnum, MLPGPARA param)
{
    long k;
    double av = 0.0, dif = 0.0;
    double vlike = -INFTY;

    if (param->vm != NODATA && param->vv != NODATA) {
	for (k = 0; k < dim2; k++)
	    calc_varstats(param->stm->data, k, dnum, &av,
			  &(param->var->data[k]), &dif);
	vlike = log(get_gauss_dia(param->vdet, 1.0, param->var,
				  param->vm, param->vv));
    }

    return vlike;
}

void sm_mvav(DMATRIX mat, long hlen)
{
    long k, l, m, p;
    double d, sd;
    DVECTOR vec = NODATA;
    DVECTOR win = NODATA;

    vec = xdvalloc(mat->row);

    // smoothing window
    win = xdvalloc(hlen * 2 + 1);
    for (k = 0, d = 1.0, sd = 0.0; k < hlen; k++, d += 1.0) {
	win->data[k] = d;	win->data[win->length - k - 1] = d;
	sd += d + d;
    }
    win->data[k] = d;	sd += d;
    for (k = 0; k < win->length; k++) win->data[k] /= sd;

    for (l = 0; l < mat->col; l++) {
	for (k = 0; k < mat->row; k++) {
	    for (m = 0, vec->data[k] = 0.0; m < win->length; m++) {
		p = k - hlen + m;
		if (p >= 0 && p < mat->row)
		    vec->data[k] += mat->data[p][l] * win->data[m];
	    }
	}
	for (k = 0; k < mat->row; k++) mat->data[k][l] = vec->data[k];
    }

    xdvfree(win);
    xdvfree(vec);

    return;
}

void get_dltmat(DMATRIX mat, DWin *dw, int dno, DMATRIX dmat)
{
    int i, j, k, tmpnum;

    tmpnum = (int)mat->row - dw->width[dno][WRIGHT];
    for (k = dw->width[dno][WRIGHT]; k < tmpnum; k++)	// time index
	for (i = 0; i < (int)mat->col; i++)	// dimension index
	    for (j = dw->width[dno][WLEFT], dmat->data[k][i] = 0.0;
		 j <= dw->width[dno][WRIGHT]; j++)
		dmat->data[k][i] += mat->data[k + j][i] * dw->coef[dno][j];

    for (i = 0; i < (int)mat->col; i++) {		// dimension index
	for (k = 0; k < dw->width[dno][WRIGHT]; k++)		// time index
	    for (j = dw->width[dno][WLEFT], dmat->data[k][i] = 0.0;
		 j <= dw->width[dno][WRIGHT]; j++)
		if (k + j >= 0)
		    dmat->data[k][i] += mat->data[k + j][i] * dw->coef[dno][j];
		else
		    dmat->data[k][i] += (2.0 * mat->data[0][i] - mat->data[-k - j][i]) * dw->coef[dno][j];
	for (k = tmpnum; k < (int)mat->row; k++)	// time index
	    for (j = dw->width[dno][WLEFT], dmat->data[k][i] = 0.0;
		 j <= dw->width[dno][WRIGHT]; j++)
		if (k + j < (int)mat->row)
		    dmat->data[k][i] += mat->data[k + j][i] * dw->coef[dno][j];
		else
		    dmat->data[k][i] += (2.0 * mat->data[mat->row - 1][i] - mat->data[mat->row - k - j + mat->row - 2][i]) * dw->coef[dno][j];
    }

    return;
}


double *dcalloc(int x, int xoff)
{
    double *ptr;

    if ((ptr = (double *) calloc(x, sizeof(*ptr))) == NULL) {
	fprintf(stderr, "Cannot Allocate Memory\n");
	exit(1);
    }
    ptr += xoff;
    return(ptr);
}

extern double **ddcalloc(int x, int y, int xoff, int yoff)
{
    double **ptr;
    register int i;

    if ((ptr = (double **) calloc(x, sizeof(*ptr))) == NULL) {
	fprintf(stderr, "Cannot Allocate Memory\n");
	exit(1);
    }
    for (i = 0; i < x; i++) ptr[i] = dcalloc(y, yoff);
    ptr += xoff;
    return(ptr);
}

double ***dddcalloc(int x, int y, int z, int xoff, int yoff, int zoff)
{
    double ***ptr;
    register int i;

    if ((ptr = (double ***) calloc(x, sizeof(*ptr))) == NULL) {
	fprintf(stderr, "Cannot Allocate Memory\n");
	exit(1);
    }
    for (i = 0; i < x; i++) ptr[i] = ddcalloc(y, z, yoff, zoff);
    ptr += xoff;
    return(ptr);
}


/////////////////////////////////////
// ML using Choleski decomposition //
/////////////////////////////////////
void InitDWin(PStreamChol *pst, char *dynwinf, char *accwinf)
{
    register int i;
    int fsize, leng;
    FILE *fp;

    pst->dw.num = 1;	// only static
    if (!strnone(dynwinf)) {
	pst->dw.num = 2;	// static + dyn
	if (!strnone(accwinf)) pst->dw.num = 3;	// static + dyn + acc
    }
    // memory allocation
    if ((pst->dw.width = (int **) calloc(pst->dw.num, sizeof(int *)))
	== NULL) {	// [1--3][2]
	fprintf(stderr, "Cannot Allocate Memory\n");
	exit(1);
    } else {
	for (i = 0; i < pst->dw.num; i++) {
	    if ((pst->dw.width[i] = (int *) calloc(2, sizeof(int))) == NULL) {
		fprintf(stderr, "Cannot Allocate Memory\n");
		exit(1);
	    }
	}
    }
    if ((pst->dw.coef = (double **) calloc(pst->dw.num, sizeof(double *)))
	== NULL) {	// [3][*]
	fprintf(stderr, "Cannot Allocate Memory\n");
	exit(1);
    }

    // window for static parameter	WLEFT = 0, WRIGHT = 1
    pst->dw.width[0][WLEFT] = pst->dw.width[0][WRIGHT] = 0;
    pst->dw.coef[0] = dcalloc(1, 0);
    pst->dw.coef[0][0] = 1.0;

    // set delta coefficients
    for (i = 1; i < pst->dw.num; i++) {
	if (i == 1) {		// read dyn window coefficients
	    fsize = get_dnum_file(dynwinf, 1);
	    if ((fp = fopen(dynwinf, "rb")) == NULL) {
		fprintf(stderr, "file %s not found\n", dynwinf);
		exit(1);
	    }
	} else if (i == 2) {	// read acc window coefficients
	    fsize = get_dnum_file(accwinf, 1);
	    if ((fp = fopen(accwinf, "rb")) == NULL) {
		fprintf(stderr, "file %s not found\n", accwinf);
		exit(1);
	    }
	} else {
	    fprintf(stderr, "Error: InitDWin\n");
	    exit(1);
	}
	pst->dw.coef[i] = dcalloc(fsize, 0);
	fread(pst->dw.coef[i], sizeof(double), fsize, fp);
	// set pointer
	leng = fsize / 2;			// L (fsize = 2 * L + 1)
	pst->dw.coef[i] += leng;		// [L] -> [0]	center
	pst->dw.width[i][WLEFT] = -leng;	// -L		left
	pst->dw.width[i][WRIGHT] = leng;	//  L		right
	if (fsize % 2 == 0) pst->dw.width[i][WRIGHT]--;
	fclose(fp);
    }

    pst->dw.maxw[WLEFT] = pst->dw.maxw[WRIGHT] = 0;
    for (i = 0; i < pst->dw.num; i++) {
	if (pst->dw.maxw[WLEFT] > pst->dw.width[i][WLEFT])
	    pst->dw.maxw[WLEFT] = pst->dw.width[i][WLEFT];
	if (pst->dw.maxw[WRIGHT] < pst->dw.width[i][WRIGHT])
	    pst->dw.maxw[WRIGHT] = pst->dw.width[i][WRIGHT];
    }

    return;
}

void InitPStreamChol(PStreamChol *pst, char *dynwinf, char *accwinf,
		     int order, int T)
{
    // order of cepstrum
    pst->order = order;

    // windows for dynamic feature
    InitDWin(pst, dynwinf, accwinf);

    // dimension of observed vector
    pst->vSize = (pst->order + 1) * pst->dw.num;	// odim = dim * (1--3)

    // memory allocation
    pst->T = T;					// number of frames
    pst->width = pst->dw.maxw[WRIGHT] * 2 + 1;	// width of R
    pst->mseq = ddcalloc(T, pst->vSize, 0, 0);	// [T][odim] 
    pst->ivseq = ddcalloc(T, pst->vSize, 0, 0);	// [T][odim]
    pst->R = ddcalloc(T, pst->width, 0, 0);	// [T][width]
    pst->r = dcalloc(T, 0);			// [T]
    pst->g = dcalloc(T, 0);			// [T]
    pst->c = ddcalloc(T, pst->order + 1, 0, 0);	// [T][dim]

    return;
}

void mlgparaChol(DMATRIX pdf, PStreamChol *pst, DMATRIX mlgp)
{
    int t, d;

    // error check
    if (pst->vSize * 2 != pdf->col || pst->order + 1 != mlgp->col) {
	fprintf(stderr, "Error mlgparaChol: Different dimension\n");
	exit(1);
    }

    // mseq: U^{-1}*M,	ifvseq: U^{-1}
    for (t = 0; t < pst->T; t++) {
	for (d = 0; d < pst->vSize; d++) {
	    pst->mseq[t][d] = pdf->data[t][d];
	    pst->ivseq[t][d] = pdf->data[t][pst->vSize + d];
	}
    } 

    // ML parameter generation
    mlpgChol(pst);

    // extracting parameters
    for (t = 0; t < pst->T; t++)
	for (d = 0; d <= pst->order; d++)
	    mlgp->data[t][d] = pst->c[t][d];

    return;
}

// generate parameter sequence from pdf sequence using Choleski decomposition
void mlpgChol(PStreamChol *pst)
{
   register int m;

   // generating parameter in each dimension
   for (m = 0; m <= pst->order; m++) {
       calc_R_and_r(pst, m);
       Choleski(pst);
       Choleski_forward(pst);
       Choleski_backward(pst, m);
   }
   
   return;
}

//------ parameter generation fuctions
// calc_R_and_r: calculate R = W'U^{-1}W and r = W'U^{-1}M
void calc_R_and_r(PStreamChol *pst, const int m)
{
    register int i, j, k, l, n;
    double   wu;
   
    for (i = 0; i < pst->T; i++) {
	pst->r[i] = pst->mseq[i][m];
	pst->R[i][0] = pst->ivseq[i][m];
      
	for (j = 1; j < pst->width; j++) pst->R[i][j] = 0.0;
      
	for (j = 1; j < pst->dw.num; j++) {
	    for (k = pst->dw.width[j][0]; k <= pst->dw.width[j][1]; k++) {
		n = i + k;
		if (n >= 0 && n < pst->T && pst->dw.coef[j][-k] != 0.0) {
		    l = j * (pst->order + 1) + m;
		    pst->r[i] += pst->dw.coef[j][-k] * pst->mseq[n][l]; 
		    wu = pst->dw.coef[j][-k] * pst->ivseq[n][l];
            
		    for (l = 0; l < pst->width; l++) {
			n = l-k;
			if (n <= pst->dw.width[j][1] && i + l < pst->T &&
			    pst->dw.coef[j][n] != 0.0)
			    pst->R[i][l] += wu * pst->dw.coef[j][n];
		    }
		}
	    }
	}
    }

    return;
}

// Choleski: Choleski factorization of Matrix R
void Choleski(PStreamChol *pst)
{
    register int t, j, k;

    pst->R[0][0] = sqrt(pst->R[0][0]);

    for (j = 1; j < pst->width; j++) pst->R[0][j] /= pst->R[0][0];

    for (t = 1; t < pst->T; t++) {
	for (j = 1; j < pst->width; j++)
	    if (t - j >= 0)
		pst->R[t][0] -= pst->R[t - j][j] * pst->R[t - j][j];
         
	pst->R[t][0] = sqrt(pst->R[t][0]);
         
	for (j = 1; j < pst->width; j++) {
	    for (k = 0; k < pst->dw.maxw[WRIGHT]; k++)
		if (j != pst->width - 1)
		    pst->R[t][j] -=
			pst->R[t - k - 1][j - k] * pst->R[t - k - 1][j + 1];
            
	    pst->R[t][j] /= pst->R[t][0];
	}
    }
   
    return;
}

// Choleski_forward: forward substitution to solve linear equations
void Choleski_forward(PStreamChol *pst)
{
    register int t, j;
    double hold;
   
    pst->g[0] = pst->r[0] / pst->R[0][0];

    for (t=1; t < pst->T; t++) {
	hold = 0.0;
	for (j = 1; j < pst->width; j++)
	    if (t - j >= 0 && pst->R[t - j][j] != 0.0)
		hold += pst->R[t - j][j] * pst->g[t - j];
	pst->g[t] = (pst->r[t] - hold) / pst->R[t][0];
    }
   
    return;
}

// Choleski_backward: backward substitution to solve linear equations
void Choleski_backward(PStreamChol *pst, const int m)
{
    register int t, j;
    double hold;
   
    pst->c[pst->T - 1][m] = pst->g[pst->T - 1] / pst->R[pst->T - 1][0];

    for (t = pst->T - 2; t >= 0; t--) {
	hold = 0.0;
	for (j = 1; j < pst->width; j++)
	    if (t + j < pst->T && pst->R[t][j] != 0.0)
		hold += pst->R[t][j] * pst->c[t + j][m];
	pst->c[t][m] = (pst->g[t] - hold) / pst->R[t][0];
   }
   
   return;
}

// Full Covariance Version
void InitPStreamCholFC(PStreamChol *pst, char *dynwinf, char *accwinf,
		       int order, int T)
{
    // order of cepstrum
    pst->order = order;

    // windows for dynamic feature
    InitDWin(pst, dynwinf, accwinf);

    // dimension of observed vector
    pst->vSize = (pst->order + 1) * pst->dw.num;	// odim = dim * (1--3)

    // memory allocation
    pst->T = T;					// number of frames
    pst->width = pst->dw.maxw[WRIGHT] * 2 + 1;	// width of R
    pst->mseq = ddcalloc(T, pst->vSize, 0, 0);	// [T][odim] 
    // [T][odim][odim]
    pst->ifvseq = dddcalloc(T, pst->vSize, pst->vSize, 0, 0, 0);
    // [T * dim][width * dim]
    pst->R = ddcalloc(T * (order + 1), pst->width * (order + 1), 0, 0);
    pst->r = dcalloc(T * (order + 1), 0);		// [T * dim]
    pst->g = dcalloc(T * (order + 1), 0);		// [T * dim]
    pst->c = ddcalloc(T, pst->order + 1, 0, 0);		// [T][dim]

    return;
}

void mlgparaCholFC(DMATRIX pdf, PStreamChol *pst, DMATRIX mlgp)
{
    int t, d, dx, dy;

    // error check
    if (pst->vSize * (pst->vSize + 1) != pdf->col
	|| pst->order + 1 != mlgp->col) {
	fprintf(stderr, "Error mlgparaCholFC: Different dimension\n");
	exit(1);
    }

    // mseq: U^{-1}*M,	ifvseq: U^{-1}
    for (t = 0; t < pst->T; t++) {
	for (dx = 0, d = 0; dx < pst->vSize; dx++) {
	    pst->mseq[t][dx] = pdf->data[t][dx];
	    for (dy = 0; dy < pst->vSize; dy++, d++)
		pst->ifvseq[t][dx][dy] = pdf->data[t][pst->vSize + d];
	}
    } 

    // ML parameter generation
    mlpgCholFC(pst);

    // extracting parameters
    for (t = 0; t < pst->T; t++)
	for (d = 0; d <= pst->order; d++)
	    mlgp->data[t][d] = pst->c[t][d];

    return;
}

// generate parameter sequence from pdf sequence using Choleski decomposition
void mlpgCholFC(PStreamChol *pst)
{
   // generating parameter in each dimension
   calc_R_and_r_FC(pst);
   CholeskiFC(pst);
   Choleski_forwardFC(pst);
   Choleski_backwardFC(pst);

   return;
}

//------ parameter generation fuctions
// calculate R = W'U^{-1}W and r = W'U^{-1}M
void calc_R_and_r_FC(PStreamChol *pst)
{
    register int t, i, j, k, l, m, n, sdim;
    double *tmpdata = NULL;

    // (1 * 2 + 1) * 2 = 6	offset = 2 -> -2 -1  0 1  2 3
    tmpdata = dcalloc((pst->dw.maxw[WRIGHT] * 2 + 1) * pst->dw.num,
		      pst->dw.maxw[WRIGHT] * pst->dw.num);
    sdim = pst->order + 1;		// dim of static vector
    for (t = 0; t < pst->T; t++) {
	for (m = 0; m < sdim; m++) {	// dim of static vector
	    // t-th frame, m-th static
	    pst->r[t * sdim + m] = pst->mseq[t][m];
	    // dynamic feat
	    for (j = 1; j < pst->dw.num; j++) {	// num of dyn feats
		// j-th dyn feat
		for (k = pst->dw.width[j][0]; k <= pst->dw.width[j][1]; k++) {
		    // k-th dyn win coef
		    n = t + k;	// n: time index
		    if (n >= 0 && n < pst->T && pst->dw.coef[j][-k] != 0.0) {
			// j * sdim + m: dim of obs vec
			pst->r[t * sdim + m] +=
			    pst->dw.coef[j][-k] * pst->mseq[n][j * sdim + m];
		    }
		}
	    }

	    for (i = 0; i < sdim; i++) {	// dim of static vector
		// W'U^{-1} vector, t-th frame, m-th row, i-th col
		// static window
		for (l = -pst->dw.maxw[WRIGHT];
		     l <= pst->dw.maxw[WRIGHT]; l++) {
		    if (l == 0) {
			tmpdata[0] = pst->ifvseq[t][m][i];
			for (j = 1; j < pst->dw.num; j++)
			    tmpdata[j] = pst->ifvseq[t][m][j * sdim + i];
		    } else {
			for (j = 0; j < pst->dw.num; j++)
			    tmpdata[l * pst->dw.num + j] = 0.0;
		    }
		}
		// delta window
		for (j = 1; j < pst->dw.num; j++) {	// j-th dyn coef
		    for (k = pst->dw.width[j][0]; k <= pst->dw.width[j][1];
			 k++) {				// k-th dyn win coef
			n = t + k;	// n: time index
			if (n >= 0 && n < pst->T && pst->dw.coef[j][-k] != 0.0)
			    for (l = 0; l < pst->dw.num; l++)
				tmpdata[k * pst->dw.num + l] +=
				    pst->dw.coef[j][-k] *
				    pst->ifvseq[n][j * sdim + m][l * sdim + i];
		    }
		}
		// W'U^{-1}W vector, t-th frame, m-th row, i-th col
		// static window
		for (l = 0; l <= pst->dw.maxw[WRIGHT] && t + l < pst->T; l++)
		    pst->R[t * sdim + m][l * sdim + i] =
			tmpdata[l * pst->dw.num];
		for (; l < pst->width; l++)
		    pst->R[t * sdim + m][l * sdim + i] = 0.0;
		// delta window
		for (l = 0; l < pst->width && t + l < pst->T; l++) {
		    for (j = 1; j < pst->dw.num; j++) {	// j-th dyn coef
			for (k = pst->dw.width[j][0]; k <= pst->dw.width[j][1];
			     k++) {			// k-th dyn win coef
			    n = -k + l;	// n-th dyn win coef
			    if (n <= pst->dw.width[j][1] &&
				n >= pst->dw.width[j][0])
				if (pst->dw.coef[j][n] != 0.0)
				    pst->R[t * sdim + m][l * sdim + i] +=
					pst->dw.coef[j][n]
					* tmpdata[k * pst->dw.num + j];
			}
		    }
		}
	    }
	}
    }

    n = pst->width * sdim;
    for (t = 0; t < pst->T; t++) {
	for (i = 0; i < sdim; i++) {
	    for (j = 0, m = i; m < n; m++, j++)
		pst->R[t * sdim + i][j] = pst->R[t * sdim + i][m];
	    for (; j < n; j++) pst->R[t * sdim + i][j] = 0.0;
	}
    }

    // memory free
    tmpdata -= pst->dw.maxw[WRIGHT] * pst->dw.num;
    free(tmpdata);

    return;
}

// Choleski: Choleski factorization of Matrix R
void CholeskiFC(PStreamChol *pst)
{
    register int t, j, k, row, col;

    row = pst->T * (pst->order + 1);
    col = pst->width * (pst->order + 1);

    pst->R[0][0] = sqrt(pst->R[0][0]);
    for (j = 1; j < col; j++) pst->R[0][j] /= pst->R[0][0];

    for (t = 1; t < row; t++) {
	for (j = 1; j < col; j++)
	    if (t - j >= 0)
		pst->R[t][0] -= pst->R[t - j][j] * pst->R[t - j][j];
	if (pst->R[t][0] < 0.0) {
	    fprintf(stderr, "Error CholeskiFC, %d\n", t);
	    exit(1);
	}
	pst->R[t][0] = sqrt(pst->R[t][0]);
	for (j = 1; j < col && t + j < row; j++) {
	    for (k = 0; k < col; k++)
		if (k != col - 1 && t - k - 1 >= 0 && j + k + 1 < col)
		    pst->R[t][j] -= pst->R[t - k - 1][k + 1]
			* pst->R[t - k - 1][j + k + 1];
	    pst->R[t][j] /= pst->R[t][0];
	}
    }
   
    return;
}

// Choleski_forward: forward substitution to solve linear equations
void Choleski_forwardFC(PStreamChol *pst)
{
    register int t, j, row, col;
    double hold;
   
    row = pst->T * (pst->order + 1);
    col = pst->width * (pst->order + 1);

    pst->g[0] = pst->r[0] / pst->R[0][0];

    for (t = 1; t < row; t++) {
	hold = 0.0;
	for (j = 1; j < col; j++)
	    if (t - j >= 0 && pst->R[t - j][j] != 0.0)
		hold += pst->R[t - j][j] * pst->g[t - j];
	pst->g[t] = (pst->r[t] - hold) / pst->R[t][0];
    }
   
    return;
}

// Choleski_backward: backward substitution to solve linear equations
void Choleski_backwardFC(PStreamChol *pst)
{
    register int t, tt, j, m, row, col, k, l;
    double hold;
   
    row = pst->T * (pst->order + 1);
    col = pst->width * (pst->order + 1);

    pst->c[pst->T - 1][pst->order] = pst->g[row - 1] / pst->R[row - 1][0];

    for (t = row - 2, tt = pst->T - 1, m = pst->order - 1; t >= 0; t--, m--) {
	if (m < 0) {
	    tt--;	m = pst->order;
	}
	hold = 0.0;
	for (j = 1; j < col; j++) {
	    if (t + j < row && pst->R[t][j] != 0.0) {
		k = (t + j) / (pst->order + 1);
		l = t + j - k * (pst->order + 1);
		hold += pst->R[t][j] * pst->c[k][l];
	    }
	}
	pst->c[tt][m] = (pst->g[t] - hold) / pst->R[t][0];
    }
    if (tt != 0 && m != -1) {
	fprintf(stderr, "Error: Choleski_backwardFC\n");
	exit(1);
    }

    return;
}

////////////////////////////////////
// ML Considering Global Variance //
////////////////////////////////////
void varconv(double **c, const int m, const int T, const double var)
{
    register int n;
    double sd, osd;
    double oav = 0.0, ovar = 0.0, odif = 0.0;

    calc_varstats(c, m, T, &oav, &ovar, &odif);
    osd = sqrt(ovar);	sd = sqrt(var);
    for (n = 0; n < T; n++)
	c[n][m] = (c[n][m] - oav) / osd * sd + oav;

    return;
}

void calc_varstats(double **c, const int m, const int T,
		   double *av, double *var, double *dif)
{
    register int i;
    register double d;

    *av = 0.0;
    *var = 0.0;
    *dif = 0.0;
    for (i = 0; i < T; i++) *av += c[i][m];
    *av /= (double)T;
    for (i = 0; i < T; i++) {
	d = c[i][m] - *av;
	*var += d * d;	*dif += d;
    }
    *var /= (double)T;

    return;
}

// Diagonal Covariance Version
void mlgparaGrad(DMATRIX pdf, PStreamChol *pst, DMATRIX mlgp, const int max,
		 double th, double e, double alpha, DVECTOR vm, DVECTOR vv,
		 XBOOL nrmflag, XBOOL extvflag)
{
    int t, d;

    // error check
    if (pst->vSize * 2 != pdf->col || pst->order + 1 != mlgp->col) {
	fprintf(stderr, "Error mlgparaChol: Different dimension\n");
	exit(1);
    }

    // mseq: U^{-1}*M,	ifvseq: U^{-1}
    for (t = 0; t < pst->T; t++) {
	for (d = 0; d < pst->vSize; d++) {
	    pst->mseq[t][d] = pdf->data[t][d];
	    pst->ivseq[t][d] = pdf->data[t][pst->vSize + d];
	}
    } 

    // ML parameter generation
    mlpgChol(pst);

    // extend variance
    if (extvflag == XTRUE)
	for (d = 0; d <= pst->order; d++)
	    varconv(pst->c, d, pst->T, vm->data[d]);

    // estimating parameters
    mlpgGrad(pst, max, th, e, alpha, vm, vv, nrmflag);

    // extracting parameters
    for (t = 0; t < pst->T; t++)
	for (d = 0; d <= pst->order; d++)
	    mlgp->data[t][d] = pst->c[t][d];

    return;
}

// generate parameter sequence from pdf sequence using gradient
void mlpgGrad(PStreamChol *pst, const int max, double th, double e,
	      double alpha, DVECTOR vm, DVECTOR vv, XBOOL nrmflag)
{
   register int m, i, t;
   double diff, n, dth;

   if (nrmflag == XTRUE)
       n = (double)(pst->T * pst->vSize) / (double)(vm->length);
   else n = 1.0;

   // generating parameter in each dimension
   for (m = 0; m <= pst->order; m++) {
       calc_R_and_r(pst, m);
       dth = th * sqrt(vm->data[m]);
       for (i = 0; i < max; i++) {
	   calc_grad(pst, m);
	   if (vm != NODATA && vv != NODATA)
	       calc_vargrad(pst, m, alpha, n, vm->data[m], vv->data[m]);
	   for (t = 0, diff = 0.0; t < pst->T; t++) {
	       diff += pst->g[t] * pst->g[t];
	       pst->c[t][m] += e * pst->g[t];
	   }
	   diff = sqrt(diff / (double)pst->T);
	   if (diff < dth || diff == 0.0) break;
       }
   }
   
   return;
}

// calc_grad: calculate -RX + r = -W'U^{-1}W * X + W'U^{-1}M
void calc_grad(PStreamChol *pst, const int m)
{
    register int i, j;

    for (i = 0; i < pst->T; i++) {
	pst->g[i] = pst->r[i] - pst->c[i][m] * pst->R[i][0];
	for (j = 1; j < pst->width; j++) {
	    if (i + j < pst->T) pst->g[i] -= pst->c[i + j][m] * pst->R[i][j];
	    if (i - j >= 0) pst->g[i] -= pst->c[i - j][m] * pst->R[i - j][j];
	}
    }

    return;
}

void calc_vargrad(PStreamChol *pst, const int m, double alpha, double n,
		  double vm, double vv)
{
    register int i;
    double vg, w1, w2;
    double av = 0.0, var = 0.0, dif = 0.0;

    if (alpha > 1.0 || alpha < 0.0) {
	w1 = 1.0;		w2 = 1.0;
    } else {
	w1 = alpha;	w2 = 1.0 - alpha;
    }

    calc_varstats(pst->c, m, pst->T, &av, &var, &dif);

    for (i = 0; i < pst->T; i++) {
	vg = -(var - vm) * (pst->c[i][m] - av) * vv * 2.0 / (double)pst->T;
	pst->g[i] = w1 * pst->g[i] / n + w2 * vg;
    }

    return;
}
