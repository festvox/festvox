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
/*  Subroutine for Speech Analysis                                   */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/basic.h"
#include "../include/fft.h"
#include "../include/fileio.h"
#include "../include/memory.h"
#include "../include/option.h"
#include "../include/voperate.h"
#include "../include/window.h"

#include "../sub/sptk_sub.h"
#include "../sub/anasyn_sub.h"
#include "analysis_sub.h"

//
//	speech analysis function
//
DMATRIX analysis_body(DVECTOR x,	// input waveform
		      double fs,	// sampling frequency (Hz)
		      double framem,	// frame length (ms)
		      double shiftm,	// frame shift (ms)
		      long fftl)	// FFT length
{
    long ii;		// counter
    long hfftl;		// half of fft length
    long framel;	// frame length point
    double shiftl;	// shift length
    double iist;	// counter
    long nframe;	// the number of frame
    double rmsp;
    double ave;
    DVECTOR xh = NODATA;
    DVECTOR rv = NODATA;
    DVECTOR xt = NODATA;	// temporary speech
    DVECTOR cx = NODATA;	// cut speech
    DVECTOR win = NODATA;	// window for cutting wave
    DVECTOR spc = NODATA;	// fft spectrum
    DMATRIX sgram = NODATA;	// smoothed spectrogram

    // initialize global parameter
    framel = (long)round(framem * fs / 1000.0);
    fftl = POW2(nextpow2(MAX(fftl, framel)));
    hfftl = fftl / 2 + 1;
    shiftl = shiftm * fs / 1000.0;

    // filtering (highpass filter 70 [Hz] < freq)
    xh = xdvclone(x);
    cleaninglownoise(xh, fs, 70.0);

    ave = dvsum(xh) / (double)(xh->length);
    for (ii = 0, rmsp = 0.0; ii < xh->length; ii++) {
	rmsp += pow(xh->data[ii] - ave, 2.0);
    }
    rmsp = sqrt(rmsp / (double)(xh->length - 1));

    // convert signal for processing
    rv = xdvrandn(framel / 2);	dvscoper(rv, "*", rmsp / 4000.0);
    xt = xdvalloc(xh->length + framel / 2 + framel);
    dvpaste(xt, rv, 0, rv->length, 0);
    dvpaste(xt, xh, framel / 2, xh->length, 0);
    xdvfree(rv);
    rv = xdvrandn(framel);	dvscoper(rv, "*", rmsp / 4000.0);
    dvpaste(xt, rv, xh->length + framel / 2, rv->length, 0);

    // memory free
    xdvfree(xh);
    xdvfree(rv);

    // get number of frame
    nframe = (long)round((double)x->length / shiftl);

    // memory allocation
    sgram = xdmzeros(nframe, hfftl);
    
    // Blackman window
    win = xdvnblackman(framel);

    // loop of smoothing in frequency domain
    for (ii = 0, iist = 0.0; ii < nframe; ii++) {
	// cut data of xt
	cx = ana_xcutsig(xt, (long)round(iist), framel);

	// check zero signal
	ana_zerocheck(cx, rmsp / 4000.0);

	// calculate fft spectrum
	spc = ana_xgetfftpow(cx, win, fftl, 1.0);

	// copy amplitude spectrum to ii-th row of matrix
	dmcopyrow(sgram, ii, spc);

	// memory free
	xdvfree(cx);
	xdvfree(spc);

	iist += shiftl;
    }

    // memory free
    xdvfree(xt);
    xdvfree(win);

    return sgram;
}


DVECTOR ana_xcutsig(DVECTOR sig, long offset, long length)
{
    DVECTOR cx = NODATA;

    // cut signal
    cx = xdvcut(sig, offset, length);
    dvscoper(cx, "-", dvmean(cx));

    return cx;
}

void ana_zerocheck(DVECTOR sig, double ncoef)
{
    long k;
    DVECTOR rv = NODATA;

    for (k = 0; k < sig->length; k++)
	if (sig->data[k] != 0.0) return;

    rv = xdvrandn(sig->length);	dvscoper(rv, "*", ncoef);
    dvpaste(sig, rv, 0, rv->length, 0);
    xdvfree(rv);

    return;
}

DVECTOR ana_xgetfftpow(DVECTOR cx, DVECTOR wxe, long fftl, double pc)
{
    DVECTOR cx2 = NODATA;
    DVECTOR pw = NODATA;			// fft power spectrum

    // get windowed data of tx
    cx2 = xdvclone(cx);
    dvoper(cx2, "*", wxe);

    // get fft amplitude spectrum
    pw = xdvfftabs(cx2, fftl);

    // pw = pw ^ pc
    dvscoper(pw, "^", pc);

    // memory free
    xdvfree(cx2);

    return pw;
}

double spvec2pow(DVECTOR vec, XBOOL db_flag)
{
    long k, fftl2, fftl;
    double pow;

    fftl2 = vec->length - 1;
    fftl = fftl2 * 2;
    for (k = 1, pow = SQUARE(vec->data[0]); k < fftl2; k++)
	pow += 2.0 * SQUARE(vec->data[k]);
    pow += SQUARE(vec->data[k]);
    pow /= (double)fftl;

    if (db_flag == XTRUE) pow = 10.0 * log10(pow);

    return pow;
}

DVECTOR xspg2pow_norm(DMATRIX spg)
{
    long k;
    double sum;
    DVECTOR sp = NODATA;
    DVECTOR pv = NODATA;

    // memory allocation
    pv = xdvalloc(spg->row);

    for (k = 0, sum = 0.0; k < spg->row; k++) {
	sp = xdmextractrow(spg, k);
	pv->data[k] = spvec2pow(sp, XFALSE);
	sum += pv->data[k];
	xdvfree(sp);
    }
    sum /= (double)k;

    for (k = 0; k < pv->length; k++)
	pv->data[k] = 10.0 * log10(pv->data[k] / sum);

    return pv;
}

DVECTOR xget_spg2powvec(DMATRIX n2sgram, XBOOL log_flag)
{
    long k, l;
    long fftl, fftl2;
    DVECTOR powv = NODATA;

    powv = xdvalloc(n2sgram->row);
    fftl2 = n2sgram->col - 1;
    fftl = fftl2 * 2;

    for (k = 0; k < n2sgram->row; k++) {
	powv->data[k] = SQUARE(n2sgram->data[k][0]);
	for (l = 1; l < fftl2; l++)
	    powv->data[k] += 2.0 * SQUARE(n2sgram->data[k][l]);
	powv->data[k] += SQUARE(n2sgram->data[k][l]);
	powv->data[k] /= (double)fftl;
	if (log_flag == XTRUE) {
	    if (powv->data[k] != 0.0) {
		powv->data[k] = log10(powv->data[k]);
	    } else {
		powv->data[k] = -5.0;
	    }
	}
    }

    return powv;
}

DMATRIX xspg2mpmcepg(DMATRIX spg, long order, long fftl, XBOOL power_flag,
		     XBOOL fast_flag)
{
    long k, l;
    DVECTOR vec = NODATA;
    DVECTOR spw = NODATA;
    DMATRIX mcepg = NODATA;

    // memory allocation
    if (power_flag == XTRUE) {
	mcepg = xdmalloc(spg->row, order + 1);
    } else {
	mcepg = xdmalloc(spg->row, order);
    }
    spw = xdvzeros(fftl);

    for (k = 0; k < spg->row; k++) {
	vec = xdmextractrow(spg, k);
	dvcopy(spw, vec);
	dvfftturn(spw);
	if (fast_flag == XTRUE) {
	    mcep(spw->data, fftl, vec->data, order, ALPHA, 1, 1, 0.001, 0.0);
	} else {
	    mcep(spw->data, fftl, vec->data, order, ALPHA, 2, 30, 0.001, 0.0);
	}
	if (power_flag == XTRUE) {
	    for (l = 0; l <= order; l++)
		mcepg->data[k][l] = vec->data[l];
	} else {
	    for (l = 0; l < order; l++)
		mcepg->data[k][l] = vec->data[l + 1];
	}
	xdvfree(vec);
    }
    xdvfree(spw);

    return mcepg;
}
