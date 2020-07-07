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
#include <string.h>
/*#include <unistd.h>*/
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include "../include/defs.h"
#include "../include/memory.h"
#include "../include/basic.h"
#include "../include/voperate.h"
#include "../include/fft.h"
#include "../include/kaiser.h"
#include "../include/filter.h"

void dvangle(DVECTOR x)
{
    long k;

    if (x->imag == NULL) {
	dvzeros(x, x->length);
	return;
    }

    for (k = 0; k < x->length; k++) {
	if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
	    x->data[k] = 0.0;
	} else {
	    x->data[k] = atan2(x->imag[k], x->data[k]);
	}
    }
    dvreal(x);

    return;
}

DVECTOR xdvangle(DVECTOR x)
{
    long k;
    DVECTOR phs;

    if (x->imag == NULL) {
	phs = xdvzeros(x->length);
	return phs;
    } else {
	phs = xdvalloc(x->length);
    }

    for (k = 0; k < phs->length; k++) {
	if (x->data[k] == 0.0 && x->imag[k] == 0.0) {
	    phs->data[k] = 0.0;
	} else {
	    phs->data[k] = atan2(x->imag[k], x->data[k]);
	}
    }

    return phs;
}

void dvunwrap(DVECTOR phs, double cutoff)
{
    long k;
    double a;
    DVECTOR f;
    
    if (cutoff <= 0.0) {
	cutoff = PI;
    }

    a = dvmin(phs, NULL);
    for (k = 0; k < phs->length; k++) {
	phs->data[k] = rem(phs->data[k] - a, 2.0 * PI) + a;
    }

    f = xdvzeros(phs->length);
    for (k = 1; k < f->length; k++) {
	a = phs->data[k] - phs->data[k - 1];
	if (a > cutoff) {
	    f->data[k] = -2.0 * PI;
	} else if (a < -cutoff) {
	    f->data[k] = 2.0 * PI;
	}
    }

    dvcumsum(f);
    dvoper(phs, "+", f);

    xdvfree(f);

    return;
}

double sinc(double x, double c)
{
    double a;

    if (x == 0.0) {
	a = c;
    } else {
	a = (c * sin(x)) / x;
    }
    return a;
}

DVECTOR xdvlowpass(double cutoff, double sidelobe, double trans, double gain)
{
    long k;
    long half, length;
    double beta;
    double value;
    DVECTOR filter;

#if 0
    /* 0 <= cufoff <= 1 */
    cutoff = MIN(cutoff, 1.0);
    cutoff = MAX(cutoff, 0.0);
#endif

    /* get kaiser parameter */
    getkaiserparam(sidelobe, trans, &beta, &length);

    /* make sure length is odd */
    length = (long)(length / 2) * 2 + 1;
    half = (length - 1) / 2;

#ifdef DEBUG
    fprintf(stderr, "lowpass: cutoff = %f, gain = %f, beta = %f, length = %ld\n",
	    cutoff, gain, beta, length);
#endif

    /* memory allocate */
    filter = xdvalloc(length);

    /* get kaiser window */
    kaiser(filter->data, filter->length, beta);

    /* calculate lowpass filter */
    for (k = 0; k < length; k++) {
	value = sinc(PI * cutoff * (double)(k - half), gain * cutoff);
	filter->data[k] *= value;
    }

    return filter;
}

#define getnumloop(x,y) MAX((int)(((x) + (y) - 1) / (y)), 1)

DVECTOR xdvfftfiltm(DVECTOR b, DVECTOR x, long fftp)
{
    int fftp_p;
    long i, nloop;
    long k;
    long pos = 0;
    long length, block;
    double real, imag;
    double *hr, *hi;
    double *xr, *xi;
    DVECTOR xo;

    /* get fft point */
    if (x->length < b->length) {
	length = b->length + x->length - 1;
    } else {
	length = b->length;
    }
    fftp_p = nextpow2(MAX(fftp, 2 * length));
    fftp = POW2(fftp_p);
	
    /* initialize */
    length = b->length / 2 - 1;
    block = fftp - b->length + 1;
    nloop = getnumloop(x->length, block);

    /* memory allocate */
    hr = xalloc(fftp, double);
    hi = xalloc(fftp, double);
    xr = xalloc(fftp, double);
    xi = xalloc(fftp, double);
    if (x->imag != NULL || b->imag != NULL) {
	xo = xdvrizeros(x->length);
    } else {
	xo = xdvzeros(x->length);
    }

    /* fft of the filter */
    for (k = 0; k < fftp; k++) {
	if (k < b->length) {
	    hr[k] = b->data[k];
	    if (b->imag != NULL) {
		hi[k] = b->imag[k];
	    } else {
		hi[k] = 0.0;
	    }
	} else {
	    hr[k] = 0.0;
	    hi[k] = 0.0;
	}
    }
    fft(hr, hi, fftp, 0);

    /* loop for every block of data */
    for (i = 0; i < nloop; i++) {
	/* fft of input speech data */
	for (k = 0; k < fftp; k++) {
	    if (k < block && (pos = i * block + k) < x->length) {
		xr[k] = x->data[pos];
		if (x->imag != NULL) {
		    xi[k] = x->imag[pos];
		} else {
		    xi[k] = 0.0;
		}
	    } else {
		xr[k] = 0.0;
		xi[k] = 0.0;
	    }
	}
	fft(xr, xi, fftp, 0);

	/* multiplication in frequency domain */
	for (k = 0; k < fftp; k++) {  
	    real = xr[k] * hr[k] - xi[k] * hi[k];
	    imag = xr[k] * hi[k] + xi[k] * hr[k];
	    xr[k] = real;
	    xi[k] = imag;
	}

	/* ifft */
	fft(xr, xi, fftp, 1);

	/* overlap adding */
	for (k = 0; k < fftp && (pos = i * block + k) < xo->length; k++) {
	    if (pos >= 0) {
		xo->data[pos] += xr[k];
		if (xo->imag != NULL) {
		    xo->imag[pos] += xi[k];
		}
	    }
	}
    }

    /* memory free */
    free(hr);
    free(hi);
    free(xr);
    free(xi);

    return xo;
}
#if 0
#endif

DVECTOR xdvfftfilt(DVECTOR b, DVECTOR x, long fftp)
{
    int fftp_p;
    long i, nloop;
    long pos;
    long length, block;
    DVECTOR sb;
    DVECTOR cx;
    DVECTOR sx;
    DVECTOR xo;

    /* get fft point */
    if (x->length < b->length) {
	length = b->length + x->length - 1;
    } else {
	length = b->length;
    }
    fftp_p = nextpow2(MAX(fftp, 2 * length));
    fftp = POW2(fftp_p);
	
    /* initialize */
    length = b->length / 2 - 1;
    block = fftp - b->length + 1;
    nloop = getnumloop(x->length, block);

    /* memory allocate */
    if (x->imag != NULL || b->imag != NULL) {
	xo = xdvrizeros(x->length + b->length - 1);
    } else {
	xo = xdvzeros(x->length + b->length - 1);
    }

    /* fft of the filter */
    sb = xdvfft(b, fftp);

    /* loop for every block of data */
    for (i = 0; i < nloop; i++) {
	pos = i * block;

	/* fft of input speech data */
	cx = xdvcut(x, pos, block);
	sx = xdvfft(cx, fftp);

	/* multiplication in frequency domain */
	dvoper(sx, "*", sb);

	/* ifft */
	dvifft(sx);

	/* overlap adding */
	dvpaste(xo, sx, pos, sx->length, 1);

	/* memory free */
	xdvfree(cx);
	xdvfree(sx);
    }

    /* memory free */
    xdvfree(sb);

    return xo;
}

DVECTOR xdvfftfiltm2(DVECTOR b, DVECTOR x, long fftp)
{
    int fftp_p;
    long i, nloop;
    long pos;
    long length, block;
    DVECTOR sb;
    DVECTOR sx;
    DVECTOR xo;

    /* get fft point */
    if (x->length < b->length) {
	length = b->length + x->length - 1;
    } else {
	length = b->length;
    }
    fftp_p = nextpow2(MAX(fftp, 2 * length));
    fftp = POW2(fftp_p);
	
    /* initialize */
    length = b->length / 2 - 1;
    block = fftp - b->length + 1;
    nloop = getnumloop(x->length, block);

    /* memory allocate */
    if (x->imag != NULL || b->imag != NULL) {
	xo = xdvrizeros(x->length);
    } else {
	xo = xdvzeros(x->length);
    }

    /* fft of the filter */
    sb = xdvfft(b, fftp);

    /* loop for every block of data */
    for (i = 0; i < nloop; i++) {
	pos = i * block;

	/* fft of input speech data */
	sx = xdvrizeros(fftp);
	dvpaste(sx, x, pos, block, 0);
	dvfft(sx);

	/* multiplication in frequency domain */
	dvoper(sx, "*", sb);

	/* ifft */
	dvifft(sx);

	/* overlap adding */
	dvpaste(xo, sx, pos, sx->length, 1);

	/* memory free */
	xdvfree(sx);
    }

    /* memory free */
    xdvfree(sb);

    return xo;
}

DVECTOR xdvconv(DVECTOR a, DVECTOR b)
{
    long i, j, pos;
    double ar, ai;
    double br, bi;
    double sum, sumi;
    DVECTOR c;

    if (a->imag != NULL || b->imag != NULL) {
	c = xdvrizeros(a->length + b->length - 1);
    } else {
	c = xdvzeros(a->length + b->length - 1);
    }

    for (i = 0; i < c->length; i++) {
	if (c->imag != NULL) {
	    for (j = 0, sum = sumi = 0.0; j < a->length; j++) {
		pos = i - j;
		if (pos >= 0 && pos < b->length) {
		    ar = a->data[j];
		    br = b->data[pos];
		    if (a->imag == NULL) {
			ai = 0.0;
		    } else {
			ai = a->imag[j];
		    }
		    if (b->imag == NULL) {
			bi = 0.0;
		    } else {
			bi = b->imag[pos];
		    }
		    sum += ar * br - ai * bi;
		    sumi += ar * bi + ai * br;
		}
	    }
	    c->data[i] = sum;
	    c->imag[i] = sumi;
	} else {
	    for (j = 0, sum = 0.0; j < a->length; j++) {
		pos = i - j;
		if (pos >= 0 && pos < b->length) {
		    sum += a->data[j] * b->data[pos];
		}
	    }
	    c->data[i] = sum;
	}
    }

    return c;
}

DVECTOR xdvfilter(DVECTOR b, DVECTOR a, DVECTOR x)
{
    long i, j, pos;
    double ar, ai;
    double br, bi;
    double xr, xi;
    double arsum, aisum;
    double brsum, bisum;
    double value;
    DVECTOR xo;

    if (a->imag != NULL || b->imag != NULL || x->imag != NULL) {
	xo = xdvrizeros(x->length);
    } else {
	xo = xdvzeros(x->length);
    }

    for (i = 0; i < xo->length; i++) {
	if (xo->imag != NULL) {
	    for (j = 1, arsum = aisum = 0.0; j < a->length; j++) {
		pos = i - j;
		if (pos >= 0 && pos < xo->length) {
		    ar = a->data[j];
		    xr = xo->data[pos];
		    if (a->imag == NULL) {
			ai = 0.0;
		    } else {
			ai = a->imag[j];
		    }
		    if (xo->imag == NULL) {
			xi = 0.0;
		    } else {
			xi = xo->imag[pos];
		    }
		    arsum += ar * xr - ai * xi;
		    aisum += ar * xi + ai * xr;
		}
	    }

	    for (j = 0, brsum = bisum = 0.0; j < b->length; j++) {
		pos = i - j;
		if (pos >= 0 && pos < x->length) {
		    br = b->data[j];
		    xr = x->data[pos];
		    if (b->imag == NULL) {
			bi = 0.0;
		    } else {
			bi = b->imag[j];
		    }
		    if (x->imag == NULL) {
			xi = 0.0;
		    } else {
			xi = x->imag[pos];
		    }
		    brsum += br * xr - bi * xi;
		    bisum += br * xi + bi * xr;
		}
	    }

	    if (a->imag == NULL || a->imag[0] == 0.0) {
		xo->data[i] = (brsum - arsum) / a->data[0];
		xo->imag[i] = (bisum - aisum) / a->data[0];
	    } else {
		value = CSQUARE(a->data[0], a->imag[0]);
		xr = brsum - arsum;
		xi = bisum - aisum;
		xo->data[i] = (xr * a->data[0] + xi * a->imag[0]) / value;
		xo->imag[i] = (-xr * a->imag[0] + xi * a->data[0]) / value;
	    }
	} else {
	    for (j = 1, arsum = 0.0; j < a->length; j++) {
		pos = i - j;
		if (pos >= 0 && pos < xo->length) {
		    arsum += a->data[j] * xo->data[pos];
		}
	    }

	    for (j = 0, brsum = 0.0; j < b->length; j++) {
		pos = i - j;
		if (pos >= 0 && pos < x->length) {
		    brsum += b->data[j] * x->data[pos];
		}
	    }

	    xo->data[i] = (brsum - arsum) / a->data[0];
	}
    }

    return xo;
}
