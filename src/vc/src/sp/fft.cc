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
#include "../include/fft.h"
#include "../include/vector.h"
#include "../include/voperate.h"
#include "../include/complex.h"
#include "../include/filter.h"

/*
 *	fft function for complex unit
 */
DCOMPLEX xvfft(DVECTOR real, DVECTOR imag, long length, int inv)
{
    DCOMPLEX cplx;

    cplx = xdccreate(real, imag, length);
    fft(cplx->real->data, cplx->imag->data, cplx->length, inv);

    return cplx;
}

void cfft(DCOMPLEX cplx, int inv)
{
    fft(cplx->real->data, cplx->imag->data, cplx->length, inv);

    return;
}

void cfftturn(DCOMPLEX cplx)
{
    fftturn(cplx->real->data, cplx->imag->data, cplx->length);

    return;
}

void cfftshift(DCOMPLEX cplx)
{
    fftshift(cplx->real->data, cplx->imag->data, cplx->length);

    return;
}

/*
 *	convolution using fft
 */
FVECTOR xfvfftconv(FVECTOR a, FVECTOR b, long fftl)
{
    FVECTOR x;
    FVECTOR ca, cb;

    /* fft of a */
    ca = xfvfft(a, fftl);
    
    /* fft of b */
    cb = xfvfft(b, fftl);

    /* convolution */
    fvoper(ca, "*", cb);

    /* ifft */
    x = xfvifft(ca, fftl);
    fvreal(x);

    /* memory free */
    xfvfree(ca);
    xfvfree(cb);

    return x;
}

DVECTOR xdvfftconv(DVECTOR a, DVECTOR b, long fftl)
{
    DVECTOR x;
    DVECTOR ca, cb;

    /* fft of a */
    ca = xdvfft(a, fftl);
    
    /* fft of b */
    cb = xdvfft(b, fftl);

    /* convolution */
    dvoper(ca, "*", cb);

    /* ifft */
    x = xdvifft(ca, fftl);
    dvreal(x);

    /* memory free */
    xdvfree(ca);
    xdvfree(cb);

    return x;
}

FVECTOR xfvfftpower(FVECTOR x, long fftl)
{
    FVECTOR y;

    /* fft */
    y = xfvfft(x, fftl);

    /* square of complex */
    fvsquare(y);

    return y;
}

DVECTOR xdvfftpower(DVECTOR x, long fftl)
{
    DVECTOR y;

    /* fft */
    y = xdvfft(x, fftl);

    /* square of complex */
    dvsquare(y);

    return y;
}

FVECTOR xfvfftabs(FVECTOR x, long fftl)
{
    FVECTOR y;

    /* fft */
    y = xfvfft(x, fftl);

    /* abs of complex */
    fvabs(y);

    return y;
}

DVECTOR xdvfftabs(DVECTOR x, long fftl)
{
    DVECTOR y;

    /* fft */
    y = xdvfft(x, fftl);

    /* abs of complex */
    dvabs(y);

    return y;
}

DVECTOR xdvfftangle(DVECTOR x, long fftl)
{
    DVECTOR y;

    /* fft */
    y = xdvfft(x, fftl);

    /* phase angle */
    dvangle(y);

    return y;
}

DVECTOR xdvfftgrpdly(DVECTOR x, long fftl)
{
    long k;
    double value;
    DVECTOR fx;
    DVECTOR dfx;
    DVECTOR gd;

    /* fft of input signal */
    fx = xdvfft(x, fftl);

    /* calculate frequency diff spectrum */
    dfx = xdvrizeros(fftl);
    for (k = 0; k < x->length; k++) {
	dfx->imag[k] = -(double)k * x->data[k];
    }
    dvfft(dfx);

    /* calculate group delay */
    gd = xdvalloc(fftl);
    for (k = 0; k < fftl; k++) {
	value = SQUARE(fx->data[k]) + SQUARE(fx->imag[k]);
	if (value == 0.0) {
	    gd->data[k] = 0.0;
	} else {
	    gd->data[k] = -(dfx->imag[k] * fx->data[k] -
			    fx->imag[k] * dfx->data[k]) / value;
	}
    }

    /* memory free */
    xdvfree(fx);
    xdvfree(dfx);

    return gd;
}

void dvspectocep(DVECTOR x)
{
    long k;

    dvabs(x);
    for (k = 0; k < x->length; k++) {
	if (x->data[k] > 0.0) {
	    x->data[k] = log(x->data[k]);
	} else {
	    x->data[k] = log(ALITTLE_NUMBER);
	}
    }

    dvifft(x);
    dvreal(x);

    return;
}

void dvceptospec(DVECTOR x)
{
    /* convert cepstrum to spectrum */
    dvfft(x);
    dvexp(x);

    return;
}
#if 0
#endif

DVECTOR xdvrceps(DVECTOR x, long fftl)
{
    DVECTOR cep;

    cep = xdvfftabs(x, fftl);
    dvspectocep(cep);

    return cep;
}

void dvlif(DVECTOR cep, long fftl, long lif)
{
    long k;
    long lif2;

    if (lif >= 0) {
	lif2 = fftl - lif;
	for (k = 0; k < cep->length; k++) {
	    if (k > lif && k < lif2) {
		cep->data[k] = 0.0;
	    }
	}
    } else {
	lif *= -1;
	lif2 = fftl - lif;
	for (k = 0; k < cep->length; k++) {
	    if (k <= lif || k >= lif2) {
		cep->data[k] = 0.0;
	    }
	}
    }

    return;
}

void dvceptompc(DVECTOR cep)
{
    long k;
    long fftl2;

    fftl2 = cep->length / 2;
    
    for (k = 0; k < cep->length; k++) {
	if (k == 0) {
	    cep->data[k] = cep->data[k];
	} else if (k < fftl2) {
	    cep->data[k] = 2.0 * cep->data[k];
	} else {
	    cep->data[k] = 0.0;
	}
    }

    return;
}

DVECTOR xdvmpceps(DVECTOR x, long fftl)
{
    DVECTOR cep;

    cep = xdvrceps(x, fftl);
    dvceptompc(cep);

    return cep;
}

DVECTOR xdvcspec(DVECTOR mag, DVECTOR phs)
{
    DVECTOR spc;
    DVECTOR phe;

    if (mag == NODATA && phs == NODATA) {
	return NODATA;
    } else if (phs == NODATA) {
	phe = xdvabs(mag);
    } else {
	/* exponential of phase */
	phe = xdvcplx(NODATA, phs);
	dvexp(phe);

	if (mag != NODATA) {
	    /* multiply phase */
	    spc = xdvabs(mag);
	    dvoper(phe, "*", spc);

	    /* memory free */
	    xdvfree(spc);
	}
    }

    return phe;
}

void fvfftshift(FVECTOR x)
{
    fftshiftf(x->data, x->imag, x->length);

    return;
}

void dvfftshift(DVECTOR x)
{
    fftshift(x->data, x->imag, x->length);

    return;
}

void fvfftturn(FVECTOR x)
{
    fftturnf(x->data, x->imag, x->length);

    return;
}

void dvfftturn(DVECTOR x)
{
    fftturn(x->data, x->imag, x->length);

    return;
}

void fvfft(FVECTOR x)
{
    if (x->imag == NULL) {
	fvizeros(x, x->length);
    }

    fftf(x->data, x->imag, x->length, 0);

    return;
}

void dvfft(DVECTOR x)
{
    if (x->imag == NULL) {
	dvizeros(x, x->length);
    }

    fft(x->data, x->imag, x->length, 0);

    return;
}

void fvifft(FVECTOR x)
{
    if (x->imag == NULL) {
	fvizeros(x, x->length);
    }

    fftf(x->data, x->imag, x->length, 1);

    return;
}

void dvifft(DVECTOR x)
{
    if (x->imag == NULL) {
	dvizeros(x, x->length);
    }

    fft(x->data, x->imag, x->length, 1);

    return;
}

FVECTOR xfvfft(FVECTOR x, long length)
{
    long fftp;
    FVECTOR y;

    fftp = POW2(nextpow2(MAX(length, x->length)));

    y = xfvrizeros(fftp);
    fvcopy(y, x);

    fftf(y->data, y->imag, fftp, 0);

    return y;
}

DVECTOR xdvfft(DVECTOR x, long length)
{
    long fftp;
    DVECTOR y;

    fftp = POW2(nextpow2(MAX(length, x->length)));

    y = xdvrizeros(fftp);
    dvcopy(y, x);

    fft(y->data, y->imag, fftp, 0);

    return y;
}

FVECTOR xfvifft(FVECTOR x, long length)
{
    long fftp;
    FVECTOR y;

    fftp = POW2(nextpow2(MAX(length, x->length)));

    y = xfvrizeros(fftp);
    fvcopy(y, x);

    fftf(y->data, y->imag, fftp, 1);

    return y;
}

DVECTOR xdvifft(DVECTOR x, long length)
{
    long fftp;
    DVECTOR y;

    fftp = POW2(nextpow2(MAX(length, x->length)));

    y = xdvrizeros(fftp);
    dvcopy(y, x);

    fft(y->data, y->imag, fftp, 1);

    return y;
}

int rfftabs(double *x, long fftp)
{
    long k;
    double *xRe, *xIm;

    xRe = xalloc(fftp, double);
    xIm = xalloc(fftp, double);
    
    for (k = 0; k < fftp; k++) {
	xRe[k] = x[k];
	xIm[k] = 0.0;
    }
	
    fft(xRe, xIm, fftp, 0);
	
    for (k = 0; k < fftp; k++) {
	x[k] = xRe[k] * xRe[k] + xIm[k] * xIm[k];
	x[k] = sqrt(x[k]);
    }

    xfree(xRe);
    xfree(xIm);
    
    return SUCCESS;
}

int rfftpow(double *x, long fftp)
{
    long k;
    double *xRe, *xIm;

    xRe = xalloc(fftp, double);
    xIm = xalloc(fftp, double);
    
    for (k = 0; k < fftp; k++) {
	xRe[k] = x[k];
	xIm[k] = 0.0;
    }
	
    fft(xRe, xIm, fftp, 0);
	
    for (k = 0; k < fftp; k++) {
	x[k] = xRe[k] * xRe[k] + xIm[k] * xIm[k];
    }

    xfree(xRe);
    xfree(xIm);
    
    return SUCCESS;
}

int rfftangle(double *x, long fftp)
{
    long k;
    double *xRe, *xIm;

    xRe = xalloc(fftp, double);
    xIm = xalloc(fftp, double);
    
    for (k = 0; k < fftp; k++) {
	xRe[k] = x[k];
	xIm[k] = 0.0;
    }
	
    fft(xRe, xIm, fftp, 0);
	
    for (k = 0; k < fftp; k++) {
	if (xRe[k] == 0.0 && xIm[k] == 0.0) {
	    x[k] = 0.0;
	} else {
	    x[k] = atan2(xIm[k], xRe[k]);
	}
    }

    xfree(xRe);
    xfree(xIm);
    
    return SUCCESS;
}

/*
 *	next power of 2
 */
int nextpow2(long n)
{
    int p;
    long value;

    for (p = 1;; p++) {
	value = (long)POW2(p);
	if (value >= n) {
	    break;
	}
    }

    return p;
}

/*
 *	fft turn for float data 
 */
void fftturnf(float *xRe, float *xIm, long fftp)
{
    long i;
    long hfftp;

    hfftp = fftp - (fftp / 2);

    if (xRe != NULL) {
	/* real part */
	for (i = 1; i < hfftp; i++) {
	    xRe[fftp - i] = xRe[i];
	}
    }
    if (xIm != NULL) {
	/* imaginary part */
	for (i = 1; i < hfftp; i++) {
	    xIm[fftp - i] = xIm[i];
	}
    }

    return;
} 

/*
 *	fft turn 
 */
void fftturn(double *xRe, double *xIm, long fftp)
{
    long i;
    long hfftp;

    hfftp = fftp - (fftp / 2);

    if (xRe != NULL) {
	/* real part */
	for (i = 1; i < hfftp; i++) {
	    xRe[fftp - i] = xRe[i];
	}
    }
    if (xIm != NULL) {
	/* imaginary part */
	for (i = 1; i < hfftp; i++) {
	    xIm[fftp - i] = -xIm[i];
	}
    }

    return;
} 

/* fft shift for float data */
void fftshiftf(float *xRe, float *xIm, long fftp)
{
    long i;
    long hfftp, hfftp2, hfftpm;
    float value, value2;

    hfftp = fftp / 2;
    hfftp2 = fftp - hfftp;
    hfftpm = hfftp - 1;

    if (xRe != NULL) {
	/* real part */
	value2 = xRe[hfftp];
	xRe[hfftp] = xRe[fftp - 1];	/* if fft point is odd */
	for (i = 0; i < hfftpm; i++) {
	    value = xRe[i];
	    xRe[i] = value2;
	    value2 = xRe[i + hfftp + 1];
	    xRe[i + hfftp2] = value;
	}
	value = xRe[i];
	xRe[i] = value2;
	xRe[i + hfftp2] = value;
    }
    if (xIm != NULL) {
	/* imaginaly part */
	value2 = xIm[hfftp];
	xIm[hfftp] = xIm[fftp - 1];	/* if fft point is odd */
	for (i = 0; i < hfftpm; i++) {
	    value = xIm[i];
	    xIm[i] = value2;
	    value2 = xIm[i + hfftp + 1];
	    xIm[i + hfftp2] = value;
	}
	value = xIm[i];
	xIm[i] = value2;
	xIm[i + hfftp2] = value;
    }

    return;
} 

/*
 *	fft shift 
 */
void fftshift(double *xRe, double *xIm, long fftp)
{
    long i;
    long hfftp, hfftp2, hfftpm;
    double value, value2;

    hfftp = fftp / 2;
    hfftp2 = fftp - hfftp;
    hfftpm = hfftp - 1;

    if (xRe != NULL) {
	/* real part */
	value2 = xRe[hfftp];
	xRe[hfftp] = xRe[fftp - 1];	/* if fft point is odd */
	for (i = 0; i < hfftpm; i++) {
	    value = xRe[i];
	    xRe[i] = value2;
	    value2 = xRe[i + hfftp + 1];
	    xRe[i + hfftp2] = value;
	}
	value = xRe[i];
	xRe[i] = value2;
	xRe[i + hfftp2] = value;
    }
    if (xIm != NULL) {
	/* imaginaly part */
	value2 = xIm[hfftp];
	xIm[hfftp] = xIm[fftp - 1];	/* if fft point is odd */
	for (i = 0; i < hfftpm; i++) {
	    value = xIm[i];
	    xIm[i] = value2;
	    value2 = xIm[i + hfftp + 1];
	    xIm[i + hfftp2] = value;
	}
	value = xIm[i];
	xIm[i] = value2;
	xIm[i + hfftp2] = value;
    }

    return;
} 

/*
 *	calculate FFT for float data 
 */
int fftf(float *xRe, float *xIm, long fftp, int inv)
{
    int p;
    long i, ip, j, k, m, me, me1, n, nv2;
    float uRe, uIm, vRe, vIm, wRe, wIm, tRe, tIm;

    p = nextpow2(fftp);
    n = (long)POW2(p);
    if (n != fftp) {
	fprintf(stderr, "fft error: fft point must be a power of 2\n");
	return FAILURE;
    }
    nv2 = n / 2;

    if (inv) {
	for (i = 0; i < n; i++) {
	    xIm[i] = -xIm[i];
	}
    }

    /* bit reversion */
    for (i = 0, j = 0; i < n - 1; i++) {
	if (j > i) {
	    tRe = xRe[j];	tIm = xIm[j];
	    xRe[j] = xRe[i];	xIm[j] = xIm[i];
	    xRe[i] = tRe;	xIm[i] = tIm;
	}
	k = nv2;
	while (j >= k) {
	    j -= k;
	    k /= 2;
	}
	j += k;
    }

    /* butterfly numeration */
    for (m = 1; m <= p; m++) {
	me = (long)POW2(m);		me1 = me / 2;
	uRe = 1.0;		uIm = 0.0;
	wRe = (float)cos(PI / (double)me1);
	wIm = (float)(-sin(PI / (double)me1));
	for (j = 0; j < me1; j++) {
	    for (i = j; i < n; i += me) {
		ip = i + me1;
		tRe = xRe[ip] * uRe - xIm[ip] * uIm;
		tIm = xRe[ip] * uIm + xIm[ip] * uRe;
		xRe[ip] = xRe[i] - tRe;
		xIm[ip] = xIm[i] - tIm;
		xRe[i] += tRe;
		xIm[i] += tIm;
	    }
	    vRe = uRe * wRe - uIm * wIm;
	    vIm = uRe * wIm + uIm * wRe;
	    uRe = vRe;
	    uIm = vIm;
	}
    }

    if (inv) {
	for (i = 0; i < n; i++) {
	    xRe[i] /= (float)n;
	    xIm[i] /= (float)(-n);

	}
    }

    return SUCCESS;
}

/*
 *	calculate FFT 
 */
int fft(double *xRe, double *xIm, long fftp, int inv)
{
    int p;
    long i, ip, j, k, m, me, me1, n, nv2;
    double uRe, uIm, vRe, vIm, wRe, wIm, tRe, tIm;

    p = nextpow2(fftp);
    n = (long)POW2(p);
    if (n != fftp) {
	fprintf(stderr, "fft error: fft point must be a power of 2\n");
	return FAILURE;
    }
    nv2 = n / 2;

    if (inv) {
	for (i = 0; i < n; i++) {
	    xIm[i] = -xIm[i];
	}
    }

    /* bit reversion */
    for (i = 0, j = 0; i < n - 1; i++) {
	if (j > i) {
	    tRe = xRe[j];	tIm = xIm[j];
	    xRe[j] = xRe[i];	xIm[j] = xIm[i];
	    xRe[i] = tRe;	xIm[i] = tIm;
	}
	k = nv2;
	while (j >= k) {
	    j -= k;
	    k /= 2;
	}
	j += k;
    }

    /* butterfly numeration */
    for (m = 1; m <= p; m++) {
	me = (long)POW2(m);	me1 = me / 2;
	uRe = 1.0;		uIm = 0.0;
	wRe = cos(PI / (double)me1);
	wIm = (-sin(PI / (double)me1));
	for (j = 0; j < me1; j++) {
	    for (i = j; i < n; i += me) {
		ip = i + me1;
		tRe = xRe[ip] * uRe - xIm[ip] * uIm;
		tIm = xRe[ip] * uIm + xIm[ip] * uRe;
		xRe[ip] = xRe[i] - tRe;
		xIm[ip] = xIm[i] - tIm;
		xRe[i] += tRe;
		xIm[i] += tIm;
	    }
	    vRe = uRe * wRe - uIm * wIm;
	    vIm = uRe * wIm + uIm * wRe;
	    uRe = vRe;
	    uIm = vIm;
	}
    }

    if (inv) {
	for (i = 0; i < n; i++) {
	    xRe[i] /= (double)n;
	    xIm[i] /= (double)(-n);

	}
    }

    return SUCCESS;
}
