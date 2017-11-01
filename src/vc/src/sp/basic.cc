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
/*#include <unistd.h>*/
#include <math.h>

#include "../include/defs.h"
#include "../include/basic.h"

#ifdef NO_WARNING
int sp_warning = 0;
#else
int sp_warning = 1;
#endif

void sp_warn_on(void)
{
    sp_warning = 1;
    return;
}

void sp_warn_off(void)
{
    sp_warning = 0;
    return;
}

/*double randun(void)
{
    static int randun_init = 0;
    double x;
    
*/    /* reset random function */
/*    if (randun_init == 0) {
	srand((int)getpid());
	randun_init = 1;
    }

    x = (double)rand() / ((double)RAND_MAX + 1.0);

    return x;
}
*/
/*
 *	mu : mean  sigma : standard deviation
 */
/*double gauss(double mu, double sigma)
{
    int i;
    double a, x;

    for (i = 0, a = 0.0; i < 12; i++) {
	a += randun();
    }

    x = (a - 6.0) * sigma + mu;

    return x;
}
*/
double round(double x)
{
    double y;

    y = floor(x + 0.5);

    return y;
}

double fix(double x)
{
    double n;
    double y;

    y = modf(x, &n);

    return n;
}

double rem(double x, double y)
{
    double n;
    double z;

    z = modf(x / y, &n);
    z = x - n * y;

    return z;
}

void cexpf(float *xr, float *xi)
{
    double a;

    if (xr == NULL) {
	return;
    } else if (*xr == 0.0) {
	*xr = (float)cos((double)*xi);
	*xi = (float)sin((double)*xi);
    } else if (xi != NULL && *xi != 0.0) {
	a = exp((double)*xr);
	*xr = (float)(a * cos((double)*xi));
	*xi = (float)(a * sin((double)*xi));
    } else {
	*xr = (float)exp((double)*xr);
    }

    return;
}

void cexp(double *xr, double *xi)
{
    double a;

    if (xr == NULL) {
	return;
    } else if (*xr == 0.0) {
	*xr = cos(*xi);
	*xi = sin(*xi);
    } else if (xi != NULL && *xi != 0.0) {
	a = exp(*xr);
	*xr = a * cos(*xi);
	*xi = a * sin(*xi);
    } else {
	*xr = exp(*xr);
    }

    return;
}

void clogf(float *xr, float *xi)
{
    double a;

    if (*xr < 0.0 || (xi != NULL && *xi != 0.0)) {
	a = CABS(*xr, *xi);
	*xi = (float)atan2((double)*xi, (double)*xr);
	*xr = (float)log(a);
    } else {
	if (*xr == 0.0) {
	    if (sp_warning)
		fprintf(stderr, "warning: clogf: log of zero\n");
	    
	    *xr = (float)log(ALITTLE_NUMBER);
	} else {
	    *xr = (float)log((double)*xr);
	}
    }

    return;
}

void clog(double *xr, double *xi)
{
    double a;

    if (*xr < 0.0 || (xi != NULL && *xi != 0.0)) {
	a = CABS(*xr, *xi);
	*xi = atan2(*xi, *xr);
	*xr = log(a);
    } else {
	if (*xr == 0.0) {
	    if (sp_warning)
		fprintf(stderr, "warning: clog: log of zero\n");
	    
	    *xr = log(ALITTLE_NUMBER);
	} else {
	    *xr = log(*xr);
	}
    }

    return;
}

void ftos(char *buf, double x)
{
    int j;
    int flag = 0;
    double n;
    double pn;
    double xi;
    char sxi[MAX_LINE];

    if (x == 0.0) {
	strcpy(buf, "0");
    } else {
	pn = floor(log10(fabs(x)));
	n = pow(10.0, pn);

	if (fabs(pn) >= 4) {
	    xi = x / n;
	    flag = 1;
	} else {
	    xi = x;
	}

	sprintf(sxi, "%f", xi);
	for (j = strlen(sxi) - 1; j >= 0; j--) {
	    if (sxi[j] == '.') {
		sxi[j] = NUL;
		break;
	    } else if (sxi[j] == '-' || sxi[j] == '+') {
		sxi[j + 2] = NUL;
		break;
	    } else if (sxi[j] != '0') {
		sxi[j + 1] = NUL;
		break;
	    }
	}

	if (flag == 1) {
	    sprintf(buf, "%se%.0f", sxi, pn);
	} else {
	    sprintf(buf, "%s", sxi);
	}
    }

    return;
}

//int randsort_numcmp(const void *x, const void *y)
int randsort_numcmp(const void *, const void *)
{
    double tx, ty;

    tx = randn();
    ty = randn();

    if (tx < ty) return (-1);
    if (tx > ty) return (1);
    return (0);
}

void randsort(void *data, int num, int size)
{
    qsort(data, (unsigned)num, (unsigned)size, randsort_numcmp);

    return;
}

long factorial(int n)
{
    long k = 1;
    
    if (n > 0) {
	while (1) {
	    k *= (long)n;
	    n--;
	    if (n <= 1) {
		break;
	    }
	}
    }

    return k;
}

void decibel(double *x, long length)
{
    long k;

    for (k = 0; k < length; k++) {
	x[k] = x[k] * x[k];
	if (x[k] <= 0.0) {
	    if (sp_warning)
		fprintf(stderr, "warning: decibel: log of zero\n");
	    
	    x[k] = 10.0 * log10(ALITTLE_NUMBER);
	} else {
	    x[k] = 10.0 * log10(x[k]);
	}
    }

    return;
}

void decibelp(double *x, long length)
{
    long k;

    for (k = 0; k < length; k++) {
	if (x[k] <= 0.0) {
	    if (sp_warning)
		fprintf(stderr, "warning: decibelp: log of zero\n");
	    
	    x[k] = 10.0 * log10(ALITTLE_NUMBER);
	} else {
	    x[k] = 10.0 * log10(x[k]);
	}
    }

    return;
}

double simple_random()
{ 
   /* a, c, m are from Numerical Recipies in C */
   static long  x = 1,
                a = 419,
                c = 6173,
                m = 29282;
   x  = ( x * a + c) % m;

   return( (double)x / (double)m );
}

double simple_gnoise(double rms)
{
    double     x, yy;

    yy = simple_random() + 1.0e-30;
    /* Gaussian noise */
    x = sqrt(-2.* log(yy)) * cos(2.* PI * simple_random())* rms;

    return x;
}
