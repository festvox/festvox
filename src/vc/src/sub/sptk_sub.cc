/*
  ----------------------------------------------------------------
	Speech Signal Processing Toolkit (SPTK): version 3.0
			 SPTK Working Group

		   Department of Computer Science
		   Nagoya Institute of Technology
				and
    Interdisciplinary Graduate School of Science and Engineering
		   Tokyo Institute of Technology
		      Copyright (c) 1984-2000
			All Rights Reserved.

  Permission is hereby granted, free of charge, to use and
  distribute this software and its documentation without
  restriction, including without limitation the rights to use,
  copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of this work, and to permit persons to whom this
  work is furnished to do so, subject to the following conditions:

    1. The code must retain the above copyright notice, this list
       of conditions and the following disclaimer.

    2. Any modifications must be clearly marked as such.

  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,
  SPTK WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM
  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF
  TECHNOLOGY, SPTK WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE
  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
  PERFORMANCE OF THIS SOFTWARE.
 ----------------------------------------------------------------
*/
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
/*              Speech Signal Processing Toolkit (SPTK)              */
/*            http://kt-lab.ics.nitech.ac.jp/~tokuda/SPTK/           */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  SPTK subroutine                                                  */
/*                                                                   */
/*  Extracted from SPTK by Tomoki Toda (tomoki@ics.nitech.ac.jp)     */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fft.h"
#include "sptk_sub.h"

// memory alloccation (double)
double *dalloc(int cc)
{
  double *ptr;

  if( (ptr=(double *)malloc(cc*sizeof(double))) == NULL){
      fprintf(stderr,"Can't malloc\n");return(0);}
  return(ptr);
}

double *dgetmem(int leng)
{
    return ( (double *)getmem(leng, sizeof(double)) );
}

char *getmem(int leng, unsigned size)
{
    char *p = NULL;

    if ((p = (char *)calloc(leng, size)) == NULL){
        fprintf(stderr, "Memory allocation error !\n");
        exit(3);
    }
    return (p);
}

void fillz(char *ptr, int size, int nitem)
{
    register long n;
    
    n = size * nitem;
    while(n--)
        *ptr++ = '\0';
}

void movem(register char *a, register char *b, int size, int nitem)
{
    register long i;

    i = size * nitem;
    if (a > b)
        while (i--) *b++ = *a++;
    else{
        a += i; b += i;
        while (i--) *--b = *--a;
    }
}

void freqt(double *c1,	// minimum phase sequence (Cepstrum)
	   long m1,	// order of minimum phase sequence (FFT / 2)
	   double *c2,	// warped sequence [0]-[m2]
	   long m2,	// order of warped sequence
	   double a)	// all-pass constant
{
    register int 	i, j;
    double		b;
    static double	*d = NULL, *g;
    static int		size;
    
    if(d == NULL){
	size = (int)m2;
	if ((d = dalloc(size+size+2)) == NULL) exit(1);
	g = d + size + 1;
    }

    if((int)m2 > size){
	free(d);
	size = (int)m2;
	if ((d = dalloc(size+size+2)) == NULL) exit(1);
	g = d + size + 1;
    }
    
    b = 1 - a*a;
    fillz((char *)g, sizeof(*g), (int)m2+1);

    for (i=-(int)m1; i<=0; i++){
	if (0 <= (int)m2)
	    g[0] = c1[-i] + a*(d[0] = g[0]);
	if (1 <= (int)m2)
	    g[1] = b*d[0] + a*(d[1] = g[1]);
	for (j=2; j<=(int)m2; j++)
	    g[j] = d[j-1] + a*((d[j]=g[j]) - g[j-1]);
    }
    
    movem((char *)g, (char *)c2, sizeof(*g), (int)m2+1);
}

// Mel-Cepstral Analysis 
//   return value : 0 -> completed by end condition
//                  -1-> completed by maximum iteration
int mcep(double *spc, 	// input amplitude spectrum
	 int flng, 	// frame (FFT) length
	 double *mc,	// mel cepstrum
	 int m,		// order of mel cepstrum
	 double a,	// alpha
	 int itr1,	// minimum number of iteration
	 int itr2,	// maximum number of iteration
	 double dd,	// end condition
	 double e)	// initial value for log-periodgram
{


    register int   i, j;
    int		   flag = 0, f2, m2;
    double	   t, s;
    static double  *x = NULL, *y, *c, *d, *al, *b;
    static int     size_x, size_d;
  
    if(x == NULL){
	x = dgetmem(3*flng);
	y = x + flng; c = y + flng;
	size_x = flng;
	
	d = dgetmem(3*m+3);
	al = d  + (m+1); b  = al + (m+1);
	size_d = m;
    }
    if(flng > size_x){
	free(x);
	x = dgetmem(3*flng);
	y = x + flng; c = y + flng;
	size_x = flng;
    }
    if(m > size_d){
	free(d);
	d = dgetmem(3*m+3);
	al = d  + (m+1); b  = al + (m+1);
	size_d = m;
    }

    f2 = flng / 2;
    m2 = m + m;

    //  log power spectrum 
    for(i=0; i<flng; i++){
	x[i] = spc[i] * spc[i];
	c[i] = log(x[i]+e);
    }
	
    //  1, (-a), (-a)^2, ..., (-a)^M 
    al[0] = 1.0;
    for(i=1; i<=m; i++)
	al[i] = -a * al[i-1];

    //  initial value of cepstrum
    fillz((char *)y, sizeof(*y), flng);
    fft(c, y, flng, 1);				//  c : IFFT[x]
    
    c[0] /= 2.0;
    c[flng/2] /= 2.0;
    freqt(c, f2, mc, m, a);			//  mc : mel cep.
    s = c[0];

    //  Newton Raphson method 
    for (j=1; j<=itr2; j++){
	fillz((char *)c, sizeof(*c), flng);
	freqt(mc, m, c, f2, -a);		//  mc : mel cep.
	fillz((char *)y, sizeof(*y), flng);
	fft(c, y, flng, 0);			//  c, y : FFT[mc]
	for(i=0; i<flng; i++)
	    c[i] = x[i] / exp(c[i] + c[i]);
	fillz((char *)y, sizeof(*y), flng);
	fft(c, y, flng, 1);
	frqtr(c, f2, c, m2, a);			//  c : r(k)

	t = c[0];
	if(j >= itr1){
	    if (fabs((t - s)/t) < dd){
		 flag = 1;
		 break;
	    }
	    s = t;
	}
	
	for(i=0; i<=m; i++)
	    b[i] = c[i] - al[i];
	for(i=0; i<=m2; i++)  y[i] = c[i];
	for(i=0; i<=m2; i+=2) y[i] -= c[0];
	for(i=2; i<=m;  i+=2) c[i] += c[0];
	c[0] += c[0];

	if (theq(c, y, d, b, m+1, -1.0)){
	    fprintf(stderr,"mcep : Error in theq() at %dth iteration !\n", j);
	    exit(1);
	}

	for (i=0; i<=m; i++) mc[i] += d[i];
    }

    if(flag) return(0);
    else     return(-1);
}

/***************************************************************

    Frequency Transformation for Calculating Coefficients

	void	frqtr(c1, m1, c2, m2, a)

	double	*c1   : minimum phase sequence
	int	m1    : order of minimum phase sequence
	double	*c2   : warped sequence
	int	m2    : order of warped sequence
	double	a     : all-pass constant

***************************************************************/
void frqtr(double *c1, int m1, double *c2, int m2, double a)
{
    register int 	i, j;
    static double	*d = NULL, *g;
    static int		size;
    
    if(d == NULL){
	size = m2;
	d = dgetmem(size+size+2);
	g = d + size + 1;
    }

    if(m2 > size){
	free(d);
	size = m2;
	d = dgetmem(size+size+2);
	g = d + size + 1;
    }

    fillz((char *)g, sizeof(*g), m2+1);

    for (i=-m1; i<=0; i++){
	if (0 <= m2){
	    d[0] = g[0];
	    g[0] = c1[-i];
	}
	for (j=1; j<=m2; j++)
	    g[j] = d[j-1] + a*((d[j] = g[j]) - g[j-1]);
    }

    movem((char *)g, (char *)c2, sizeof(*g), m2+1);
}

/******************************************************************

    $Id: sptk_sub.cc,v 1.1.1.1 2009/06/14 17:05:49 awb Exp $

    Subroutine for Solving a Toeplitz plus Hankel		
	Coefficient Matrix System of Equations  ( T + H ) a = b	


    int	theq(t, h, a, b, n, eps)

    double   *t  : Toeplitz elements -> T(i,j) = t(|i-j|) t[0]..t[n-1]
    double   *h  : Hankel elements -> H(i,j) = h(i+j) 	  h[0]...h[2*n-2]
    double   *a  : solution vector of equation		  a[0]...a[n-1] 
    double   *b  : known vector  			  b[0]...b[n-1]
    int      n   : system order
    double   eps : singular check (eps(if -1.0, 1.0e-6 is assumed))

    return value :
	0  : normally completed	
	-1 : abnormally completed

****************************************************************/
int theq(double *t, double *h, double *a, double *b, int n, double eps)
{
    static double **r = NULL, **x, **xx, **p;
    static int    size;
    double	  ex[4], ep[2], vx[4], bx[4], g[2];
    int		  i;

    if(r == NULL){
	r  = mtrx2(n, 4); x  = mtrx2(n, 4);
	xx = mtrx2(n, 4); p  = mtrx2(n, 2);
	size = n;
    }
    if(n > size){
	for(i = 0; i < n; i++){
	    free((char*)r[i]);  free((char*)x[i]);
	    free((char*)xx[i]); free((char*)p[i]);
	}
	free((char*)r);  free((char*)x);
	free((char*)xx); free((char*)p);

	r  = mtrx2(n, 4); x  = mtrx2(n, 4);
	xx = mtrx2(n, 4); p  = mtrx2(n, 2);
	size = n;
    }
	
    if(eps < 0.0) eps = 1.0e-6;

    // make r
    for( i = 0; i < n; i++){
	r[i][0] = r[i][3] = t[i];
	r[i][1] = h[n-1+i];
	r[i][2] = h[n-1-i];
    }

    // step 1
    x[0][0] = x[0][3] = 1.0;
    if(cal_p0(p, r, b, n, eps) == -1) return(-1);

    vx[0] = r[0][0];
    vx[1] = r[0][1];
    vx[2] = r[0][2];
    vx[3] = r[0][3];

    // step 2
    for(i=1; i<n; i++){
	cal_ex(ex, r, x, i);
	cal_ep(ep, r, p, i);
	if(cal_bx(bx, vx, ex, eps) == -1) return(-1);
	cal_x(x, xx, bx, i);
	cal_vx(vx, ex, bx);
	if(cal_g(g, vx, b, ep, i, n, eps) == -1) return(-1) ;
	cal_p(p, x, g, i);
    }
    
    // step 3
    for(i=0; i<n; i++) a[i] = p[i][0];
    
    return(0);
}

void mm_mul(double *t, double *x, double *y)
{
    t[0] = x[0] * y[0] + x[1] * y[2];
    t[1] = x[0] * y[1] + x[1] * y[3];
    t[2] = x[2] * y[0] + x[3] * y[2];
    t[3] = x[2] * y[1] + x[3] * y[3];
}

int cal_p0(double **p, double **r, double *b, int n, double eps)
{
    double	t[4], s[2];

    if(inverse(t, r[0], eps) == -1) return(-1);
    s[0] = b[0];
    s[1] = b[n-1];
    mv_mul(p[0], t, s);
    return(0);
}

void cal_ex(double *ex, double **r, double **x, int i)
{
    int		j;
    double	t[4], s[4];

    s[0] = s[1] = s[2] = s[3] = 0.;

    for(j=0; j<i; j++){
	mm_mul(t, r[ i - j ], x[j]);
	s[0] += t[0]; s[1] += t[1];
	s[2] += t[2]; s[3] += t[3];
    }

    ex[0] = s[0]; ex[1] = s[1];
    ex[2] = s[2]; ex[3] = s[3];
}

void cal_ep(double *ep, double **r, double **p, int i)
{
    int		j;
    double	t[2], s[2];

    s[0] = s[1] = 0.;
    
    for(j=0; j<i; j++){
	mv_mul(t, r[ i - j ], p[j]);
	s[0] += t[0]; s[1] += t[1];
    }
    ep[0] = s[0]; ep[1] = s[1];
}

int cal_bx(double *bx, double *vx, double *ex, double eps)
{
    double	t[4], s[4];

    crstrns(t, vx);
    if(inverse(s, t, eps) == -1) return(-1);
    mm_mul(bx, s, ex);
    return(0);
}

void cal_x(double **x, double **xx, double *bx, int i)
{
    int		j;
    double	t[4], s[4];

    for(j=1; j<i; j++){
	crstrns(t, xx[i-j]);
	mm_mul(s, t, bx);
	x[j][0] -= s[0]; x[j][1] -= s[1];
	x[j][2] -= s[2]; x[j][3] -= s[3];
    }

    for(j=1; j<i; j++){
	xx[j][0] = x[j][0]; xx[j][1] = x[j][1];
	xx[j][2] = x[j][2]; xx[j][3] = x[j][3];
    }
    
    x[i][0] = xx[i][0] = -bx[0];
    x[i][1] = xx[i][1] = -bx[1];
    x[i][2] = xx[i][2] = -bx[2];
    x[i][3] = xx[i][3] = -bx[3];
}

void cal_vx(double *vx, double *ex, double *bx)
{
    double	t[4], s[4];

    crstrns(t, ex);
    mm_mul(s, t, bx);
    vx[0] -= s[0]; vx[1] -= s[1];
    vx[2] -= s[2]; vx[3] -= s[3];
}

int cal_g(double *g, double *vx, double *b, double *ep, int i, int n,
	      double eps)
{
    double	t[2], s[4], u[4];

    t[0] = b[i] - ep[0];
    t[1] = b[n-1-i] - ep[1];
    crstrns(s, vx);
    
    if(inverse(u, s, eps) == -1) return(-1);
    mv_mul(g, u, t);
    return(0);
}

void cal_p(double **p, double **x, double *g, int i)
{
    double	t[4], s[2];
    int		j;

    for(j=0; j<i; j++){
	crstrns(t, x[i-j]);
	mv_mul(s, t, g);
	p[j][0] += s[0];
	p[j][1] += s[1];
    }
    
    p[i][0] = g[0];
    p[i][1] = g[1];
}

int inverse(double *x, double *y, double eps)
{
    double	det;

    det = y[0] * y[3] - y[1] * y[2];

    if(fabs(det) < eps) return(-1);
    
    x[0] = y[3]  / det;
    x[1] = -y[1] / det;
    x[2] = -y[2] / det;
    x[3] = y[0]  / det;
    return(0);
}

void crstrns(double *x, double *y)
{
    x[0] = y[3];
    x[1] = y[2];
    x[2] = y[1];
    x[3] = y[0];
}

void mv_mul(double *t, double *x, double *y)
{
    t[0] = x[0] * y[0] + x[1] * y[1];
    t[1] = x[2] * y[0] + x[3] * y[1];
}

double **mtrx2(int a, int b)
{
    int		i;
    double	**x;

    if(! (x = (double**)calloc((unsigned)a, sizeof(*x)))){
	fprintf(stderr, "mtrx2() in theq() : memory allocation error !\n");
	exit(3);
    }
    for(i=0; i<a; i++)
	if(! (x[i] = (double*)calloc((unsigned)b, sizeof(**x)))){
	    fprintf(stderr, "mtrx2() in theq() : memory allocation error !\n");
	    exit(3);
	}

    return(x);
}

