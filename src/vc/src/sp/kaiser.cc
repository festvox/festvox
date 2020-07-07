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
#include "../include/kaiser.h"

#define MIN_TRANS 0.0001

void getkaiserparam(double sidelobe, double trans, double *beta, long *length)
{
    double value;

    if (sidelobe < 21) {
	*beta = 0;
    } else if (sidelobe > 50) {
	*beta = 0.1102 * ((double)sidelobe - 8.7);
    } else {
	value = (double)sidelobe - 21.0;
	*beta = 0.5842 * pow(value, 0.4) + 0.07886 * value;
    }

    if (trans == 0.0) trans = MIN_TRANS;
    *length = (long)(((double)sidelobe - 8.0) / (2.285 * PI * trans));
    
    return;
}

int kaiser_org(double w[], long n, double beta)
{
    double an1, t, rms;
    long i;

    if (n <= 1) 
	return FAILURE;
    rms = 0.0;
    an1 = 1.0 / (double)(n - 1);
    for(i = 0; i < n; ++i) {
	t = ((double)( i + i - n + 1 )) * an1;
	w[i] = ai0((double)(beta * sqrt(1.0 - t * t)));
	rms += w[i] * w[i];
    }

    /* Normalize w[i] to have a unity power gain. */
    rms = sqrt((double)n / rms);
    while(n-- > 0) *w++ *= rms;

    return SUCCESS;
}

/* This function is buggy. */
double ai0_org(double x)
{
    int i;
    double y, e, de, sde, t;

    y = x / 2.0;
    t = 1.0e-12;
    e = 1.0;
    de = 1.0;
    for (i = 1; i <= 100; i++) {
        de *= y / (double)i;
        sde = de * de;
        e += sde;
        if (sde < e * t)
            break;
    }

    return e;
}

int kaiser(double w[], long n, double beta)
{
    double an1, t;
    long i;

    if (n <= 1) 
	return FAILURE;

    an1 = 1.0 / (double)(n - 1);
    for(i = 0; i < n; i++) {
	t = ((double)(2 * i - n + 1)) * an1;
	w[i] = ai0(beta * sqrt(1.0 - t * t));
	w[i] /= ai0(beta);
    }

    return SUCCESS;
}

double ai0(double x)
{
    double d, ds, s;

    ds = 1;
    d = 2;
    s = ds;
    ds = (x * x) / 4.0;
    while (ds >= 0.2e-8*s) {
	d += 2;
	s += ds;
	ds *= (x * x) / (d * d);
    }
    
    return s;
}
