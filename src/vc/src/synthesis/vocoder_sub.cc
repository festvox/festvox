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
/*   vocoder.c : mel-cepstral vocoder                                */
/*              (pulse/noise excitation & MLSA filter)               */
/*                                                                   */
/*                                    2003/12/26 by Heiga Zen        */
/*  ---------------------------------------------------------------  */

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
/*  NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN        */
/*  CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.         */
/*                                                                   */
/*********************************************************************/
/*                                                                   */
/*  Mel-cepstral vocoder (pulse/noise excitation & MLSA filter)      */
/*                                    2003/12/26 by Heiga Zen        */
/*                                                                   */
/*  Extracted from HTS and slightly modified                         */
/*   by Tomoki Toda (tomoki@ics.nitech.ac.jp)                        */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../sub/sptk_sub.h"
#include "vocoder_sub.h"

void init_vocoder(double fs, int framel, int m, VocoderSetup *vs)
{
    // initialize global parameter
    vs->fprd = framel;
    vs->iprd = 1;
    vs->seed = 1;
    vs->pd   = 5;

    vs->next =1;
    vs->gauss = TRUE;

    vs->pade[ 0]=1.0;
    vs->pade[ 1]=1.0; vs->pade[ 2]=0.0;
    vs->pade[ 3]=1.0; vs->pade[ 4]=0.0;      vs->pade[ 5]=0.0;
    vs->pade[ 6]=1.0; vs->pade[ 7]=0.0;      vs->pade[ 8]=0.0;      vs->pade[ 9]=0.0;
    vs->pade[10]=1.0; vs->pade[11]=0.4999273; vs->pade[12]=0.1067005; vs->pade[13]=0.01170221; vs->pade[14]=0.0005656279;
    vs->pade[15]=1.0; vs->pade[16]=0.4999391; vs->pade[17]=0.1107098; vs->pade[18]=0.01369984; vs->pade[19]=0.0009564853;
    vs->pade[20]=0.00003041721;

    vs->rate = fs;
    vs->c = dgetmem(3 * (m + 1) + 3 * (vs->pd + 1) + vs->pd * (m + 2));
   
    vs->p1 = -1;
    vs->sw = 0;
    vs->x  = 0x55555555;
   
    // for postfiltering
    vs->mc = NULL;
    vs->o  = 0;
    vs->d  = NULL;
    vs->irleng= 64;
   
    return;
}

void vocoder(double p, double *mc, int m, double a, double beta,
	     VocoderSetup *vs, double *wav, long *pos)
{
    double inc, x, e1, e2;
    int i, j, k; 
   
    if (p != 0.0) 
	p = vs->rate / p;  // f0 -> pitch
   
    if (vs->p1 < 0) {
	if (vs->gauss & (vs->seed != 1)) 
	    vs->next = srnd((unsigned)vs->seed);
         
	vs->p1   = p;
	vs->pc   = vs->p1;
	vs->cc   = vs->c + m + 1;
	vs->cinc = vs->cc + m + 1;
	vs->d1   = vs->cinc + m + 1;
      
	mc2b(mc, vs->c, m, a);
      
	if (beta > 0.0 && m > 1) {
	    e1 = b2en(vs->c, m, a, vs);
	    vs->c[1] -= beta * a * mc[2];
	    for (k=2;k<=m;k++)
		vs->c[k] *= (1.0 + beta);
	    e2 = b2en(vs->c, m, a, vs);
	    vs->c[0] += log(e1/e2)/2;
	}

	return;
    }

    mc2b(mc, vs->cc, m, a); 
    if (beta>0.0 && m > 1) {
	e1 = b2en(vs->cc, m, a, vs);
	vs->cc[1] -= beta * a * mc[2];
	for (k = 2; k <= m; k++)
	    vs->cc[k] *= (1.0 + beta);
	e2 = b2en(vs->cc, m, a, vs);
	vs->cc[0] += log(e1 / e2) / 2.0;
    }

    for (k=0; k<=m; k++)
	vs->cinc[k] = (vs->cc[k] - vs->c[k]) *
	    (double)vs->iprd / (double)vs->fprd;

    if (vs->p1!=0.0 && p!=0.0) {
	inc = (p - vs->p1) * (double)vs->iprd / (double)vs->fprd;
    } else {
	inc = 0.0;
	vs->pc = p;
	vs->p1 = 0.0;
    }

    for (j = vs->fprd, i = (vs->iprd + 1) / 2; j--;) {
	if (vs->p1 == 0.0) {
	    if (vs->gauss)
		x = (double) nrandom(vs);
	    else
		x = mseq(vs);
	} else {
	    if ((vs->pc += 1.0) >= vs->p1) {
		x = sqrt (vs->p1);
		vs->pc = vs->pc - vs->p1;
	    } else x = 0.0;
	}

	x *= exp(vs->c[0]);

	x = mlsadf(x, vs->c, m, a, vs->pd, vs->d1, vs);

	wav[*pos] = x;
	*pos += 1;

	if (!--i) {
	    vs->p1 += inc;
	    for (k = 0; k <= m; k++) vs->c[k] += vs->cinc[k];
	    i = vs->iprd;
	}
    }
   
    vs->p1 = p;
    movem((char *)vs->cc, (char *)vs->c, sizeof(double), m+1);
   
    return;
}

void vocoder(double p, double *mc, double dpow, int m, double a, double beta,
	     VocoderSetup *vs, double *wav, long *pos)
{
    double inc, x, e1, e2;
    int i, j, k; 
   
    if (p != 0.0) 
	p = vs->rate / p;  // f0 -> pitch
   
    if (vs->p1 < 0) {
	if (vs->gauss & (vs->seed != 1)) 
	    vs->next = srnd((unsigned)vs->seed);
         
	vs->p1   = p;
	vs->pc   = vs->p1;
	vs->cc   = vs->c + m + 1;
	vs->cinc = vs->cc + m + 1;
	vs->d1   = vs->cinc + m + 1;
      
	mc2b(mc, vs->c, m, a);
	vs->c[0] += dpow;
      
	if (beta > 0.0 && m > 1) {
	    e1 = b2en(vs->c, m, a, vs);
	    vs->c[1] -= beta * a * mc[2];
	    for (k=2;k<=m;k++)
		vs->c[k] *= (1.0 + beta);
	    e2 = b2en(vs->c, m, a, vs);
	    vs->c[0] += log(e1/e2)/2;
	}

	return;
    }

    mc2b(mc, vs->cc, m, a);
    vs->cc[0] += dpow;
    if (beta>0.0 && m > 1) {
	e1 = b2en(vs->cc, m, a, vs);
	vs->cc[1] -= beta * a * mc[2];
	for (k = 2; k <= m; k++)
	    vs->cc[k] *= (1.0 + beta);
	e2 = b2en(vs->cc, m, a, vs);
	vs->cc[0] += log(e1 / e2) / 2.0;
    }

    for (k=0; k<=m; k++)
	vs->cinc[k] = (vs->cc[k] - vs->c[k]) *
	    (double)vs->iprd / (double)vs->fprd;

    if (vs->p1!=0.0 && p!=0.0) {
	inc = (p - vs->p1) * (double)vs->iprd / (double)vs->fprd;
    } else {
	inc = 0.0;
	vs->pc = p;
	vs->p1 = 0.0;
    }

    for (j = vs->fprd, i = (vs->iprd + 1) / 2; j--;) {
	if (vs->p1 == 0.0) {
	    if (vs->gauss)
		x = (double) nrandom(vs);
	    else
		x = mseq(vs);
	} else {
	    if ((vs->pc += 1.0) >= vs->p1) {
		x = sqrt (vs->p1);
		vs->pc = vs->pc - vs->p1;
	    } else x = 0.0;
	}

	x *= exp(vs->c[0]);

	x = mlsadf(x, vs->c, m, a, vs->pd, vs->d1, vs);

	wav[*pos] = x;
	*pos += 1;

	if (!--i) {
	    vs->p1 += inc;
	    for (k = 0; k <= m; k++) vs->c[k] += vs->cinc[k];
	    i = vs->iprd;
	}
    }
   
    vs->p1 = p;
    movem((char *)vs->cc, (char *)vs->c, sizeof(double), m+1);
   
    return;
}

double mlsadf(double x, double *b, int m, double a, int pd, double *d, VocoderSetup *vs)
{
    double mlsadf1(double, double*, int, double, int, double*, VocoderSetup*);
    double mlsadf2(double, double*, int, double, int, double*, VocoderSetup*);

   vs->ppade = &(vs->pade[pd*(pd+1)/2]);
    
   x = mlsadf1 (x, b, m, a, pd, d, vs);
   x = mlsadf2 (x, b, m, a, pd, &d[2*(pd+1)], vs);

   return(x);
}

double mlsadf1(double x, double *b, int m, double a, int pd, double *d, VocoderSetup *vs)
{
   double v, out = 0.0, *pt, aa;
   register int i;

   aa = 1 - a*a;
   pt = &d[pd+1];

   for (i=pd; i>=1; i--) {
      d[i] = aa*pt[i-1] + a*d[i];
      pt[i] = d[i] * b[1];
      v = pt[i] * vs->ppade[i];
      x += (1 & i) ? v : -v;
      out += v;
   }

   pt[0] = x;
   out += x;

   return(out);
}

double mlsadf2 (double x, double *b, int m, double a, int pd, double *d, VocoderSetup *vs)
{
   double v, out = 0.0, *pt, aa;
   double mlsafir(double, double*, int, double, double*);
   register int i;
    
   aa = 1 - a*a;
   pt = &d[pd * (m+2)];

   for (i=pd; i>=1; i--) {
      pt[i] = mlsafir (pt[i-1], b, m, a, &d[(i-1)*(m+2)]);
      v = pt[i] * vs->ppade[i];

      x  += (1&i) ? v : -v;
      out += v;
   }
    
   pt[0] = x;
   out  += x;

   return(out);
}

double mlsafir (double x, double *b, int m, double a, double *d)
{
   double y = 0.0;
   double aa;
   register int i;

   aa = 1 - a*a;

   d[0] = x;
   d[1] = aa*d[0] + a*d[1];

   for (i=2; i<=m; i++) {
      d[i] = d[i] + a*(d[i+1]-d[i-1]);
      y += d[i]*b[i];
   }

   for (i=m+1; i>1; i--) 
      d[i] = d[i-1];

   return(y);
}

double nrandom (VocoderSetup *vs)
{
   if (vs->sw == 0) {
      vs->sw = 1;
      do {
         vs->r1 = 2.0 * rnd(&vs->next) - 1.0;
         vs->r2 = 2.0 * rnd(&vs->next) - 1.0;
         vs->s  = vs->r1 * vs->r1 + vs->r2 * vs->r2;
      } while (vs->s > 1 || vs->s == 0);

      vs->s = sqrt (-2 * log(vs->s) / vs->s);
      
      return(vs->r1*vs->s);
   }
   else {
      vs->sw = 0;
      
      return (vs->r2*vs->s);
   }
}

double rnd (unsigned long *next)
{
   double r;

   *next = *next * 1103515245L + 12345;
   r = (*next / 65536L) % 32768L;

   return(r/RANDMAX); 
}

unsigned long srnd ( unsigned long seed )
{
   return(seed);
}

int mseq (VocoderSetup *vs)
{
   register int x0, x28;

   vs->x >>= 1;

   if (vs->x & B0)
      x0 = 1;
   else
      x0 = -1;

   if (vs->x & B28)
      x28 = 1;
   else
      x28 = -1;

   if (x0 + x28)
      vs->x &= B31_;
   else
      vs->x |= B31;

   return(x0);
}

// mc2b : transform mel-cepstrum to MLSA digital fillter coefficients
void mc2b (double *mc, double *b, int m, double a)
{
   b[m] = mc[m];
    
   for (m--; m>=0; m--)
      b[m] = mc[m] - a * b[m+1];
   
   return;
}


double b2en (double *b, int m, double a, VocoderSetup *vs)
{
   double en;
   int k;
   
   if (vs->o<m) {
      if (vs->mc != NULL)
         free(vs->mc);
    
      vs->mc = dgetmem((m + 1) + 2 * vs->irleng);
      vs->cep = vs->mc + m+1;
      vs->ir  = vs->cep + vs->irleng;
   }

   b2mc(b, vs->mc, m, a);
   freqt(vs->mc, m, vs->cep, vs->irleng-1, -a, vs);
   c2ir(vs->cep, vs->irleng, vs->ir, vs->irleng);
   en = 0.0;
   
   for (k=0;k<vs->irleng;k++)
      en += vs->ir[k] * vs->ir[k];

   return(en);
}


// b2bc : transform MLSA digital filter coefficients to mel-cepstrum
void b2mc (double *b, double *mc, int m, double a)
{
  double d, o;
        
  d = mc[m] = b[m];
  for (m--; m>=0; m--) {
    o = b[m] + a * d;
    d = b[m];
    mc[m] = o;
  }
  
  return;
}

// freqt : frequency transformation
void freqt (double *c1, int m1, double *c2, int m2, double a, VocoderSetup *vs)
{
   register int i, j;
   double b;
    
   if (vs->d==NULL) {
      vs->size = m2;
      vs->d    = dgetmem(vs->size + vs->size + 2);
      vs->g    = vs->d+vs->size+1;
   }

   if (m2>vs->size) {
      free(vs->d);
      vs->size = m2;
      vs->d    = dgetmem(vs->size + vs->size + 2);
      vs->g    = vs->d+vs->size+1;
   }
    
   b = 1-a*a;
   for (i=0; i<m2+1; i++)
      vs->g[i] = 0.0;

   for (i=-m1; i<=0; i++) {
      if (0 <= m2)
         vs->g[0] = c1[-i]+a*(vs->d[0]=vs->g[0]);
      if (1 <= m2)
         vs->g[1] = b*vs->d[0]+a*(vs->d[1]=vs->g[1]);
      for (j=2; j<=m2; j++)
         vs->g[j] = vs->d[j-1]+a*((vs->d[j]=vs->g[j])-vs->g[j-1]);
   }
    
   movem((char *)vs->g, (char *)c2, sizeof(double), m2+1);
   
   return;
}

// c2ir : The minimum phase impulse response is evaluated from the minimum phase cepstrum
void c2ir (double *c, int nc, double *h, int leng)
{
   register int n, k, upl;
   double  d;

   h[0] = exp(c[0]);
   for (n=1; n<leng; n++) {
      d = 0;
      upl = (n>=nc) ? nc-1 : n;
      for (k=1; k<=upl; k++)
         d += k*c[k]*h[n-k];
      h[n] = d/n;
   }
   
   return;
}

double get_dpow(double *rmcep, double *cmcep, int m, double a,
		VocoderSetup *vs)
{
    double e1, e2, dpow;

    if (vs->p1 < 0) {
	vs->p1 = 1;
	vs->cc = vs->c + m + 1;
	vs->cinc = vs->cc + m + 1;
	vs->d1   = vs->cinc + m + 1;
    }

    mc2b(rmcep, vs->c, m, a); 
    e1 = b2en(vs->c, m, a, vs);

    mc2b(cmcep, vs->cc, m, a); 
    e2 = b2en(vs->cc, m, a, vs);

    dpow = log(e1 / e2) / 2.0;

    return dpow;
}

void free_vocoder(VocoderSetup *vs)
{
    free(vs->c);
    free(vs->mc);
    free(vs->d);
 
    vs->c = NULL;
    vs->mc = NULL;
    vs->d = NULL;
    vs->ppade = NULL;
    vs->cc = NULL;
    vs->cinc = NULL;
    vs->d1 = NULL;
    vs->g = NULL;
    vs->cep = NULL;
    vs->ir = NULL;
   
    return;
}

// -------------------- End of "vocoder.c" --------------------
