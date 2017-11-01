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

#ifndef __VOPERATE_H
#define __VOPERATE_H

#include "vector.h"

extern void fvoper(FVECTOR a, const char *op, FVECTOR b);
extern void dvoper(DVECTOR a, const char *op, DVECTOR b);
extern FVECTOR xfvoper(FVECTOR a, const char *op, FVECTOR b);
extern DVECTOR xdvoper(DVECTOR a, const char *op, DVECTOR b);

extern void fvscoper(FVECTOR a, const char *op, float t);
extern void dvscoper(DVECTOR a, const char *op, double t);
extern FVECTOR xfvscoper(FVECTOR a, const char *op, float t);
extern DVECTOR xdvscoper(DVECTOR a, const char *op, double t);

extern void svoper(SVECTOR a, const char *op, SVECTOR b);
extern void lvoper(LVECTOR a, const char *op, LVECTOR b);
extern SVECTOR xsvoper(SVECTOR a, const char *op, SVECTOR b);
extern LVECTOR xlvoper(LVECTOR a, const char *op, LVECTOR b);
extern void svscoper(SVECTOR a, const char *op, double t);
extern void lvscoper(LVECTOR a, const char *op, double t);
extern SVECTOR xsvscoper(SVECTOR a, const char *op, double t);
extern LVECTOR xlvscoper(LVECTOR a, const char *op, double t);

extern void svabs(SVECTOR x);
extern void lvabs(LVECTOR x);
extern void fvabs(FVECTOR x);
extern void dvabs(DVECTOR x);
extern SVECTOR xsvabs(SVECTOR x);
extern LVECTOR xlvabs(LVECTOR x);
extern FVECTOR xfvabs(FVECTOR x);
extern DVECTOR xdvabs(DVECTOR x);

extern void svsquare(SVECTOR x);
extern void lvsquare(LVECTOR x);
extern void fvsquare(FVECTOR x);
extern void dvsquare(DVECTOR x);
extern SVECTOR xsvsquare(SVECTOR x);
extern LVECTOR xlvsquare(LVECTOR x);
extern FVECTOR xfvsquare(FVECTOR x);
extern DVECTOR xdvsquare(DVECTOR x);

extern void svsign(SVECTOR x);
extern void lvsign(LVECTOR x);
extern void fvsign(FVECTOR x);
extern void dvsign(DVECTOR x);
extern SVECTOR xsvsign(SVECTOR x);
extern LVECTOR xlvsign(LVECTOR x);
extern FVECTOR xfvsign(FVECTOR x);
extern DVECTOR xdvsign(DVECTOR x);

extern SVECTOR xsvremap(SVECTOR x, LVECTOR map);
extern LVECTOR xlvremap(LVECTOR x, LVECTOR map);
extern FVECTOR xfvremap(FVECTOR x, LVECTOR map);
extern DVECTOR xdvremap(DVECTOR x, LVECTOR map);

extern SVECTOR xsvcodiff(SVECTOR x, double coef);
extern LVECTOR xlvcodiff(LVECTOR x, double coef);
extern FVECTOR xfvcodiff(FVECTOR x, double coef);
extern DVECTOR xdvcodiff(DVECTOR x, double coef);

extern LVECTOR xsvfind(SVECTOR x);
extern LVECTOR xlvfind(LVECTOR x);
extern LVECTOR xfvfind(FVECTOR x);
extern LVECTOR xdvfind(DVECTOR x);

extern DVECTOR xdvfindv(DVECTOR x);
extern DVECTOR xdvsceval(DVECTOR x, const char *op, double t);
extern LVECTOR xdvscfind(DVECTOR x, const char *op, double t);
extern DVECTOR xdvscfindv(DVECTOR x, const char *op, double t);

extern void lvcumsum(LVECTOR x);
extern void fvcumsum(FVECTOR x);
extern void dvcumsum(DVECTOR x);
extern LVECTOR xlvcumsum(LVECTOR x);
extern FVECTOR xfvcumsum(FVECTOR x);
extern DVECTOR xdvcumsum(DVECTOR x);

extern void lvcumprod(LVECTOR x);
extern void fvcumprod(FVECTOR x);
extern void dvcumprod(DVECTOR x);
extern LVECTOR xlvcumprod(LVECTOR x);
extern FVECTOR xfvcumprod(FVECTOR x);
extern DVECTOR xdvcumprod(DVECTOR x);

extern void fvexp(FVECTOR x);
extern void dvexp(DVECTOR x);
extern FVECTOR xfvexp(FVECTOR x);
extern DVECTOR xdvexp(DVECTOR x);

extern void fvlog(FVECTOR x);
extern void dvlog(DVECTOR x);
extern FVECTOR xfvlog(FVECTOR x);
extern DVECTOR xdvlog(DVECTOR x);

extern void fvdecibel(FVECTOR x);
extern void dvdecibel(DVECTOR x);
extern FVECTOR xfvdecibel(FVECTOR x);
extern DVECTOR xdvdecibel(DVECTOR x);

extern FVECTOR xfvrandn(long length);
extern DVECTOR xdvrandn(long length);

extern void dvsort(DVECTOR x);

extern long svsum(SVECTOR x);
extern long lvsum(LVECTOR x);
extern float fvsum(FVECTOR x);
extern double dvsum(DVECTOR x);
extern long svsqsum(SVECTOR x);
extern long lvsqsum(LVECTOR x);
extern float fvsqsum(FVECTOR x);
extern double dvsqsum(DVECTOR x);
extern long svabssum(SVECTOR x);
extern long lvabssum(LVECTOR x);
extern float fvabssum(FVECTOR x);
extern double dvabssum(DVECTOR x);

extern short svmax(SVECTOR x, long *index);
extern long lvmax(LVECTOR x, long *index);
extern float fvmax(FVECTOR x, long *index);
extern double dvmax(DVECTOR x, long *index);

extern short svmin(SVECTOR x, long *index);
extern long lvmin(LVECTOR x, long *index);
extern float fvmin(FVECTOR x, long *index);
extern double dvmin(DVECTOR x, long *index);

extern void svscmax(SVECTOR x, short a);
extern void lvscmax(LVECTOR x, long a);
extern void fvscmax(FVECTOR x, float a);
extern void dvscmax(DVECTOR x, double a);

extern void svscmin(SVECTOR x, short a);
extern void lvscmin(LVECTOR x, long a);
extern void fvscmin(FVECTOR x, float a);
extern void dvscmin(DVECTOR x, double a);

extern float fvdot(FVECTOR x, FVECTOR y);
extern double dvdot(DVECTOR x, DVECTOR y);

extern void dvmorph(DVECTOR x, DVECTOR y, double rate);
extern DVECTOR xdvmorph(DVECTOR x, DVECTOR y, double rate);

#define svmean(x) ((double)svsum(x) / (double)x->length)
#define lvmean(x) ((double)lvsum(x) / (double)x->length)
#define fvmean(x) ((double)fvsum(x) / (double)x->length)
#define dvmean(x) (dvsum(x) / (double)x->length)
#define xsvdiff(x) xsvcodiff(x, 1.0);
#define xlvdiff(x) xlvcodiff(x, 1.0);
#define xfvdiff(x) xfvcodiff(x, 1.0);
#define xdvdiff(x) xdvcodiff(x, 1.0);

#endif /* __VOPERATE_H */
