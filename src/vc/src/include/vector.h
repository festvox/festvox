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

#ifndef __VECTOR_H
#define __VECTOR_H

typedef struct SVECTOR_STRUCT {
    long length;
    short *data;
    short *imag;
} *SVECTOR;

typedef struct LVECTOR_STRUCT {
    long length;
    long *data;
    long *imag;
} *LVECTOR;

typedef struct FVECTOR_STRUCT {
    long length;
    float *data;
    float *imag;
} *FVECTOR;

typedef struct DVECTOR_STRUCT {
    long length;
    double *data;
    double *imag;
} *DVECTOR;

typedef struct SVECTORS_STRUCT {
    long num_vector;
    SVECTOR *vector;
} *SVECTORS;

typedef struct LVECTORS_STRUCT {
    long num_vector;
    LVECTOR *vector;
} *LVECTORS;

typedef struct FVECTORS_STRUCT {
    long num_vector;
    FVECTOR *vector;
} *FVECTORS;

typedef struct DVECTORS_STRUCT {
    long num_vector;
    DVECTOR *vector;
} *DVECTORS;

#if 0
typedef struct WAVE_STRUCT {
    char *name;
    int format;
    int headlen;
    float samp_freq;
    SVECTOR wave;
    short min;
    short max;
} *WAVE;
#endif

#define vlength(x) ((x)->length)
#define vdata(x) ((x)->data)
#define vreal(x) ((x)->data)
#define vimag(x) ((x)->data)

extern SVECTOR xsvalloc(long length);
extern LVECTOR xlvalloc(long length);
extern FVECTOR xfvalloc(long length);
extern DVECTOR xdvalloc(long length);
extern void xsvfree(SVECTOR vector);
extern void xlvfree(LVECTOR vector);
extern void xfvfree(FVECTOR vector);
extern void xdvfree(DVECTOR vector);

extern void svialloc(SVECTOR x);
extern void lvialloc(LVECTOR x);
extern void fvialloc(FVECTOR x);
extern void dvialloc(DVECTOR x);
extern void svifree(SVECTOR x);
extern void lvifree(LVECTOR x);
extern void fvifree(FVECTOR x);
extern void dvifree(DVECTOR x);

extern SVECTOR xsvrialloc(long length);
extern LVECTOR xlvrialloc(long length);
extern FVECTOR xfvrialloc(long length);
extern DVECTOR xdvrialloc(long length);

extern SVECTOR xsvrealloc(SVECTOR x, long length);
extern LVECTOR xlvrealloc(LVECTOR x, long length);
extern FVECTOR xfvrealloc(FVECTOR x, long length);
extern DVECTOR xdvrealloc(DVECTOR x, long length);

extern SVECTORS xsvsalloc(long num);
extern LVECTORS xlvsalloc(long num);
extern FVECTORS xfvsalloc(long num);
extern DVECTORS xdvsalloc(long num);
extern void xsvsfree(SVECTORS xs);
extern void xlvsfree(LVECTORS xs);
extern void xfvsfree(FVECTORS xs);
extern void xdvsfree(DVECTORS xs);

extern SVECTOR xsvcplx(SVECTOR xr, SVECTOR xi);
extern LVECTOR xlvcplx(LVECTOR xr, LVECTOR xi);
extern FVECTOR xfvcplx(FVECTOR xr, FVECTOR xi);
extern DVECTOR xdvcplx(DVECTOR xr, DVECTOR xi);

extern void svreal(SVECTOR x);
extern void lvreal(LVECTOR x);
extern void fvreal(FVECTOR x);
extern void dvreal(DVECTOR x);

extern void svimag(SVECTOR x);
extern void lvimag(LVECTOR x);
extern void fvimag(FVECTOR x);
extern void dvimag(DVECTOR x);

extern SVECTOR xsvreal(SVECTOR x);
extern LVECTOR xlvreal(LVECTOR x);
extern FVECTOR xfvreal(FVECTOR x);
extern DVECTOR xdvreal(DVECTOR x);

extern SVECTOR xsvimag(SVECTOR x);
extern LVECTOR xlvimag(LVECTOR x);
extern FVECTOR xfvimag(FVECTOR x);
extern DVECTOR xdvimag(DVECTOR x);

extern void svconj(SVECTOR x);
extern void lvconj(LVECTOR x);
extern void fvconj(FVECTOR x);
extern void dvconj(DVECTOR x);
extern SVECTOR xsvconj(SVECTOR x);
extern LVECTOR xlvconj(LVECTOR x);
extern FVECTOR xfvconj(FVECTOR x);
extern DVECTOR xdvconj(DVECTOR x);

extern void svriswap(SVECTOR x);
extern void lvriswap(LVECTOR x);
extern void fvriswap(FVECTOR x);
extern void dvriswap(DVECTOR x);
extern SVECTOR xsvriswap(SVECTOR x);
extern LVECTOR xlvriswap(LVECTOR x);
extern FVECTOR xfvriswap(FVECTOR x);
extern DVECTOR xdvriswap(DVECTOR x);

extern void svcopy(SVECTOR y, SVECTOR x);
extern void lvcopy(LVECTOR y, LVECTOR x);
extern void fvcopy(FVECTOR y, FVECTOR x);
extern void dvcopy(DVECTOR y, DVECTOR x);

extern SVECTOR xsvclone(SVECTOR x);
extern LVECTOR xlvclone(LVECTOR x);
extern FVECTOR xfvclone(FVECTOR x);
extern DVECTOR xdvclone(DVECTOR x);

extern SVECTOR xsvcat(SVECTOR x, SVECTOR y);
extern LVECTOR xlvcat(LVECTOR x, LVECTOR y);
extern FVECTOR xfvcat(FVECTOR x, FVECTOR y);
extern DVECTOR xdvcat(DVECTOR x, DVECTOR y);

extern void svinit(SVECTOR x, long j, long incr, long n);
extern void lvinit(LVECTOR x, long j, long incr, long n);
extern void fvinit(FVECTOR x, float j, float incr, float n);
extern void dvinit(DVECTOR x, double j, double incr, double n);

extern SVECTOR xsvinit(long j, long incr, long n);
extern LVECTOR xlvinit(long j, long incr, long n);
extern FVECTOR xfvinit(float j, float incr, float n);
extern DVECTOR xdvinit(double j, double incr, double n);

extern void sviinit(SVECTOR x, long j, long incr, long n);
extern void lviinit(LVECTOR x, long j, long incr, long n);
extern void fviinit(FVECTOR x, float j, float incr, float n);
extern void dviinit(DVECTOR x, double j, double incr, double n);

extern SVECTOR xsvriinit(long j, long incr, long n);
extern LVECTOR xlvriinit(long j, long incr, long n);
extern FVECTOR xfvriinit(float j, float incr, float n);
extern DVECTOR xdvriinit(double j, double incr, double n);

extern SVECTOR xsvcut(SVECTOR x, long offset, long length);
extern LVECTOR xlvcut(LVECTOR x, long offset, long length);
extern FVECTOR xfvcut(FVECTOR x, long offset, long length);
extern DVECTOR xdvcut(DVECTOR x, long offset, long length);

extern void svpaste(SVECTOR y, SVECTOR x, long offset, long length, int overlap);
extern void lvpaste(LVECTOR y, LVECTOR x, long offset, long length, int overlap);
extern void fvpaste(FVECTOR y, FVECTOR x, long offset, long length, int overlap);
extern void dvpaste(DVECTOR y, DVECTOR x, long offset, long length, int overlap);

extern LVECTOR xsvtol(SVECTOR x);
extern FVECTOR xsvtof(SVECTOR x);
extern DVECTOR xsvtod(SVECTOR x);
extern DVECTOR xfvtod(FVECTOR x);
extern SVECTOR xdvtos(DVECTOR x);
extern LVECTOR xdvtol(DVECTOR x);
extern FVECTOR xdvtof(DVECTOR x);

extern SVECTOR xsvset(short *data, long length);
extern SVECTOR xsvsetnew(short *data, long length);
extern LVECTOR xlvset(long *data, long length);
extern LVECTOR xlvsetnew(long *data, long length);
extern FVECTOR xfvset(float *data, long length);
extern FVECTOR xfvsetnew(float *data, long length);
extern DVECTOR xdvset(double *data, long length);
extern DVECTOR xdvsetnew(double *data, long length);

#define xsvnums(length, value) xsvinit((long)(value), 0, (long)(length))
#define xlvnums(length, value) xlvinit((long)(value), 0, (long)(length))
#define xfvnums(length, value) xfvinit((float)(value), 0.0, (float)(length))
#define xdvnums(length, value) xdvinit((double)(value), 0.0, (double)(length))

#define xsvzeros(length) xsvnums(length, 0)
#define xlvzeros(length) xlvnums(length, 0)
#define xfvzeros(length) xfvnums(length, 0.0)
#define xdvzeros(length) xdvnums(length, 0.0)

#define xsvones(length) xsvnums(length, 1)
#define xlvones(length) xlvnums(length, 1)
#define xfvones(length) xfvnums(length, 1.0)
#define xdvones(length) xdvnums(length, 1.0)

#define xsvnull() xsvalloc(0)
#define xlvnull() xlvalloc(0)
#define xfvnull() xfvalloc(0)
#define xdvnull() xdvalloc(0)

#define xsvrinums(length, value) xsvriinit((long)(value), 0, (long)(length))
#define xlvrinums(length, value) xlvriinit((long)(value), 0, (long)(length))
#define xfvrinums(length, value) xfvriinit((float)(value), 0.0, (float)(length))
#define xdvrinums(length, value) xdvriinit((double)(value), 0.0, (double)(length))

#define xsvrizeros(length) xsvrinums(length, 0)
#define xlvrizeros(length) xlvrinums(length, 0)
#define xfvrizeros(length) xfvrinums(length, 0.0)
#define xdvrizeros(length) xdvrinums(length, 0.0)

#define xsvriones(length) xsvrinums(length, 1)
#define xlvriones(length) xlvrinums(length, 1)
#define xfvriones(length) xfvrinums(length, 1.0)
#define xdvriones(length) xdvrinums(length, 1.0)

#define svnums(x, length, value) svinit(x, (long)(value), 0, (long)(length))
#define lvnums(x, length, value) lvinit(x, (long)(value), 0, (long)(length))
#define fvnums(x, length, value) fvinit(x, (float)(value), 0.0, (float)(length))
#define dvnums(x, length, value) dvinit(x, (double)(value), 0.0, (double)(length))

#define svzeros(x, length) svnums(x, length, 0)
#define lvzeros(x, length) lvnums(x, length, 0)
#define fvzeros(x, length) fvnums(x, length, 0.0)
#define dvzeros(x, length) dvnums(x, length, 0.0)

#define svones(x, length) svnums(x, length, 1)
#define lvones(x, length) lvnums(x, length, 1)
#define fvones(x, length) fvnums(x, length, 1.0)
#define dvones(x, length) dvnums(x, length, 1.0)

#define svinums(x, length, value) sviinit(x, (long)(value), 0, (long)(length))
#define lvinums(x, length, value) lviinit(x, (long)(value), 0, (long)(length))
#define fvinums(x, length, value) fviinit(x, (float)(value), 0.0, (float)(length))
#define dvinums(x, length, value) dviinit(x, (double)(value), 0.0, (double)(length))

#define svizeros(x, length) svinums(x, length, 0.0)
#define lvizeros(x, length) lvinums(x, length, 0.0)
#define fvizeros(x, length) fvinums(x, length, 0.0)
#define dvizeros(x, length) dvinums(x, length, 0.0)

#define sviones(x, length) svinums(x, length, 1.0)
#define lviones(x, length) lvinums(x, length, 1.0)
#define fviones(x, length) fvinums(x, length, 1.0)
#define dviones(x, length) dvinums(x, length, 1.0)

#endif /* __VECTOR_H */
