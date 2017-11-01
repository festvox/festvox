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

#ifndef __MATRIX_H
#define __MATRIX_H

#include "vector.h"

typedef struct SMATRIX_STRUCT {
    long row;
    long col;
    short **data;
    short **imag;
} *SMATRIX;

typedef struct LMATRIX_STRUCT {
    long row;
    long col;
    long **data;
    long **imag;
} *LMATRIX;

typedef struct FMATRIX_STRUCT {
    long row;
    long col;
    float **data;
    float **imag;
} *FMATRIX;

typedef struct DMATRIX_STRUCT {
    long row;
    long col;
    double **data;
    double **imag;
} *DMATRIX;

typedef struct SMATRICES_STRUCT {
    long num_matrix;
    SMATRIX *matrix;
} *SMATRICES;

typedef struct LMATRICES_STRUCT {
    long num_matrix;
    LMATRIX *matrix;
} *LMATRICES;

typedef struct FMATRICES_STRUCT {
    long num_matrix;
    FMATRIX *matrix;
} *FMATRICES;

typedef struct DMATRICES_STRUCT {
    long num_matrix;
    DMATRIX *matrix;
} *DMATRICES;

extern LMATRIX xlmalloc(long row, long col);
extern DMATRIX xdmalloc(long row, long col);
extern void xlmfree(LMATRIX matrix);
extern void xdmfree(DMATRIX matrix);

extern void lmialloc(LMATRIX x);
extern void dmialloc(DMATRIX x);
extern void lmifree(LMATRIX x);
extern void dmifree(DMATRIX x);

extern LMATRIX xlmrialloc(long row, long col);
extern DMATRIX xdmrialloc(long row, long col);

extern LMATRICES xlmsalloc(long num);
extern DMATRICES xdmsalloc(long num);
extern void xlmsfree(LMATRICES xs);
extern void xdmsfree(DMATRICES xs);

extern void lmreal(LMATRIX x);
extern void dmreal(DMATRIX x);
extern void lmimag(LMATRIX x);
extern void dmimag(DMATRIX x);

extern LMATRIX xlmnums(long row, long col, long value);
extern DMATRIX xdmnums(long row, long col, double value);
extern void lmnums(LMATRIX mat, long row, long col, long value);
extern void dmnums(DMATRIX mat, long row, long col, double value);
extern void lminums(LMATRIX mat, long row, long col, long value);
extern void dminums(DMATRIX mat, long row, long col, double value);
extern LMATRIX xlmrinums(long row, long col, long value);
extern DMATRIX xdmrinums(long row, long col, double value);

extern DMATRIX xdminitrow(long nrow, double j, double incr, double n);
extern DMATRIX xdminitcol(long ncol, double j, double incr, double n);

extern LVECTOR xlmcutrow(LMATRIX mat, long row, long offset, long length);
extern DVECTOR xdmcutrow(DMATRIX mat, long row, long offset, long length);
extern LVECTOR xlmcutcol(LMATRIX mat, long col, long offset, long length);
extern DVECTOR xdmcutcol(DMATRIX mat, long col, long offset, long length);

extern void lmpasterow(LMATRIX mat, long row, LVECTOR vec,
		       long offset, long length, int overlap);
extern void dmpasterow(DMATRIX mat, long row, DVECTOR vec,
		       long offset, long length, int overlap);
extern void lmpastecol(LMATRIX mat, long col, LVECTOR vec,
		       long offset, long length, int overlap);
extern void dmpastecol(DMATRIX mat, long col, DVECTOR vec,
		       long offset, long length, int overlap);

extern LVECTOR xlmrmax(LMATRIX mat);
extern LVECTOR xdmrmax(DMATRIX mat);
extern LVECTOR xlmrmin(LMATRIX mat);
extern LVECTOR xdmrmin(DMATRIX mat);
extern LVECTOR xlmrextract(LMATRIX mat, LVECTOR idx);
extern DVECTOR xdmrextract(DMATRIX mat, LVECTOR idx);
extern LVECTOR xlmcmax(LMATRIX mat);
extern LVECTOR xdmcmax(DMATRIX mat);
extern LVECTOR xlmcmin(LMATRIX mat);
extern LVECTOR xdmcmin(DMATRIX mat);
extern LVECTOR xlmcextract(LMATRIX mat, LVECTOR idx);
extern DVECTOR xdmcextract(DMATRIX mat, LVECTOR idx);

#define xlmextractrow(mat, k) xlmcutrow(mat, (long)(k), 0, mat->col)
#define xdmextractrow(mat, k) xdmcutrow(mat, (long)(k), 0, mat->col)
#define xlmextractcol(mat, k) xlmcutcol(mat, (long)(k), 0, mat->row)
#define xdmextractcol(mat, k) xdmcutcol(mat, (long)(k), 0, mat->row)

#define lmcopyrow(mat, k, vec) lmpasterow(mat, (long)(k), vec, 0, vec->length, 0)
#define dmcopyrow(mat, k, vec) dmpasterow(mat, (long)(k), vec, 0, vec->length, 0)
#define lmcopycol(mat, k, vec) lmpastecol(mat, (long)(k), vec, 0, vec->length, 0)
#define dmcopycol(mat, k, vec) dmpastecol(mat, (long)(k), vec, 0, vec->length, 0)

#define xlmzeros(row, col) xlmnums(row, col, 0)
#define xdmzeros(row, col) xdmnums(row, col, 0.0)
#define xlmones(row, col) xlmnums(row, col, 1)
#define xdmones(row, col) xdmnums(row, col, 1.0)

#define lmzeros(mat, row, col) lmnums(mat, row, col, 0)
#define dmzeros(mat, row, col) dmnums(mat, row, col, 0.0)
#define lmones(mat, row, col) lmnums(mat, row, col, 1)
#define dmones(mat, row, col) dmnums(mat, row, col, 1.0)

#define lmizeros(mat, row, col) lminums(mat, row, col, 0)
#define dmizeros(mat, row, col) dminums(mat, row, col, 0.0)
#define lmiones(mat, row, col) lminums(mat, row, col, 1)
#define dmiones(mat, row, col) dminums(mat, row, col, 1.0)

#define xlmrizeros(row, col) xlmrinums(row, col, 0)
#define xdmrizeros(row, col) xdmrinums(row, col, 0.0)
#define xlmriones(row, col) xlmrinums(row, col, 1)
#define xdmriones(row, col) xdmrinums(row, col, 1.0)

#define xlmnull() xlmalloc(0, 0)
#define xdmnull() xdmalloc(0, 0)

#endif /* __MATRIX_H */
