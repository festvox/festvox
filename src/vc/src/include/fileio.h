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

#ifndef __FILEIO_H
#define __FILEIO_H

#include "vector.h"
#include "matrix.h"

extern long getfilesize(const char *filename, int headlen);

extern void swapshort(short *data, long length);
extern void swaplong(long *data, long length);
extern void swapfloat(float *data, long length);
extern void swapdouble(double *data, long length);

extern void freadshort(short *data, long length, int swap, FILE *fp);
extern void freadlong(long *data, long length, int swap, FILE *fp);
extern void freadfloat(float *data, long length, int swap, FILE *fp);
extern void freaddouble(double *data, long length, int swap, FILE *fp);
extern void freadshorttod(double *data, long length, int swap, FILE *fp);
extern void fwriteshort(short *data, long length, int swap, FILE *fp);
extern void fwritelong(long *data, long length, int swap, FILE *fp);
extern void fwritefloat(float *data, long length, int swap, FILE *fp);
extern void fwritedouble(double *data, long length, int swap, FILE *fp);
extern void fwritedoubletos(double *data, long length, int swap, FILE *fp);

extern SVECTOR xreadssignal(const char *filename, int headlen, int swap);
extern FVECTOR xreadfsignal(const char *filename, int headlen, int swap);
extern DVECTOR xdvreadssignal(const char *filename, int headlen, int swap);
extern DVECTOR xreaddsignal(const char *filename, int headlen, int swap);
extern DVECTOR xreadf2dsignal(const char *filename, int headlen, int swap);
extern void writessignal(const char *filename, SVECTOR vector, int swap);
extern void dvwritessignal(const char *filename, DVECTOR vector, int swap);
extern void writedsignal(const char *filename, DVECTOR vector, int swap);
extern void writed2fsignal(const char *filename, DVECTOR vector, int swap);

extern LMATRIX xreadlmatrix(const char *filename, long ncol, int swap);
extern DMATRIX xreaddmatrix(const char *filename, long ncol, int swap);
extern DMATRIX xreadf2dmatrix(const char *filename, long ncol, int swap);
extern void writelmatrix(const char *filename, LMATRIX mat, int swap);
extern void writedmatrix(const char *filename, DMATRIX mat, int swap);
extern void writed2fmatrix(const char *filename, DMATRIX mat, int swap);

extern long getfilesize_txt(const char *filename);
extern int readdvector_txt(const char *filename, DVECTOR vector);
extern DVECTOR xreaddvector_txt(const char *filename);
extern int writedvector_txt(const char *filename, DVECTOR vector);

extern int getnumrow_txt(const char *filename);
extern int getnumcol_txt(const char *filename);
extern int fgetcol_txt(char *buf, int col, FILE *fp);
extern int sgetcol(char *buf, int col, const char *line);
extern int fgetline(char *buf, FILE *fp);
extern int getline(char *buf);
extern const char *gets0(const char *buf, int size);
extern int sscanf_setup(char *line, char *name, char *value);
extern int dvreadcol_txt(const char *filename, int col, DVECTOR vector);
extern DVECTOR xdvreadcol_txt(const char *filename, int col);

#define fgetnumrow getnumrow_txt
#define fgetnumcol getnumcol_txt
#define fgetcol fgetcol_txt

extern void svdump(SVECTOR vec);
extern void lvdump(LVECTOR vec);
extern void fvdump(FVECTOR vec);
extern void dvdump(DVECTOR vec);

extern void svfdump(SVECTOR vec, FILE *fp);
extern void lvfdump(LVECTOR vec, FILE *fp);
extern void fvfdump(FVECTOR vec, FILE *fp);
extern void dvfdump(DVECTOR vec, FILE *fp);

extern void lmfdump(LMATRIX mat, FILE *fp);
extern void dmfdump(DMATRIX mat, FILE *fp);

#ifdef VARARGS
extern void dvnfdump();
#else
extern void dvnfdump(FILE *fp, DVECTOR vec, ...);
#endif

#define xreadsvector(filename, swap) xreadssignal((filename), 0, (swap))
#define writesvector writessignal

#define xreaddvector(filename, swap) xreaddsignal((filename), 0, (swap))
#define writedvector writedsignal

#define read_dvector_txt readdvector_txt
#define write_dvector_txt writedvector_txt

#define getsiglen(filename, headlen, type) (getfilesize(filename, headlen) / (long)sizeof(type))
#define getisiglen(filename, headlen) (getsiglen(filename, headlen, int))
#define getssiglen(filename, headlen) (getsiglen(filename, headlen, short))
#define getlsiglen(filename, headlen) (getsiglen(filename, headlen, long))
#define getfsiglen(filename, headlen) (getsiglen(filename, headlen, float))
#define getdsiglen(filename, headlen) (getsiglen(filename, headlen, double))

extern void check_dir(const char *file);
extern long get_dnum_file(const char *file, long dim);
extern long get_fnum_file(const char *file, long dim);
extern long get_lnum_file(const char *file, long dim);


#endif /* __FILEIO_H */
