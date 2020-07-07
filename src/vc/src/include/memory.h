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

#ifndef __MEMORY_H
#define __MEMORY_H

extern char *safe_malloc(unsigned int nbytes);
extern char *safe_realloc(char *p, unsigned int nbytes);

extern int **imatalloc(int row, int col);
extern void imatfree(int **mat, int row);

extern short **smatalloc(int row, int col);
extern void smatfree(short **mat, int row);

extern long **lmatalloc(int row, int col);
extern void lmatfree(long **mat, int row);

extern float **fmatalloc(int row, int col);
extern void fmatfree(float **mat, int row);

extern double **dmatalloc(int row, int col);
extern void dmatfree(double **mat, int row);

extern char *strclone(const char *string);

#define xalloc(n, type) (type *)safe_malloc((unsigned)(n)*sizeof(type))
#define xrealloc(p, n, type) (type *)safe_realloc((char *)(p),(unsigned)(n)*sizeof(type))
#define xfree(p) {free((char *)(p));(p)=NULL;}

#define arrcpy(p1, p2, n, type) memmove((char *)(p1),(char *)(p2),(unsigned)(n)*sizeof(type))

#define strrepl(s1, s2) {if ((s1) != NULL) xfree(s1); (s1) = (((s2) != NULL) ? strclone(s2) : NULL);}
#define strreplace strrepl

#define xsmatalloc(row, col) smatalloc((int)(row), (int)(col))
#define xlmatalloc(row, col) lmatalloc((int)(row), (int)(col))
#define xfmatalloc(row, col) fmatalloc((int)(row), (int)(col))
#define xdmatalloc(row, col) dmatalloc((int)(row), (int)(col))

#define xsmatfree(x, row) smatfree(x, (int)(row))
#define xlmatfree(x, row) lmatfree(x, (int)(row))
#define xfmatfree(x, row) fmatfree(x, (int)(row))
#define xdmatfree(x, row) dmatfree(x, (int)(row))

#endif /* __MEMORY_H */
