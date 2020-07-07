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
#include <string.h>

#include "../include/defs.h"
#include "../include/memory.h"

#define MEMORY_SERIES

char *safe_malloc(unsigned int nbytes)
{
    char *p;

    if (nbytes <= 0) {
	nbytes = 1;
    }

    p = (char *)malloc(nbytes);

    if (p == NULL) {
	fprintf(stderr, "can't malloc %d bytes\n", nbytes);
	exit(-1);
    }

    return p;
}

char *safe_realloc(char *p, unsigned int nbytes)
{
    if (nbytes <= 0) {
	nbytes = 1;
    }

    if (p == NULL) {
	return safe_malloc(nbytes);
    }

#ifdef EXIST_REALLOC_BUG
    p = (char *)realloc(p, nbytes + 1);	/* reason of +1 is realloc's bug */
#else
    p = (char *)realloc(p, nbytes);
#endif

    if (p == NULL) {
	fprintf(stderr, "can't realloc %d bytes\n",nbytes);
	exit(-1);
    }

    return p;
}

int **imatalloc(int row, int col)
{
    int i;
    int **mat;
	
    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, int *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, int);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, int);
    }
#endif

    return mat;
}

void imatfree(int **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

short **smatalloc(int row, int col)
{
    int i;
    short **mat;

    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, short *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, short);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, short);
    }
#endif

    return mat;
}

void smatfree(short **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

long **lmatalloc(int row, int col)
{
    int i;
    long **mat;
	
    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, long *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, long);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, long);
    }
#endif

    return mat;
}

void lmatfree(long **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

float **fmatalloc(int row, int col)
{
    int i;
    float **mat;

    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, float *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, float);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, float);
    }
#endif

    return mat;
}

void fmatfree(float **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

double **dmatalloc(int row, int col)
{
    int i;
    double **mat;
	
    row = MAX(row, 1);
    col = MAX(col, 1);
	
    mat = xalloc(row, double *);

#ifdef MEMORY_SERIES
    *mat = xalloc(row * col, double);
    for (i = 0; i < row; i++) {
	*(mat + i) = *mat + i * col;
    }
#else 
    for (i = 0; i < row; i++) {
	*(mat + i) = xalloc(col, double);
    }
#endif

    return mat;
}

void dmatfree(double **mat, int row)
{
    row = MAX(row, 1);

#ifdef MEMORY_SERIES
    xfree(*mat);
#else
    {
	int i;
	for (i = 0; i < row; i++) {
	    xfree(*(mat + i));
	}
    }
#endif

    xfree(mat);
}

char *strclone(const char *string)
{
    char *buf;

    if (string == NULL)
	return NULL;

    buf = xalloc((strlen(string) + 1), char);
    strcpy(buf, string);

    return buf;
}
