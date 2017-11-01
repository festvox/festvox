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

#ifndef __DEFS_H
#define __DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NO_WARNING

#define XBOOL int
#define XTRUE 1
#define XFALSE 0

#ifndef XtRXBOOL 
#define XtRXBOOL XtRInt
#endif
#ifndef XTRUE_STRING
#define XTRUE_STRING "1"
#endif
#ifndef XFALSE_STRING
#define XFALSE_STRING "0"
#endif

#define MAX_LINE	192
#define MAX_PATHNAME	256
#define MAX_MESSAGE	512
#define SUCCESS  	1
#define FAILURE        	0
#define ON		1
#define OFF		0
#define UNKNOWN		(-1)
#define ALITTLE_NUMBER 1.0e-10

#ifdef NUL
#undef NUL
#endif
#define NUL		'\0'

#ifdef M_PI
#define PI		M_PI
#else
#define PI		3.1415926535897932385
#endif

#ifndef NULL
#define NULL		0
#endif

#define NODATA NULL

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define SQUARE(x) ((x) * (x))
#define CSQUARE(xr, xi) ((xr)*(xr)+(xi)*(xi))
#define POW2(p) (1 << (int)(p))
#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define FABS(x) ((x) >= 0.0 ? (x) : -(x))
#define CABS(xr, xi) sqrt((double)(xr)*(double)(xr)+(double)(xi)*(double)(xi))
#define dB(x) (20.0 * log10((double)(x)))
#define dBpow(x) (10.0 * log10((double)(x)))

#define streq(s1, s2) ((s1 != NULL) && (s2 != NULL) && (strcmp((s1), (s2)) == 0) ? 1 : 0)
#define strneq(s1, s2, n) ((s1 != NULL) && (s2 != NULL) && (strncmp((s1), (s2), n) == 0) ? 1 : 0)
#define strveq(s1, s2) ((s1 != NULL) && (s2 != NULL) && (strncmp((s1), (s2), strlen(s2)) == 0) ? 1 : 0)
#define arraysize(array) ((unsigned int)(sizeof(array) / sizeof(array[0])))
#define samesign(a, b) ((a) * (b) >= 0)
#define eqtrue(value) (((value) == XTRUE) ? 1 : 0)
#define strnone(string) (((string) == NULL || *(string) == NUL) ? 1 : 0)
#define strnull strnone

#endif /* __DEFS_H */
