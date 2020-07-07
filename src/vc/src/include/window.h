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

#ifndef __WINDOW_H
#define __WINDOW_H

#include "vector.h"

extern void blackmanf(float window[], long length);
extern void blackman(double window[], long length);
extern FVECTOR xfvblackman(long length);
extern DVECTOR xdvblackman(long length);

extern void hammingf(float window[], long length);
extern void hamming(double window[], long length);
extern FVECTOR xfvhamming(long length);
extern DVECTOR xdvhamming(long length);

extern void hanningf(float window[], long length);
extern void hanning(double window[], long length);
extern FVECTOR xfvhanning(long length);
extern DVECTOR xdvhanning(long length);

extern void nblackmanf(float window[], long length);
extern void nblackman(double window[], long length);
extern FVECTOR xfvnblackman(long length);
extern DVECTOR xdvnblackman(long length);

extern void nhammingf(float window[], long length);
extern void nhamming(double window[], long length);
extern FVECTOR xfvnhamming(long length);
extern DVECTOR xdvnhamming(long length);

extern void nhanningf(float window[], long length);
extern void nhanning(double window[], long length);
extern FVECTOR xfvnhanning(long length);
extern DVECTOR xdvnhanning(long length);

extern void nboxcarf(float window[], long length);
extern void nboxcar(double window[], long length);
extern FVECTOR xfvnboxcar(long length);
extern DVECTOR xdvnboxcar(long length);

#endif /* __WINDOW_H */
