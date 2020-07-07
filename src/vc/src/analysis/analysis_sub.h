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
/*          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          */
/*          Date   :  June 2004                                      */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  Subroutine for Speech Analysis                                   */
/*                                                                   */
/*-------------------------------------------------------------------*/

#ifndef __ANALYSIS_SUB_H
#define __ANALYSIS_SUB_H

#define ALPHA 0.42

extern DMATRIX analysis_body(DVECTOR x, double fs, double framem,
			     double shiftm, long fftl);

extern DVECTOR ana_xcutsig(DVECTOR sig, long offset, long length);
extern void ana_zerocheck(DVECTOR sig, double ncoef);
extern DVECTOR ana_xgetfftpow(DVECTOR cx, DVECTOR wxe, long fftl, double pc);
extern double spvec2pow(DVECTOR vec, XBOOL db_flag);
extern DVECTOR xspg2pow_norm(DMATRIX spg);
extern DVECTOR xget_spg2powvec(DMATRIX n2sgram, XBOOL log_flag);
extern DMATRIX xspg2mpmcepg(DMATRIX spg, long order, long fftl,
			    XBOOL power_flag, XBOOL fast_flag);

#endif /* _ANALYSIS_SUB_H_ */
