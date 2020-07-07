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
/*  Subroutine for DTW                                               */
/*                                                                   */
/*-------------------------------------------------------------------*/

#ifndef __DTW_BODY_H
#define __DTW_BODY_H

extern DMATRIX xget_dist_mat(DMATRIX orgmat, DMATRIX tarmat, long startdim,
			     long lastdim, long cep, XBOOL noprint_flag);
extern DMATRIX xget_dtw_mat(DMATRIX orgmat, LMATRIX twfunc);
extern DMATRIX xget_dtw_orgmat_dbl(DMATRIX orgmat, LMATRIX twfunc);
extern DMATRIX xget_dtw_tarmat_dbl(DMATRIX tarmat, LMATRIX twfunc);
extern DMATRIX xget_sumdistmat_asym(DMATRIX distmat, long startl,
				    LMATRICES pathmats);
extern LMATRIX dtw_body_asym(DMATRIX distmat, double shiftm, double startm,
				 double endm, XBOOL noprint_flag,
				 XBOOL sumdistprint_flag, XBOOL sd_flag);
extern LMATRIX xget_twfunc_asym(long ci, LMATRICES pathmats,
				    DMATRIX sumdistmat, XBOOL noprint_flag,
				    XBOOL sumdistprint_flag, XBOOL sd_flag);
extern LMATRIX dtw_body_fixed(DMATRIX distmat, XBOOL noprint_flag,
				  XBOOL sumdistprint_flag, XBOOL sd_flag);
DMATRIX xget_sumdistmat_fixed(DMATRIX distmat, LMATRIX pathmat);
LMATRIX xget_twfunc_fixed(DMATRIX sumdistmat, LMATRIX pathmat);
extern LMATRIX dtw_body_free(DMATRIX distmat, double shiftm, double startm,
				 double endm, XBOOL noprint_flag,
				 XBOOL sumdistprint_flag, XBOOL sd_flag);
extern DMATRIX xget_sumdistmat_free(DMATRIX distmat, long startl,
					LMATRICES pathmats);
extern double get_meandist_free(long ris, long cis, DMATRIX sumdistmat,
				    LMATRICES pathmats);
extern LMATRIX xget_twfunc_free(long ri, long ci, LMATRICES pathmats,
				    DMATRIX sumdistmat, XBOOL noprint_flag,
				    XBOOL sumdistprint_flag, XBOOL sd_flag);
extern void getcd(DMATRIX orgmat, DMATRIX tarmat, long startdim, long lastdim);
extern void getcd(DMATRIX orgmat, DMATRIX tarmat, long startdim, long lastdim,
		  char *outf);

#endif /* __DTW_BODY_H */
