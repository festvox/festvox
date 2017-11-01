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
/*  GMM subroutine                                                   */
/*                                                                   */
/*-------------------------------------------------------------------*/

#ifndef __GMM_BODY_SUB_H
#define __GMM_BODY_SUB_H

extern DVECTOR xget_paramvec(DMATRIX weightmat, DMATRIX meanmat,
			     DMATRIX covmat, long ydim, XBOOL dia_flag);
extern DVECTOR xget_paramvec_yx(DMATRIX weightmat, DMATRIX meanmat,
				DMATRIX covmat, long ydim, XBOOL dia_flag);
extern void get_paramvec(DVECTOR param, DMATRIX weight, DMATRIX xmean,
			 DMATRIX ymean, DMATRIX xxcov, DMATRIX yxcov,
			 DMATRIX yycov, XBOOL dia_flag, XBOOL msg_flag);
extern void get_paramvec(DVECTOR param, long xdim, long ydim, DMATRIX weight,
			 DMATRIX mean, DMATRIX cov, XBOOL dia_flag,
			 XBOOL msg_flag);
extern long get_clsnum(DVECTOR param, long xdim, long ydim, XBOOL dia_flag);
extern DVECTOR xget_detvec_diamat2inv(DMATRIX covmat);
extern DVECTOR xget_detvec_mat2inv(long clsnum, DMATRIX covmat,
				   XBOOL dia_flag);
extern DVECTOR xget_detvec_mat2inv_jde(long clsnum, DMATRIX covmat,
				       XBOOL dia_flag);
extern double cal_xmcxmc(DVECTOR x, DVECTOR m, DMATRIX c);
extern double cal_xmcxmc(long clsidx, DVECTOR x, DMATRIX mm, DMATRIX cm);
extern double get_gauss_full(long clsidx, DVECTOR vec, DVECTOR detvec,
			     DMATRIX weightmat, DMATRIX meanvec,
			     DMATRIX invcovmat);
extern double get_gauss_dia(double det, double weight, DVECTOR vec,
			    DVECTOR meanvec, DVECTOR invcovvec);
extern double get_gauss_dia(long clsidx, DVECTOR vec, DVECTOR detvec,
			    DMATRIX weightmat, DMATRIX meanmat,
			    DMATRIX invcovmat);
extern double get_gauss_jde_dia(long clsidx, DVECTOR vec, DVECTOR detvec,
				DMATRIX weightmat, DMATRIX meanmat,
				DMATRIX invcovmat);
extern DVECTOR xget_gaussvec_jde(DVECTOR vec, DVECTOR detvec,
				 DMATRIX weightmat, DMATRIX meanmat,
				 DMATRIX invcovmat, XBOOL dia_flag);
extern DVECTOR xget_gaussvec_dia(DVECTOR vec, DVECTOR detvec,
				 DMATRIX weightmat, DMATRIX meanmat,
				 DMATRIX invcovmat);
extern DVECTOR xget_gaussvec_full(DVECTOR vec, DVECTOR detvec,
				  DMATRIX weightmat, DMATRIX meanmat,
				  DMATRIX invcovmat);
extern void get_gaussmat_jde_file(DVECTOR detvec, DMATRIX gaussm,
				  char *vecmatfile, long dnum, long dim,
				  DMATRIX weightmat, DMATRIX meanmat,
				  DMATRIX invcovmat, XBOOL dia_flag);
extern void get_gaussmat_jde_file(DVECTOR detvec, char *gaussmatfile,
				  char *vecmatfile, long dnum, long dim,
				  DMATRIX weightmat, DMATRIX meanmat,
				  DMATRIX invcovmat, XBOOL dia_flag);
extern DVECTOR xget_gammavec(DVECTOR gaussvec);
extern DVECTOR xget_sumgvec_gammamat(DMATRIX gaussm, long row, long col,
				     DMATRIX gammam);
extern DVECTOR xget_sumgvec_gammamat_file(char *gaussmatfile, long row,
					  long col, char *gammamatfile);
extern void estimate_weight(DMATRIX weightmat, DVECTOR sumgvec);
extern void estimate_mean_file(char *vecmatfile, long dim, DMATRIX meanmat,
			       DMATRIX gammam, DVECTOR sumgvec, long clsnum);
extern void estimate_mean_file(char *vecmatfile, long dim, DMATRIX meanmat,
			       char *gammamatfile, DVECTOR sumgvec,
			       long clsnum);
extern void estimate_cov_jde_file(char *vecmatfile, long dim, DMATRIX meanmat,
				  DMATRIX covmat, DMATRIX gammam,
				  DVECTOR sumgvec, long clsnum,
				  XBOOL dia_flag);
extern void estimate_cov_jde_file(char *vecmatfile, long dim, DMATRIX meanmat,
				  DMATRIX covmat, char *gammamatfile,
				  DVECTOR sumgvec, long clsnum,
				  XBOOL dia_flag);
extern double get_likelihood(long datanum, long clsnum, DMATRIX gaussmat);
extern double get_likelihood_file(long datanum, long clsnum,
				  char *gaussmatfile);

#endif /* __GMM_BODY_SUB_H */
