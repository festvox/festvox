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
/*  Subroutine for GMM Mapping                                       */
/*                                                                   */
/*-------------------------------------------------------------------*/

#ifndef __GMMMAP_SUB_H
#define __GMMMAP_SUB_H

typedef struct GMMPARA_STRUCT {
    long clsnum;
    long xdim;
    long ydim;
    XBOOL dia_flag;
    DVECTOR detvec;
    DMATRIX wght;
    DMATRIX xmean;
    DMATRIX xxcov;
    DMATRIX lm;
    DMATRIX dm;
} *GMMPARA;

extern GMMPARA xgmmpara(char *gmmfile, long xdim, long ydim, char *xcovfile,
			 XBOOL dia_flag, XBOOL msg_flag);
extern void xgmmparafree(GMMPARA gmmpara);
extern void get_gmmmappara(DMATRIX lm, DMATRIX dm, DMATRIX xmean,
			   DMATRIX ymean, DMATRIX xxicov, DMATRIX yxcov,
			   DMATRIX yycov, XBOOL dia_flag, XBOOL msg_flag);
extern void gmmmap(char *inf, char *outf, char *wseqf, char *mseqf,
		   char *covf, char *xmseqf, GMMPARA gmmpara, XBOOL msg_flag);
extern void gmmmap_file(char *inf, char *outf, char *wseqf, char *mseqf,
			char *covf, GMMPARA gmmpara, XBOOL msg_flag);
extern void gmmmap_vit(char *inf, char *outf, char *wseqf, char *covf,
		       char *clsseqf, char *xmseqf, GMMPARA gmmpara,
		       XBOOL msg_flag);
extern void gmmmap_vit_file(char *inf, char *outf, char *wseqf, char *covf,
			    char *clsseqf, GMMPARA gmmpara, XBOOL msg_flag);
extern DVECTOR xget_gmmmap_clsvec(DVECTOR x, long clsidx, long ydim,
				  DMATRIX lm, XBOOL dia_flag);
extern DVECTOR xget_gmmmapvec(DVECTOR x, DVECTOR detvec, DMATRIX wghtmat,
			      DMATRIX xmean, DMATRIX xxicov, DMATRIX lm,
			      long ydim, XBOOL dia_flag);
extern DMATRIX xget_gmmmapmat(DMATRIX xm, DVECTOR detvec, DMATRIX wghtmat,
			      DMATRIX xmean, DMATRIX xxicov, DMATRIX lm,
			      long ydim, XBOOL dia_flag);
extern void get_gmmmapmat_file(char *xfile, DVECTOR detvec, DMATRIX wghtmat,
			       DMATRIX xmean, DMATRIX xxicov, DMATRIX lm,
			       long ydim, char *mfile, XBOOL dia_flag,
			       XBOOL msg_flag);
extern DMATRIX xget_gmmmap_wghtseq(DMATRIX xm, DVECTOR detvec,
				   DMATRIX wghtmat, DMATRIX xmean,
				   DMATRIX xxicov, XBOOL dia_flag);
extern DMATRIX xget_gmmmap_wghtseq_vit(DMATRIX xm, DVECTOR detvec,
				       DMATRIX wghtmat, DMATRIX xmean,
				       DMATRIX xxicov, XBOOL dia_flag,
				       LVECTOR clsidxv);
extern void get_gmmmap_wghtseq_file(char *xfile, DVECTOR detvec,
				    DMATRIX wghtmat, DMATRIX xmean,
				    DMATRIX xxicov, char *mfile,
				    XBOOL dia_flag, XBOOL msg_flag);
extern void get_gmmmap_wghtseq_vit_file(char *xfile, DVECTOR detvec,
					DMATRIX wghtmat, DMATRIX xmean,
					DMATRIX xxicov, char *mfile,
					XBOOL dia_flag, LVECTOR clsidxv,
					XBOOL msg_flag);
extern DMATRIX xget_gmmmap_meanseq(DMATRIX xm, DMATRIX lm, long ydim,
				   long clsnum, XBOOL dia_flag);
extern DMATRIX xget_gmmmap_meanseq_vit(DMATRIX xm, DMATRIX lm, long ydim,
				       long clsnum, XBOOL dia_flag,
				       LVECTOR clsidxv);
extern void get_gmmmap_meanseq_file(char *xfile, DMATRIX lm, long xdim,
				    long ydim, long clsnum, char *mfile,
				    XBOOL dia_flag, XBOOL msg_flag);
extern void get_gmmmap_meanseq_vit_file(char *xfile, DMATRIX lm, long xdim,
					long ydim, long clsnum, char *mfile,
					XBOOL dia_flag, LVECTOR clsidxv,
					XBOOL msg_flag);

#endif /* __GMMMAP_SUB_H */
