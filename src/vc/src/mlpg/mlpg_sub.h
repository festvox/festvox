/*  ---------------------------------------------------------------  */
/*      The HMM-Based Speech Synthesis System (HTS): version 1.1.1   */
/*                        HTS Working Group                          */
/*                                                                   */
/*                   Department of Computer Science                  */
/*                   Nagoya Institute of Technology                  */
/*                                and                                */
/*    Interdisciplinary Graduate School of Science and Engineering   */
/*                   Tokyo Institute of Technology                   */
/*                      Copyright (c) 2001-2003                      */
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
/*                                                                   */
/*    2. Any modifications must be clearly marked as such.           */
/*                                                                   */
/*  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,  */
/*  HTS WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM    */
/*  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL       */
/*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF        */
/*  TECHNOLOGY, HTS WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE    */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY        */
/*  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  */
/*  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS   */
/*  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR          */
/*  PERFORMANCE OF THIS SOFTWARE.                                    */
/*                                                                   */
/*  ---------------------------------------------------------------  */
/*    mlpg.c : speech parameter generation from pdf sequence         */
/*                                                                   */
/*                                    2003/12/26 by Heiga Zen        */
/*  ---------------------------------------------------------------  */

/*********************************************************************/
/*                                                                   */
/*            Nagoya Institute of Technology, Aichi, Japan,          */
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
/*  NAGOYA INSTITUTE OF TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, AND  */
/*  THE CONTRIBUTORS TO THIS WORK DISCLAIM ALL WARRANTIES WITH       */
/*  REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF     */
/*  MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL NAGOYA INSTITUTE  */
/*  OF TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, NOR THE CONTRIBUTORS  */
/*  BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR  */
/*  ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR       */
/*  PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER   */
/*  TORTUOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE    */
/*  OR PERFORMANCE OF THIS SOFTWARE.                                 */
/*                                                                   */
/*********************************************************************/
/*                                                                   */
/*  ML-Based Parameter Generation                                    */
/*                                    2003/12/26 by Heiga Zen        */
/*                                                                   */
/*  Basic functions are extracted from HTS and                       */
/*   modified by Tomoki Toda (tomoki@ics.nitech.ac.jp)               */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          */
/*          Date   :  June 2004                                      */
/*                                                                   */
/*-------------------------------------------------------------------*/

#ifndef __MLPG_SUB_H
#define __MLPG_SUB_H

#define	LENGTH 256
#define	INFTY ((double) 1.0e+38)
#define	INFTY2 ((double) 1.0e+19)
#define	INVINF ((double) 1.0e-38)
#define	INVINF2 ((double) 1.0e-19)

#define	WLEFT 0
#define	WRIGHT 1

#define	abs(x) ((x) > 0.0 ? (x) : -(x))
#define	sign(x) ((x) >= 0.0 ? 1 : -1)
#define	finv(x) (abs(x) <= INVINF2 ? sign(x)*INFTY : (abs(x) >= INFTY2 ? 0 : 1.0/(x)))

typedef struct _DWin {
    int	num;		/* number of static + deltas */
    int **width;	/* width [0..num-1][0(left) 1(right)] */
    double **coef;	/* coefficient [0..num-1][length[0]..length[1]] */
    int maxw[2];	/* max width [0(left) 1(right)] */
} DWin;

typedef struct _PStreamChol {
    int vSize;		// size of ovserved vector
    int order;		// order of cepstrum
    int T;		// number of frames
    int width;		// width of WSW
    DWin dw;
    double **mseq;	// sequence of mean vector
    double **ivseq;	// sequence of invarsed covariance vector
    double ***ifvseq;	// sequence of invarsed full covariance vector
    double **R;		// WSW[T][range]
    double *r;		// WSM [T]
    double *g;		// g [T]
    double **c;		// parameter c
} PStreamChol;


typedef struct MLPGPARA_STRUCT {
    DVECTOR ov;
    DVECTOR iuv;
    DVECTOR iumv;
    DVECTOR flkv;
    DMATRIX stm;
    DMATRIX dltm;
    DMATRIX pdf;
    DVECTOR detvec;
    DMATRIX wght;
    DMATRIX mean;
    DMATRIX cov;
    LVECTOR clsidxv;
    DVECTOR clsdetv;
    DMATRIX clscov;
    double vdet;
    DVECTOR vm;
    DVECTOR vv;
    DVECTOR var;
} *MLPGPARA;

extern MLPGPARA xmlpgpara(long dim, long dim2, long dnum, long clsnum,
			  char *dynwinf, char *wseqf, char *mseqf, char *covf,
			  char *stf, char *vmfile, char *vvfile,
			  PStreamChol *pst, XBOOL dia_flag, XBOOL msg_flag);
extern MLPGPARA xmlpgpara_vit(long dim, long dim2, long dnum, long clsnum,
			      char *dynwinf, char *cseqf, char *mseqf,
			      char *covf, char *vmfile, char *vvfile,
			      PStreamChol *pst, XBOOL dia_flag,
			      XBOOL msg_flag);
extern void xmlpgparafree(MLPGPARA param);
extern void get_cseq_mlpgpara(MLPGPARA param, char *cseqf, long dnum);
extern void get_stm_mlpgpara(MLPGPARA param, char *mseqf,
			     long dim, long dim2, long dnum);
extern void get_stm_mlpgpara(MLPGPARA param, char *stf, char *wseqf,
			     char *mseqf, long dim, long dim2, long dnum,
			     long clsnum, XBOOL msg_flag);
extern void get_gmm_mlpgpara(MLPGPARA param, PStreamChol *pst, char *dynwinf,
			     char *covf, long dim, long dim2, long dnum,
			     long clsnum, XBOOL dia_flag);
extern void get_gv_mlpgpara(MLPGPARA param, char *vmfile, char *vvfile,
			    long dim2, XBOOL msg_flag);
extern double get_like_pdfseq(long dim, long dim2, long dnum, long clsnum,
			      MLPGPARA param, FILE *wfp, FILE *mfp,
			      XBOOL dia_flag, XBOOL vit_flag);
extern double get_like_pdfseq_vit(long dim, long dim2, long dnum, long clsnum,
				  MLPGPARA param, FILE *wfp, FILE *mfp,
				  XBOOL dia_flag);
extern double get_like_gv(long dim2, long dnum, MLPGPARA param);
extern void sm_mvav(DMATRIX mat, long hlen);
extern void get_dltmat(DMATRIX mat, DWin *dw, int dno, DMATRIX dmat);


extern double *dcalloc(int x, int xoff);
extern double **ddcalloc(int x, int y, int xoff, int yoff);
extern double ***dddcalloc(int x, int y, int z, int xoff, int yoff, int zoff);

/***********************************/
/* ML using Choleski decomposition */
/***********************************/
/* Diagonal Covariance Version */
extern void InitDWin(PStreamChol *pst, char *dynwinf, char *accwinf);
extern void InitPStreamChol(PStreamChol *pst, char *dynwinf, char *accwinf,
			    int order, int T);
extern void mlgparaChol(DMATRIX pdf, PStreamChol *pst, DMATRIX mlgp);
extern void mlpgChol(PStreamChol *pst);
extern void calc_R_and_r(PStreamChol *pst, const int m);
extern void Choleski(PStreamChol *pst);
extern void Choleski_forward(PStreamChol *pst);
extern void Choleski_backward(PStreamChol *pst, const int m);
/* Full Covariance Version */
extern void InitPStreamCholFC(PStreamChol *pst, char *dynwinf, char *accwinf,
			      int order, int T);
extern void mlgparaCholFC(DMATRIX pdf, PStreamChol *pst, DMATRIX mlgp);
extern void mlpgCholFC(PStreamChol *pst);
extern void calc_R_and_r_FC(PStreamChol *pst);
extern void CholeskiFC(PStreamChol *pst);
extern void Choleski_forwardFC(PStreamChol *pst);
extern void Choleski_backwardFC(PStreamChol *pst);

/**********************************/
/* ML Considering Global Variance */
/**********************************/
extern void varconv(double **c, const int m, const int T, const double var);
extern void calc_varstats(double **c, const int m, const int T,
			  double *av, double *var, double *dif);
/* Diagonal Covariance Version */
extern void mlgparaGrad(DMATRIX pdf, PStreamChol *pst, DMATRIX mlgp,
			const int max, double th, double e, double alpha,
			DVECTOR vm, DVECTOR vv, XBOOL nrmflag, XBOOL extvflag);
extern void mlpgGrad(PStreamChol *pst, const int max, double th, double e,
		     double alpha, DVECTOR vm, DVECTOR vv, XBOOL nrmflag);
extern void calc_grad(PStreamChol *pst, const int m);
extern void calc_vargrad(PStreamChol *pst, const int m, double alpha, double n,
			 double vm, double vv);

#endif /* __MLPG_SUB_H */
