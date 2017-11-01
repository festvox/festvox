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
/*  VQ with LBG Algorithm                                            */
/*                                    1997/11/26 by S. Nakamura      */
/*                                                                   */
/*  Modified by Tomoki Toda (tomoki@ics.nitech.ac.jp)                */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/

#ifndef __VQ_H
#define __VQ_H

#define MAXBOOK     1
#define SPLIT_COEF1 0.8
#define SPLIT_COEF2 0.2
#define LBG_LOOP    35
#define LBG_EPS     0.0001
#define MAXMEM      500000000
#define streq(s1, s2) ((s1 != NULL) && (s2 != NULL) && (strcmp((s1), (s2)) == 0) ? 1 : 0)

struct codebook {
    int    *vlab;
    int    *vemp;
    double *vpara;
    int    vecno;
    int    vecsize;
    int    vecorder;
    int    dim_s;
    double dist;
};

struct sample {
    double *buff;
    int    nfrms;
};

struct analysis {
    int nts;
};

struct avpara {
    double *avp;
    int    tnum;
};


extern int *ialloc(int cc);
extern double *dalloc(int cc);
extern float *falloc(int cc);
extern void init_vqlabel(struct codebook *bookp, struct sample *smpl);
extern void free_vqlabel(struct codebook *bookp);
extern void lbg(struct analysis *condana, struct codebook *bookp,
		struct sample *smpl, char *outflbl, XBOOL float_flag,
		XBOOL msg_flag);
extern void centroid(struct analysis *condana, struct codebook *bookp,
		     struct sample *smpl);
extern void init_vparam(struct avpara *avpr, struct codebook *bookp);
extern void sum_param(struct avpara *avpr, struct sample *smpl, int no,
		      struct codebook *bookp, struct analysis *condana) ;
extern void set_vparam(struct codebook *bookp, struct sample *smpl,
		       struct analysis *condana, int no1, int no2);
extern void normalize_cb(struct codebook *bookp, int no, struct avpara *avpr);
extern void label(struct analysis *condana, struct codebook *bookp,
		  struct sample *smpl);
extern double dist2(struct analysis *condana, struct codebook *bookp,
		    struct sample *smpl, int no1, int no2);
extern void splitting(struct analysis *condana, struct codebook *bookp,
		      struct sample *smpl);
extern void splitcode(struct analysis *condana, struct codebook *bookp,
		      struct sample *smpl, int k1, int k2, int i1, int i2);
extern double dist3(struct analysis *condana, struct codebook *bookp,
		    struct sample *smpl, int no1, int no2);


#endif /* _VQ_H_ */
