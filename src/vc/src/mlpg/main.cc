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
/*  TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE    */
/*  OR PERFORMANCE OF THIS SOFTWARE.                                 */
/*                                                                   */
/*********************************************************************/
/*                                                                   */
/*          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          */
/*          Date   :  June 2004                                      */
/*                                                                   */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  ML-Based Parameter Generation                                    */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "../sub/matope_sub.h"
#include "../sub/gmm_sub.h"
#include "mlpg_sub.h"

typedef struct CONDITION_STRUCT {
    long dim;
    long clsnum;
    long maxitnum;
    double thresh;
    double vthresh;
    char *dynwinf;
    char *stfile;
    char *vmfile;
    char *vvfile;
    char *flkfile;
    XBOOL dia_flag;
    XBOOL sm_flag;
    XBOOL last_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {28, 1, 20, 0.00001, 0.1, NULL, NULL, NULL, NULL, NULL,
		  XFALSE, XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 4
ARGFILE argfile_struct[] = {
    {"[inweightfile]", NULL},
    {"[inmeanfile]", NULL},
    {"[incovfile]", NULL},
    {"[outfile]", NULL},
};

#define NUM_OPTION 15
OPTION option_struct[] ={
    {"-dim", NULL, "dimension of joint vector", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-clsnum", NULL, "number of mixtures", "clsnum",
	 NULL, TYPE_LONG, &cond.clsnum, XFALSE},
    {"-maxitnum", NULL, "maximum number of iterations", "maxitnum",
	 NULL, TYPE_LONG, &cond.maxitnum, XFALSE},
    {"-thresh", NULL, "threshold of increasing likelihood", "thresh",
	 NULL, TYPE_DOUBLE, &cond.thresh, XFALSE},
    {"-vthresh", NULL, "threshold of variance", "vthresh",
	 NULL, TYPE_DOUBLE, &cond.vthresh, XFALSE},
    {"-dynwinf", NULL, "window file for dynamic feature", "dynwinf",
	 NULL, TYPE_STRING, &cond.dynwinf, XFALSE},
    {"-stfile", NULL, "static sequence files", "stfile",
	 NULL, TYPE_STRING, &cond.stfile, XFALSE},
    {"-vmfile", NULL, "mean of variance files", "vmfile",
	 NULL, TYPE_STRING, &cond.vmfile, XFALSE},
    {"-vvfile", NULL, "variance of variance files", "vvfile",
	 NULL, TYPE_STRING, &cond.vvfile, XFALSE},
    {"-flkfile", NULL, "frame-based likelihood files", "flkfile",
	 NULL, TYPE_STRING, &cond.flkfile, XFALSE},
    {"-dia", NULL, "using diagonal covariance", NULL,
	 NULL, TYPE_BOOLEAN, &cond.dia_flag, XFALSE},
    {"-sm", NULL, "smoothing by moving average", NULL,
	 NULL, TYPE_BOOLEAN, &cond.sm_flag, XFALSE},
    {"-last", NULL, "using variance after iteration", NULL,
	 NULL, TYPE_BOOLEAN, &cond.last_flag, XFALSE},
    {"-nmsg", NULL, "no message", NULL,
	 NULL, TYPE_BOOLEAN, &cond.last_flag, XFALSE},
    {"-help", "-h", "display this message", NULL,
	 NULL, TYPE_BOOLEAN, &cond.help_flag, XFALSE},
};

OPTIONS options_struct = {
    NULL, 1, NUM_OPTION, option_struct, NUM_ARGFILE, argfile_struct,
};


int main(int argc, char *argv[])
{
    int i, fc;
    long dim2, dnum, cnt;
    double like, blike, bvlike, vlike;
    PStreamChol pst;
    MLPGPARA param = NODATA;
    FILE *wfp = NULL, *mfp = NULL;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);
    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);
    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "ML-Based Parameter Generation");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    // check options
    if (strnone(cond.dynwinf)) {	// delta window
	fprintf(stderr, "Error need -dynwinf option\n");
	exit(1);
    }
    dim2 = cond.dim / 2;	// dimension of static feature
    if (dim2 * 2 != cond.dim) {
	fprintf(stderr, "Error dimension: %ld\n", cond.dim);
	exit(1);
    }
    // Number of frames
    //    weight sequence [dnum * clsnum][1]
    //    means [dnum * clsnum][dim]
    dnum = get_dnum_file(options_struct.file[0].name, cond.clsnum);
    if (dnum != get_dnum_file(options_struct.file[1].name,
			      cond.dim * cond.clsnum)) {
	fprintf(stderr,
		"Error: different number of frames of input features\n");
	exit(1);
    }

    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "dynamic feature window: %s\n", cond.dynwinf);
	fprintf(stderr, "Input weight: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Input mean: %s\n", options_struct.file[1].name);
	fprintf(stderr, "Input covariance: %s\n", options_struct.file[2].name);
	fprintf(stderr, "Output file: %s\n", options_struct.file[3].name);
	fprintf(stderr, "Dimension of feature %ld [(st)%ld (dlt)%ld]\n",
		cond.dim, dim2, dim2);
	fprintf(stderr, "Number of frames: %ld\n", dnum);
	fprintf(stderr, "Number of mixtures: %ld\n", cond.clsnum);
	if (cond.sm_flag == XTRUE)
	    fprintf(stderr, "applying smoothing\n");
    }

    // memory allocation
    param = xmlpgpara(cond.dim, dim2, dnum, cond.clsnum, cond.dynwinf,
		      options_struct.file[0].name,
		      options_struct.file[1].name,
		      options_struct.file[2].name,
		      cond.stfile, cond.vmfile, cond.vvfile,
		      &pst, cond.dia_flag, cond.msg_flag);

    // open files
    if ((wfp = fopen(options_struct.file[0].name, "rb")) == NULL) {
	fprintf(stderr, "Can't read file: %s\n", options_struct.file[0].name);
	exit(1);
    }
    if ((mfp = fopen(options_struct.file[1].name, "rb")) == NULL) {
	fprintf(stderr, "Can't read file: %s\n", options_struct.file[1].name);
	exit(1);
    }

    if (cond.msg_flag == XTRUE) {
	if (cond.dia_flag == XTRUE &&
	    param->vm != NODATA && param->vv != NODATA) {
	    fprintf(stderr, "# ML considering gloval variance\n");
	}
	printf("#Number of Mixtures: %ld\n", cond.clsnum);
	printf("#Static Features: [%ld][%ld]\n",
	       param->stm->row, param->stm->col);
	if (param->vm != NODATA && param->vv != NODATA)
	    printf("#Iter.	Likelihood	Increase	GV Likelihood	Increase\n");
	else printf("#Iter.	Likelihood	Increase\n");
    }

    // EM algorithm
    cnt = 0;	blike = -INFTY;	bvlike = -INFTY;
    while (1) {
	// calculating dynamic feature sequence
	get_dltmat(param->stm, &pst.dw, 1, param->dltm);

	// likelihood for PDF sequence
	like = get_like_pdfseq(cond.dim, dim2, dnum, cond.clsnum, param,
			       wfp, mfp, cond.dia_flag, XFALSE);

	// likelihood on global variance
	vlike = get_like_gv(dim2, dnum, param);

	// print likelihoods
	if (cond.msg_flag == XTRUE) {
	    printf("%ld", cnt);
	    if (like < -1.0e+10) printf("	---	---");
	    else if (FABS(like - blike) > 1.0e+10)
		printf("	%f	---", like);
	    else printf("	%f	%f", like, like - blike);
	    if (param->vm != NODATA && param->vv != NODATA) {
		if (vlike < -1.0e+10) printf("	---	---");
		else if (FABS(vlike - bvlike) > 1.0e+10)
		    printf("	%f	---", vlike);
		else printf("	%f	%f", vlike, vlike - bvlike);
	    }
	    printf("\n");
	}

	// comparing likelihoods
	if (FABS(like - blike) < cond.thresh) break;
	blike = like;
	bvlike = vlike;

	// parameter generation based on ML
	if (cond.dia_flag == XTRUE) {
	    if (cond.last_flag == XFALSE &&
		param->vm != NODATA && param->vv != NODATA)
		mlgparaGrad(param->pdf, &pst, param->stm, 1000, cond.vthresh,
			    1.0e-3, -1.0, param->vm, param->vv, XTRUE,
			    XFALSE);
	    else mlgparaChol(param->pdf, &pst, param->stm);
	} else mlgparaCholFC(param->pdf, &pst, param->stm);
	if (cond.sm_flag == XTRUE) sm_mvav(param->stm, 1);
	cnt++;

	if (cnt > cond.maxitnum && cond.maxitnum >= 0) break;
    }
    // close file
    fclose(wfp);	fclose(mfp);

    if (cond.last_flag == XTRUE &&
	cond.dia_flag == XTRUE && param->vm != NODATA && param->vv != NODATA) {
	mlgparaGrad(param->pdf, &pst, param->stm, 1000, cond.vthresh, 1.0e-3,
		    -1.0, param->vm, param->vv, XTRUE, XFALSE);
	if (cond.sm_flag == XTRUE) sm_mvav(param->stm, 1);
    }

    // write file
    writedmatrix(options_struct.file[3].name, param->stm, 0);

    if (cond.msg_flag == XTRUE)
	if (like < blike) printf("#likelihood decreases\n");

    if (!strnone(cond.flkfile)) writedsignal(cond.flkfile, param->flkv, 0);

    // memory free
    xmlpgparafree(param);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
