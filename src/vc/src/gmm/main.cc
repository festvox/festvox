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
/*  Training GMM on Joint Probability                                */
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

typedef struct CONDITION_STRUCT {
    long dim;
    long itnum;
    double thresh;
    char *wghtfile;
    char *meanfile;
    char *covfile;
    XBOOL dia_flag;
    char *bcovfile;
    double flcoef;
    XBOOL file_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {50, 20, 0.000001, NULL, NULL, NULL, XFALSE, NULL, 0.001,
		  XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 4
ARGFILE argfile_struct[] = {
    {"[infile]", NULL},
    {"[outwghtfile]", NULL},
    {"[outmeanfile]", NULL},
    {"[outcovfile]", NULL},
};

#define NUM_OPTION 12
OPTION option_struct[] ={
    {"-dim", NULL, "dimension of vector", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-itnum", NULL, "max number of iteration", "itnum",
	 NULL, TYPE_LONG, &cond.itnum, XFALSE},
    {"-thresh", NULL, "threshold of likelihood", "thresh",
	 NULL, TYPE_DOUBLE, &cond.thresh, XFALSE},
    {"-wghtfile", NULL, "initial weight file", "wghtfile",
	 NULL, TYPE_STRING, &cond.wghtfile, XFALSE},
    {"-meanfile", NULL, "initial mean file", "meanfile",
	 NULL, TYPE_STRING, &cond.meanfile, XFALSE},
    {"-covfile", NULL, "initial covariance file", "covfile",
	 NULL, TYPE_STRING, &cond.covfile, XFALSE},
    {"-dia", NULL, "use diagonal covariance matrix", NULL,
	 NULL, TYPE_BOOLEAN, &cond.dia_flag, XFALSE},
    {"-bcovfile", NULL, "basic covariance file", "bcovfile",
	 NULL, TYPE_STRING, &cond.bcovfile, XFALSE},
    {"-flcoef", NULL, "flooring coefficient", "flcoef",
	 NULL, TYPE_DOUBLE, &cond.flcoef, XFALSE},
    {"-file", NULL, "file processing (output temporary files)", NULL,
	 NULL, TYPE_BOOLEAN, &cond.file_flag, XFALSE},
    {"-nmsg", NULL, "no message", NULL,
	 NULL, TYPE_BOOLEAN, &cond.msg_flag, XFALSE},
    {"-help", "-h", "display this message", NULL,
	 NULL, TYPE_BOOLEAN, &cond.help_flag, XFALSE},
};

OPTIONS options_struct = {
    NULL, 1, NUM_OPTION, option_struct, NUM_ARGFILE, argfile_struct,
};

// main
int main(int argc, char *argv[])
{
    int i, fc;
    long k, ri, dnum, clsnum;
    double like, blike;
    char tmpstr[MAX_MESSAGE] = "";
    char tmpgauss[MAX_MESSAGE] = "";
    char tmpgamma[MAX_MESSAGE] = "";
    DVECTOR detvec = NODATA;
    DVECTOR sumgvec = NODATA;
    DMATRIX wghtmat = NODATA;
    DMATRIX meanmat = NODATA;
    DMATRIX covmat = NODATA;
    DMATRIX bcovmat = NODATA;
    DMATRIX cpcovmat = NODATA;
    DMATRIX gaussm = NODATA;
    DMATRIX gammam = NODATA;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Training GMM on Joint Probability");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Input File: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Output Weight file: %s\n",
		options_struct.file[1].name);
	fprintf(stderr, "Output Mean file: %s\n", options_struct.file[2].name);
	fprintf(stderr, "Output Covariance File: %s\n",
		options_struct.file[3].name);
    }

    // number of input vectors
    dnum = get_dnum_file(options_struct.file[0].name, cond.dim);

    // read initial parameter files
    if (strnone(cond.wghtfile)) {
	fprintf(stderr, "You must use initial weight file (-wghtfile)\n");
	exit(1);
    } else {
	if ((wghtmat = xreaddmatrix(cond.wghtfile, 1, 0)) == NODATA) {
	    fprintf(stderr, "Can't read init weight file\n");
	    exit(1);
	}
    }
    if (strnone(cond.meanfile)) {
	fprintf(stderr, "You must use initial mean file (-meanfile)\n");
	exit(1);
    } else {
	if ((meanmat = xreaddmatrix(cond.meanfile,
				    cond.dim, 0)) == NODATA) {
	    fprintf(stderr, "Can't read init mean file\n");
	    exit(1);
	}
    }
    if (strnone(cond.covfile)) {
	fprintf(stderr, "You must use initial covariance file (-covfile)\n");
	exit(1);
    } else {
	if ((covmat = xreaddmatrix(cond.covfile,
				    cond.dim, 0)) == NODATA) {
	    fprintf(stderr, "Can't read init cov file\n");
	    exit(1);
	}
    }
    if (strnone(cond.bcovfile)) {
	bcovmat = xdmzeros(covmat->col, covmat->col);
    } else {
	if ((bcovmat = xreaddmatrix(cond.bcovfile,
				    cond.dim, 0)) == NODATA) {
	    fprintf(stderr, "Can't read basic cov file\n");
	    exit(1);
	}
	if (bcovmat->row != bcovmat->col || bcovmat->row != meanmat->col) {
	  fprintf(stderr, "Format error -bcovf\n");
	  exit(1);
	}
	for (ri = 0; ri < covmat->col; ri++)
	    bcovmat->data[ri][ri] *= FABS(cond.flcoef);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "Flooring coefficient: %f\n", FABS(cond.flcoef));
    }

    // error check
    if ((clsnum = wghtmat->row) != meanmat->row) {
	fprintf(stderr, "different class between weight and mean, %ld, %ld\n",
		wghtmat->row, meanmat->row);
	exit(1);
    }
    if (covmat->row != (cond.dim * clsnum)) {
	fprintf(stderr, "different covfile\n");
	exit(1);
    }

    if (cond.file_flag == XTRUE) {
	strcpy(tmpgauss, options_struct.file[1].name);
	strcpy(tmpgamma, options_struct.file[1].name);
	strcat(tmpgauss, ".tmp_gauss");
	strcat(tmpgamma, ".tmp_gamma");
	if (cond.msg_flag == XTRUE) {
	    fprintf(stderr, "Temporary files\n");
	    fprintf(stderr, "   %s\n", tmpgauss);
	    fprintf(stderr, "   %s\n", tmpgamma);
	}
    } else {
	gaussm = xdmalloc(dnum, clsnum);
	gammam = xdmalloc(dnum, clsnum);
    }

    // -dia option : get diagonal covariance
    if (cond.dia_flag == XTRUE) {
	get_diamat_jde(covmat);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "get diagonal covariance matrix\n");
    }
    // flooring
    floor_diamat(covmat, bcovmat);

    // get determinant and inverse matrix (covmat = inverse)
    if ((detvec = xget_detvec_mat2inv_jde(clsnum, covmat, cond.dia_flag))
	== NODATA) {
	fprintf(stderr, "Can't calculate determinant\n");
	exit(1);
    }
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "calculating likelihood ...\n");
    // get gauss matrix
    if (cond.file_flag == XTRUE) {
	get_gaussmat_jde_file(detvec, tmpgauss, options_struct.file[0].name,
			      dnum, cond.dim, wghtmat, meanmat, covmat,
			      cond.dia_flag);
	// calculate likelihood log(p(x,y))
	like = get_likelihood_file(dnum, clsnum, tmpgauss);
    } else {
	get_gaussmat_jde_file(detvec, gaussm, options_struct.file[0].name,
			      dnum, cond.dim, wghtmat, meanmat, covmat,
			      cond.dia_flag);
	// calculate likelihood log(p(x,y))
	like = get_likelihood(dnum, clsnum, gaussm);
    }
    if (cond.msg_flag == XTRUE) {
	printf("#initial likelihood\n");
	printf("%f\n", like);
    }
    blike = like;

    // estimating parameter with EM algorithm
    for (k = 0; k < cond.itnum; k++) {
	// get gamma matrix
	if (cond.file_flag == XTRUE) {
	    sumgvec = xget_sumgvec_gammamat_file(tmpgauss, dnum, clsnum,
						 tmpgamma);
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "wrote gammamat [%ld][%ld]\n", dnum, clsnum);
	} else {
	    sumgvec = xget_sumgvec_gammamat(gaussm, dnum, clsnum, gammam);
	}

	// estimate weight matrix(vector)
	estimate_weight(wghtmat, sumgvec);

	// estimate mean matrix
	if (cond.file_flag == XTRUE)
	    estimate_mean_file(options_struct.file[0].name, cond.dim,
			       meanmat, tmpgamma, sumgvec, clsnum);
	else
	    estimate_mean_file(options_struct.file[0].name, cond.dim,
			       meanmat, gammam, sumgvec, clsnum);

	// estimate covariance matrix
	if (cond.file_flag == XTRUE)
	    estimate_cov_jde_file(options_struct.file[0].name, cond.dim,
				  meanmat, covmat, tmpgamma, sumgvec, clsnum,
				  cond.dia_flag);
	else
	    estimate_cov_jde_file(options_struct.file[0].name, cond.dim,
				  meanmat, covmat, gammam, sumgvec, clsnum,
				  cond.dia_flag);
	// flooring
	floor_diamat(covmat, bcovmat);

	// memory free
	xdvfree(detvec);
	xdvfree(sumgvec);

	// copy covariance matrix
	cpcovmat = xdmclone(covmat);
	
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "%ld-th iteration done\n", k+1);

	// get determinant and inverse matrix (covmat = inverse)
	if ((detvec = xget_detvec_mat2inv_jde(clsnum, covmat, cond.dia_flag))
	    == NODATA) {
	    fprintf(stderr, "Can't calculate determinant\n");
	    exit(1);
	}
	// get gauss matrix
	if (cond.file_flag == XTRUE) {
	    get_gaussmat_jde_file(detvec, tmpgauss,
				  options_struct.file[0].name, dnum, cond.dim,
				  wghtmat, meanmat, covmat, cond.dia_flag);
	    // calculate likelihood log(p(x,y))
	    like = get_likelihood_file(dnum, clsnum, tmpgauss);
	} else {
	    get_gaussmat_jde_file(detvec, gaussm, options_struct.file[0].name,
				  dnum, cond.dim, wghtmat, meanmat, covmat,
				  cond.dia_flag);
	    // calculate likelihood log(p(x,y))
	    like = get_likelihood(dnum, clsnum, gaussm);
	}
	if (cond.msg_flag == XTRUE) {
	    printf("#[%ld] likelihood\n", k+1);
	    printf("%f\n", like);
	}

	if (like >= blike) {
	    // write weight file
	    writedmatrix(options_struct.file[1].name, wghtmat, 0);
	    // write mean file
	    writedmatrix(options_struct.file[2].name, meanmat, 0);
	    // write covariance file
	    writedmatrix(options_struct.file[3].name, cpcovmat, 0);
	    // memory free
	    xdmfree(cpcovmat);	    cpcovmat = NODATA;
	} else {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "likelihood is decrease\n");
	    break;
	}
	if ((like - blike) < cond.thresh) {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "Training done\n");
	    break;
	}
	blike = like;
    }
    if (cond.file_flag == XTRUE) {
	sprintf(tmpstr, "rm -f %s %s", tmpgauss, tmpgamma);
	system(tmpstr);
    } else {
	xdmfree(gaussm);
	xdmfree(gammam);
    }
    
    // memory free
    xdvfree(detvec);
    xdmfree(wghtmat);
    xdmfree(meanmat);
    xdmfree(covmat);
    xdmfree(bcovmat);
    if (cpcovmat != NODATA) xdmfree(cpcovmat);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");
    return 0;
}
