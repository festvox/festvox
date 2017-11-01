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
/*  DTW                                                              */
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
#include "dtw_sub.h"

typedef struct CONDITION_STRUCT {
    double shiftm;
    double startm;
    double endm;
    long dim;
    long sdim;
    long ldim;
    char *twffile;
    char *twfdat;
    char *intwf;
    char *dtwofile;
    char *dtwtfile;
    char *dtwotfile;
    char *frmcdfile;
    XBOOL fixed_flag;
    XBOOL getcd_flag;
    XBOOL notwf_flag;
    XBOOL dtworg_flag;
    XBOOL nmsg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {5.0, 0.0, 0.0, 40, 0, 39,
		  NULL, NULL, NULL, NULL, NULL, NULL, NULL,
		  XFALSE, XFALSE, XFALSE, XFALSE, XFALSE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[orgfile]", NULL},
    {"[tarfile]", NULL},
};

#define NUM_OPTION 19
OPTION option_struct[] ={
    {"-shift", NULL, "frame shift [ms]", "shift",
	 NULL, TYPE_DOUBLE, &cond.shiftm, XFALSE},
    {"-start", NULL, "start length [ms]", "start",
	 NULL, TYPE_DOUBLE, &cond.startm, XFALSE},
    {"-end", NULL, "end length [ms]", "end",
	 NULL, TYPE_DOUBLE, &cond.endm, XFALSE},
    {"-dim", NULL, "dimension", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-sdim", NULL, "start dimension", "sdim",
	 NULL, TYPE_LONG, &cond.sdim, XFALSE},
    {"-ldim", NULL, "last dimension", "ldim",
	 NULL, TYPE_LONG, &cond.ldim, XFALSE},
    {"-twffile", NULL, "get time warping function file", "twffile",
	 NULL, TYPE_STRING, &cond.twffile, XFALSE},
    {"-twfdat", NULL, "get time warping function plot file", "twfdat",
	 NULL, TYPE_STRING, &cond.twfdat, XFALSE},
    {"-intwf", NULL, "input time warping function file", "intwf",
	 NULL, TYPE_STRING, &cond.intwf, XFALSE},
    {"-dtwofile", NULL, "get DTWed original file", "dtwofile",
	 NULL, TYPE_STRING, &cond.dtwofile, XFALSE},
    {"-dtwtfile", NULL, "get DTWed target file", "dtwtfile",
	 NULL, TYPE_STRING, &cond.dtwtfile, XFALSE},
    {"-dtwotfile", NULL, "get DTWed joint (org-tar) file", "dtwotfile",
	 NULL, TYPE_STRING, &cond.dtwotfile, XFALSE},
    {"-frmcdfile", NULL, "get frame CD file", "frmcdfile",
	 NULL, TYPE_STRING, &cond.frmcdfile, XFALSE},
    {"-fixed", NULL, "fixed start and end points", NULL,
	 NULL, TYPE_BOOLEAN, &cond.fixed_flag, XFALSE},
    {"-getcd", NULL, "calculate CD", NULL,
	 NULL, TYPE_BOOLEAN, &cond.getcd_flag, XFALSE},
    {"-notwf", NULL, "calculate CD without DTW", "notwf",
	 NULL, TYPE_BOOLEAN, &cond.notwf_flag, XFALSE},
    {"-dtworg", NULL, "DTW original vectors", NULL,
	 NULL, TYPE_BOOLEAN, &cond.dtworg_flag, XFALSE},
    {"-nmsg", NULL, "no message", NULL,
	 NULL, TYPE_BOOLEAN, &cond.nmsg_flag, XFALSE},
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
    long k, j, dim;
    LMATRIX twf = NODATA;
    DMATRIX orgmat = NODATA;
    DMATRIX tarmat = NODATA;
    DMATRIX dtwomat = NODATA;
    DMATRIX dtwtmat = NODATA;
    DMATRIX dtwotmat = NODATA;
    DMATRIX distmat = NODATA;
    FILE *fp;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Dynamic Time Warping");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if (!strnone(cond.frmcdfile)) cond.getcd_flag = XTRUE;
    
    // read inputfile
    if ((orgmat = xreaddmatrix(options_struct.file[0].name,
			       cond.dim, 0)) == NODATA) {
	fprintf(stderr, "DTW: Can't read Original File\n");
	exit(1);
    }
    if ((tarmat = xreaddmatrix(options_struct.file[1].name,
			       cond.dim, 0)) == NODATA) {
	fprintf(stderr, "DTW: Can't read Target File\n");
	exit(1);
    }
    if (cond.nmsg_flag == XFALSE) {
	fprintf(stderr, "Original File [%ld][%ld]: %s\n",
		orgmat->row, orgmat->col, options_struct.file[0].name);
	fprintf(stderr, "Target File [%ld][%ld]: %s\n",
		tarmat->row, tarmat->col, options_struct.file[1].name);
    }

    // error check
    if (cond.ldim >= orgmat->col || cond.ldim >= tarmat->col ||
	cond.sdim > cond.ldim) {
	fprintf(stderr, "sdim or ldim error\n");
	exit(1);
    }
    dim = cond.ldim - cond.sdim + 1;

    // calculate CD or minus likelihood without time warping
    if (cond.notwf_flag == XTRUE) {
	if (orgmat->row != tarmat->row) {
	    fprintf(stderr, "Error different number of frames [%ld][%ld], [%ld][%ld]\n", orgmat->row, orgmat->col, tarmat->row, tarmat->col);
	    exit(1);
	}
	if (!strnone(cond.frmcdfile)) {
	    getcd(orgmat, tarmat, cond.sdim, cond.ldim,
		  cond.frmcdfile);
	} else {
	    getcd(orgmat, tarmat, cond.sdim, cond.ldim);
	}
	// memory free
	xdmfree(orgmat);
	xdmfree(tarmat);
	
	if (cond.nmsg_flag == XFALSE) fprintf(stderr, "done\n");

	exit(0);
    }

    if (strnone(cond.intwf)) {
	// get distance between original and target
	distmat = xget_dist_mat(orgmat, tarmat, cond.sdim,
				cond.ldim, 1, cond.nmsg_flag);
	// write distance file
	if (0) {
	    writedmatrix("dist.mat", distmat, 0);
	    fprintf(stderr, "write distance file [%ld][%ld]: dist.mat\n",
		    distmat->row, distmat->col);
	}
	// DTW
	if (cond.fixed_flag == XTRUE) {
	    if (cond.dtworg_flag == XFALSE) {
		if ((twf = dtw_body_fixed(distmat, cond.nmsg_flag,
					  XFALSE, XFALSE)) == NODATA) {
		    fprintf(stderr, "DTW failed: Can't get twf\n");
		    exit(1);
		}
	    } else {
		if ((twf = dtw_body_asym(distmat, cond.shiftm,
					 0.0, 0.0, cond.nmsg_flag,
					 XFALSE, XFALSE)) == NODATA) {
		    fprintf(stderr, "DTW failed: Can't get twf\n");
		    exit(1);
		}
	    }
	} else {
	    if (cond.dtworg_flag == XFALSE) {
		if ((twf = dtw_body_free(distmat, cond.shiftm,
					 cond.startm, cond.endm,
					 cond.nmsg_flag,
					 XFALSE, XFALSE)) == NODATA) {
		    fprintf(stderr, "DTW failed: Can't get twf\n");
		    exit(1);
		}
	    } else {
		if ((twf = dtw_body_asym(distmat, cond.shiftm,
					 cond.startm, cond.endm,
					 cond.nmsg_flag,
					 XFALSE, XFALSE)) == NODATA) {
		    fprintf(stderr, "DTW failed: Can't get twf\n");
		    exit(1);
		}
	    }
	}
	// memory free
	xdmfree(distmat);
    } else {
	// read input time warping function file
	if ((twf = xreadlmatrix(cond.intwf, 2, 0)) == NODATA) {
	    fprintf(stderr, "Can't read input time warping function file\n");
	    exit(1);
	}
    }
    if (cond.getcd_flag == XFALSE &&
	strnone(cond.dtwofile) && strnone(cond.dtwtfile) &&
	strnone(cond.dtwotfile)) {
	// memory free
	xdmfree(orgmat);	xdmfree(tarmat);
    }

    // write time warping function file
    if (!strnone(cond.twffile)) {
	writelmatrix(cond.twffile, twf, 0);
	if (cond.nmsg_flag == XFALSE)
	    fprintf(stderr, "write time warping function file: %s\n",
		    cond.twffile);
    }
    // write time warping function plot file
    if (!strnone(cond.twfdat)) {
	// open file
	if ((fp = fopen(cond.twfdat, "wt")) == NULL) {
	    fprintf(stderr, "Can't open file\n");
	    exit(1);
	}
	// write file (x:original, y:target)
	for (k = 0; k < twf->row; k++)
	    for (j = 0; j < twf->data[k][1]; j++)
		if (twf->data[k][0] >= 0)
		    fprintf(fp, "%f %f\n", (double)(twf->data[k][0] + j)
			    * cond.shiftm, (double)k * cond.shiftm);
	if (cond.nmsg_flag == XFALSE)
	    fprintf(stderr, "write time warping function plot file: %s\n",
		    cond.twfdat);
	// close file
	fclose(fp);
    }

    // get cepstrum distortion
    if (cond.getcd_flag == XTRUE || !strnone(cond.dtwofile) ||
	!strnone(cond.dtwtfile) || !strnone(cond.dtwotfile)) {
	if (cond.dtworg_flag == XTRUE) {
	    // DTW
	    if ((dtwomat = xget_dtw_mat(orgmat, twf)) == NODATA) {
		fprintf(stderr, "DTW failed matrix (getting CD)\n");
		exit(1);
	    } else if (cond.nmsg_flag == XFALSE)
		fprintf(stderr, "DTWed original matrix [%ld][%ld]\n",
			dtwomat->row, dtwomat->col);
	    // memory free
	    xdmfree(orgmat);
	    // copy
	    dtwtmat = xdmclone(tarmat);
	    // memory free
	    xdmfree(tarmat);
	} else {
	    // DTW
	    if ((dtwomat = xget_dtw_orgmat_dbl(orgmat, twf)) == NODATA) {
		fprintf(stderr, "DTW failed matrix (getting CD)\n");
		exit(1);
	    } else if (cond.nmsg_flag == XFALSE)
		fprintf(stderr, "DTWed original matrix [%ld][%ld]\n",
			dtwomat->row, dtwomat->col);
	    // memory free
	    xdmfree(orgmat);
	    // DTW
	    if ((dtwtmat = xget_dtw_tarmat_dbl(tarmat, twf)) == NODATA) {
		fprintf(stderr, "DTW failed matrix (getting CD)\n");
		exit(1);
	    } else if (cond.nmsg_flag == XFALSE)
		fprintf(stderr, "DTWed target matrix [%ld][%ld]\n",
			dtwtmat->row, dtwtmat->col);
	    // memory free
	    xdmfree(tarmat);
	}

	// calculate distortion
	if (cond.getcd_flag == XTRUE) {
	    // get cepstrum distortion
	    if (!strnone(cond.frmcdfile)) {
		getcd(dtwomat, dtwtmat, cond.sdim, cond.ldim,
		      cond.frmcdfile);
	    } else {
		getcd(dtwomat, dtwtmat, cond.sdim, cond.ldim);
	    }
	}

	// write the DTWed file
	if (!strnone(cond.dtwofile)) {
	    if (cond.nmsg_flag == XFALSE)
		fprintf(stderr, "write DTWed original file\n");
	    writedmatrix(cond.dtwofile, dtwomat, 0);
	}
	if (!strnone(cond.dtwtfile)) {
	    if (cond.nmsg_flag == XFALSE)
		fprintf(stderr, "write DTWed target file\n");
	    writedmatrix(cond.dtwtfile, dtwtmat, 0);
	}
	if (!strnone(cond.dtwotfile)) {
	    if (cond.nmsg_flag == XFALSE)
		fprintf(stderr, "write DTWed joint file\n");
	    dtwotmat = xjoint_matrow(dtwomat, dtwtmat);
	    writedmatrix(cond.dtwotfile, dtwotmat, 0);
	    // memory free
	    xdmfree(dtwotmat);
	}
	// memory free
	xdmfree(dtwomat);
	xdmfree(dtwtmat);
    }

    // memory free
    xlmfree(twf);

    if (cond.nmsg_flag == XFALSE)
	fprintf(stderr, "done\n");

    return 0;
}
