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
/*  Constructing Parameter File for GMM Mapping                      */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "../sub/gmm_sub.h"

typedef struct CONDITION_STRUCT {
    long dim;
    long ydim;
    XBOOL dia_flag;
    XBOOL yx_flag;
    XBOOL inv_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {40, 20, XFALSE, XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 4
ARGFILE argfile_struct[] = {
    {"[weightfile]", NULL},
    {"[meanfile]", NULL},
    {"[covfile]", NULL},
    {"[outfile]", NULL},
};

#define NUM_OPTION 7
OPTION option_struct[] ={
    {"-dim", NULL, "dimension of joint vector", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-ydim", NULL, "dimension of vector Y", "ydim",
	 NULL, TYPE_LONG, &cond.ydim, XFALSE},
    {"-dia", NULL, "use diagonal covariance matrix", NULL,
	 NULL, TYPE_BOOLEAN, &cond.dia_flag, XFALSE},
    {"-yx", NULL, "parameter for inverse conversion, Y-to-X", NULL,
	 NULL, TYPE_BOOLEAN, &cond.yx_flag, XFALSE},
    {"-inv", NULL, "output weight, mean and cov files from param file", NULL,
	 NULL, TYPE_BOOLEAN, &cond.inv_flag, XFALSE},
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
    long clsnum;
    DVECTOR paramvec = NODATA;
    DMATRIX weightmat = NODATA;
    DMATRIX meanmat = NODATA;
    DMATRIX covmat = NODATA;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct,
		  "Constructing Parameter File for GMM Mapping");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if (cond.dim <= cond.ydim) {
	fprintf(stderr, "Error dimension parameters\n");
	exit(1);
    } else if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Joint vector [%ld] = [X(%ld) Y(%ld)]\n",
		cond.dim, cond.dim - cond.ydim, cond.ydim);
    }

    if (cond.inv_flag == XFALSE) {
	if (cond.msg_flag == XTRUE) {
	    fprintf(stderr, "weight File: %s\n", options_struct.file[0].name);
	    fprintf(stderr, "mean File: %s\n", options_struct.file[1].name);
	    fprintf(stderr, "covariance File: %s\n",
		    options_struct.file[2].name);
	    fprintf(stderr, "Output File: %s\n", options_struct.file[3].name);
	}
	// read initial parameter file
	if ((weightmat = xreaddmatrix(options_struct.file[0].name, 1, 0))
	    == NODATA) {
	    fprintf(stderr, "Can't read file: %s\n",
		    options_struct.file[0].name);
	    exit(1);
	}
	if ((meanmat = xreaddmatrix(options_struct.file[1].name, cond.dim, 0))
	    == NODATA) {
	    fprintf(stderr, "Can't read file: %s\n",
		    options_struct.file[1].name);
	    exit(1);
	}
	if ((covmat = xreaddmatrix(options_struct.file[2].name, cond.dim, 0))
	    == NODATA) {
	    fprintf(stderr, "Can't read file: %s\n",
		    options_struct.file[1].name);
	    exit(1);
	}
	if (cond.msg_flag == XTRUE) {
	    fprintf(stderr, "Read weight vectors [%ld][%ld]\n",
		    weightmat->row, weightmat->col);
	    fprintf(stderr, "Read mean vectors [%ld][%ld]\n",
		    meanmat->row, meanmat->col);
	    fprintf(stderr, "Read covariance matrices [%ld][%ld]\n",
		    covmat->row, covmat->col);
	}

	if (cond.yx_flag == XFALSE)
	    paramvec = xget_paramvec(weightmat, meanmat, covmat, cond.ydim,
				     cond.dia_flag);
	else
	    paramvec = xget_paramvec_yx(weightmat, meanmat, covmat, cond.ydim,
					cond.dia_flag);
	writedvector(options_struct.file[3].name, paramvec, 0);
	if (cond.msg_flag == XTRUE) {
	    if (cond.yx_flag == XFALSE)
		fprintf(stderr, "Write parameter vector: [%ld]\n",
			paramvec->length);
	    else
		fprintf(stderr, "Write parameter vector (YX): [%ld]\n",
			paramvec->length);
	}
    } else {
	if (cond.msg_flag == XTRUE) {
	    fprintf(stderr, "Input parameter File: %s\n",
		    options_struct.file[3].name);
	    fprintf(stderr, "Output weight File: %s\n",
		    options_struct.file[0].name);
	    fprintf(stderr, "Output mean File: %s\n",
		    options_struct.file[1].name);
	    fprintf(stderr, "Output covariance File: %s\n",
		    options_struct.file[2].name);
	}
	// read GMM parameter file
	if ((paramvec = xreaddvector(options_struct.file[3].name, 0))
	    == NODATA) {
	    fprintf(stderr, "Error GMM parameter; %s\n",
		    options_struct.file[3].name);
	    exit(1);
	}
	clsnum = get_clsnum(paramvec, cond.dim - cond.ydim, cond.ydim,
			    cond.dia_flag);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "Number of classes: %ld\n", clsnum);
	// memory allocation
	weightmat = xdmalloc(clsnum, 1);
	meanmat = xdmalloc(clsnum, cond.dim);
	covmat = xdmzeros(clsnum * cond.dim, cond.dim);
	get_paramvec(paramvec, cond.dim - cond.ydim, cond.ydim, weightmat,
		     meanmat, covmat, cond.dia_flag, XTRUE);
	writedmatrix(options_struct.file[0].name, weightmat, 0);
	writedmatrix(options_struct.file[1].name, meanmat, 0);
	writedmatrix(options_struct.file[2].name, covmat, 0);
    }

    // memory free
    xdmfree(weightmat);
    xdmfree(meanmat);
    xdmfree(covmat);
    xdvfree(paramvec);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");
    return 0;
}
