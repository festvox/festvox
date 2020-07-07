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
/*  VQ Labeling and Calculation of Weights and Covariances           */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"

typedef struct CONDITION_STRUCT {
    long dim;
    char *cbookfile;
    char *weightfile;
    char *covfile;
    XBOOL ldist_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {48, NULL, NULL, NULL, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 7
OPTION option_struct[] ={
    {"-dim", NULL, "dimension of vector", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-cbookfile", NULL, "codebook file", "cbookfile",
	 NULL, TYPE_STRING, &cond.cbookfile, XFALSE},
    {"-weightfile", NULL, "get weight matrix file", "weightfile",
	 NULL, TYPE_STRING, &cond.weightfile, XFALSE},
    {"-covfile", NULL, "get covariance matrix file", "covfile",
	 NULL, TYPE_STRING, &cond.covfile, XFALSE},
    {"-ldist", NULL, "allowing large distance", NULL,
	 NULL, TYPE_BOOLEAN, &cond.ldist_flag, XFALSE},
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
    long cri, bri, ci, ri, cori, coci;
    long num = 0;
    double dist, mindist;
    LMATRIX labelmat = NODATA;
    DVECTOR centroid = NODATA;
    DVECTOR cepvec = NODATA;
    DVECTOR weight = NODATA;
    DMATRIX cbook = NODATA;
    DMATRIX cepmat = NODATA;
    DMATRIX cov = NODATA;
    DMATRIX weightmat = NODATA;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++) {
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN) {
	    getargfile(argv[i], &fc, &options_struct);
	}
    }

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "VQ Labeling and Calculation of Weights and Covariances\n");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    // error check
    if (strnone(cond.cbookfile)) {
	fprintf(stderr, "need code book file\n");
	exit(1);
    } else {
	if ((cbook = xreaddmatrix(cond.cbookfile, cond.dim, 0)) == NODATA) {
	    fprintf(stderr, "Can't read code book file\n");
	    exit(1);
	}
    }
    if ((cepmat = xreaddmatrix(options_struct.file[0].name,
			       cond.dim, 0)) == NODATA) {
	fprintf(stderr, "Can't read input file\n");
	exit(1);
    }
    if (cbook->col != cepmat->col) {
	fprintf(stderr, "different dimension\n");
	exit(1);
    }

    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Input File: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Output File: %s\n", options_struct.file[1].name);
    }

    // memory allocation
    labelmat = xlmzeros(cepmat->row, 1);

    // labeling
    for (cri = 0; cri < cepmat->row; cri++) {	// cep file
	mindist = 99999.9;
	for (bri = 0; bri < cbook->row; bri++) {	// code book
	    //calculate distance between cri-th cepstrum and bri-th centroid
	    for (ci = 0, dist = 0.0; ci < cepmat->col; ci++) {
	      	dist += pow((cepmat->data[cri][ci] - cbook->data[bri][ci]), 2.0);
	    }
	    //	    mindist = MIN(dist, mindist);
	    if (dist < mindist) mindist = dist;
	    if (mindist == dist) {
		num = bri;	// code book(centroid) number
	    }
	}
	if (mindist > 99999.0 && cond.ldist_flag == XFALSE) {
	    fprintf(stderr, "error: distance is too large\n");
	    exit(1);
	}
	labelmat->data[cri][0] = num;
	//	printf("%ld, %ld\n", cri, num);
    }

    // calculate both weight and covariance matrix inter class
    if (!strnone(cond.covfile) || !strnone(cond.weightfile)) {
	// memory allocation
	weight = xdvzeros(cbook->row);
	cov = xdmzeros((cond.dim * cbook->row), cond.dim);
	for (ri = 0; ri < cepmat->row; ri++) {
	    cepvec = xdmextractrow(cepmat, ri);
	    num = labelmat->data[ri][0];
	    weight->data[num] += 1.0;
	    centroid = xdmextractrow(cbook, num);
	    dvoper(cepvec, "-", centroid);
	    xdvfree(centroid);
	    for (cori = (num*cond.dim); cori < ((num+1)*cond.dim); cori++) {
		for (coci = 0; coci < cond.dim; coci++) {
		    cov->data[cori][coci]+=cepvec->data[cori-num*cond.dim]*cepvec->data[coci];
		}
	    }
	    xdvfree(cepvec);
	}
	// normalize
	for (ri = 0, num = 0; ri < weight->length; ri++) {
	    num += (long)weight->data[ri];
	    if (weight->data[ri] != 0.0) {
		for (cori = (ri * cond.dim); cori < ((ri+1)*cond.dim); cori++) {
		    for (coci = 0; coci < cond.dim; coci++) {
			cov->data[cori][coci] /= weight->data[ri];
		    }
		}
	    }
	}
	dvscoper(weight, "/", num);

	// write weight file
	if (!strnone(cond.weightfile)) {
	    weightmat = xdmzeros(weight->length, 1);
	    dmcopycol(weightmat, 0, weight);
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "write weight file\n");
	    writedmatrix(cond.weightfile, weightmat, 0);
	    xdmfree(weightmat);
	}
	// write covariance file
	if (!strnone(cond.covfile)) {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "write covariance file\n");
	    writedmatrix(cond.covfile, cov, 0);
	}
	// memory free
	xdvfree(weight);
	xdmfree(cov);
    }

    // memory free
    xdmfree(cbook);
    xdmfree(cepmat);
    // write output (label)file
    writelmatrix(options_struct.file[1].name, labelmat, 0);
    // memory free
    xlmfree(labelmat);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
