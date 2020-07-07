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
/*  Converting Covariance Matrices to Diagonal Covariance Vectors    */
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
    XBOOL dia2cov_flag;
    XBOOL sd_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {24, XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[infile]", NULL},
    {"[outfile]", NULL},
};

#define NUM_OPTION 5
OPTION option_struct[] ={
    {"-dim", NULL, "dimension of vector", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-dia2cov", NULL, "change diagonal vectors into diagonal matrices", NULL,
	 NULL, TYPE_BOOLEAN, &cond.dia2cov_flag, XFALSE},
    {"-sd", NULL, "standard deviation", NULL,
	 NULL, TYPE_BOOLEAN, &cond.sd_flag, XFALSE},
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
    long ri, ci, cls = 0;
    DMATRIX mat1 = NODATA;
    DMATRIX mat2 = NODATA;

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
	printhelp(options_struct, "Converting Covariance Matrices to Diagonal Covariance Vectors");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    // read input file
    if ((mat1 = xreaddmatrix(options_struct.file[0].name,
			     cond.dim, 0)) == NODATA) {
	fprintf(stderr, "Can't read input file\n");
    } else {
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "Input matrix [%ld][%ld]\n", mat1->row, mat1->col);
	if (cond.dia2cov_flag == XFALSE) {
	    cls = mat1->row / mat1->col;
	} else {
	    cls = mat1->row;
	}
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "Number of Class: %ld\n", cls);
    }
    // change matrix
    if (cond.dia2cov_flag == XFALSE) {
	mat2 = xdmalloc(cls, mat1->col);
	if (cond.sd_flag == XTRUE && cond.msg_flag == XTRUE)
	    fprintf(stderr, "Convert to standard deviation\n");
	for (ri = 0; ri < cls; ri++) {
	    for (ci = 0; ci < mat1->col; ci++) {
		if (cond.sd_flag == XFALSE)
		    mat2->data[ri][ci] = mat1->data[ci + ri * mat1->col][ci];
		else
		    mat2->data[ri][ci] = sqrt(mat1->data[ci + ri * mat1->col][ci]);
	    }
	}
    } else {
	mat2 = xdmzeros(cls * mat1->col, mat1->col);
	for (ri = 0; ri < cls; ri++) {
	    for (ci = 0; ci < mat1->col; ci++) {
		mat2->data[ci + ri * mat1->col][ci] = mat1->data[ri][ci];
	    }
	}
    }
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "Output matrix [%ld][%ld]\n", mat2->row, mat2->col);
    // write output file
    writedmatrix(options_struct.file[1].name, mat2, 0);
    // memory free
    xdmfree(mat1);
    xdmfree(mat2);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
