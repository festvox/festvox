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
/*  Frame Extraction Based on Power                                  */
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
    long dim;		// Dimension
    double lp;		// Limit of lower power
    double up;		// Limit of upper power
    char *npowfile;	// Normalized power file
    XBOOL float_flag;	// float
    XBOOL msg_flag;	// print message
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {24, -100.0, 100.0, NULL, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 7
OPTION option_struct[] = {
    {"-dim", NULL, "dimension", "dim", 
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-lp", NULL, "limit of lower power", "lp", 
	 NULL, TYPE_DOUBLE, &cond.lp, XFALSE},
    {"-up", NULL, "limit of upper power", "up", 
	 NULL, TYPE_DOUBLE, &cond.up, XFALSE},
    {"-npowfile", NULL, "normalized power file", "npowfile", 
	 NULL, TYPE_STRING, &cond.npowfile, XFALSE},
    {"-float", NULL, "float format", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.float_flag, XFALSE},
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
    long k, num;
    DVECTOR powvec = NODATA;
    DVECTOR dvec = NODATA;
    FVECTOR fvec = NODATA;
    FILE *ifp, *ofp;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);
    
    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Frame Extraction Based on Power");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    // read normalized power vector
    if (strnone(cond.npowfile)) {
	fprintf(stderr, "Error: Need -npowfile option\n");
	exit(1);
    } else {
	if (cond.float_flag == XFALSE) {
	    if ((powvec = xreaddsignal(cond.npowfile, 0, 0)) == NODATA) {
		fprintf(stderr, "Error: file format: %s\n", cond.npowfile);
		exit(1);
	    }
	} else {
	    if ((powvec = xreadf2dsignal(cond.npowfile, 0, 0)) == NODATA) {
		fprintf(stderr, "Error: file format: %s\n", cond.npowfile);
		exit(1);
	    }
	}
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "power vector [%ld]\n", powvec->length);
    }

    // error check
    if (cond.float_flag == XFALSE)
	num = get_dnum_file(options_struct.file[0].name, cond.dim);
    else
	num = get_fnum_file(options_struct.file[0].name, cond.dim);
    if (num != powvec->length) {
	fprintf(stderr, "Error: no correspondence of the number of frames\n");
	exit(1);
    }
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "input data [%ld][%ld]\n", num, cond.dim);

    // open file
    if ((ifp = fopen(options_struct.file[0].name, "rb")) == NULL) {
	fprintf(stderr, "can't open file: %s\n", options_struct.file[0].name);
	exit(1);
    }
    if ((ofp = fopen(options_struct.file[1].name, "wb")) == NULL) {
	fprintf(stderr, "can't open file: %s\n", options_struct.file[1].name);
	exit(1);
    }
    // memory allocation
    if (cond.float_flag == XFALSE) dvec = xdvalloc(cond.dim);
    else fvec = xfvalloc(cond.dim);
    // extraction frames
    for (k = 0, num = 0; k < powvec->length; k++) {
	if (cond.float_flag == XFALSE)
	    fread(dvec->data, sizeof(double), (int)cond.dim, ifp);
	else
	    fread(fvec->data, sizeof(float), (int)cond.dim, ifp);
	if ((powvec->data[k] >= cond.lp || cond.lp == -100.0) &&
	    (powvec->data[k] <= cond.up || cond.up == 100.0)) {
	    if (cond.float_flag == XFALSE)
		fwrite(dvec->data, sizeof(double), (int)cond.dim, ofp);
	    else
		fwrite(fvec->data, sizeof(float), (int)cond.dim, ofp);
	    num++;
	}
    }
    // close file
    fclose(ifp);
    fclose(ofp);

    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "extracted frame [%ld]->[%ld]\n", powvec->length, num);

    // memory free
    if (cond.float_flag == XFALSE) xdvfree(dvec);
    else xfvfree(fvec);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
