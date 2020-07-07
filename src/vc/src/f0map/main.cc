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
/*  F0 Linear Mapping                                                */
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
    char *ostfile;
    char *tstfile;
    XBOOL log_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {NULL, NULL, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 5
OPTION option_struct[] = {
    {"-ostfile", NULL, "original statistics file", "ostfile", 
	 NULL, TYPE_STRING, &cond.ostfile, XFALSE},
    {"-tstfile", NULL, "target statistics file", "tstfile", 
	 NULL, TYPE_STRING, &cond.tstfile, XFALSE},
    {"-log", NULL, "statistics of log F0", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.log_flag, XFALSE},
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
    long k;
    DVECTOR f0v = NODATA;
    DVECTOR ost = NODATA;
    DVECTOR tst = NODATA;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "F0 Linear Mapping");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Input: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Output: %s\n", options_struct.file[1].name);
    }

    if ((f0v = xdvreadcol_txt(options_struct.file[0].name, 0)) == NODATA) {
	fprintf(stderr, "can't read f0 file\n");
	exit(1);
    }
    if (!strnone(cond.ostfile)) {
	if ((ost = xdvreadcol_txt(cond.ostfile, 0)) == NODATA) {
	    fprintf(stderr, "can't read file: %s\n", cond.ostfile);
	    exit(1);
	} else if (ost->length != 2) {
	    fprintf(stderr, "Error: file format: %s\n", cond.ostfile);
	    exit(1);
	}
    } else {
	fprintf(stderr, "Error: Need -ostfile option\n");
	exit(1);
    }
    if (!strnone(cond.tstfile)) {
	if ((tst = xdvreadcol_txt(cond.tstfile, 0)) == NODATA) {
	    fprintf(stderr, "can't read file: %s\n", cond.tstfile);
	    exit(1);
	} else if (tst->length != 2) {
	    fprintf(stderr, "Error: file format: %s\n", cond.tstfile);
	    exit(1);
	}
    } else {
	fprintf(stderr, "Error: Need -tstfile option\n");
	exit(1);
    }

    if (cond.msg_flag == XTRUE) {
	if (cond.log_flag == XTRUE)
	    fprintf(stderr, "exp((log F0 - %f) / %f * %f + %f)\n",
		    ost->data[0], ost->data[1], tst->data[1], tst->data[0]);
	else
	    fprintf(stderr, "(F0 - %f) / %f * %f + %f\n",
		    ost->data[0], ost->data[1], tst->data[1], tst->data[0]);
    }
    for (k = 0; k < f0v->length; k++) {
	if (f0v->data[k] > 0.0) {
	    if (cond.log_flag == XTRUE) {
		f0v->data[k] = exp((log(f0v->data[k]) - ost->data[0]) /
				   ost->data[1] * tst->data[1] + tst->data[0]);
	    } else {
		f0v->data[k] = (f0v->data[k] - ost->data[0]) /
		    ost->data[1] * tst->data[1] + tst->data[0];
	    }
	}
    }
    writedvector_txt(options_struct.file[1].name, f0v);

    // memory free
    xdvfree(f0v);
    xdvfree(ost);
    xdvfree(tst);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
