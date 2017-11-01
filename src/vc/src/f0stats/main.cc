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
/*  Calculating Statistics of F0                                     */
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
    XBOOL log_flag;
    XBOOL all_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 4
OPTION option_struct[] = {
    {"-log", NULL, "statistics of log F0", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.log_flag, XFALSE},
    {"-all", NULL, "using all values", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.all_flag, XFALSE},
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
    double av, sd;
    DVECTOR f0v = NODATA;
    FILE *fp;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Calculating Statistics of F0");
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

    for (k = 0, av = 0.0, num = 0; k < f0v->length; k++) {
	if (f0v->data[k] > 0.0 || cond.all_flag == XTRUE) {
	    if (cond.log_flag == XTRUE) {
		if (f0v->data[k] > 0.0) av += log(f0v->data[k]);
		else {
		    fprintf(stderr, "Error log(zero)\n");
		    exit(1);
		}
	    } else av += f0v->data[k];
	    num++;
	}
    }
    if (num == 0) {
	fprintf(stderr, "Error: zero F0s\n");
	exit(1);
    }
    av /= (double)num;
    for (k = 0, sd = 0.0; k < f0v->length; k++) {
	if ((f0v->data[k] != 0.0 && finite(f0v->data[k]))
	    || cond.all_flag == XTRUE) {
	    if (cond.log_flag == XTRUE) {
		if (f0v->data[k] > 0.0)
		    sd += (log(f0v->data[k]) - av) * (log(f0v->data[k]) - av);
		else {
		    fprintf(stderr, "Error log(zero)\n");
		    exit(1);
		}
	    } else sd += (f0v->data[k] - av) * (f0v->data[k] - av);
	}
    }
    sd = sqrt(sd / (double)num);

    if ((fp = fopen(options_struct.file[1].name, "wt")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", options_struct.file[1].name);
	exit(1);
    }
    fprintf(fp, "%f\n", av);
    fprintf(fp, "%f\n", sd);
    fclose(fp);

    // memory free
    xdvfree(f0v);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
