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
/*  Creating Joint Feature Vectors                                   */
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

typedef struct CONDITION_STRUCT {
    long dim1;
    long dim2;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {1, 24, XTRUE, XFALSE};

#define NUM_ARGFILE 3
ARGFILE argfile_struct[] = {
    {"[mat1file]", NULL},
    {"[mat2file]", NULL},
    {"[outfile]", NULL},
};

#define NUM_OPTION 4
OPTION option_struct[] ={
    {"-dim1", NULL, "dimension of vector (mat1file)", "dim1",
	 NULL, TYPE_LONG, &cond.dim1, XFALSE},
    {"-dim2", NULL, "dimension of vector (mat2file)", "dim2",
	 NULL, TYPE_LONG, &cond.dim2, XFALSE},
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
    DMATRIX mat = NODATA;

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
	printhelp(options_struct, "Creating Joint Feature Vectors");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Input 1 File: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Input 2 File: %s\n", options_struct.file[1].name);
	fprintf(stderr, "Output File: %s\n", options_struct.file[2].name);
    }

    if ((mat = xjoint_matrow_file(options_struct.file[0].name,
				  cond.dim1,
				  options_struct.file[1].name,
				  cond.dim2)) == NODATA) {
	    fprintf(stderr, "Can't joint matrix\n");
	    exit(1);
    } else if (cond.msg_flag == XTRUE)
	fprintf(stderr, "got matrix [%ld][%ld]\n", mat->row, mat->col);

    // write output file
    writedmatrix(options_struct.file[2].name, mat, 0);

    // memory free
    xdmfree(mat);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
