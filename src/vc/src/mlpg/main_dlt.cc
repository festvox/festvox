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
/*  Calculation of Dynamic Feature                                   */
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
#include "mlpg_sub.h"

typedef struct CONDITION_STRUCT {
    long dim;
    char *dynwinf;
    XBOOL jnt_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {24, NULL, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[infile]", NULL},
    {"[outfile]", NULL},
};

#define NUM_OPTION 5
OPTION option_struct[] ={
    {"-dim", NULL, "dimension", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-dynwinf", NULL, "window file for dynamic feature", "dynwinf",
	 NULL, TYPE_STRING, &cond.dynwinf, XFALSE},
    {"-jnt", NULL, "output joint matrix", NULL,
	 NULL, TYPE_BOOLEAN, &cond.jnt_flag, XFALSE},
    {"-nmsg", NULL, "no message", NULL,
	 NULL, TYPE_BOOLEAN, &cond.msg_flag, XFALSE},
    {"-help", "-h", "display this message", NULL,
	 NULL, TYPE_BOOLEAN, &cond.help_flag, XFALSE},
};

OPTIONS options_struct = {
    NULL, 1, NUM_OPTION, option_struct, NUM_ARGFILE, argfile_struct,
};


int main(int argc, char *argv[])
{
    int i, fc;
    DMATRIX stm = NODATA, dltm = NODATA, jntm = NODATA;
    PStreamChol pst;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);
    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);
    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Calculation of Dynamic Feature");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    // read data
    if ((stm = xreaddmatrix(options_struct.file[0].name, cond.dim, 0))
	== NODATA) {
	fprintf(stderr, "Can't read covariance file\n");
	exit(1);
    }
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "read matrix[%ld][%ld]\n", stm->row, stm->col);

    // read window file
    if (!strnone(cond.dynwinf)) {
	InitPStreamChol(&pst, cond.dynwinf, NULL, (int)1, (int)1);
    } else {
	fprintf(stderr, "Error: need -dynwinf option\n");
	exit(1);
    }
    // calculating dynamic feature sequence
    dltm = xdmalloc(stm->row, stm->col);
    get_dltmat(stm, &pst.dw, 1, dltm);

    // write file
    if (cond.jnt_flag == XTRUE) {
	jntm = xjoint_matrow(stm, dltm);
	writedmatrix(options_struct.file[1].name, jntm, 0);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "write joint matrix[%ld][%ld]\n",
		    jntm->row, jntm->col);
	xdmfree(jntm);
    } else {
	writedmatrix(options_struct.file[1].name, dltm, 0);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "write delta matrix[%ld][%ld]\n",
		    dltm->row, dltm->col);
    }

    // memory free
    xdmfree(stm);	xdmfree(dltm);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
