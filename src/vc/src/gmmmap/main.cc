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
/*  GMM Mapping                                                      */
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
#include "../sub/matope_sub.h"

#include "gmmmap_sub.h"

typedef struct CONDITION_STRUCT {
    long xdim;		// dimension (source vector)
    long ydim;		// dimension (target vector)
    char *gmmfile;	// GMM mapping parameter file
    char *wseqfile;
    char *mseqfile;
    char *covfile;
    char *clsseqfile;
    XBOOL dia_flag;	// use diagonal covariance matrix
    XBOOL file_flag;
    XBOOL vit_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {20, 20, NULL, NULL, NULL, NULL, NULL,
		  XFALSE, XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 12
OPTION option_struct[] = {
    {"-xdim", NULL, "dimension of source vector", "xdim",
	 NULL, TYPE_LONG, &cond.xdim, XFALSE},
    {"-ydim", NULL, "dimension of target vector", "ydim",
	 NULL, TYPE_LONG, &cond.ydim, XFALSE},
    {"-gmmfile", NULL, "GMM parameter file", "gmmfile", 
	 NULL, TYPE_STRING, &cond.gmmfile, XFALSE},
    {"-wseqfile", NULL, "get weight sequence file", "wseqfile", 
	 NULL, TYPE_STRING, &cond.wseqfile, XFALSE},
    {"-mseqfile", NULL, "get mean sequence file", "mseqfile", 
	 NULL, TYPE_STRING, &cond.mseqfile, XFALSE},
    {"-covfile", NULL, "get covariance file", "covfile", 
	 NULL, TYPE_STRING, &cond.covfile, XFALSE},
    {"-clsseqfile", NULL, "get class sequence file", "clsseqfile", 
	 NULL, TYPE_STRING, &cond.clsseqfile, XFALSE},
    {"-dia", NULL, "diagonal covariance", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.dia_flag, XFALSE},
    {"-file", NULL, "file processing", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.file_flag, XFALSE},
    {"-vit", NULL, "viterbi search", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.vit_flag, XFALSE},
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
    DVECTOR paramvec = NODATA;
    GMMPARA gmmpara = NODATA;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE) printhelp(options_struct, "GMM Mapping");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Input: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Output: %s\n", options_struct.file[1].name);
	if (cond.xdim == cond.ydim && cond.dia_flag == XTRUE)
	    fprintf(stderr, "Using all diagonal covariances\n");
	else if (cond.dia_flag == XTRUE)
	    fprintf(stderr, "Using XX diagonal covariance\n");
    }

    // parameters for GMM mapping
    gmmpara = xgmmpara(cond.gmmfile, cond.xdim, cond.ydim, NULL,
		       cond.dia_flag, cond.msg_flag);

    // GMM mapping
    if (cond.file_flag == XFALSE) {
	if (cond.vit_flag == XTRUE) {
	    gmmmap_vit(options_struct.file[0].name,
		       options_struct.file[1].name, cond.wseqfile,
		       cond.covfile, cond.clsseqfile, NULL,
		       gmmpara, cond.msg_flag);
	} else {
	    gmmmap(options_struct.file[0].name, options_struct.file[1].name,
		   cond.wseqfile, cond.mseqfile, cond.covfile, NULL,
		   gmmpara, cond.msg_flag);
	}
    } else {
	if (cond.vit_flag == XTRUE) {
	    gmmmap_vit_file(options_struct.file[0].name,
			    options_struct.file[1].name, cond.wseqfile,
			    cond.covfile, cond.clsseqfile, gmmpara,
			    cond.msg_flag);
	} else {
	    gmmmap_file(options_struct.file[0].name,
			options_struct.file[1].name, cond.wseqfile,
			cond.mseqfile, cond.covfile, gmmpara, cond.msg_flag);
	}
    }

    // memory free
    xdvfree(paramvec);
    xgmmparafree(gmmpara);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
