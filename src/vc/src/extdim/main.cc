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
/*  Component Extraction of Specific Dimension                       */
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
    long sd;
    long ed;
    long sf;
    long ef;
    long jf;
    char *chinf;
    XBOOL txt_flag;
    XBOOL eprt_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {25, 0, -1, 0, -1, 0, NULL, XFALSE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 11
OPTION option_struct[] ={
    {"-dim", NULL, "dimension of vector", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-sd", NULL, "start dim", "sd",
	 NULL, TYPE_LONG, &cond.sd, XFALSE},
    {"-ed", NULL, "end dim", "ed",
	 NULL, TYPE_LONG, &cond.ed, XFALSE},
    {"-sf", NULL, "start frame", "sf",
	 NULL, TYPE_LONG, &cond.sf, XFALSE},
    {"-ef", NULL, "end frame", "ef",
	 NULL, TYPE_LONG, &cond.ef, XFALSE},
    {"-jf", NULL, "jump frame", "jf",
	 NULL, TYPE_LONG, &cond.jf, XFALSE},
    {"-chinf", NULL, "channel file", "chinf",
	 NULL, TYPE_STRING, &cond.chinf, XFALSE},
    {"-txt", NULL, "text output", NULL,
	 NULL, TYPE_BOOLEAN, &cond.txt_flag, XFALSE},
    {"-eprt", NULL, "e print", NULL,
	 NULL, TYPE_BOOLEAN, &cond.eprt_flag, XFALSE},
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
    long k, l, nk;
    long sr, er, sc, ec;
    DMATRIX mat = NODATA;
    DVECTOR chiv = NODATA;
    FILE *fp;
    
    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct,
		  "Component Extraction of Specific Dimension");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if ((mat = xreaddmatrix(options_struct.file[0].name, cond.dim, 0))
	== NODATA) {
	fprintf(stderr, "Can't read Input File\n");
	exit(1);
    } else if (cond.msg_flag == XTRUE)
	fprintf(stderr, "read Input matrix [%ld][%ld]\n", mat->row, mat->col);

    if (0) {	// Error check
	if (cond.sf < 0 || cond.ef >= mat->row || cond.sf > cond.ef) {
	    printf("Error: %s\n", options_struct.file[0].name);
	    exit(1);
	}
    }

    sr = MIN(MAX(cond.sf, 0), mat->row - 1);		// 0 <= sr < row
    if (cond.ef < 0) cond.ef = mat->row - 1;
    er = MAX(MIN(cond.ef + 1, mat->row), sr + 1);	// sr < er <= row

    sc = MIN(MAX(cond.sd, 0), mat->col - 1);		// 0 <= sc < col
    if (cond.ed < 0) cond.ed = mat->col - 1;
    ec = MAX(MIN(cond.ed + 1, mat->col), sc + 1);	// sc < ec <= col

    if (!strnone(cond.chinf)) {
	if ((chiv = xreaddvector_txt(cond.chinf)) == NULL) {
	    fprintf(stderr, "Can't read File: %s\n", cond.chinf);
	    exit(1);
	}
	if (chiv->length != cond.dim) {
	    fprintf(stderr, "file format: %s\n", cond.chinf);
	    exit(1);
	}
	if (cond.msg_flag == XTRUE) {
	    fprintf(stderr, "Extracted channel:");
	    for (k = 0; k < cond.dim; k++)
		if (chiv->data[k] != 0.0) fprintf(stderr, " %ld", k);
	    fprintf(stderr, "\n");
	}
    }

    if ((fp = fopen(options_struct.file[1].name, "wb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n",
		options_struct.file[1].name);
	exit(1);
    }

    if (cond.txt_flag == XFALSE) {
	for (k = sr, nk = 0; k < er; k += 1 + cond.jf, nk++) {
	    for (l = sc; l < ec; l++) {
		if (chiv != NODATA) {
		    if (chiv->data[l] != 0.0)
			fwrite(&(mat->data[k][l]), sizeof(double), (size_t)1,
			       fp);
		} else {
		    fwrite(&(mat->data[k][l]), sizeof(double), (size_t)1, fp);
		}
	//	if (mat->data[k][l] == 1.0) printf("Error\n");
	    }
	}
    } else {
	for (k = sr, nk = 0; k < er; k += 1 + cond.jf, nk++) {
	    for (l = sc; l < ec; l++) {
		if (chiv != NODATA) {
		    if (chiv->data[l] != 0.0)
			if (cond.eprt_flag == XFALSE)
			    fprintf(fp, "%f", mat->data[k][l]);
			else
			    fprintf(fp, "%e", mat->data[k][l]);
		} else {
		    if (cond.eprt_flag == XFALSE)
			fprintf(fp, "%f", mat->data[k][l]);
		    else
			fprintf(fp, "%e", mat->data[k][l]);
		}
		if (l + 1 < ec) fprintf(fp, " ");
		else fprintf(fp, "\n");
	    }
	}
    }
    fclose(fp);
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "Extract data [%ld (%ld)->(%ld)][%ld (%ld)->(%ld)]\n",
		nk, sr, er - 1, ec - sc, sc, ec - 1);

    // memory free
    if (chiv != NODATA) xdvfree(chiv);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
