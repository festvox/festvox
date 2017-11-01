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
/*  Extracting F0 from Utterance File                                */
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
    double framem;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {5.0, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inuttfile]", NULL},
    {"[outf0file]", NULL},
};

#define NUM_OPTION 3
OPTION option_struct[] ={
    {"-framem", NULL, "frame length [ms]", "framem",
	 NULL, TYPE_DOUBLE, &cond.framem, XFALSE},
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
    long k, l, t, uvidx;
    long tmplen = MAX_MESSAGE - 1;
    long phnum, f0num, frmnum;
    double x1, x2, y1, y2, pos;
    long vvec[10000];
    long phidx[10000];
    long f0idx[10000];
    double endvec[10000];
    double f0vec[10000];
    double posvec[10000];
    char tmpstr[64] = "";
    char str[MAX_MESSAGE] = "";
    DVECTOR f0v = NODATA;
    //unvoiced phonemes
    char uvph[128] = " ch f hh k p s sh t th pau h# brth ";
    FILE *fp;

    void quicksort(double*, long, long, long*);

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Extracting F0 from Utterance File");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Input File: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Output File: %s\n", options_struct.file[1].name);
    }

    // open input file
    if ((fp = fopen(options_struct.file[0].name, "rt")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n", options_struct.file[0].name);
	exit(1);
    }
    // read utterance file
    phnum = 0;	f0num = 0;
    while (fgets(str, tmplen, fp) != NULL) {
	if (strstr(str, "; dur_factor ") != NULL) {
	    // duration
	    for (i = 0; *(strstr(str, " ; end ") + 7 + i) != ' '; i++)
		tmpstr[i] = *(strstr(str, " ; end ") + 7 + i);
	    tmpstr[i] = '\0';
	    endvec[phnum] = atof(tmpstr);
	    // UV
	    tmpstr[0] = ' ';
	    for (i = 0; *(strstr(str, " ; name ") + 8 + i) != ' '; i++)
		tmpstr[i + 1] = *(strstr(str, " ; name ") + 8 + i);
	    tmpstr[i + 1] = ' ';	tmpstr[i + 2] = '\0';
	    if (strstr(uvph, tmpstr) == NULL) vvec[phnum] = 1;
	    else vvec[phnum] = -1;
	    phidx[phnum] = phnum;
	    phnum++;
	}
	if (strstr(str, " ; f0 ") != NULL && strstr(str, " ; pos ") != NULL) {
	    // F0
	    for (i = 0; *(strstr(str, " ; f0 ") + 6 + i) != ' '; i++)
		tmpstr[i] = *(strstr(str, " ; f0 ") + 6 + i);
	    tmpstr[i] = '\0';
	    f0vec[f0num] = atof(tmpstr);
	    // pos of F0
	    for (i = 0; *(strstr(str, " ; pos ") + 7 + i) != ' '; i++)
		tmpstr[i] = *(strstr(str, " ; pos ") + 7 + i);
	    tmpstr[i] = '\0';
	    posvec[f0num] = atof(tmpstr);
	    f0idx[f0num] = f0num;
	    f0num++;
	}
    }
    // close file
    fclose(fp);

    // sort
    quicksort(endvec, 0, phnum - 1, phidx);
    quicksort(posvec, 0, f0num - 1, f0idx);

    frmnum = (long)(endvec[phnum - 1] * 1000.0 / cond.framem);
    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Number of phonemes: %ld\n", phnum);
	fprintf(stderr, "Number of F0 points: %ld\n", f0num);
	fprintf(stderr, "Number of frames: %ld\n", frmnum);
    }

    // memory allocation
    f0v = xdvzeros(frmnum);

    // interpolation
    for (pos = 0.0, t = 0, k = 0, l = 0;
	 t < frmnum && k < f0num - 1 && pos < posvec[f0num - 1];
	 pos += cond.framem / 1000.0, t++) {
	if (pos >= posvec[k]) {
	    while (k < f0num - 1) {
		if (posvec[k] == posvec[k + 1]) k++;
		else if (pos >= posvec[k + 1]) k++;
		else break;
	    }
	    if (k >= f0num) break;

	    while (l < phnum) {
		if (pos >= endvec[l]) l++;
		else break;
	    }
	    if (l >= phnum) uvidx = -1;
	    else uvidx = vvec[phidx[l]];

	    if (uvidx == 1) {
		x1 = posvec[k];
		x2 = posvec[k + 1];
		y1 = f0vec[f0idx[k]];
		y2 = f0vec[f0idx[k + 1]];
		f0v->data[t] = (y2 - y1) / (x2 - x1) * (pos - x1) + y1;
	    }
	}
    }

    // wirte file
    writedvector_txt(options_struct.file[1].name, f0v);

    // memory free
    xdvfree(f0v);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}


void quicksort(double *array, long lower, long upper, long *idx)
{
    long l, u, ltmp;
    double bound, tmp;

    bound = array[lower];
    l = lower;	u = upper;
    do {
	while (array[l] < bound) l++;
	while (array[u] > bound) u--;
	if (l <= u) {
	    tmp = array[u];
	    array[u] = array[l];	array[l] = tmp;
	    ltmp = idx[u];
	    idx[u] = idx[l];		idx[l] = ltmp;
	    l++;			u--;
	}
    } while (l < u);
    if (lower < u) quicksort(array, lower, u, idx);
    if (l < upper) quicksort(array, l, upper, idx);

    return;
}
