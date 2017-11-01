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
/*  VQ with LBG Algorithm                                            */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include  <stdio.h>
#include  <stdlib.h>
#include  <string.h>
#include  <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "./vq_sub.h"

typedef struct CONDITION_STRUCT {
    long dim;
    long sd;
    long cls;
    XBOOL float_flag;
    XBOOL msg_flag;
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {25, 0, 1, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[infile]", NULL},
    {"[outlabel]", NULL},
};

#define NUM_OPTION 6
OPTION option_struct[] ={
    {"-dim", NULL, "dimension of vector", "dim",
	 NULL, TYPE_LONG, &cond.dim, XFALSE},
    {"-sd", NULL, "start dim", "sd",
	 NULL, TYPE_LONG, &cond.sd, XFALSE},
    {"-cls", NULL, "codebook size", "cls",
	 NULL, TYPE_LONG, &cond.cls, XFALSE},
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
int main(int  argc, char *argv[])
{
    int i, fc = 0;
    int j, u, nts, ni, tmp;
    double data[1];
    float fdata[1];
    float *tmpfdata = NULL;
    char outf[MAX_MESSAGE] = "";
    FILE  *fpi, *fpo;
    struct codebook *cbook;
    struct sample smpl;
    struct analysis condana;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);

    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "VQ with LBG Algorithm");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    cbook = (struct codebook *)malloc( 1 * sizeof(struct codebook) );
    cbook->dim_s = (int)cond.sd;
    cbook->vecorder = (int)cond.dim;
    nts = cbook->vecorder;
    cbook->vecsize = (int)cond.cls;
    sprintf(outf, "%s%d.mat", options_struct.file[1].name, cbook->vecsize);
    condana.nts    = nts;
    if (cond.msg_flag == XTRUE) {
	fprintf(stderr, "Input File: %s\n", options_struct.file[0].name);
	fprintf(stderr, "Output File: %s\n", outf);
    }

    if ((fpi = fopen(options_struct.file[0].name, "rb")) == NULL) {
	fprintf(stderr, "Can't open file: %s\n",
		options_struct.file[0].name);
	exit(1);
    }

    tmp = 0;
    if (cond.float_flag == XFALSE) {
	while (1 == fread(data, sizeof(double), 1, fpi)) tmp++;
    } else {
	while (1 == fread(fdata, sizeof(float), 1, fpi)) tmp++;
    }
    fseek(fpi, 0, SEEK_SET);

    ni = tmp / nts;
    if (tmp != ni * nts) {
	fprintf(stderr, "Error dimension [%d][%d]\n", ni, nts);
	exit(1);
    }
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "all vector[%d][%d]\n", ni, nts);

    if (cond.float_flag == XFALSE) {
	if (ni > MAXMEM / (int)sizeof(double) / nts)
	    ni = MAXMEM / (int)sizeof(double) / nts;
    } else {
	if (ni > MAXMEM / (int)sizeof(float) / nts)
	    ni = MAXMEM / (int)sizeof(float) / nts;
    }

    smpl.buff = dalloc(ni*nts);

    if (cond.float_flag == XFALSE) {
	for(i=0;i<ni;i++){
	    u=fread(&smpl.buff[i*nts],sizeof(double),nts,fpi);
	    if ( feof(fpi) )   break;
	}
    } else {
	tmpfdata = falloc(nts);
	for(i=0;i<ni;i++){
	    u=fread(tmpfdata,sizeof(float),nts,fpi);
	    for (j = 0; j < nts; j++)
		smpl.buff[i * nts + j] = (double)tmpfdata[j];
	    if ( feof(fpi) )   break;
	}
	free(tmpfdata);	tmpfdata = NULL;
    }
    smpl.nfrms = i;
    fclose(fpi);
 
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "read vector[%d][%d]\n", ni, nts);
    if (cond.msg_flag == XTRUE) printf("ok\n");

    if (cond.float_flag == XFALSE) {
	lbg(&condana, cbook, &smpl, options_struct.file[1].name,
	    cond.float_flag, cond.msg_flag);
	if ((fpo = fopen(outf, "wb")) == NULL) {
	    fprintf(stderr, "Can't open file: %s\n", outf);
	    exit(1);
	}
	fwrite(cbook->vpara, sizeof(double),
	       cbook->vecsize * cbook->vecorder, fpo);
	fclose(fpo);
    } else {
	tmpfdata = falloc(cbook->vecsize * cbook->vecorder);
	tmp = cbook->vecsize * cbook->vecorder;
	lbg(&condana, cbook, &smpl, options_struct.file[1].name,
	    cond.float_flag, cond.msg_flag);
	for (j = 0; j < tmp; j++) tmpfdata[j] = (float)cbook->vpara[j];
	if ((fpo = fopen(outf, "wb")) == NULL) {
	    fprintf(stderr, "Can't open file: %s\n", outf);
	    exit(1);
	}
	fwrite(tmpfdata, sizeof(float),
	       cbook->vecsize * cbook->vecorder, fpo);
	fclose(fpo);
	free(tmpfdata);	tmpfdata = NULL;
    }
    if (cond.msg_flag == XTRUE) fprintf(stderr, "wrote %s\n", outf);

//  for(i=0; i < smpl.nfrms ; i++)
//      printf("lab[%d] = %d\n",i,cbook->vlab[i]);

    // memory free
    free_vqlabel(cbook);
    free(cbook);
    free(smpl.buff);
  
    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
