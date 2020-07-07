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
/*  Speech Synthesis                                                 */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "../sub/anasyn_sub.h"
#include "synthesis_sub.h"

typedef struct CONDITION_STRUCT {
    double fs;		// Sampling frequency [Hz]
    double framem;	// Frame length (ms)
    long order;		// Cepstrum order
    char *rmcepfile;	// reference mel-cep file
    XBOOL wav_flag;	// wav file
    XBOOL float_flag;	// float
    XBOOL msg_flag;	// print message
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {16000.0, 5.0, 24, NULL, XTRUE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 3
ARGFILE argfile_struct[] = {
    {"[inf0file]", NULL},
    {"[inmcepfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 8
OPTION option_struct[] = {
    {"-f", NULL, "sampling frequency [Hz]", "samp_freq", 
	 NULL, TYPE_DOUBLE, &cond.fs, XFALSE},
    {"-frame", NULL, "frame length [ms]", "frame", 
	 NULL, TYPE_DOUBLE, &cond.framem, XFALSE},
    {"-order", NULL, "cepstrum order", "order", 
	 NULL, TYPE_LONG, &cond.order, XFALSE},
    {"-rmcepfile", NULL, "reference mel-cep file", "rmcepfile", 
	 NULL, TYPE_STRING, &cond.rmcepfile, XFALSE},
    {"-raw", NULL, "input raw file (16bit short)", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.wav_flag, XFALSE},
    {"-float", NULL, "input float", NULL, 
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
    SVECTOR xo = NODATA;
    DVECTOR sy = NODATA;
    DVECTOR f0v = NODATA;
    DVECTOR dpow = NODATA;
    DMATRIX mcep = NODATA;
    DMATRIX rmcep = NODATA;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);
    
    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Speech Synthesis Using MLSA filter");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");

    // read f0 file
    if ((f0v = xdvreadcol_txt(options_struct.file[0].name, 0)) == NODATA) {
	fprintf(stderr, "Can't read f0 file\n");
	exit(1);
    } else if (cond.msg_flag == XTRUE)
	fprintf(stderr, "Read F0 sequence [%ld]\n", f0v->length);
    // read mel-cepstrum file
    if (cond.float_flag == XFALSE) {
	if ((mcep = xreaddmatrix(options_struct.file[1].name,
				 cond.order + 1, 0)) == NODATA) {
	    fprintf(stderr, "Can't read mel-cep file\n");
	    exit(1);
	}
	if (!strnone(cond.rmcepfile)) {
	    if ((rmcep = xreaddmatrix(cond.rmcepfile, cond.order + 1, 0))
		== NODATA) {
		fprintf(stderr, "Can't read mel-cep file\n");
		exit(1);
	    }
	}
    } else {
	if ((mcep = xreadf2dmatrix(options_struct.file[1].name,
				 cond.order + 1, 0)) == NODATA) {
	    fprintf(stderr, "Can't read mel-cep file\n");
	    exit(1);
	}
	if (!strnone(cond.rmcepfile)) {
	    if ((rmcep = xreadf2dmatrix(cond.rmcepfile, cond.order + 1, 0))
		== NODATA) {
		fprintf(stderr, "Can't read mel-cep file\n");
		exit(1);
	    }
	}
    }
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "Read Mel-cep sequence [%ld][%ld]\n",
		mcep->row, mcep->col);

    if (rmcep != NODATA) {
	// differential power
	dpow = get_dpowvec(rmcep, mcep);
	// memory free
	xdmfree(rmcep);
    }

    // SYNTHESIS
    if (cond.msg_flag == XTRUE) fprintf(stderr, "=== Speech Synthesis ===\n");
    sy = synthesis_body(mcep, f0v, dpow, cond.fs, cond.framem);

    // write wave data
    xo = xdvtos(sy);
    if (cond.wav_flag == XTRUE)
	writessignal_wav(options_struct.file[2].name, xo, cond.fs);
    else
	writessignal(options_struct.file[2].name, xo, 0);
    if (cond.msg_flag == XTRUE)
	fprintf(stderr, "Wrie waveform [%ld]: %s\n",
		xo->length, options_struct.file[2].name);

    // memory free
    xdmfree(mcep);
    xdvfree(f0v);
    xdvfree(sy);
    xsvfree(xo);
    if (dpow != NODATA) xdvfree(dpow);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
