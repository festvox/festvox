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
/*  Speech Analysis                                                  */
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
#include "analysis_sub.h"

typedef struct CONDITION_STRUCT {
    double fs;		// Sampling frequency [Hz]
    double framem;	// Frame length (ms)
    double shiftm;	// Frame shift (ms)
    long fftl;		// FFT length
    long order;		// Cepstrum order
    char *npowfile;	// Normalized power file
    XBOOL mcep_flag;	// Mel Cepstrogram
    XBOOL pow_flag;	// include Power coefficient
    XBOOL lpow_flag;	// include Linear Power coefficient
    XBOOL fast_flag;	// fast version
    XBOOL wav_flag;	// wav file
    XBOOL float_flag;	// float
    XBOOL msg_flag;	// print message
    XBOOL help_flag;
} CONDITION;

CONDITION cond = {16000.0, 25.0, 5.0, 512, 24, NULL, XFALSE, XFALSE, XFALSE,
		  XFALSE, XTRUE, XFALSE, XTRUE, XFALSE};

#define NUM_ARGFILE 2
ARGFILE argfile_struct[] = {
    {"[inputfile]", NULL},
    {"[outputfile]", NULL},
};

#define NUM_OPTION 14
OPTION option_struct[] = {
    {"-f", NULL, "sampling frequency [Hz]", "samp_freq", 
	 NULL, TYPE_DOUBLE, &cond.fs, XFALSE},
    {"-frame", NULL, "frame length [ms]", "frame", 
	 NULL, TYPE_DOUBLE, &cond.framem, XFALSE},
    {"-shift", NULL, "frame shift [ms]", "shift", 
	 NULL, TYPE_DOUBLE, &cond.shiftm, XFALSE},
    {"-fftl", NULL, "fft length", "fft_length", 
	 NULL, TYPE_LONG, &cond.fftl, XFALSE},
    {"-order", NULL, "cepstrum order", "order", 
	 NULL, TYPE_LONG, &cond.order, XFALSE},
    {"-npowfile", NULL, "get normalized power file", "npowfile", 
	 NULL, TYPE_STRING, &cond.npowfile, XFALSE},
    {"-mcep", NULL, "mel cepstrogram", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.mcep_flag, XFALSE},
    {"-pow", NULL, "include power coefficient", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.pow_flag, XFALSE},
    {"-lpow", NULL, "include linear power coefficient", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.lpow_flag, XFALSE},
    {"-fast", NULL, "fast version", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.fast_flag, XFALSE},
    {"-raw", NULL, "input raw file (16bit short)", NULL, 
	 NULL, TYPE_BOOLEAN, &cond.wav_flag, XFALSE},
    {"-float", NULL, "output float", NULL, 
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
    SVECTOR x = NODATA;
    DVECTOR xd = NODATA;
    DVECTOR powvec = NODATA;
    DMATRIX sgram = NODATA;
    DMATRIX mcepg = NODATA;

    // get program name
    options_struct.progname = xgetbasicname(argv[0]);

    // get option
    for (i = 1, fc = 0; i < argc; i++)
	if (getoption(argc, argv, &i, &options_struct) == UNKNOWN)
	    getargfile(argv[i], &fc, &options_struct);
    
    // display message
    if (cond.help_flag == XTRUE)
	printhelp(options_struct, "Speech Analysis for Extracting Spectra");
    if (fc != options_struct.num_file)
	printerr(options_struct, "not enough files");
    
    if (cond.wav_flag == XTRUE) {
	xd = xread_wavfile(options_struct.file[0].name, &cond.fs,
			   cond.msg_flag);
    } else {
	// read wave data
	if ((x = xreadssignal(options_struct.file[0].name, 0, 0)) == NODATA) {
	    fprintf(stderr, "Can't read wave data\n");
	    exit(1);
	} else {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "read wave: %s\n",
			options_struct.file[0].name);
	    xd = xsvtod(x);
	    xsvfree(x);
	}
    }

    // ANALYSIS
    if (cond.msg_flag == XTRUE) fprintf(stderr, "=== Speech Analysis ===\n");
    // calculate spectrogram
    if ((sgram = analysis_body(xd, cond.fs, cond.framem, cond.shiftm,
			       cond.fftl)) == NODATA) {
	fprintf(stderr, "Error: analysis_body failed\n");
	exit(1);
    } else {
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "Extracting spectrogram done\n");
    }

    // normalized power sequence
    if (!strnone(cond.npowfile)) {
	powvec = xspg2pow_norm(sgram);
	if (cond.float_flag == XFALSE) writedsignal(cond.npowfile, powvec, 0);
	else writed2fsignal(cond.npowfile, powvec, 0);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "write normalized power file [%ld]: %s\n",
		    powvec->length, cond.npowfile);
	// memory free
	xdvfree(powvec);
    }

    // write analysis file
    if (cond.mcep_flag == XFALSE) {
	if (cond.float_flag == XFALSE)
	    writedmatrix(options_struct.file[1].name, sgram, 0);
	else
	    writed2fmatrix(options_struct.file[1].name, sgram, 0);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "write spectrogram [%ld][%ld]: %s\n",
		    sgram->row, sgram->col, options_struct.file[1].name);
	// memory free
	xdmfree(sgram);
    } else {
	if (cond.lpow_flag == XTRUE) {
	    cond.pow_flag = XTRUE;
	    powvec = xget_spg2powvec(sgram, XTRUE);
	}
	// change spectrogram into cepstrogram
	mcepg = xspg2mpmcepg(sgram, cond.order, cond.fftl, cond.pow_flag,
			     cond.fast_flag);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "change Spectrogram [%ld][%ld] into Mel Cepstrogram [%ld][%ld]\n", sgram->row, sgram->col, mcepg->row, mcepg->col);
	// memory free
	xdmfree(sgram);

	if (cond.lpow_flag == XTRUE) {
	    if (cond.msg_flag == XTRUE)
		fprintf(stderr, "substitute to power on linear domain\n");
	    dmpastecol(mcepg, 0, powvec, 0, powvec->length, 0);
	    // memory free
	    xdvfree(powvec);
	}
	if (cond.float_flag == XFALSE)
	    writedmatrix(options_struct.file[1].name, mcepg, 0);
	else
	    writed2fmatrix(options_struct.file[1].name, mcepg, 0);
	if (cond.msg_flag == XTRUE)
	    fprintf(stderr, "write Mel cepstrogram [%ld][%ld]: %s\n",
		    mcepg->row, mcepg->col, options_struct.file[1].name);
	// memory free
	xdmfree(mcepg);
    }
    // memory free
    xdvfree(xd);

    if (cond.msg_flag == XTRUE) fprintf(stderr, "done\n");

    return 0;
}
