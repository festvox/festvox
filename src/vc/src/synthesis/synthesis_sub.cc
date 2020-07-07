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
/*  Subroutine for Speech Synthesis                                  */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/basic.h"
#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"
#include "../include/window.h"

#include "../sub/anasyn_sub.h"
#include "vocoder_sub.h"
#include "synthesis_sub.h"

//
//	speech synthesis function
//
DVECTOR synthesis_body(DMATRIX mcep,	// input mel-cep sequence
		       DVECTOR f0v,	// input F0 sequence
		       DVECTOR dpow,	// input diff-pow sequence
		       double fs,	// sampling frequency (Hz)
		       double framem)	// FFT length
{
    long t, pos;
    int framel;
    double f0;
    VocoderSetup vs;
    DVECTOR xd = NODATA;
    DVECTOR syn = NODATA;

    framel = (int)round(framem * fs / 1000.0);
    init_vocoder(fs, framel, mcep->col - 1, &vs);

    // synthesize waveforms by MLSA filter
    xd = xdvalloc(mcep->row * (framel + 2));
    for (t = 0, pos = 0; t < mcep->row; t++) {
	if (t >= f0v->length) f0 = 0.0;
	else f0 = f0v->data[t];
	if (dpow == NODATA)
	    vocoder(f0, mcep->data[t], mcep->col - 1, ALPHA, 0.0, &vs,
		    xd->data, &pos);
	else
	    vocoder(f0, mcep->data[t], dpow->data[t], mcep->col - 1, ALPHA,
		    0.0, &vs, xd->data, &pos);
    }
    syn = xdvcut(xd, 0, pos);

    // normalized amplitude
    waveampcheck(syn, XFALSE);

    // memory free
    xdvfree(xd);
    free_vocoder(&vs);

    return syn;
}

DVECTOR get_dpowvec(DMATRIX rmcep, DMATRIX cmcep)
{
    long t;
    DVECTOR dpow = NODATA;
    VocoderSetup pvs;

    // error check
    if (rmcep->col != cmcep->col) {
	fprintf(stderr, "Error: Different number of dimensions\n");
	exit(1);
    }
    if (rmcep->row != cmcep->row) {
	fprintf(stderr, "Error: Different number of frames\n");
	exit(1);
    }

    // memory allocation
    dpow = xdvalloc(rmcep->row);
    init_vocoder(16000.0, 80, rmcep->col - 1, &pvs);

    // calculate differential power
    for (t = 0; t < rmcep->row; t++)
	dpow->data[t] = get_dpow(rmcep->data[t], cmcep->data[t],
				 rmcep->col - 1, ALPHA, &pvs);

    // memory free
    free_vocoder(&pvs);

    return dpow;
}

void waveampcheck(DVECTOR wav, XBOOL msg_flag)
{
    double value;

    value = MAX(FABS(dvmax(wav, NULL)), FABS(dvmin(wav, NULL)));
    if (value >= 32000.0) {
	if (msg_flag == XTRUE) {
	    fprintf(stderr, "amplitude is too big: %f\n", value);
	    fprintf(stderr, "execute normalization\n");
	}
	dvscoper(wav, "*", 32000.0 / value);
    }

    return;
}
