/*************************************************************************/
/*                                                                       */
/*                  Language Technologies Institute                      */
/*                     Carnegie Mellon University                        */
/*                        Copyright (c) 2006                             */
/*                        All Rights Reserved.                           */
/*                                                                       */
/*  Permission is hereby granted, free of charge, to use and distribute  */
/*  this software and its documentation without restriction, including   */
/*  without limitation the rights to use, copy, modify, merge, publish,  */
/*  distribute, sublicense, and/or sell copies of this work, and to      */
/*  permit persons to whom this work is furnished to do so, subject to   */
/*  the following conditions:                                            */
/*   1. The code must retain the above copyright notice, this list of    */
/*      conditions and the following disclaimer.                         */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*   4. The authors' names are not used to endorse or promote products   */
/*      derived from this software without specific prior written        */
/*      permission.                                                      */
/*                                                                       */
/*  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
/*  THIS SOFTWARE.                                                       */
/*                                                                       */
/*************************************************************************/
/*             Author:  Alan W Black (awb@cs.cmu.edu)                    */
/*               Date:  August 2006                                      */
/*************************************************************************/
/*                                                                       */
/*  Resynthesize a track/lpc/residual into a waveform file               */
/*                                                                       */
/*************************************************************************/
#include <stdio.h>
#include <math.h>
#include <string.h>

/* Because some things are const in real flite */
#define const

#include "cst_wave.h"
#include "cst_track.h"
#include "cst_sigpr.h"

#define PT_LPC_PARAMS_START 3
#define PT_NUM_LPC_PARAMS 16
#define PT_RESSIZE 2
#define PT_RESSTART (PT_LPC_PARAMS_START+PT_NUM_LPC_PARAMS)

int main(int argc, char **argv)
{
    cst_track *param_track;
    cst_lpcres *lpcres;
    int samples, i, j, r, start;
    cst_wave *wav;
    float ptime;

    lpcres = new_lpcres();
    param_track = new_track();
    cst_track_load_est(param_track,argv[1]);

    lpcres = new_lpcres();
    lpcres_resize_frames(lpcres,param_track->num_frames);
    lpcres->num_channels = PT_NUM_LPC_PARAMS;
    lpcres->lpc_min = -12.206278;
    lpcres->lpc_range =  17.588343;
    lpcres->sample_rate = 16000;

    ptime = 0.0;
    for (i=0; i<param_track->num_frames; i++)
    {
        lpcres->frames[i] = cst_alloc(unsigned short,PT_NUM_LPC_PARAMS);
        for (j=0; j<PT_NUM_LPC_PARAMS; j++)
            lpcres->frames[i][j] = 
                param_track->frames[i][PT_LPC_PARAMS_START+j];
        lpcres->sizes[i] = param_track->frames[i][PT_RESSIZE];
#if 0
        lpcres->sizes[i] = 
            (int)(((float)lpcres->sample_rate)*(param_track->times[i]-ptime));
        ptime = param_track->times[i];
#endif
#if 0
        if (param_track->frames[i][0] > 0)
            lpcres->sizes[i] = 
                (int)(((float)lpcres->sample_rate)/param_track->frames[i][0]);
        else
            lpcres->sizes[i] = param_track->frames[i][PT_RESSIZE];
#endif
    }

    for (samples=0,i=0; i<lpcres->num_frames; i++)
        samples += lpcres->sizes[i];
    start = lpcres->sizes[0]/2;
    start = 0;
    samples += start;
    
    lpcres_resize_samples(lpcres,samples*2);
    for (r=start,i=0; i<lpcres->num_frames; i++)
    {
	for (j=0; j<lpcres->sizes[i]; j++,r++)
        {
            if (j < 256)
                lpcres->residual[r] = param_track->frames[i][PT_RESSTART+j];
            else
                lpcres->residual[r] = 255;
        }
    }

    wav = lpc_resynth(lpcres);

    cst_wave_save_riff(wav,argv[2]);

    return 0;
}
