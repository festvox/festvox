/*************************************************************************/
/*                                                                       */
/*                   Carnegie Mellon University and                      */
/*                   Alan W Black and Kevin A. Lenzo                     */
/*                      Copyright (c) 1998-2000                          */
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
/*                                                                       */
/*  Power normalised a waveform with a power contour specified as a      */
/*  file of factors at points in the file                                */
/*                                                                       */
/*  This makes osme assumptions that aren't full general:                */
/*     it doesn't normalized the from last label to end of waveform      */
/*     (which is probably silence)                                       */
/*                                                                       */
/*************************************************************************/


#include "EST.h"

int main(int argc,char **argv)
{
    EST_Wave w;
    EST_Relation powcont;
    EST_Option al;
    EST_StrList files;
    EST_Item *powerpoint;
    int i, end_sample, start_sample;
    float factor,increment;
    float maxthres = -1;

    parse_command_line
	(argc,argv,
	 EST_String("[options]")+
	 "Summary: normalise a waveform with a power contour labeled file\n"+
	 "-h        Options help\n"+
	 "-wave <ifile>\n"+
	 "-powcont <ifile>\n"+
	 "-max <float> Absolute maximum/minimum (1/max) power modification\n"+
	 "-o <ofile> Output pm file\n",
	 files,al);

    w.load(al.val("-wave"));
    powcont.load(al.val("-powcont"));
    if (al.present("-max"))
    {
	maxthres = al.fval("-max");
	for (powerpoint = powcont.head(); powerpoint; 
	     powerpoint = next_item(powerpoint))
	{
	    if (fabs(powerpoint->F("name")) > maxthres)
		powerpoint->set("name",maxthres);
	    if (fabs(powerpoint->F("name")) < 1.0/maxthres)
		powerpoint->set("name",1.0/maxthres);
	}
    }

    powerpoint = powcont.head();
    increment = 0.0;
    end_sample = w.num_samples();
    if (powerpoint == 0)
	factor = 1.0;
    else
    {
	factor = powerpoint->F("name");
	end_sample = (int)(powerpoint->F("end")*(float)w.sample_rate());
    }

    for (i=0; i<w.num_samples(); i++,factor+=increment)
    {
	w.a(i) = (short)((float)w.a(i) * factor);
//	printf("factor %f increment %f end_sample %d %d\n",
//	       factor,increment,end_sample,i);
	if (i == end_sample)
	{
	    powerpoint = next_item(powerpoint);
	    if (powerpoint == 0)
		break;
	    start_sample = end_sample;
	    end_sample = (int)(powerpoint->F("end") * (float)w.sample_rate());
//	    printf("es %d ss %d end %f sample_rate %d\n",
//		   end_sample,start_sample,powerpoint->F("end"),
//		   w.sample_rate());
	    increment = (powerpoint->F("name")-factor)/
		((float)(end_sample-start_sample));
//	    printf("f %f nf %f icr %f\n",
//		   factor,
//		   powerpoint->F("name"),
//		   increment);
	}
    }

    w.save(al.val("-o"));

    return 0;
}
