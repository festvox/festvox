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
/*  This is in speech_tools 1.5.0's pda but not in the 1.4.1 version so  */
/*  we temporatily duplicate he smoothing functions here.                */
/*                                                                       */
/*  based on pault's code in the new pda (though we deal with            */
/*  like the ancient icda did                                            */
/*                                                                       */
/*************************************************************************/

#include <math.h>
#include "EST.h"
#include "EST_cmd_line_options.h"

void option_override(EST_Features &op, EST_Option al, 
		     const EST_String &option, const EST_String &arg);

void smooth_portion(EST_Track &c, EST_Features &op);

static void set_parameters(EST_Features &a_list, EST_Option &al);
static void interpolate(const EST_Track &c, 
			const EST_Track &speech,
			EST_Track &interp);
static void pm_to_f0(const EST_Track &pm, EST_Track &f0, 
		     float minf0, float shift);

int main(int argc,char **argv)
{
    EST_Track f0_in, f0_interpolated, pm_in;
    EST_Option al;
    EST_Features op;
    EST_StrList files;
    EST_String silences;
    EST_Utterance utt; /* to hold labels */
    int i;

    parse_command_line
	(argc,argv,
	 EST_String("[options]")+
	 "Summary: smooth and interporlate through unvoiced sections\n"+
	 "-h        Options help\n"+
	 options_track_input()+ "\n"+
	 options_track_output()+"\n"
	 "-pm_input\n"+
         "     Input is from (non-filled) pitchmark file rather than\n"+
         "     an F0 file.\n"+
         "-pm_min_f0 <float> {110.0}\n"+
         "     If pm_input is used, this defined when pm distances are to\n"+
         "     treated as unvoiced regions\n"+
         "-pm_f0_shift <float> {0.005}\n"+
         "     If pm_input is used, this specifies the F0 shift for the\n"+
         "     generated F0 track\n"+
	 "-interpolate\n"+
         "     Interpolate between unvoices section (use -lab if you want\n"+
         "     silences to remain with F0 of zero\n"+
	 "-lab <ifile>\n"+
         "     label file identifying phone breaks used to find, silence\n"+
         "     and non-silence parts of the file\n"+
	 "-silences <string>\n"+
         "     comma separated list of silence names\n"+
	 "-presmooth\n"+
         "     smooth before interpolating\n"+
	 "-postsmooth\n"+
         "     smooth after interpolating\n"+
	 "-prewindow <float> {0.05}\n"+
         "     size of window used for smoothing before interpolation\n"+
	 "-postwindow <float> {0.05}\n"+
         "     size of window used for smoothing after interpolation\n",
	 files,al);

    default_pda_options(op);
    set_parameters(op, al);

    if (al.present("-pm_input"))
    {
	if (read_track(pm_in, files.first(), al) != format_ok)
	    exit(-1);
	pm_to_f0(pm_in,f0_in,
		 al.fval("-pm_min_f0"),
		 al.fval("-pm_f0_shift"));
    }
    else if (read_track(f0_in, files.first(), al) != format_ok)
	exit(-1);

    // EST_Track should preserve file_type, it doesn't
    if (!al.present("-otype"))
	al.add_item("-otype","ssff");
    if (al.present("-silences"))
	silences = al.val("-silences");

    /* Presmooth */
    if (al.present("-presmooth"))
    {
	int bbb = (int)(al.fval("-prewindow") / f0_in.shift());
	op.set("point_window_size", bbb);
	printf("smooth window is %d\n",bbb);
	smooth_portion(f0_in, op);
    }

    /* Interpolate */
    if (al.present("-interpolate"))
    {
	EST_Track spsil;
	EST_Relation *r;
	EST_Item *item;
	r = utt.create_relation("labs");
	if (al.present("-lab"))
	{
	    r->load(al.val("-lab"));
	    // If labels go passed end of F0 (common for PM based F0 files)
  	    if (r->rlast()->F("end") > f0_in.end())
  	    {
  		int cn = f0_in.num_frames();
  		f0_in.resize(
  		    (int)(cn-1+((r->rlast()->F("end")-f0_in.end())/f0_in.shift())),
  		    f0_in.num_channels());
  		f0_in.fill_time(f0_in.shift());
  		for (i=cn; i<f0_in.num_frames(); i++)
  		{
  		    f0_in.a(i,0) = 0;
  		    f0_in.a(i,1) = 0;
  		}
  	    }
	}
	if ((r->rlast() == 0) ||
	    (r->rlast()->F("end") < f0_in.end()))
	{
	    item = r->append(0);
	    item->set("name","something"); // fake name
	    item->set("end",f0_in.end());
	}
//	f0_in.save("afterlab.f0");
	/* Mark which frames are speech vs silence */
	spsil = f0_in;
	for (item = r->first(),i=0; i<spsil.num_frames(); i++)
	{
	    while (item && (item->F("end") < spsil.t(i)))
	    {
		item = inext(item);
	    }
	    if ((item == 0) ||
		(item->name() == silences) ||
		(item->name() == "h#")  ||
		(item->name() == "ssil")  ||
		(item->name() == "H#"))
		spsil.a(i) = 0;
	    else
		spsil.a(i) = 1;
	}
//	spsil.save("spsil.f0");
	/* File in F0 at speech (non silence) parts */
	interpolate(f0_in, spsil, f0_interpolated);
//	f0_interpolated.save("fred.f0");
    }
    else
	f0_interpolated = f0_in;

    /* Postsmooth */
    if (al.present("-postsmooth"))
    {
	op.set("point_window_size", 
	       (int)(al.fval("-postwindow") / f0_in.shift()));
	smooth_portion(f0_interpolated, op);
	for (i=0; i < f0_interpolated.num_frames(); i++)
	{
	    if (f0_interpolated.a(i,1) == 0)
		f0_interpolated.a(i,0) = 0;
	}
    }

    if (f0_interpolated.save(al.val("-o"),
			     al.val("-otype")) != write_ok)
	exit(-1);

    return 0;
}

static void set_parameters(EST_Features &op, EST_Option &al)
{
    op.set("srpd_resize", 1);
    op.set("window_length", 0.05);
    op.set("second_length", 0.05);

    // general options
    option_override(op, al, "pda_frame_shift", "-shift");
    option_override(op, al, "pda_frame_length", "-length");
    option_override(op, al, "max_pitch", "-fmax");
    option_override(op, al, "min_pitch", "-fmin");

    // low pass filtering options.
    option_override(op, al, "lpf_cutoff", "-u");
    option_override(op, al, "lpf_order", "-forder");

    option_override(op, al, "decimation", "-d");
    option_override(op, al, "noise_floor", "-n");
    option_override(op, al, "min_v2uv_coef_thresh", "-m");
    option_override(op, al, "v2uv_coef_thresh_ratio", "-R");
    option_override(op, al, "v2uv_coef_thresh", "-H");
    option_override(op, al, "anti_doubling_thresh", "-t");
    option_override(op, al, "peak_tracking", "-P");

    option_override(op, al, "f0_file_type", "-otype");
    option_override(op, al, "wave_file_type", "-itype");

    if (al.val("-L", 0) == "true")
	op.set("do_low_pass", "true");
    if (al.val("-R", 0) == "true")
	op.set("do_low_pass", "false");
    
}

static void interpolate(const EST_Track &c, 
			const EST_Track &speech,
			EST_Track &interp)
{
    // Interpolate between unvoiced sections, and ensure breaks
    // during silences
    int i, n, p;
    float m;
    float n_val, p_val;

    interp = c;  // copy track

    if (speech.num_frames() < c.num_frames())
	interp.resize(speech.num_frames(), interp.num_channels());

    for (i = 1; i < interp.num_frames()-1; i++)
    {
	if (speech.a(i) == 1)
	{
	    if (interp.a(i) < 1)
	    {
		p = i-1;
		for (n=i+1; n < interp.num_frames()-1; n++)
		{
		    n_val = interp.a(n);
		    if (speech.a(n) == 0)
			break;
		    if (n_val > 0)
			break;
		}
		n_val = interp.a(n);
		p_val = interp.a(p);
		if (n_val <= 0) n_val = p_val;
		if (p_val <= 0) p_val = n_val;
		if (p_val == 0)  // there isn't any F0 at all
		    n_val = p_val = 110;
		m = (n_val - p_val) / ( n - p);
		
		for( ; i < n; i++)
		{
		    interp.a(i) = (m * (i-p)) + p_val;
		    interp.a(i,1) = 1;
		    interp.set_value(i);
		}
		i--;
	    }
	}
	else
	{
	    interp.set_break(i);
	    interp.a(i) = 0;
	    interp.a(i,1) = 0;
	}
    }
    // Last one is unvoiced, because, thats my definition
    interp.a(i) = 0;
    interp.a(i,1) = 0;
}

static void pm_to_f0(const EST_Track &pm, EST_Track &f0, 
		     float minf0, float shift)
{
    // Converts a pm to a fixed framed F0 track.
    int i,pm_pos;
    float maxshift;

    f0.resize((int)(pm.end()/shift),2);
    f0.set_channel_name("F0",0);
    f0.set_channel_name("prob_voice",1);
    f0.fill_time(shift);
    f0.set_equal_space(TRUE);
    maxshift = 1.0 / minf0;

    for (i=0; i < f0.num_frames(); i++)
    {
	pm_pos = pm.index_below(f0.t(i));
//	printf("f0.t(i)-pm.t(pm_pos) %f\n",f0.t(i)-pm.t(pm_pos));
//	printf("pm.t(pm_pos) %f\n",pm.t(pm_pos));
//	printf("pm.t(pm_pos+1) %f\n",pm.t(pm_pos+1));
	if ((fabs(f0.t(i)-pm.t(pm_pos)) > maxshift) ||
	    (pm_pos+1 >= pm.num_frames()) ||
	    ((pm.t(pm_pos+1) - pm.t(pm_pos)) > maxshift))
	{   // its unvoiced
	    f0.a(i,0) = 0;
	    f0.a(i,1) = 0;  
	}
	else // its voiced
	{
	    f0.a(i,0) = 1.0 / (pm.t(pm_pos+1) - pm.t(pm_pos));
	    f0.a(i,1) = 1;  
	}
    }

}
