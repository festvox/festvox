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
/*  Move the pitch mark to the nearest peak.  This shouldn't really be   */
/*  necessary but it does help even for EGG extracted pms                */
/*                                                                       */
/*************************************************************************/


#include "EST.h"

int main(int argc,char **argv)
{
    EST_Wave w;
    EST_Track pm_in, pm_out, pm_out2;
    EST_Option al;
    EST_StrList files;
    int i,j,window,prewindow,max;
    float maxperiod, minperiod,t;
    int k;
    int verbose=FALSE;

    parse_command_line
	(argc,argv,
	 EST_String("[options]")+
	 "Summary: move pitchmark with respect to wave\n"+
	 "-h        Options help\n"+
	 "-wave <ifile>\n"+
	 "-pm <ifile>\n"+
	 "-window <int> {16}\n"+
	 "-prewindow <int> {16}\n"+
	 "-verbose\n"+
	 "-max <float> {0.020}\n"+
	 "-min <float> {0.0025}\n"+
	 "-o <ofile> Output pm file\n",
	 files,al);

    if (al.present("-verbose"))
	verbose = TRUE;
    window = al.ival("-window");
    maxperiod = al.fval("-max");
    minperiod = al.fval("-min");
    prewindow = al.ival("-prewindow");
    window = al.ival("-window");
    w.load(al.val("-wave"));
    pm_in.load(al.val("-pm"));
    pm_out.resize(pm_in.num_frames()+30,pm_in.num_channels());
    pm_out.copy_setup(pm_in);

    for (i=0; i<pm_in.num_frames(); i++)
    {
	int pos = (int)(pm_in.t(i)*w.sample_rate());
	for (max=pos,j=pos-prewindow; 
	     (j > 0) && (j < w.num_samples()) && (j < pos+window);
	     j++)
	    if (w(j) > w(max))
		max = j;
	pm_out.t(i) = ((float)max)/w.sample_rate();
/*	printf("%f %d %f %d\n",
	       pm_in.t(i),pos,
	       pm_out.t(i),max); */
    }

    pm_out2.resize(pm_in.num_frames()*4,pm_out.num_channels());
    pm_out2.copy_setup(pm_out);

    for (k=i=0,t=0.0; t<(float)w.num_samples()/(float)w.sample_rate();
	 i++)
    {
/*	printf("working on t %f i %d out.t(i) %f k %d out2.t(k) %f\n",
	t,i,pm_out.t(i),k,pm_out2.t(k)); */
	if ((i > pm_out.num_frames()) ||
	    ((pm_out.t(i)-t) > maxperiod))
	{
	    if (verbose)
	    {
		if (i > pm_out.num_frames())
		    printf("adding at end (%f)\n", t);
		else
		    printf("splitting long period (%f) at time %f\n",
			   (pm_out.t(i)-t), pm_out.t(i));
	    }
	    if (k>=pm_out2.num_frames())
	    {
		if (verbose)
		    printf("getting more pm space %d\n",k);
		pm_out2.resize((int)((float)k*1.2),pm_out2.num_channels());
	    }
	    pm_out2.t(k) = t + ((minperiod+maxperiod)/2.0);
	    i--;
	}
	else if ((pm_out.t(i) - t) < minperiod)
	{
	    if (verbose)
		printf("skipping short period (%f) at time %f\n",
		       (pm_out.t(i)-t), pm_out.t(i));
	    continue;
	}
	else
	    pm_out2.t(k) = pm_out.t(i);
	t = pm_out2.t(k);
	k++;
    }

    pm_out2.resize(k-1,pm_in.num_channels());
    pm_out2.save(al.val("-o"));

    return 0;
}
