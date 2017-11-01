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
/*  Simple alignment program takes, a label file and track file from a   */
/*  a prompt and a track file from the response.  Produces a new label   */
/*  file form the response using a simple dp alignment.                  */
/*                                                                       */
/*  This is very crude but hopefully adequate for autolabelling diphones */
/*  which are very constrained                                           */
/*                                                                       */
/*************************************************************************/

#include <EST.h>
#include <EST_math.h>

static double frame_distance(const EST_Track &itrack, int i,
			     const EST_Track &otrack, int j);
static int aligntracks(const EST_Track &itrack,
		       const EST_Relation &ilabel,
		       const EST_Track &otrack,
		       EST_Relation &olabel);
static void zscore_normalize(EST_Track &track);
static int find_distance(const EST_IMatrix &dpp,
			 int i0, int j0,
			 int i1, int j1);
static void map_relation(const EST_Track &itrack,
			 const EST_Track &otrack,
			 const EST_IVector &map,
			 const EST_FMatrix &dpt,
			 const EST_IMatrix &dpp,
			 const EST_Relation &ilabel,
			 EST_Relation &olabel);

static int pa_verbose=0;
static int pa_withcosts=0;

int main(int argc, char **argv)
{
    // Phone alignment given cepstrum original
    EST_Option al;
    EST_StrList files;
    EST_Track itrack, otrack;
    EST_Relation ilabel, olabel;

    parse_command_line(argc, argv,
       EST_String("Usage: \n")+
       "phonealign <options>\n"+
       "Align phone labels from existing track to new track\n"+
       "-itrack <ifile>   Input track file\n"+
       "-ilabel <ifile>   Input label file\n"+
       "-otrack <ifile>   Track to be labelled\n"+
       "-olabel <ofile> {-} Output file for label for otrack\n"+
       "                  (default stdout)\n"+
       "-normalize        Normalize parameters in trackfiles\n"+
       "-verbose          Display more information\n"+
       "-withcosts        Include cost for labels\n"+
       "-beam <int>       Beam width\n",
		       files, al);


    if ((!al.present("-itrack")) ||
	(itrack.load(al.val("-itrack")) != read_ok))
    {
	cerr << "phonealign: can't read itrack file \"" << 
	    al.val("-itrack") << "\"" << endl;
	exit(-1);
    }

    if ((!al.present("-otrack")) ||
	(otrack.load(al.val("-otrack")) != read_ok))
    {
	cerr << "phonealign: can't read otrack file \"" << 
	    al.val("-otrack") << "\"" << endl;
	exit(-1);
    }

    if ((!al.present("-ilabel")) ||
	(ilabel.load(al.val("-ilabel")) != read_ok))
    {
	cerr << "phonealign: can't read ilabel file \"" << 
	    al.val("-ilabel") << "\"" << endl;
	exit(-1);
    }
    
    if (al.present("-verbose"))
	pa_verbose = 1;
    if (al.present("-withcosts"))
	pa_withcosts = 1;

    if (al.present("-normalize"))
    {
	zscore_normalize(itrack);
	zscore_normalize(otrack);
    }

    if (aligntracks(itrack,ilabel,otrack,olabel) != 0)
	exit(-1);
    
    if (olabel.save(al.val("-olabel")) != write_ok)
    {
	cerr << "phonealign: can't write olabel file \"" << 
	    al.val("-olabel") << "\"" << endl;
	exit(-1);
    }

    return 0;
}    

static void zscore_normalize(EST_Track &track)
{
    EST_SuffStats *ss = new EST_SuffStats[track.num_channels()];
    int i,j;

    for (i=0; i<track.num_frames(); i++)
	for (j=0; j<track.num_channels(); j++)
	    ss[j] += track.a_no_check(i,j);

    for (i=0; i<track.num_frames(); i++)
	for (j=0; j<track.num_channels(); j++)
	    track.a_no_check(i,j) = 
		(track.a_no_check(i,j) - ss[j].mean())/ss[j].stddev();

    delete [] ss;
}

static int aligntracks(const EST_Track &itrack,
		       const EST_Relation &ilabel,
		       const EST_Track &otrack,
		       EST_Relation &olabel)
{
    // Align itrack to otrack and then map labels in ilabel to new times
    // creating olabel.  Will be bad if the tracks aren't simmilar enough
    int i,j;

    if (itrack.num_channels() != otrack.num_channels())
    {
	cerr << "phonealign: tracks have different number of channels" << endl;
	return -1;
    }

//    printf("itrack.num_frames %d otrack.num_frames %d\n",
//	   itrack.num_frames(), otrack.num_frames());

    EST_FMatrix dpt(itrack.num_frames(),otrack.num_frames());
    EST_IMatrix dpp(itrack.num_frames(),otrack.num_frames());
//    float skew = (float)itrack.num_frames()/(float)otrack.num_frames();
    // Initialise first row and column
    dpt(0,0) = frame_distance(itrack,0,otrack,0);
    dpp(0,0) = -1;
    for (i=1; i < itrack.num_frames(); i++)
    {
	dpt(i,0) = frame_distance(itrack,i,otrack,0) + dpt(i-1,0);
	dpp(i,0) = -1;
    }
    for (j=1; j < otrack.num_frames(); j++)
    {
	dpt(0,j) = frame_distance(itrack,0,otrack,j) + dpt(0,j-1);
	dpp(0,j) = 1;
    }

    for (i=1; i < itrack.num_frames(); i++)
    {
	for (j=1; j < otrack.num_frames(); j++)
	{
	    dpt(i,j) = frame_distance(itrack,i,otrack,j);
	    if (dpt(i-1,j) < dpt(i-1,j-1))
	    {
		if (dpt(i,j-1) < dpt(i-1,j))
		{   
		    dpt(i,j) += dpt(i,j-1);
		    dpp(i,j) = 1; // hold
		}
		else
		{   // horizontal best 
		    dpt(i,j) += dpt(i-1,j);
		    dpp(i,j) = -1; // jump
		}
	    }
	    else if (dpt(i,j-1) < dpt(i-1,j-1))
	    {
		dpt(i,j) += dpt(i,j-1);
		dpp(i,j) = 1; // hold
	    }
	    else
	    {
		dpt(i,j) += dpt(i-1,j-1);
		dpp(i,j) = 0;
	    }
	}
	
    }
    
    EST_IVector map(itrack.num_frames());
    float cost = -1;

    for (i=itrack.num_frames()-1,j=otrack.num_frames()-1;
	 i >= 0; i--)
    {
	if ((cost == -1) && (!isnanf(dpt(i,j))))
	    cost = dpt(i,j);
	map[i] = j;
	while (dpp(i,j) == 1)
	    j--;
	if (dpp(i,j) == 0)
	    j--;
	if (pa_verbose)
	    cout << i << " " << j << " " << dpt(i,j) << endl;
    }

    if (pa_verbose)
	cout << "cost is " << cost << endl;

    map_relation(itrack,otrack,map,dpt,dpp,ilabel,olabel);

    return 0;
}

static int find_distance(const EST_IMatrix &dpp,
			 int i0, int j0,
			 int i1, int j1)
{
    // The number of steps between i1,j1 and i0,i0 in the match
    int i,j;
    int count = 0;

    for (i=i1,j=j1; (i >= i0) && (j >= j0); count++ )
    {
	if ((i < i0)||(j < j0))
	{
	    cerr << "can't find distance between to not connected points "
		 << endl;
	    break;
	}
	if (dpp(i,j) == 1)
	    j--;
	else if (dpp(i,j) == 0)
	    i--,j--;
	else // its -1
	    i--;
    }
    return count;
}

static void map_relation(const EST_Track &itrack,
			 const EST_Track &otrack,
			 const EST_IVector &map,
			 const EST_FMatrix &dpt,
			 const EST_IMatrix &dpp,
			 const EST_Relation &ilabel,
			 EST_Relation &olabel)
{
    // Build new Relation from ilabel mapping end time through
    // the map.
    EST_Item *i;
    float end;
    float lastcost = 0, cost;
    int thisi, thisj;
    int lasti=0, lastj=0;
    int dist;

    copy_relation(ilabel,olabel);

    // Map the end feature values
    for (i=olabel.head(); i != 0; i=next_item(i))
    {
	end = i->F("end");
	thisi = itrack.index(end);
	thisj = map(itrack.index(end));
	cost = dpt(thisi,thisj);
	end = otrack.t(map(itrack.index(end)));
	i->set("end",end);
	if (pa_withcosts)
	{
	    // Find number of steps to get here
	    dist = find_distance(dpp,lasti, lastj,thisi, thisj);
	    i->set("cost",(cost-lastcost)/(float)dist);
	}
	lastcost = cost;
	lasti = thisi;
	lastj = thisj;
    }
}

static double frame_distance(const EST_Track &itrack, int i,
			     const EST_Track &otrack, int j)
{
    // Euclidean distance between frames
    double sum = 0;
    double d;
    int n;

    if ((i < 0) || (j < 0))
	return 0;
    
    for (n=0; n < itrack.num_channels(); n++)
    {
	d = itrack.a_no_check(i,n) - otrack.a_no_check(j,n);
	sum += d*d;
    }
    return sqrt(sum);
}
		       

