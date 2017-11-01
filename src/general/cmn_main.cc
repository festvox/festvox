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
/*  Cepstral mean normalization.                                         */
/*  This is done in two passes, first to find means and std then to      */
/*  apply these to a given track                                         */
/*                                                                       */
/*************************************************************************/

#include "EST.h"

static void find_meanstd(EST_Track &ss, EST_StrList &files);
static void cep_normalize(EST_Track &tt, const EST_Track &ss, EST_Relation &rr);

#define num_phones 44
const char *phonetab [num_phones] = {
    "other", "aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", 
    "d", "dh", "eh", "er", "ey", "f", "g", "hh", "ih", "iy",
    "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "pau", 
    "r", "s", "sh", "t", "th", "uh", "uw", "v", "w", "y", 
    "z", "zh", "ssil", NULL };

static int get_phone_id(const char* phone)
{
    int i;

    for (i=1; phonetab[i]; i++)
	if (streq(phone,phonetab[i]))
	    return i;

    printf("treating %s as other\n",phone);
    return 0;
}

int main(int argc,char **argv)
{
    EST_Option al;
    EST_StrList files;
    EST_Litem *p;
    EST_Track ss,tt;
    EST_Relation rr;

    parse_command_line
	(argc,argv,
	 EST_String("[options] mcep/*.mcep\n")+
	 "Summary: find (and apply) phone based means/stddev for \n"+
         "         cepstral normalization.  Outputs to nmcep/\n"+
	 "-h        Options help\n",
	 files,al);

    printf("Finding stats\n");
    find_meanstd(ss,files);

    printf("Applying stats\n");
    for (p=files.head(); p != 0; p=p->next())
    {
	tt.load(files(p));
	printf("%s\n",(const char *)files(p));
	rr.load(EST_String("lab/")+basename(files(p)).before(".")+".lab");
	cep_normalize(tt,ss,rr);
	tt.save(EST_String("n")+files(p),"est_binary");
	rr.clear();
    }

    return 0;
}

static void find_meanstd(EST_Track &ss, EST_StrList &files)
{
    // Find means and stddev for each coefficient
    int i,j;
    float v;
    FILE *fd;
    EST_Litem *p;
    EST_Track tt;
    EST_Item *s;
    EST_Relation rr;
    EST_SuffStats **sstable = 0;
    int phoneid;
    EST_String lll;

    p = files.head();
    if (p == NULL)
    {
	cerr << "cmn: no files to build stats from" << endl;
	exit(-1);
    }
    tt.load(files(p));

    sstable = new EST_SuffStats*[num_phones];
    for (i=0; i< num_phones; i++)
	sstable[i] = new EST_SuffStats[tt.num_channels()];
    
    for (p=files.head(); p != 0; p=p->next())
    {
	tt.load(files(p));
	lll = EST_String("lab/")+basename(files(p)).before(".")+".lab";
        /*	printf("%s\n",(const char *)lll); */
	rr.clear();
	rr.load(lll);
	phoneid = get_phone_id("pau");
	for (s=rr.head(),i=0; i<tt.num_frames(); i++)
	{
/*	    printf("%s %f %f\n", phonetab[phoneid], s->F("end"),tt.t(i)); */
            if (tt.a_no_check(i,0) > 0) /* F0 */
                sstable[phoneid][0] += tt.a_no_check(i,0);
	    for (j=1; j<tt.num_channels(); j++)
	    {
		v = tt.a_no_check(i,j);
		if (!finite(v))   // sigh, better safe that sorry
		    v = 100;
		if (fabs(v) > 100)
		    v = 100;
		sstable[phoneid][j] += v;
	    }
	    if (s && next_item(s) && (s->F("end") < tt.t(i)))
	    {
		s = next_item(s);
		phoneid = get_phone_id(s->S("name"));
/*		printf("%s\n",phonetab[phoneid]); */
	    }
	}
    }

    fd = fopen("etc/phone.cmn","wb");
    ss.resize(2*num_phones,tt.num_channels());
    for (i=0; phonetab[i]; i++)
    {
	fprintf(fd,"%s ",phonetab[i]);
	for (j=0; j<ss.num_channels(); j++)
	{
	    fprintf(fd,"%f %f ",sstable[i][j].mean(), 
		   sstable[i][j].stddev());
	    ss.a_no_check((i*2)+0,j) = sstable[i][j].mean();
	    ss.a_no_check((i*2)+1,j) = sstable[i][j].stddev();
	}
        fprintf(fd,"\n");
	delete [] sstable[i];
    }
    delete [] sstable;
    fclose(fd);
}

static void cep_normalize(EST_Track &tt, const EST_Track &ss, EST_Relation &rr)
{
    // Normalize coefficients in tt from means and stddevs in ss
    int i,j;
    EST_Item *s;
    int phoneid;

    if (tt.num_channels() != ss.num_channels())
    {
	cerr << "cmn: meanstd files has " << ss.num_channels() <<
	    " while cep track has " << tt.num_channels() << endl;
	exit(-1);
    }

    phoneid = get_phone_id("pau");
    s = rr.head();
    for (i=0; i < tt.num_frames(); i++)
    {
        if (tt.a_no_check(i,0) > 0) /* F0 */
	    tt.a_no_check(i,0) = 10 + 
		(tt.a_no_check(i,0) - 
		 ss.a_no_check((phoneid*2)+0,0)) / 
		ss.a_no_check((phoneid*2)+1,0);
	for (j=1; j < tt.num_channels(); j++)
	{
	    tt.a_no_check(i,j) = 
		(tt.a_no_check(i,j) - 
		 ss.a_no_check((phoneid*2)+0,j)) / 
		ss.a_no_check((phoneid*2)+1,j);
//	    if (tt.a_no_check(i,j) > 50)
//		tt.a_no_check(i,j) = 50;
//	    else if (tt.a_no_check(i,j) < -50)
//		tt.a_no_check(i,j) = -50;
	}
	while (s && next_item(s) && (s->F("end") < tt.t(i)))
	{
	    s = next_item(s);
	    phoneid = get_phone_id(s->S("name"));
	}
    }
}

