#!/bin/sh
###########################################################################
##                                                                       ##
##                                                                       ##
##              Carnegie Mellon University, Pittsburgh, PA               ##
##                      Copyright (c) 2007-2010                          ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
##          Author :  S P Kishore (skishore@cs.cmu.edu)                  ##
##          Date   :  February 2007                                      ##
##                                                                       ##
###########################################################################

LANG=C; export LANG

if [ ! "$ESTDIR" ]
then
   echo "environment variable ESTDIR is unset"
   echo "set it to your local speech tools directory e.g."
   echo '   bash$ export ESTDIR=/home/awb/projects/speech_tools/'
   echo or
   echo '   csh% setenv ESTDIR /home/awb/projects/speech_tools/'
   exit 1
fi

if [ ! "$FESTVOXDIR" ]
then
   echo "environment variable FESTVOXDIR is unset"
   echo "set it to your local festvox directory e.g."
   echo '   bash$ export FESTVOXDIR=/home/awb/projects/festvox/'
   echo or
   echo '   csh% setenv FESTVOXDIR /home/awb/projects/festvox/'
   exit 1
fi

if [ "$SIODHEAPSIZE" = "" ]
then
   SIODHEAPSIZE=20000000
   export SIODHEAPSIZE
fi
HEAPSIZE=$SIODHEAPSIZE

islicedir=$FESTVOXDIR/src/interslice
rawF=etc/raw.done.data
prompF=etc/txt.done.data
phseqF=etc/txt.phseq.data
phseqIF=etc/txt.phseq.data.int
phlist=etc/ph_list.int
bwav=bwav/bwav.wav
logF=etc/tlog.txt

if [ $1 = "help" ]
then 
  echo 'Splits long files with long text into utterance sized waveforms with'
  echo 'the appropriate words ready for further phoneme alignment'
  echo '   $FESTVOXDIR/src/interslice/scripts/do_islice_v2.sh setup'
  echo '  ./bin/do_islice_v2.sh islice BIGFILE.txt BIGFILE.wav'
  echo 'Will generate BIGFILE.data and wav/*.wav split waveforms.'
  echo '  Manual copy'
  echo '  copy / link large audio file to bwav/bwav.wav'
  echo '  copy the text onto etc/raw.done.data'
  echo 'sh bin/do_islice_v2.sh getprompts [prefix]'
  echo 'sh bin/do_islice_v2.sh phseq'
  echo 'sh bin/do_islice_v2.sh phseqint'
  echo 'sh bin/do_islice_v2.sh doseg'
  echo 'sh bin/do_islice_v2.sh slice'
fi

if [ $1 = "setup" ]
then
   mkdir etc
   mkdir tmp
   mkdir feat
   mkdir lab
   mkdir bwav
   mkdir twav

   model=model
   www=""
   if [ $# = 2 ]
   then
       model=ml_model
       www="with $model"
   fi

   # Default English models or ml_models
   cp -p $FESTVOXDIR/src/interslice/$model/ph_list.int etc/
   cp -p $FESTVOXDIR/src/interslice/$model/model101.txt etc/
   cp -p $FESTVOXDIR/src/interslice/$model/global_mn_vr.txt etc/

   cp -r $islicedir/bin ./
   cp -p $islicedir/scripts/do_islice_v2.sh ./bin/do_islice_v2.sh
   cp -p $FESTVOXDIR/src/ehmm/bin/FeatureExtraction bin/FeatureExtraction.exe
   cp -r $islicedir/scripts ./

   echo Ready to slice $www ... use
   echo "   ./bin/do_islice_v2 islice BIGFILE.txt BIGFILE.wav"

  exit 
fi

if [ $1 = "islice" ]
then
   dbname=`basename $2 .txt`
   tfiledirname=`dirname $2`

   echo Generating prompts from $2 text file
   $0 getprompts $2 $dbname $tfiledirname/$dbname.data

   $ESTDIR/bin/ch_wave -F 16000 $3 -o bwav/bwav.wav || exit -1
   cp -pr $tfiledirname/$dbname.data etc/txt.done.data

   $0 phseq
   $0 phseqint
   $0 doseg
   $0 slice

   exit
fi

if [ $1 = "getprompts" ]
then

   $FESTVOXDIR/src/promptselect/text2utts -all $2 -dbname $3 -o $4

   exit
fi

if [ $1 = "getprompts_lang" ]
then
   lang_dir=$5

   cp -pr $2 $lang_dir/prompts_txtfile.$$
   ( cd $lang_dir;
     . ./etc/voice.defs;
     $FESTVOXDIR/src/promptselect/text2utts -all \
     -eval festvox/${FV_VOICENAME}_cg.scm \
     -eval '(voice_'${FV_VOICENAME}'_cg)' \
     prompts_txtfile.$$ \
     -dbname $3 -o prompts_data.$$ )

   mv $lang_dir/prompts_data.$$ $4
   rm -f $lang_dir/prompts_txtfile.$$

   exit
fi

if [ $1 = "phseq" ]
then
   # Output phone sequence (with ssil) for alignment
   # Simplified and changed to allow different base voice/language
   $ESTDIR/../festival/bin/festival -b scripts/phseq.scm '(gen_phone_seq "'$prompF'" "'$phseqF'")'
   exit
fi


if [ $1 = "phseq_lang" ]
then
   # Generate phone sequence (with ssil) based on give voice/lang
   lang_dir=$2

   cp -pr $prompF $lang_dir/prompts_islice.$$
   # may be parallel threads so have a thread specific phseq.scm file
   cp -pr scripts/phseq.scm $lang_dir/phseq.scm.$$

   (cd $lang_dir;
      # Output phone sequence (with ssil) for alignment
      # Simplified and changed to allow different base voice/language
      . ./etc/voice.defs
      $ESTDIR/../festival/bin/festival -b phseq.scm.$$ \
          festvox/${FV_VOICENAME}_cg.scm \
          '(set! '${FV_VOICENAME}'::clunits_prompting_stage t)' \
          '(voice_'${FV_VOICENAME}'_cg)' \
          '(gen_phone_seq "'prompts_islice.$$'" "'phseq_islice.$$'")'
   )
 
   cp -pr $lang_dir/phseq_islice.$$ $phseqF

   rm -f $lang_dir/prompts_islice.$$ $lang_dir/phseq.scm.$$ $lang_dir/phseq_islice.$$
   
   exit
fi


if [ $1 = "phseqint" ]
then
   perl scripts/prmp2int.pl $phseqF $phlist
   exit
fi

if [ $1 = "doseg" ]
then
   perl scripts/do_seg.pl $phseqIF $phlist $bwav
   exit
fi

if [ $1 = "slice" ]
then
   perl scripts/slice_wav.pl $logF twav $bwav
   exit
fi

echo unknown option $*
exit


exit 1;

