#!/usr/local/bin/perl
use strict;
###########################################################################
##                                                                       ##
##                                                                       ##
##              Carnegie Mellon University, Pittsburgh, PA               ##
##                         Copyright (c) 2007                            ##
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

my $tmpD = "tmp";
my $labD = "lab";
my $etcD = "etc";
my $featD = "feat";
my $featE = "bin/FeatureExtraction.exe";
my $palignE = "bin/edec.exe";
my $wavchop = "bin/wavchop.exe"; 
my $pauS = "pau";
my $ssilS = "ssil";

my $nargs = $#ARGV + 1;
if ($nargs != 3) {
  print "Usage: perl file.pl <ph-seq-int> <ph-mod-file> <large-wave-file>\n";
  print "EHMM model is assumed to be under etc/model101.txt\n";
  exit;
}

my $phF = $ARGV[0];
my $phmodF = $ARGV[1];
my $wavF = $ARGV[2];

my $pauID = &locate_ID($phmodF, $pauS);
if ($pauID == -1) {
  print "Could not locate $pauID in $phF\n";
  $pauID = &locate_ID($phmodF, "SIL");
  if ($pauID == -1) {
    print "Could not locate SIL in $phF\n";
    exit;
  }
}
my $ssilID = &locate_ID($phmodF, $ssilS);
if ($ssilID == -1) {
  print "Could not locate $ssilID in $phF\n";
  exit;
}

print "PAU ID: $pauID and SSIL ID: $ssilID\n";

#Create a temporary directory..
#system("mkdir $tmpD");
#system("mkdir $labD");

my @phL = &Get_ProcessedLines($phF);

my $t = 0;
my $logF = $etcD."/tlog.txt";
open(fp_log, ">$logF");

my $trnF  = $tmpD."/phseq_int.data"; #create a file to store the phseq.int file..

for (my $i = 1; $i < $#phL; $i++) {  #ignore the first line...
#for (my $i = 1; $i < 6; $i++) {  #ignore the first line...

  my @wrd1 = &Get_Words($phL[$i]); 
  my $wavid1 = $wrd1[0];
  my $nph1 = $wrd1[1];
  my @wrd2 = &Get_Words($phL[$i+1]);
  my $wavid2 = $wrd2[0];
  my $nph2 = $wrd2[1];

  my $nph = $nph1 + $nph2;
  my $cnt1 = 0;
  my $cnt2 = 0;

  open(fp_trn, ">$trnF");
  print fp_trn "1\n$wavid1 $nph ";
  for (my $j = 2; $j <= $#wrd1; $j++) {
    print fp_trn "$wrd1[$j] ";

    if ($wrd1[$j] != $pauID && $wrd1[$j] != $ssilID) {
      $cnt1++;
    }
  }
  for (my $j = 2; $j <= $#wrd2; $j++) {
    print fp_trn "$wrd2[$j] ";
    if ($wrd2[$j] != $pauID && $wrd2[$j] != $ssilID) {
      $cnt2++;
    }
  }
  print fp_trn "\n";
  close(fp_trn);

  print "Act counts: $cnt1 and $cnt2 \n";
  
  #Estimate the duration of the phone seq..
  my $dt = &Estimate_Dur($nph);

  my $wavid = $wavid1;
  print "processing $i: ($wavid) with no of phones: $nph and estimated dur: $dt \n";
  do_segment($dt, $wavF, $trnF, $wavid, 2);

  my $labF  = $labD."/$wavid.lab";     #create a lab file...
  my $dt1 = process_labfile($labF, $cnt1, $pauID, $ssilID);

  my $estD = $t + $dt1;
  print fp_log "$wavid $t $estD\n";
  $t = $t + $dt1;
  print "END: Duration for prompt $i $wavid as found by edec is: $dt1\n";
  print "****************************** \n\n\n";

}
#handle last utterance.... 
#A normal forced-alignment....
  my @wrd = &Get_Words($phL[$#phL]); 
  my $wavid = $wrd[0];
  my $nph = $wrd[1];
  open(fp_trn, ">$trnF");
  print fp_trn "1\n$wavid $nph ";
  for (my $j = 2; $j <= $#wrd; $j++) {
    print fp_trn "$wrd[$j] ";
  }
  print fp_trn "\n";
  close(fp_trn);
  my $dt = &Estimate_Dur($nph);

  print "processing ($wavid) with no of phones: $nph and estimated dur: $dt \n";
  do_segment($dt, $wavF, $trnF, $wavid, 0); #0 indicates normal forced alignment

  my $labF  = $labD."/$wavid.lab";     #create a lab file...
  my @labL = &Get_ProcessedLines($labF);
  my @tw = &Get_Words($labL[$#labL]);
  my $dt1 = $tw[0];

  my $estD = $t + $dt1;
  print fp_log "$wavid $t $estD\n";
  print "END: Duration for prompt $wavid as found by edec is: $dt1\n";
  print "****************************** \n\n\n";

close(fp_log);

sub process_labfile() {
  my $labF = shift(@_);
  my $cnt = shift(@_);
  my $pauID = shift(@_);
  my $ssilID = shift(@_);
  my @labL = &Get_ProcessedLines($labF);

  print "PAUID: $pauID , SSILID: $ssilID\n";

  my @tw = &Get_Words($labL[$#labL]);
  print "Est. dur. for combined utt: $tw[0] - count for utt1: $cnt\n";
  
  my $tc = 0; 
  my $eid1 = 0;
  my $et1 = 0;
  my $eid2 = 0;
  my $et2 = 0;

  for (my $i = 1; $i <= $#labL; $i++) {
     my @wrd = &Get_Words($labL[$i]);
     my $tim = $wrd[0];
     my $ph = $wrd[2];
     
     if ($ph != $pauID && $ph != $ssilID) {
        $tc++;
       if ($tc == $cnt) {
         $eid1 = $i;
         $et1 = $tim;
         print "est tim1: $et1\n";
       }elsif ($tc == $cnt + 1) {  #catch the first phone after the CNT (required no);
         $eid2 = $i - 1;  #move back by 1 to catch the PAU ending
         last;
       }
     }
  }
  print "eid1: $eid1 and eid2: $eid2\n";
  my @wrd = &Get_Words($labL[$eid2]);
  $et2 = $wrd[0];

  my $et = $et1 + ($et2 - $et1) / 2.0;
  open(fp_lab, ">$labF");
  print fp_lab "#\n";
  for (my $i = 1; $i <= $eid1; $i++) {
    print fp_lab "$labL[$i]\n";
  }
  print fp_lab "$et 125 $pauID\n";
  close(fp_lab);

  return $et;

}

sub do_segment() {

  #INPUT:
  my $dt = shift(@_);  #est time
  my $wavF = shift(@_); #wav file
  my $trnF = shift(@_); #transcription file.
  my $wavid = shift(@_); #wavid ...
  my $ndeF = shift(@_); # 2 or 0

  #OUTPUT: labels to the wavfile, and the corresponding transcription..

  #Chop the wave-file from t to t+dt and dump it in a separate file..
  my $wvE = "raw";
  my $twvF = "tmp_wav";
  my $fwavF = $tmpD."/".$twvF.".".$wvE;
  my $eeT = $t + $dt;
  my $estdir = $ENV{ESTDIR};

  print "$wavchop $wavF $t $eeT $fwavF \n"; #this module handles boundary conditions...
  system("$wavchop $wavF $t $eeT $fwavF"); #this module handles boundary conditions ...
  # You have to make this robust to end of file sub selection
#  print "$estdir/bin/ch_wave $wavF -start $t -end $eeT -otype raw -o $fwavF\n"; #this module handles boundary conditions...
#  system("$estdir/bin/ch_wave $wavF -start $t -end $eeT -otype raw -o $fwavF"); #this module handles boundary conditions ...
                                           #if eeT (est. end time) is more than the length of the file
                                           #then eeT is set to length of the file

  print "wav chop done.. output: $fwavF \n";
  
  #Extract features from the wavefile..
  my $setF = $etcD."/my_spsettings";
  my $wvLF = $etcD."/my_wavelist";
  dump_spsettings($setF, $wvLF, $twvF);

  #Run feature extraction program..
  system("$featE $setF $wvLF");  
  print "$featE $setF $wvLF";  
  
  #File where the output MFCCs features are stored...
  my $ftF = $featD."/".$twvF.".mfcc";

  system("perl scripts/comp_dcep.pl $wvLF $featD mfcc ft 0 0");
    # (0, 0):  delta-cepstrals and delta-delta-cepstrals
  system("perl scripts/scale_feat.pl $wvLF $featD $etcD ft 4");
    # (4): Scaling factor 
  system("cp $featD/$twvF.ft $featD/$wavid.ft");

  #Align the features with the phone sequence..
  #Usage: ./a.out <ph-list.int> <prompt-file> <seq-flag> <feat-dir> <extn> <settings-file> <mod-dir> <nde-flag> <labD>
  #$EHMMDIR/bin/edec ehmm/etc/ph_list.int ehmm/etc/txt.phseq.data.int 1 ehmm/feat ft ehmm/etc/mysp_settings ehmm/mod 0 lab

  print "$palignE $phmodF $trnF 1 $featD ft $setF $etcD $ndeF $labD\n";
  system("$palignE $phmodF $trnF 1 $featD ft $setF $etcD $ndeF $labD");
  
}

sub locate_ID() {

  my $phF = shift(@_);
  my $str = shift(@_);
  my $ustr = uc($str);
  my @phL = &Get_ProcessedLines($phF);
  
  my $rv = -1;
  for (my $i = 3; $i <= $#phL; $i++) { #ignore 3 lines which are header.
    my @wrd = &Get_Words($phL[$i]);
    my $id = $wrd[0];
    my $ph = $wrd[1];
    if ($ph eq $str || $ph eq $ustr) {
      $rv = $id;
      last;
    }
  }
  #if ($rv == -1) {
  #  print "Could not locate $str in $phF\n";
  #  exit;
  #}
  return $rv;
}

sub dump_spsettings() {
  
  my $setF = shift(@_);
  my $wavLF = shift(@_);
  my $wavF = shift(@_);

  open(fp_set, ">$setF");
  print fp_set "WaveDir: ./$tmpD\n";
  print fp_set "HeaderBytes: 0\n";
  print fp_set "SamplingFreq: 16000\n";
  print fp_set "FrameSize: 160\n";
  print fp_set "FrameShift: 80\n";
  print fp_set "Lporder: 12\n";
  print fp_set "CepsNum: 16\n";
  print fp_set "FeatDir: ./feat\n"; 
  print fp_set "Ext: .raw\n";
  close(fp_set);

  open(fp_wav, ">$wavLF");
  print fp_wav "Nooffiles: 1\n";
  print fp_wav "$wavF\n";
  close(fp_wav);

}


sub Estimate_Dur() {
  my $_minAvg = 0.13; 
  my $nph = shift(@_);
  print "No of phones: $nph\n";
  my $rv  = $nph * $_minAvg;
  $rv = $rv + 20; #add 10 seconds to account for any pauses.
  return $rv;
}


sub Make_SingleSpace() {
   chomp(${$_[0]});
   ${$_[0]} =~ s/[\s]+$//;
   ${$_[0]} =~ s/^[\s]+//;
   ${$_[0]} =~ s/[\s]+/ /g;
   ${$_[0]} =~ s/[\t]+/ /g;
}

sub Check_FileExistence() {
  my $inF = shift(@_); 
  if (!(-e $inF)) { 
    print "Cannot open $inF \n";
    exit;
  } 
  return 1;
}

sub Get_Lines() {
  my $inF = shift(@_); 
  &Check_FileExistence($inF);
  open(fp_llr, "<$inF");
  my @dat = <fp_llr>;
  close(fp_llr);
  return @dat;
}

sub Get_Words() {
  my $ln = shift(@_);
  &Make_SingleSpace(\$ln);
  my @wrd = split(/ /, $ln);
  return @wrd;
}

sub Get_ProcessedLines() {
  my $inF = shift(@_);
  &Check_FileExistence($inF);
  open(fp_llr, "<$inF");
  my @dat = <fp_llr>;
  close(fp_llr);

  my @nd;
  for (my $i = 0; $i <= $#dat; $i++) {
     my $tl = $dat[$i];
     &Make_SingleSpace(\$tl);
     $nd[$i]  = $tl;
  }
  return @nd;
}
