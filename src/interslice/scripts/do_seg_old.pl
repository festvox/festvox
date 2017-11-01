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

my $nargs = $#ARGV + 1;
if ($nargs != 3) {
  print "Usage: perl file.pl <ph-seq-int> <ph-mod-file> <large-wave-file>\n";
  print "EHMM model is assumed to be under etc/model101.txt\n";
  exit;
}

my $phF = $ARGV[0];
my $phmodF = $ARGV[1];
my $wavF = $ARGV[2];

#Create a temporary directory..
system("mkdir $tmpD");
system("mkdir $labD");

my @phL = &Get_ProcessedLines($phF);

my $t = 0;
my $logF = $etcD."/tlog.txt";
open(fp_log, ">$logF");

#for (my $i = 1; $i <= $#phL; $i++) {  #ignore the first line...
for (my $i = 1; $i <= 5; $i++) {  #ignore the first line...

  my $nl = $phL[$i];
  if ($i < $#phL) {
    $nl = $nl." ".$phL[$i+1];
  }

  my @wrd1 = &Get_Words($phL[$i]); 
  my $wavid1 = $wrd1[0];
  my $nph1 = $wrd1[1];

  my $wav
  
  #Estimate the duration of the phone seq..
  my $dt = &Estimate_Dur($nph);

  print "processing $i with no of phones: $nph and estimated dur: $dt \n";

  #Chop the wave-file from t to t+dt and dump it in a separate file..
  my $wvE = "raw";
  my $twvF = "tmp_wav";
  my $fwavF = $tmpD."/".$twvF.".".$wvE;
  my $eeT = $t + $dt;

  print "$wavchop $wavF $t $eeT $fwavF \n"; #this module handles boundary conditions...
  system("$wavchop $wavF $t $eeT $fwavF"); #this module handles boundary conditions...

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

  system("perl bin/comp_dcep.pl $wvLF $featD mfcc ft 0 0");
    # (0, 0):  delta-cepstrals and delta-delta-cepstrals
  system("perl bin/scale_feat.pl $wvLF $featD $etcD ft 4");
    # (4): Scaling factor 
  system("cp $featD/$twvF.ft $featD/$wavid.ft");

  #Align the features with the phone sequence..
  my $labF  = $labD."/$wavid.lab";     #create a lab file...
  my $trnF  = $tmpD."/phseq_int.data"; #create a file to store the phseq.int file..
   
  #dump the transcription into the file...
  open(fp_trn, ">$trnF");
  print fp_trn "1 \n $phL[$i]\n";
  #print "1\n$phL[$i]\n";
  close(fp_trn);

  #Usage: ./a.out <ph-list.int> <prompt-file> <seq-flag> <feat-dir> <extn> <settings-file> <mod-dir> <nde-flag> <labD>
  #$EHMMDIR/bin/edec ehmm/etc/ph_list.int ehmm/etc/txt.phseq.data.int 1 ehmm/feat ft ehmm/etc/mysp_settings ehmm/mod 0 lab

  system("$palignE $phmodF $trnF 1 $featD ft $setF $etcD 2 $labD");
  print "$palignE $phmodF $trnF 1 $featD ft $setF $etcD 2 $labD\n";
  
  
  #Obtain the duration from the last line of the label file.. 
  my @dl = &Get_ProcessedLines($labF);
  my $tl = $dl[$#dl];
  my @tw = &Get_Words($tl);
  my $td1 = $tw[0];
  
  my $estD = $t + $td1;
  print fp_log "$wavid $t $estD\n";

  $t = $t + $td1;
  print "END: Duration for prompt $i $wavid as found by edec is: $td1\n";
  print "****************************** \n\n\n";
}
close(fp_log);

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
  my $_minAvg = 0.09; 
  my $nph = shift(@_);
  print "No of phones: $nph\n";
  my $rv  = $nph * $_minAvg;
  $rv = $rv + 5; #add 5 seconds pause...
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
