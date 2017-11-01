#!/usr/local/bin/perl
use strict;
###########################################################################
##                                                                       ##
##                                                                       ##
##              Carnegie Mellon University, Pittsburgh, PA               ##
##                      Copyright (c) 2004-2005                          ##
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
##          Date   :  June 2005                                          ##
##                                                                       ##
###########################################################################

my $nargs = $#ARGV + 1;
if ($nargs != 3) {
  print "Usage: perl file.pl <file1> <prompt-file> <prefix>\n";
  exit;
}

my $inF = $ARGV[0];
my $prmF = $ARGV[1];
my $pfx = $ARGV[2];;

my @ln = &Get_ProcessedLines($inF);
my @para;
my $p = 0;

for (my $i = 0; $i <= $#ln; $i++) {
  my @wrd = &Get_Words($ln[$i]);
  my $nw = $#wrd + 1;
  if ($nw == 0) { 
    #print "$para[$p] \n ************\n";
    $p = $p + 1; 
  }else {
    $para[$p] = $para[$p]." ".$ln[$i];
  }
}

my @mpara;
my $z = 0;
for (my $j = 0; $j <= $#para; $j++) {
  my @wrd = &Get_Words($para[$j]);
  my $nw = $#wrd + 1;
  if ($nw == 0) {
  } else {
    $mpara[$z] = $para[$j];
    $z++;
  }
}
@para = @mpara;

#Divide the paragraphs into sub-paras with a length of around 100 words!
#Step taken to assist faster running of edec, and also minimize the memory
#requirements time!

my $k = 0;
my @modP;
my $maxW = 100;

for (my $j = 0; $j <= $#para; $j++) {
  my @wrd = &Get_Words($para[$j]);
  my $nw = $#wrd + 1;
  if ($nw <= $maxW) {
    $modP[$k] = $para[$j];
    $k++;
  } else {
    $modP[$k] = $wrd[0];
    my $flag = 0;

    for (my $z = 1; $z < $#wrd; $z++) {
      if ($z % $maxW == 0) { 
        $flag = 1;
      }

      if ($flag == 0) {
        $modP[$k] = $modP[$k]." ".$wrd[$z];
      } else {
        my $cw = $wrd[$z];
        my $nw = $wrd[$z+1];
        my @lcw = split(//,$cw);
        my $lch = $lcw[$#lcw];

        my @lnw = split(//,$nw);
	my $nch = $lnw[0];
        
        #print "LCH: $lch NCH: $nch NOW: $#lcw \n";   
        #if ($nch =~ m/[A-Z]/) {
        # print "UPPER CASE \n";
        #}
        if ($lch eq "." && $nch =~ m/[A-Z]/ && $#lcw >= 2) { 
           #check whether the current word has a period.
           #whether the next word starts with a capital
           #whether the current word has more than 2 characters.
           
          $modP[$k] = $modP[$k]." ".$wrd[$z];
          $k++;
          $flag = 0;
          #print "ENTER THIS CONDITION.... \n";
           
        } else {
          $modP[$k] = $modP[$k]." ".$wrd[$z];
        }
 
      }
      
    }
    $modP[$k] = $modP[$k]." ".$wrd[$#wrd];
    $k++;
  }

}
my @para1 = @modP;

my $min = 1.0e+35;
my $max = -1.0e+35;
my $avg = 0;

open(fp_prm, ">$prmF");

for (my $j = 0; $j <= $#para1; $j++) {
  my @wrd = &Get_Words($para1[$j]);
  my $nw = $#wrd + 1;
  print "Para $j - $nw\n";
  if ($min > $nw) { $min = $nw; }
  if ($max < $nw) { $max = $nw; }
  $avg = $avg + $nw;

  my $sid = &Get_ID($j);
  my $cln = $para1[$j];
  &Make_SingleSpace(\$cln);
  &Handle_Quote(\$cln);
  print fp_prm "( $pfx\_$sid \" $cln \") \n";
  
}
close(fp_prm);
$avg = $avg / ($#para1 + 1);
$avg = int($avg);

print "Min / Max: $min / $max ; Avg: $avg\n";

sub Get_ID() {

  my $id = shift(@_);
  my $rv = "";
  if ($id < 10) {
    $rv = "000";
  }elsif ($id < 100) {
    $rv = "00";
  } elsif ($id < 1000) {
    $rv = "0";
  }
  $rv = $rv.$id;
  return $rv;
}

sub Handle_Quote() {
   chomp(${$_[0]});
   ${$_[0]} =~ s/[\"]/\\"/g;
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
