#!/usr/bin/perl
use strict;
use scripts::syllabifier;

my $nargs = $#ARGV + 1;

if($nargs != 2) {
	print "Usage: perl file.pl <promptfile> <txt.phseq.data>\n";
	exit;
}

my $prmF = $ARGV[0];
my $phseqF = $ARGV[1];

open(fp_o, ">$phseqF");

my @prm = &Get_ProcessedLines($prmF);
for (my $i = 0; $i <= $#prm; $i++) {
   my $ln = $prm[$i];
   &Make_SingleSpace(\$ln);
   $ln =~ s/[\)]$//;
   $ln =~ s/^[\(]//;
   $ln =~ s/[\"]$//;
   $ln =~ s/^[\"]//;
   $ln =~ s/[\"]//g;
   $ln =~ s/[\;]//g;
   $ln =~ s/[\.]//g;
   $ln =~ s/[\,]//g;
   $ln =~ s/[\!]//g;
   $ln =~ s/[?]//g;
   #print "$ln \n";
   my @wrd = &Get_Words($ln);
   print "Processing $wrd[0]\n";
   print fp_o "$wrd[0] SIL ";
   for (my $j = 1; $j <= $#wrd; $j++) {
     my $cw = $wrd[$j]; 
     #$cw =~ s/[\.]$//;
     #$cw =~ s/[\,]$//;
     #$cw =~ s/[?]$//;
     #$cw =~ s/[\!]$//;
     my $ph = &Phonification($cw);
     print fp_o "$ph ssil ";
   }
   print fp_o "SIL\n";
}
close(fp_o);

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

