#!/usr/bin/perl
use strict;

my $nargs = $#ARGV + 1;
my $tmpD = "tmp";

if($nargs != 2) {
	print "Usage: perl file.pl <promptfile> <txt.phseq.data>\n";
	exit;
}

my $prmF = $ARGV[0];
my $phseqF = $ARGV[1];

my $ESTDIR = `echo \$ESTDIR`;
if ($ESTDIR eq "") {
   print "DEFINE ESTDIR in the shell to proceed.. \n";
   exit;
}else {
  &Make_SingleSpace(\$ESTDIR);
  print "SPEECH TOOLS DIR Found as: $ESTDIR \n";
}


open(fp_o, ">$phseqF");

my @prm = &Get_ProcessedLines($prmF);
for (my $i = 0; $i <= $#prm; $i++) {
   my $ln = $prm[$i];
   &Make_SingleSpace(\$ln);
   $ln =~ s/[\)]$//;
   $ln =~ s/^[\(]//;
   my @wrd = &Get_Words($ln);
   print "Processing $wrd[0]\n";
   my $rln;
   for (my $k = 1; $k <= $#wrd; $k++) {
     $rln = $rln." ".$wrd[$k];
   }
   &Make_SingleSpace(\$rln);

   my $tmpF = $tmpD."/blkf.pmt";
   my $tmpU = $tmpD."/blkf.utt";
   my $tmpS = $tmpD."/blkf.seq";

   open(fp_tmp, ">$tmpF");
   print fp_tmp "( blkf $rln ) \n";
   close(fp_tmp);

   system("$ESTDIR/../festival/bin/festival -b scripts/phseq.scm \'\(genutt \"$tmpF\" \"$tmpU\"\)\'");
   system("$ESTDIR/../festival/bin/festival -b scripts/phseq.scm \'\(phseq \"$tmpF\" \"$tmpS\"\)\'");

   my @ts = &Get_ProcessedLines($tmpS);
   my $rl = $ts[0];
   my @rw = &Get_Words($rl);
   print fp_o "$wrd[0] ";
   for (my $k = 1; $k <= $#rw; $k++) {
     print fp_o "$rw[$k] ";
   }
   print fp_o "\n";

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

