#! /bin/csh -f
#######################################################################
##                                                                   ##
##            Nagoya Institute of Technology, Aichi, Japan,          ##
##       Nara Institute of Science and Technology, Nara, Japan       ##
##                                and                                ##
##             Carnegie Mellon University, Pittsburgh, PA            ##
##                      Copyright (c) 2003-2004                      ##
##                        All Rights Reserved.                       ##
##                                                                   ##
##  Permission is hereby granted, free of charge, to use and         ##
##  distribute this software and its documentation without           ##
##  restriction, including without limitation the rights to use,     ##
##  copy, modify, merge, publish, distribute, sublicense, and/or     ##
##  sell copies of this work, and to permit persons to whom this     ##
##  work is furnished to do so, subject to the following conditions: ##
##                                                                   ##
##    1. The code must retain the above copyright notice, this list  ##
##       of conditions and the following disclaimer.                 ##
##    2. Any modifications must be clearly marked as such.           ##
##    3. Original authors' names are not deleted.                    ##
##                                                                   ##    
##  NAGOYA INSTITUTE OF TECHNOLOGY, NARA INSTITUTE OF SCIENCE AND    ##
##  TECHNOLOGY, CARNEGIE MELLON UNiVERSITY, AND THE CONTRIBUTORS TO  ##
##  THIS WORK DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,  ##
##  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, ##
##  IN NO EVENT SHALL NAGOYA INSTITUTE OF TECHNOLOGY, NARA           ##
##  INSTITUTE OF SCIENCE AND TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, ##
##  NOR THE CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR      ##
##  CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM   ##
##  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,  ##
##  NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN        ##
##  CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.         ##
##                                                                   ##
#######################################################################
##                                                                   ##
##          Author :  Tomoki Toda (tomoki@ics.nitech.ac.jp)          ##
##          Date   :  June 2004                                      ##
##                                                                   ##
#######################################################################
##                                                                   ##
##  Voice Conversion with GMM                                        ##
##  (Fast Version for Festival Speech Synthesis System)              ##
##  This version is used in embedded GMM transform for CG voices     ##
##  It generates a converted Track file, and doesn't do full MLSA    ##
#######################################################################

set src_dir = $argv[1]		# src directory
set paramlist = $argv[2]	# parameter file list
set inf = $argv[3]		# input wav file (never used)
set inuttf = $argv[4]		# input utt file
set cgparams = $argv[5]		# CG generated params
set outf = $argv[6]		# output track file

# error check
if (!(-r $paramlist)) then
	echo "Error: Can't find "$paramlist
	exit
endif
if (!(-r $inuttf)) then
	echo "Error: Can't find "$inuttf
	exit
endif

##############################
set utt2f0 = $src_dir/utt2f0/utt2f0
set analysis = $src_dir/analysis/analysis
set extdim = $src_dir/extdim/extdim
set delta = $src_dir/mlpg/delta
set gmmmap = $src_dir/gmmmap/gmmmap
set mlpg = $src_dir/mlpg/mlpg_vit
set jdmat = $src_dir/jdmat/jdmat
set f0map = $src_dir/f0map/f0map
set synthesis = $src_dir/synthesis/synthesis
##############################
set order = 24
@ dim = $order + 1
@ dim2 = $order + 2
@ jdim = $order + $order
##############################

# Parameter Files
set param = `cat $paramlist`
set win = $param[1]
set clsnum = $param[2]
set gmmf = $param[3]
set vmf = $param[4]
set vvf = $param[5]
set of0sts = $param[6]
set tf0sts = $param[7]

# F0 Extraction
$ESTDIR/bin/ch_track -otype ascii -c 0 $cgparams >$outf.org.f0
# F0 Conversion
$f0map \
	-nmsg \
	-log \
	-ostfile $of0sts \
	-tstfile $tf0sts \
	$outf.org.f0 \
	$outf.conv.f0

# Spectral Extraction
$ESTDIR/bin/ch_track -otype ascii $cgparams | \
awk '{for (i=2; i<=NF; i+=1) printf("%s\n",$i); }' | \
perl $FESTVOXDIR/src/clustergen/a2d.pl >$outf.org.mcep

# Power
$extdim \
	-nmsg \
	-dim $dim \
	-ed 0 \
	$outf.org.mcep \
	$outf.org.mc0
# Spectral Envelope
$extdim \
	-nmsg \
	-dim $dim \
	-sd 1 \
	$outf.org.mcep \
	$outf.org.mcep24
# Dynamic Feature
$delta \
	-nmsg \
	-jnt \
	-dynwinf $win \
	-dim $order \
	$outf.org.mcep24 \
	$outf.org.sdmcep
# PDF Sequence Estimation Using GMM
$gmmmap \
	-nmsg \
	-vit \
	-dia \
	-xdim $jdim \
	-ydim $jdim \
	-gmmfile $gmmf \
	-clsseqfile $outf.conv.cseq \
	-wseqfile $outf.conv.wseq \
	-covfile $outf.conv.cov \
	$outf.org.sdmcep \
	$outf.conv.mseq
# Spectral Sequence Estimation Based on ML Using Global Variance
$mlpg \
	-nmsg \
	-fast \
	-sm \
	-dia \
	-dim $jdim \
	-clsnum $clsnum \
	-dynwinf $win \
	-vmfile $vmf \
	-vvfile $vvf \
	$outf.conv.cseq \
	$outf.conv.wseq \
	$outf.conv.mseq \
	$outf.conv.cov \
	$outf.conv.mcep24
# Spectrum and Power
$jdmat \
	-nmsg \
	-dim1 1 \
	-dim2 $order \
	$outf.org.mc0 \
	$outf.conv.mcep24 \
	$outf.conv.mcep

#Adding F0 too 
perl $FESTVOXDIR/src/clustergen/a2d.pl < $outf.conv.f0 > $outf.conv.f0bin 

$jdmat \
	-nmsg \
	-dim1 1 \
	-dim2 $dim \
	$outf.conv.f0bin \
	$outf.conv.mcep \
	$outf.conv.bintrack

# Convert the output track back to EST format

perl $FESTVOXDIR/src/clustergen/d2a.pl < $outf.conv.bintrack | \
awk '{if (NR%'$dim2' == 0 ){printf("%s\n", $1);} else { printf ("%s ",$1);}}' > $outf.conv.rawtrack


$ESTDIR/bin/ch_track -itype ascii -s 0.005 -otype est $outf.conv.rawtrack > $outf


# Removed Temporaly Files
rm -f $outf.org.f0 $outf.org.mc0
rm -f $outf.org.mcep $outf.org.mcep24 $outf.org.sdmcep
rm -f $outf.conv.cseq $outf.conv.wseq $outf.conv.mseq
rm -f $outf.conv.cov $outf.conv.mcep24 $outf.conv.bintrack
rm -f $outf.conv.rawtrack $outf.conv.f0bin
