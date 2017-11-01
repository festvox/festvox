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
##  (Standard Version for Using Natural Speech as Input)             ##
##                                                                   ##
#######################################################################

set src_dir = $argv[1]		# src directory
set paramlist = $argv[2]	# parameter file list
set inf = $argv[3]		# input wav file
set outf = $argv[4]		# output wav file

# error check
if (!(-r $paramlist)) then
	echo "Error: Can't find "$paramlist
	exit
endif
if (!(-r $inf)) then
	echo "Error: Can't find "$inf
	exit
endif

##############################
set tempo = $src_dir/tempo/tempo
set analysis = $src_dir/analysis/analysis
set extdim = $src_dir/extdim/extdim
set delta = $src_dir/mlpg/delta
set gmmmap = $src_dir/gmmmap/gmmmap
set mlpg = $src_dir/mlpg/mlpg_vit
set jdmat = $src_dir/jdmat/jdmat
set f0map = $src_dir/f0map/f0map
set synthesis = $src_dir/synthesis/synthesis
##############################
set uf0 = 600
set lf0 = 70
set order = 24
@ dim = $order + 1
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
if (-r $tempo) then
	$tempo \
		-nmsg \
		-uf0 $uf0 \
		-lf0 $lf0 \
		$inf \
		$outf.org.f0
else
	mkdir -p $outf:h
	$ESTDIR/bin/pda \
		$inf \
		-o $outf.org.f0 \
		-fmin $lf0 \
		-fmax $uf0 \
		-L
endif
# F0 Conversion
$f0map \
	-nmsg \
	-log \
	-ostfile $of0sts \
	-tstfile $tf0sts \
	$outf.org.f0 \
	$outf.conv.f0

# Spectral Extraction
$analysis \
	-nmsg \
	-mcep \
	-pow \
	-order $order \
	$inf \
	$outf.org.mcep
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

# Speech Synthesis
$synthesis \
	-nmsg \
	-order $order \
	-rmcepfile $outf.org.mcep \
	$outf.conv.f0 \
	$outf.conv.mcep \
	$outf

# Removed Temporaly Files
rm -f $outf.org.f0 $outf.org.mc0
rm -f $outf.org.mcep $outf.org.mcep24 $outf.org.sdmcep
rm -f $outf.conv.f0 $outf.conv.cseq $outf.conv.wseq $outf.conv.mseq
rm -f $outf.conv.cov $outf.conv.mcep24 $outf.conv.mcep
