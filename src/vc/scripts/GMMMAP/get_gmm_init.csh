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
##  Calculating Initial Parameters of GMM                            ##
##                                                                   ##
#######################################################################

set src_dir = $argv[1]	# src directory
set clsnum = $argv[2]	# the number of classes
set jnt = $argv[3]	# joint features
set cblbl = $argv[4]	# codebook file label
set initlbl = $argv[5]	# initial parameter file label

##############################
set vqlbl = $src_dir/vqlbl/vqlbl
##############################
set order = 24
@ jdim = $order + $order + $order + $order
##############################

if (!(-r $initlbl"1.cov")) then
	$vqlbl \
		-nmsg \
		-dim $jdim \
		-cbookfile $cblbl"1.mat" \
		-weightfile $initlbl.tmp.wght \
		-covfile $initlbl"1.cov" \
		$jnt \
		$initlbl.tmp.lbl
	rm -f $initlbl.tmp.wght $initlbl.tmp.lbl
endif
if (!(-r $initlbl"1.cov")) then
	echo ERROR > DUMMY-ERROR-LOG
endif

if (!(-r $initlbl$clsnum.cov) || !(-r $initlbl$clsnum.wght)) then
	$vqlbl \
		-nmsg \
		-dim $jdim \
		-cbookfile $cblbl$clsnum.mat \
		-weightfile $initlbl$clsnum.wght \
		-covfile $initlbl$clsnum.cov \
		$jnt \
		$initlbl.tmp.lbl
	rm -f $initlbl.tmp.lbl
endif
if (!(-r $initlbl$clsnum.cov) || !(-r $initlbl$clsnum.wght)) then
	echo ERROR > DUMMY-ERROR-LOG
endif
