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
##  Calculating Statistics of Global Variance                        ##
##     of Mel-Cepstrum Sequence                                      ##
##                                                                   ##
#######################################################################

set work_dir = $argv[1]	# work directory
set src_dir = $argv[2]	# src directory
set list = $argv[3]	# file list
set stlbl = $argv[4]	# statistics file label

##############################
set lbg = $src_dir/vq/lbg
set vqlbl = $src_dir/vqlbl/vqlbl
set cov2dia = $src_dir/cov2dia/cov2dia
##############################
set var_dir = $work_dir/mcep
set order = 24
##############################

set no = 1
foreach lbl (`cat $list`)
	set varf = $var_dir/$lbl.var
	if (!(-r $varf)) then
		echo ERROR > DUMMY-ERROR-LOG
	endif
	if ($no == 1) then
		mv $varf $stlbl.data
	else
		mv $stlbl.data $stlbl.data.tmp
		cat $stlbl.data.tmp $varf > $stlbl.data
		rm -f $stlbl.data.tmp $varf
	endif
	@ no = $no + 1
end
# error check
if (-r DUMMY-ERROR-LOG) then
	exit
endif

if (!(-r $stlbl.mean) || !(-r $stlbl.dcov)) then
	$lbg \
		-nmsg \
		-dim $order \
		-cls 1 \
		$stlbl.data \
		$stlbl.cls
	mv $stlbl.cls1.mat $stlbl.mean

	$vqlbl \
		-nmsg \
		-dim $order \
		-cbookfile $stlbl.mean \
		-covfile $stlbl.cov \
		$stlbl.data \
		$stlbl.lbl

	$cov2dia \
		-nmsg \
		-dim $order \
		$stlbl.cov \
		$stlbl.dcov

	rm -f $stlbl.cov
	rm -f $stlbl.lbl
endif
rm -f $stlbl.data
if (!(-r $stlbl.mean) || !(-r $stlbl.dcov)) then
	echo ERROR > DUMMY-ERROR-LOG
endif
