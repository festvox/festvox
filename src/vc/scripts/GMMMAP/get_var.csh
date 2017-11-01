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
##  Calculating Global Variance of Mel-Cepstrum Sequence             ##
##                                                                   ##
#######################################################################

set work_dir = $argv[1]	# work directory
set src_dir = $argv[2]	# src directory
set list = $argv[3]	# file list

##############################
set lbg = $src_dir/vq/lbg
set vqlbl = $src_dir/vqlbl/vqlbl
set cov2dia = $src_dir/cov2dia/cov2dia
##############################
set mcep_dir = $work_dir/mcep
set order = 24
##############################

foreach flbl (`cat $list`)
	set mcepf = $mcep_dir/$flbl.mcep
	set varf = $mcep_dir/$flbl.var

	if (!(-r $varf) && (-r $mcepf)) then
		$lbg \
			-nmsg \
			-dim $order \
			-cls 1 \
			$mcepf \
			$varf.cls \

		$vqlbl \
			-nmsg \
			-dim $order \
			-cbookfile $varf.cls1.mat \
			-covfile $varf.cov \
			$mcepf \
			$varf.lbl

		$cov2dia \
			-nmsg \
			-dim $order \
			$varf.cov \
			$varf

		rm -f $varf.cls1.mat
		rm -f $varf.cov
		rm -f $varf.lbl
	endif
	if (!(-r $varf)) then
		echo ERROR > DUMMY-ERROR-LOG
	endif
end
