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
##  Time Alignment with DTW                                          ##
##                                                                   ##
#######################################################################

set mcep_dir = $argv[1]	# mel-cep directory
set dtw_dir = $argv[2]	# DTW directory
set src_dir = $argv[3]	# src directory
set olist = $argv[4]	# original file list
set tlist = $argv[5]	# target file list
set jnt = $argv[6]	# joint feature file

##############################
set dtw = $src_dir/dtw/dtw
##############################
set order = 24
##############################
@ dim = $order + $order
@ ldim = $dim - 1

set no = 1
set tlbl = `cat $tlist`
foreach olbl (`cat $olist`)
	set orgf = $mcep_dir/$olbl.exmcep
	set tarf = $mcep_dir/$tlbl[$no].exmcep
	set twff = $dtw_dir/$olbl.twf
	set dtwf = $dtw_dir/$olbl.jmcep
	$dtw \
		-nmsg \
		-fixed \
		-dim $dim \
		-ldim $ldim \
		-intwf $twff \
		-dtwotfile $dtwf \
		$orgf \
		$tarf
	rm -f $orgf $tarf $twff
	if (!(-r $dtwf)) then
		echo ERROR > DUMMY-ERROR-LOG
	endif

	if ($no == 1) then
		mv $dtwf $jnt
	else
		mv $jnt $jnt.tmp
		cat $jnt.tmp $dtwf > $jnt
		rm -f $jnt.tmp $dtwf
	endif

	@ no = $no + 1
end
