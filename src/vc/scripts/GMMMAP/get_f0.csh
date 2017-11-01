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
##  F0 Extraction                                                    ##
##                                                                   ##
#######################################################################

set work_dir = $argv[1]	# work directory
set src_dir = $argv[2]	# src directory
set list = $argv[3]	# file list

##############################
set tempo = $src_dir/tempo/tempo
##############################
set wav_dir = $work_dir/wav
set f0_dir = $work_dir/f0
set uf0 = 300
set lf0 = 70
##############################

foreach flbl (`cat $list`)
	set wavf = $wav_dir/$flbl.wav
	set f0f = $f0_dir/$flbl.f0
        echo $f0f
	if (!(-r $f0f)) then
#		if (-r $tempo) then
#			echo $tempo -nmsg -uf0 $uf0 -lf0 $lf0 $wavf $f0f
#			$tempo -nmsg -uf0 $uf0 -lf0 $lf0 $wavf $f0f
#		else
			mkdir -p $f0f:h
			$ESTDIR/bin/pda \
				$wavf \
				-o $f0f \
				-fmin $lf0 \
				-fmax $uf0 \
				-L
#		endif
	endif
	if (!(-r $f0f)) then
		echo ERROR > DUMMY-ERROR-LOG
	endif
end
